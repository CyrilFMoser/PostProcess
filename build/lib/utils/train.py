from torch.utils.data import  DataLoader
import torch.cuda
from data.dataloader.scannetppv2_dataloader import ScanNetPPV2DataSet
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time
from external.PointTransformerV3.model import PointTransformerV3
from utils.ioumetric import IoUMetric
import torch.nn.functional as F
from utils.neighbor_voting import neighbor_voting
from datetime import datetime
import gc
import sys
import argparse
from collections import deque

N_EPOCHS = 300
CHECKPOINT_DIR = "/iopsstor/scratch/cscs/cymoser/ckpt"

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--scene_root",type=str,default="/users/cymoser/storage/datasets/GaussianWorld")
    argparser.add_argument("--metadata_root",type=str,default="/users/cymoser/storage/datasets/GaussianWorld/metadata")
    argparser.add_argument("--load_chkpt",default=False,action="store_true")
    argparser.add_argument("--chkpt_path",type=str,default="/cluster/home/cymoser/cymoser/postprocess/ckpt/medium_skip52_classes_step_6118_epoch_22.pt")
    argparser.add_argument("--shuffle",action="store_true",default=False,help="Randomly shuffle class order. If False it will sort them on a scene level")
    argparser.add_argument("--class_subset",type=int,default=100)
    argparser.add_argument("--model_typ",type=str,default="medium_skip")

    args = argparser.parse_args()
    scene_root = args.scene_root
    metadata_root = args.metadata_root
    model_typ = args.model_typ
    load = args.load_chkpt
    shuffle_classes = args.shuffle
    class_subset = args.class_subset
    train_dataset = ScanNetPPV2DataSet(
         scene_root, metadata_root, device,mode="train",subset=100000,shuffle_classes=shuffle_classes,class_subset=class_subset)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, collate_fn=train_dataset.collate_fn)
    
    val_filtered_scene_names = None#["09c1414f1b","578511c8a9"] # scenes that are currently too large for the gpu

    val_dataset = ScanNetPPV2DataSet(
         scene_root, metadata_root, device,mode="validation",subset=100000,filtered_scenes=val_filtered_scene_names,shuffle_classes=shuffle_classes,class_subset=class_subset)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, collate_fn=val_dataset.collate_fn)
    
    num_channels = train_dataset.num_channels
    if 100 != class_subset:
        model_typ += f"_{class_subset}_classes"
    else:
        model_typ += "_fullclasses"
    if shuffle_classes:
        model_typ += "shuffled"
    config = build_model(num_channels,model_typ,device,class_subset)
    if load:
        path = args.chkpt_path
        d:dict = torch.load(path,map_location=device)
        for key in ["model","optimizer","scheduler"]:
            new_key = key + "_state"
            if new_key in d.keys():
                config[key].load_state_dict(d[new_key])
        config["epoch"] = d["epoch"]
        config["global_step"] = d["global_step"]
        for state in config["optimizer"].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    # summary(model, input_data=(x_dict,))  # 1670470
    config["model"].to(device)
    config["class_subset"] = class_subset
    config["num_classes"] = 100
    train(train_dataloader,val_dataloader, config)


def train(train_dataloader: DataLoader,val_dataloader:DataLoader, config:dict):
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    model_name = config["name"]
    val_every = config["val_every"]
    ckpt_every = config["ckpt_every"]
    label_map = config["label_map"]
    excluded_classes= config["excluded_classes"]
    last_scheduler_step = config["last_scheduler_step"]
    start_epoch = config["epoch"]
    class_subset = config["class_subset"] # size of subset of classes per scene
    num_classes = config["num_classes"] # number of all classes available
    
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_name}_epoch-{start_epoch}"

    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    model.train()
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)

    

    epoch_train_iou = IoUMetric(num_classes,class_subset,excluded_classes)
    epoch_base_iou = IoUMetric(num_classes,class_subset,excluded_classes)
    loss_window = deque(maxlen=100)

    global_step = config["global_step"]
    best_val_loss = float("inf")
    print(f"Start training with {num_params} number of parameters")
    for epoch in range(start_epoch,N_EPOCHS):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9)
        epoch_loss = 0.0
        epoch_train_iou.reset()
        epoch_base_iou.reset()
        for batch_id, data in pbar:
            allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            reserved = torch.cuda.memory_reserved() / 1024**2    # in MB
            print(f"Iteration {batch_id}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            batch_train_iou = IoUMetric(num_classes,class_subset,excluded_classes)
            batch_base_iou = IoUMetric(num_classes,class_subset,excluded_classes)

            gs_valid_mask = data["valid_mask"]
            ignore_index = data["ignore_index"].to(gs_valid_mask.device)
            gs_valid_feat_mask = data["valid_feat_mask"]
            full_sorted_mask = data["full_sorted_mask"]

            # --- Base mIoU Computation ---
            with torch.no_grad():
                base_logits = data["feat"][:,:class_subset]
                base_logits[:] = base_logits[:,data["inv_perm"][0]] # shuffle back to original positions. Assumes all batches use the same ordering
                # Get the base predictions
                base_probs,base_pred = torch.max(torch.sigmoid(base_logits),dim=1)
                #base_pred[base_probs < 0.1] = ignore_index
                
                # neighbor voting to get from GS -> PC
                pc_segment = data["pc_segment"]
                pc_valid_mask = pc_segment!=ignore_index 
                pc_query = data["pc_coord"][pc_valid_mask] 
                pc_base_pred = neighbor_voting(data,base_pred,pc_query,gs_valid_feat_mask)
                # Filter out the GT labels that are ignored, and the points that got voted to ignore index            
                pc_segment_valid = pc_segment[pc_valid_mask]
                batch_base_iou.update(pc_base_pred,pc_segment_valid,full_sorted_mask)
                epoch_base_iou.combine(batch_base_iou)
            #  --- Run the model ---
            optimizer.zero_grad()
            new_feat = model(data)
            ordered_feat = new_feat.feat[:,data["inv_perm"][0]] # shuffle logits back to original positions
            valid_new_feat = ordered_feat[gs_valid_mask]
            gs_labels = data["segment"]
            gs_valid_labels = gs_labels[gs_valid_mask]
            #one_hot = F.one_hot(gs_valid_labels,num_classes=num_classes).float()
            output = loss(valid_new_feat,gs_valid_labels)

            output.backward()
            optimizer.step()

            epoch_loss += output.item()
            loss_window.append(output.item())
            # Show running average loss in tqdm
            pbar.set_description(f"Epoch {epoch} | Loss {output.item() :.4f}")
            writer.add_scalar("train/loss", output.item(), global_step)
            writer.add_scalar("epoch/loss", sum(loss_window)/ len(loss_window), global_step)
            # --- Get mIoU ---

            # Get the newly produced predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            new_pred = torch.argmax(ordered_feat,dim=-1)
            # Neighbor voting to get from GS -> PC
            pc_pred = neighbor_voting(data,new_pred,pc_query)
            batch_train_iou.update(pc_pred, pc_segment_valid,full_sorted_mask)
            batch_train_mIoU_val = batch_train_iou.compute()
            writer.add_scalar("train/mIoU",batch_train_mIoU_val , global_step)
            writer.add_scalar("train/acc",batch_train_iou.compute_acc(),global_step)
            # Get Delta mIoU
            batch_base_mIoU_val = batch_base_iou.compute()
            batch_delta_mIoU_val =  batch_train_mIoU_val-batch_base_mIoU_val
            writer.add_scalar("train/delta_mIoU",batch_delta_mIoU_val,global_step)
            writer.add_scalar("train/base/mIoU",batch_base_mIoU_val,global_step)
            writer.add_scalar("train/base/acc",batch_base_iou.compute_acc(),global_step)

            if batch_base_mIoU_val > 0 :
                batch_rel_mIoU_val = batch_train_mIoU_val/batch_base_mIoU_val
            else:
                batch_rel_mIoU_val = 0
            writer.add_scalar("train/relative_mIoU",batch_rel_mIoU_val,global_step)
            # Accumulate per epoch mIoU
            epoch_train_iou.combine(batch_train_iou)

            postfix_dict = {
                "mIoU": f"{batch_train_mIoU_val:.3f}",
                "base": f"{batch_base_mIoU_val:.3f}",
                "ΔmIoU": f"{batch_delta_mIoU_val:.3f}",
            }
            checkpoint_saved = False
            # ---- VALIDATION ----
            if global_step % val_every == 0 and global_step > 0:
                val_loss,val_miou = validate(model, val_dataloader, loss, writer, global_step,num_classes,class_subset,excluded_classes)
                postfix_dict.update({"val_loss": f"{val_loss:.4f}", "val_mIoU": f"{val_miou:.3f}"})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        val_loss,
                        f"{CHECKPOINT_DIR}/best.pt"
                    )
                    checkpoint_saved = True
                    postfix_dict.update({"ckpt": f"saved @ {global_step}"})


            # ---- CHECKPOINTING ----
            if global_step % ckpt_every == 0 and global_step > 0 and not checkpoint_saved:
                ckpt_path = f"{CHECKPOINT_DIR}/{model_name}_step_{global_step}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    val_loss=None,
                    path=ckpt_path
                )
                postfix_dict.update({"ckpt": f"saved @ {global_step}"})
             # Further Logging
            epoch_train_mIoU_val = epoch_train_iou.compute()
            epoch_base_mIoU_val = epoch_base_iou.compute()
            writer.add_scalar("epoch/mIoU",epoch_train_mIoU_val,global_step)
            writer.add_scalar("epoch/acc",epoch_train_iou.compute_acc(),global_step)
            writer.add_scalar("epoch/base/acc",epoch_base_iou.compute_acc(),global_step)
            writer.add_scalar("epoch/base/mIoU",epoch_base_mIoU_val,global_step)
            writer.add_scalar("epoch/delta_mIoU",epoch_train_mIoU_val-epoch_base_mIoU_val,global_step)
            if epoch_base_mIoU_val > 0 :
                epoch_rel_mIoU_val = epoch_train_mIoU_val/epoch_base_mIoU_val
            else:
                epoch_rel_mIoU_val = 1.+epoch_train_mIoU_val
            writer.add_scalar("epoch/relative_mIoU",epoch_rel_mIoU_val,global_step)
            writer.add_scalar("train/lr",scheduler.get_last_lr()[0],global_step)
            epoch_train_iou.log_all("train_classIoU/",writer,label_map,global_step)
            epoch_base_iou.log_all("base_classIoU/",writer,label_map,global_step)
            pbar.set_postfix(postfix_dict)
            global_step += 1
        if epoch <= last_scheduler_step:
            scheduler.step()
       
        ckpt_path = f"{CHECKPOINT_DIR}/{model_name}_step_{global_step}_epoch_{epoch}.pt"
        save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch+1,
                    global_step,
                    val_loss=None,
                    path=ckpt_path
                )
        gc.collect()
        torch.cuda.empty_cache()

def validate(model, val_loader, loss_fn, writer=None, global_step=None,num_classes=100,class_subset=100,excluded_classes=None):
    model.eval()
    val_loss = 0.0
    iou_metric_post = IoUMetric(num_classes,class_subset,excluded_classes)
    iou_metric_base = IoUMetric(num_classes,class_subset,excluded_classes)
    

    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validating", leave=False) as pbar:
        for data in val_loader:
            # base mIoU
            valid_mask = data["valid_mask"]
            labels = data["segment"]
            valid_labels = labels[valid_mask]
            ignore_index = data["ignore_index"].to(labels.device)
            gs_valid_feat_mask = data["valid_feat_mask"]
            full_sorted_mask = data["full_sorted_mask"]

            base_logits = data["feat"][:,:class_subset]
            base_logits[:] = base_logits[:,data["inv_perm"][0]] # shuffle back to original positions. Assumes all batches use the same ordering
            # Get the base predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            base_pred = torch.argmax(base_logits,dim=-1)
            pc_segment  = data["pc_segment"]
            pc_valid_mask = pc_segment!=ignore_index 
            pc_query = data["pc_coord"][pc_valid_mask]
            pc_base_pred = neighbor_voting(data,base_pred,pc_query,gs_valid_feat_mask)

            pc_base_pred_valid = pc_base_pred
            pc_segment_valid = pc_segment[pc_valid_mask]

            iou_metric_base.update(pc_base_pred_valid,pc_segment_valid,full_sorted_mask)
            # Run the model
            new_feat = model(data)
            ordered_feat = new_feat.feat[:,data["inv_perm"][0]] # shuffle logits back to original positions

            valid_new_feat = ordered_feat[valid_mask]
            #one_hot = F.one_hot(valid_labels,num_classes=num_classes).float()
            loss = loss_fn(valid_new_feat, valid_labels)
            val_loss += loss.item()
            # Get the new updated predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            new_pred = torch.argmax(ordered_feat,dim=-1)
            # Neighbor voting to get from GS -> PC
            pc_new_pred = neighbor_voting(data,new_pred,pc_query)
            iou_metric_post.update(pc_new_pred, pc_segment_valid,full_sorted_mask)
            pbar.update(1)

    val_loss /= len(val_loader)
    val_miou = iou_metric_post.compute()
    base_miou = iou_metric_base.compute()
    delta_miou = val_miou-base_miou
    if writer is not None:
        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/mIoU", val_miou, global_step)
        writer.add_scalar("val/acc", iou_metric_post.compute_acc(), global_step)
        writer.add_scalar("val/base/acc", iou_metric_base.compute_acc(), global_step)
        writer.add_scalar("val/base/mIoU", base_miou, global_step)
        writer.add_scalar("val/delta_mIoU", delta_miou, global_step)
        if base_miou > 0 :
            rel_mIoU_val = val_miou/base_miou
        else:
            rel_mIoU_val = 0
        writer.add_scalar("val/relative_mIoU",rel_mIoU_val,global_step)



    model.train()
    return val_loss,val_miou



def build_model(k:int,model_typ="medium",device="cpu",class_subset=100)->dict:
    d = {}
    if "tiny" in model_typ:
        model =PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
        in_channels=k,
        stride=             [2],
        enc_depths=         (2, 2),
        enc_channels=       (112, 200),
        enc_num_head=       (2, 4),
        enc_patch_size=     (1024, 1024),
        dec_depths=         [2],
        dec_channels=       [class_subset],
        dec_num_head=       [4],
        dec_patch_size=     [1024],
        mlp_ratio=1,
        enable_flash=True,
        num_classes=class_subset)
    elif "medium_skip" in model_typ:
        model = PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
            in_channels=k,
            stride=             (2, 2),
            enc_depths=         (2, 2, 2),
            enc_channels=       (112, 200, 400),
            enc_num_head=       (2, 4, 8),
            enc_patch_size=     (1024, 1024, 1024),
            dec_depths=         (2, 2),
            dec_channels=       (class_subset, 200),
            dec_num_head=       (4, 8),
            dec_patch_size=     (1024, 1024),
            enable_flash=True,
            enable_skip=True,
            num_classes=class_subset)
    elif "medium" in model_typ:
        model = PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
            in_channels=k,
            stride=             (2, 2),
            enc_depths=         (2, 2, 2),
            enc_channels=       (112, 200, 400),
            enc_num_head=       (2, 4, 8),
            enc_patch_size=     (1024, 1024, 1024),
            dec_depths=         (2, 2),
            dec_channels=       (class_subset, 200),
            dec_num_head=       (4, 8),
            dec_patch_size=     (1024, 1024),
            enable_flash=True,
            num_classes=class_subset)
    elif "full_skip" in model_typ:
        model = PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
            in_channels=k,
            stride=             (2, 2, 2),
            enc_depths=         (2, 2, 2, 2),
            enc_channels=       (112, 200, 400, 800),
            enc_num_head=       (2, 4, 8, 16),
            enc_patch_size=     (1024, 1024, 1024, 1024),
            dec_depths=         (2, 2, 2),
            dec_channels=       (class_subset, 200, 400),
            dec_num_head=       (4, 8, 16),
            dec_patch_size=     (1024, 1024, 1024),
            enable_flash=True,
            enable_skip=True,
            num_classes=class_subset)
    d["name"] = model_typ
    d["model"]=model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS)
    d["last_scheduler_step"] = N_EPOCHS

    d["optimizer"] = optimizer
    d["scheduler"] = scheduler

    loss = nn.CrossEntropyLoss().to(device)
    d["loss"] = loss
    d["val_every"] = 2000
    d["ckpt_every"] = 1000
    d["epoch"] = 0
    d["global_step"] = 0

    label_map_file = os.path.join("/users/cymoser/storage/datasets/GaussianWorld/metadata","top100.txt")
    with open(label_map_file, "r") as f:
        d["label_map"] = [line.strip() for line in f]
    d["excluded_classes"] = [0,1,2] # wall floor and ceiling
    return d

def save_checkpoint(model, optimizer,scheduler, epoch, global_step, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


if __name__ == '__main__':
    main()

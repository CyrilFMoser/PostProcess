from torch.utils.data import  DataLoader
import torch.cuda
from data.dataloader.scannetppv2_dataloader import ScanNetPPV2DataSet
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from external.PointTransformerV3.model import PointTransformerV3
from utils.ioumetric import IoUMetric
import torch.nn.functional as F
from utils.neighbor_voting import neighbor_voting

N_EPOCHS = 10
CHECKPOINT_DIR = "/cluster/home/cymoser/cymoser/postprocess/ckpt"

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_scene_names = None
    scene_root = "/cluster/home/cymoser/data/GaussianWorld/scannetpp_v2_mcmc_3dgs_preprocessed"
    metadata_root = "/cluster/home/cymoser/projects/SceneSplat_Benchmark/gaussian_world_3d_semseg_benchmarks/embeddings"

    train_dataset = ScanNetPPV2DataSet(
        train_scene_names, scene_root, metadata_root, device,mode="train",subset=100000)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, collate_fn=train_dataset.collate_fn)
    
    val_scene_names = None

    val_dataset = ScanNetPPV2DataSet(
        val_scene_names, scene_root, metadata_root, device,mode="validation",subset=100000)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, collate_fn=val_dataset.collate_fn)
    
    num_channels = train_dataset.num_channels
    num_classes = train_dataset.num_classes

    config = build_model(num_channels,False,device)
    load = False
    if load:
        d:dict = torch.load("/cluster/home/cymoser/cymoser/postprocess/ckpt/medium_step_300.pt",map_location=device)
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

    train(train_dataloader,val_dataloader, config,num_classes)


def train(train_dataloader: DataLoader,val_dataloader:DataLoader, config:dict,num_classes):
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    model_name = config["name"]
    val_every = config["val_every"]
    ckpt_every = config["ckpt_every"]

    excluded_classes= config["excluded_classes"]


    model.train()
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)
    writer = SummaryWriter()

    epoch_train_iou = IoUMetric(num_classes,excluded_classes)
    epoch_base_iou = IoUMetric(num_classes,excluded_classes)

    global_step = config["global_step"]
    start_epoch = config["epoch"]
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
            #print(f"Iteration {batch_id}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            batch_train_iou = IoUMetric(num_classes,excluded_classes)
            batch_base_iou = IoUMetric(num_classes,excluded_classes)
            gs_valid_mask = data["valid_mask"]
            ignore_index = data["ignore_index"].to(gs_valid_mask.device)
            gs_valid_feat_mask = data["valid_feat_mask"]

            # --- Base mIoU Computation ---
            base_logits = data["feat"][:,:num_classes]
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
            batch_base_iou.update(pc_base_pred,pc_segment_valid)
            epoch_base_iou.combine(batch_base_iou)
            #  --- Run the model ---
            optimizer.zero_grad()
            new_feat = model(data)
            valid_new_feat = new_feat.feat[gs_valid_mask]
            
            gs_labels = data["segment"]
            gs_valid_labels = gs_labels[gs_valid_mask]
            #one_hot = F.one_hot(gs_valid_labels,num_classes=num_classes).float()
            output = loss(valid_new_feat,gs_valid_labels)

            output.backward()
            optimizer.step()

            epoch_loss += output.item()

            # Show running average loss in tqdm
            pbar.set_description(f"Epoch {epoch} | Loss {output.item() :.4f}")
            writer.add_scalar("train/loss", output.item(), global_step)
            writer.add_scalar("epoch/loss", epoch_loss/ (batch_id+1), global_step)
            # --- Get mIoU ---

            # Get the newly produced predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            new_pred = torch.argmax(new_feat.feat,dim=-1)
            # Neighbor voting to get from GS -> PC
            pc_pred = neighbor_voting(data,new_pred,pc_query)
            batch_train_iou.update(pc_pred, pc_segment_valid)
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
                val_loss,val_miou = validate(model, val_dataloader, loss, writer, global_step,num_classes,excluded_classes)
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
                epoch_rel_mIoU_val = 0
            writer.add_scalar("epoch/relative_mIoU",epoch_rel_mIoU_val,global_step)

            pbar.set_postfix(postfix_dict)
            global_step += 1
        scheduler.step()
       
        ckpt_path = f"{CHECKPOINT_DIR}/{model_name}_step_{global_step}_epoch_{epoch}.pt"
        save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    val_loss=None,
                    path=ckpt_path
                )


def validate(model, val_loader, loss_fn, writer=None, global_step=None,num_classes=100,excluded_classes=None):
    model.eval()
    val_loss = 0.0
    iou_metric_post = IoUMetric(num_classes,excluded_classes)
    iou_metric_base = IoUMetric(num_classes,excluded_classes)
    

    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validating", leave=False) as pbar:
        for data in val_loader:
            # base mIoU
            valid_mask = data["valid_mask"]
            labels = data["segment"]
            valid_labels = labels[valid_mask]
            ignore_index = data["ignore_index"].to(labels.device)
            gs_valid_feat_mask = data["valid_feat_mask"]

            base_logits = data["feat"][:,:num_classes]
            # Get the base predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            base_pred = torch.argmax(base_logits,dim=-1)
            pc_segment  = data["pc_segment"]
            pc_valid_mask = pc_segment!=ignore_index 
            pc_query = data["pc_coord"][pc_valid_mask]
            pc_base_pred = neighbor_voting(data,base_pred,pc_query,gs_valid_feat_mask)

            pc_base_pred_valid = pc_base_pred[pc_valid_mask]
            pc_segment_valid = pc_segment[pc_valid_mask]

            iou_metric_base.update(pc_base_pred_valid,pc_segment_valid)
            # Run the model
            new_feat = model(data)

            valid_new_feat = new_feat.feat[valid_mask]
            #one_hot = F.one_hot(valid_labels,num_classes=num_classes).float()
            loss = loss_fn(valid_new_feat, valid_labels)
            val_loss += loss.item()
            # Get the new updated predictions. Since we don't do any confidence thresholding here
            # this is always a valid class, and never ignore_index
            new_pred = torch.argmax(new_feat.feat,dim=-1)
            # Neighbor voting to get from GS -> PC
            pc_new_pred = neighbor_voting(data,new_pred,pc_query)
            iou_metric_post.update(pc_new_pred, pc_segment_valid)
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



def build_model(k:int,tiny=False,device="cpu")->dict:
    d = {}
    if tiny:
        model =PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
        in_channels=k,
        stride=             [2],
        enc_depths=         (2, 2),
        enc_channels=       (112, 200),
        enc_num_head=       (2, 4),
        enc_patch_size=     (1024, 1024),
        dec_depths=         [2],
        dec_channels=       [100],
        dec_num_head=       [4],
        dec_patch_size=     [1024],
        mlp_ratio=1,
        enable_flash=True)
        d["name"] = "tiny"
    else:
        model = PointTransformerV3( # [1, N, C] -> [N x C, 1] -> 
            in_channels=k,
            stride=             (2, 2),
            enc_depths=         (2, 2, 2),
            enc_channels=       (112, 200, 400),
            enc_num_head=       (2, 4, 8),
            enc_patch_size=     (1024, 1024, 1024),
            dec_depths=         (2, 2),
            dec_channels=       (100, 200),
            dec_num_head=       (4, 8),
            dec_patch_size=     (1024, 1024),
            enable_flash=True)
        d["name"] = "medium"
    d["model"]=model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.006,
        weight_decay=0.05
    )
    d["optimizer"] = optimizer

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.7)
    d["scheduler"] = scheduler
    weights = np.load("/cluster/home/cymoser/projects/PostProcess/data/scannetppv2/weights/weights.npy")
    loss = nn.CrossEntropyLoss().to(device)
    d["loss"] = loss

    d["val_every"] = 5000
    d["ckpt_every"] = 50
    d["epoch"] = 0
    d["global_step"] = 0

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

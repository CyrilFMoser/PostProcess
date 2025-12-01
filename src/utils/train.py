from torch.utils.data import Dataset, DataLoader
import torch.cuda
from data.dataloader.scannetppv2_dataloader import ScanNetPPV2DataSet
from tqdm import tqdm
import warnings
from torchinfo import summary
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

from external.PointTransformerV3.model import PointTransformerV3
from utils.ioumetric import IoUMetric

N_EPOCHS = 10
CHECKPOINT_DIR = "ckpt"

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_scene_names = ["09c1414f1b_mini"]
    scene_root = "data/scenesplat/scannetppv2/scenes"
    feature_root = "data/scenesplat/scannetppv2/features"
    metadata_root = "data/scannetppv2/metadata"
    label_root = "data/scannetppv2/gaussianlabels"

    train_dataset = ScanNetPPV2DataSet(
        train_scene_names, scene_root, feature_root, metadata_root,label_root, device,mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, collate_fn=train_dataset.collate_fn)
    
    val_scene_names = ["13c3e046d7_mini"]

    val_dataset = ScanNetPPV2DataSet(
        val_scene_names, scene_root, feature_root, metadata_root,label_root, device,mode="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, collate_fn=val_dataset.collate_fn)
    
    num_classes = train_dataset.num_classes 

    config = build_model(num_classes)    

    # summary(model, input_data=(x_dict,))  # 1670470
    config["model"].to(device)

    train(train_dataloader,val_dataloader, config,num_classes)


def train(train_dataloader: DataLoader,val_dataloader:DataLoader, config:dict,num_classes):
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]

    val_every = config["val_every"]
    ckpt_every = config["ckpt_every"]


    model.train()
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)
    writer = SummaryWriter()

    epoch_train_iou = IoUMetric(num_classes)
    epoch_base_iou = IoUMetric(num_classes)

    global_step = 0
    best_val_loss = float("inf")

    print(f"Start training with {num_params} number of parameters")
    for epoch in range(N_EPOCHS):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9)
        epoch_loss = 0.0
        epoch_train_iou.reset()
        epoch_base_iou.reset()
        for batch_id, data in pbar:
            batch_train_iou = IoUMetric(num_classes)
            batch_base_iou = IoUMetric(num_classes)
            valid_mask = data["valid_mask"]
            labels = data["label"]
            valid_labels = labels[valid_mask]

            # Base mIoU Computation
            valid_feat = data["feat"][valid_mask]
            base_preds = torch.argmax(valid_feat, dim=-1)
            batch_base_iou.update(base_preds,valid_labels)
            epoch_base_iou.combine(batch_base_iou)
            # Run the model
            optimizer.zero_grad()
            new_feat = model(data)

            valid_new_feat = new_feat.feat[valid_mask]

            output = loss(valid_new_feat,valid_labels)

            output.backward()
            optimizer.step()

            epoch_loss += output.item()

            # Show running average loss in tqdm
            pbar.set_description(f"Epoch {epoch} | Loss {epoch_loss / (batch_id+1):.4f}")
            writer.add_scalar("train/loss", output.item(), epoch * len(train_dataloader) + batch_id)
            # Get mIoU
            preds = torch.argmax(valid_new_feat, dim=-1)
            batch_train_iou.update(preds, valid_labels)
            batch_train_mIoU_val = batch_train_iou.compute()
            writer.add_scalar("train/mIoU",batch_train_mIoU_val , global_step)
            # Get Delta mIoU
            batch_base_mIoU_val = batch_base_iou.compute()
            batch_delta_mIoU_val =  batch_train_mIoU_val-batch_base_mIoU_val
            writer.add_scalar("train/delta_mIoU",batch_delta_mIoU_val,global_step)
            writer.add_scalar("train/base_mIoU",batch_base_mIoU_val,global_step)
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
                val_loss,val_miou = validate(model, val_dataloader, loss, writer, global_step,num_classes)
                postfix_dict.update({"val_loss": f"{val_loss:.4f}", "val_mIoU": f"{val_miou:.3f}"})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        val_loss,
                        f"{CHECKPOINT_DIR}/best.pt"
                    )
                    checkpoint_saved = True
                    postfix_dict.update({"ckpt": f"saved @ {global_step}"})


            # ---- CHECKPOINTING ----
            if global_step % ckpt_every == 0 and global_step > 0 and not checkpoint_saved:
                ckpt_path = f"{CHECKPOINT_DIR}/step_{global_step}.pt"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    val_loss=None,
                    path=ckpt_path
                )
                postfix_dict.update({"ckpt": f"saved @ {global_step}"})
            pbar.set_postfix(postfix_dict)
            global_step += 1
        scheduler.step()
        # Further Logging
        writer.add_scalar("train/epoch_mIoU",epoch_train_iou.compute(),global_step)
        writer.add_scalar("train/epoch_delta_mIoU",epoch_train_iou.compute()-epoch_base_iou.compute(),global_step)


def validate(model, val_loader, loss_fn, writer=None, global_step=None,num_classes=100):
    model.eval()
    val_loss = 0.0
    iou_metric_post = IoUMetric(num_classes)
    iou_metric_base = IoUMetric(num_classes)


    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validating", leave=False) as pbar:
        for data in val_loader:
            # base mIoU
            valid_mask = data["valid_mask"]
            labels = data["label"]
            valid_labels = labels[valid_mask]

            valid_feat = data["feat"][valid_mask]
            base_preds = torch.argmax(valid_feat, dim=-1)
            iou_metric_base.update(base_preds,valid_labels)
            # Run the model
            new_feat = model(data)

            valid_new_feat = new_feat.feat[valid_mask]

            loss = loss_fn(valid_new_feat, valid_labels)
            val_loss += loss.item()

            preds = torch.argmax(valid_new_feat, dim=-1)   # shape [N]
            iou_metric_post.update(preds, valid_labels)
            pbar.update(1)

    val_loss /= len(val_loader)
    val_miou = iou_metric_post.compute()
    base_miou = iou_metric_base.compute()
    delta_miou = val_miou-base_miou
    if writer is not None:
        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/mIoU", val_miou, global_step)
        writer.add_scalar("val/base_mIoU", base_miou, global_step)
        writer.add_scalar("val/delta_mIoU", delta_miou, global_step)



    model.train()
    return val_loss,val_miou



def build_model(k:int)->dict:
    d = {}
    model = PointTransformerV3(
        in_channels=k,
        stride=             (2, 2),
        enc_depths=         (2, 2, 2),
        enc_channels=       (100, 200, 400),
        enc_num_head=       (2, 4, 8),
        enc_patch_size=     (1024, 1024, 1024),
        dec_depths=         (2, 2),
        dec_channels=       (100, 200),
        dec_num_head=       (4, 8),
        dec_patch_size=     (1024, 1024),
        enable_flash=False)
    d["model"]=model

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    d["optimizer"] = optimizer

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.7)
    d["scheduler"] = scheduler

    loss = nn.CrossEntropyLoss()
    d["loss"] = loss    

    d["val_every"] = 5
    d["ckpt_every"] = 10
    return d

def save_checkpoint(model, optimizer, epoch, global_step, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)


if __name__ == '__main__':
    main()

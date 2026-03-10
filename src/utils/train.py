from torch.utils.data import DataLoader
import torch.cuda
from data.dataloader.scannetppv2_dataloader import ScanNetPPV2DataSet
from tqdm import tqdm
import torch.nn as nn
import wandb
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
import re
from collections import deque
import logging
from pathlib import Path
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import multiprocessing as mp
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*timm\.models\.layers.*"
)

import hashlib
import json
import fcntl

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


def compute_identity_hash(cfg: DictConfig) -> tuple[str, dict]:
    """Compute a short hash identifying a unique training configuration.

    Included: model architecture, class_subset, optimizer params, lr,
              scheduler, excluded_classes, run_tag.
    Excluded: N_EPOCHS, val/ckpt/eval_every, wandb settings, paths.
    """
    m = OmegaConf.to_container(cfg.model, resolve=True)
    m.pop("batch_size", None)  # batch size doesn't define a distinct experiment
    identity = {
        "model": m,
        "class_subset": int(cfg.class_subset),
        "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
        "lr": float(cfg.training.lr),
        "excluded_classes": sorted(int(c) for c in cfg.training.excluded_classes),
        "run_tag": cfg.get("run_tag", ""),
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
    }
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8]
    return config_hash, identity


def get_or_create_run_dir(model_name: str, base_checkpoint_dir: str, cfg: DictConfig, is_main_process: bool) -> str:
    """Return the checkpoint subdirectory for this config, creating a registry entry if needed.

    Layout: <base_checkpoint_dir>/<model_name>/run_NNNN/
    Registry: <base_checkpoint_dir>/<model_name>/registry.json
    Only rank 0 writes; the result is broadcast to all ranks.
    """
    config_hash, identity = compute_identity_hash(cfg)
    model_dir = Path(base_checkpoint_dir) / model_name
    registry_path = model_dir / "registry.json"
    lock_path = model_dir / "registry.lock"

    result = [None]
    if is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
                for entry in registry:
                    if entry["hash"] == config_hash:
                        result[0] = entry["checkpoint_dir"]
                        break
                else:
                    run_id = len(registry)
                    checkpoint_dir = str(model_dir / f"run_{run_id:04d}")
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(cfg, Path(checkpoint_dir) / "config.yaml")
                    registry.append({
                        "run_id": run_id,
                        "hash": config_hash,
                        "checkpoint_dir": checkpoint_dir,
                    })
                    registry_path.write_text(json.dumps(registry, indent=2))
                    result[0] = checkpoint_dir
                    print(f"Registered new run run_{run_id:04d} (hash {config_hash}) -> {checkpoint_dir}")
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    dist.broadcast_object_list(result, src=0)
    return result[0]


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, pred, target):
        return sum(w * l(pred, target) for l, w in zip(self.losses, self.weights))


def _build_single_loss(cfg_entry):
    loss_type = cfg_entry.get("type", "ce")
    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            label_smoothing=float(cfg_entry.get("label_smoothing", 0.0))
        )
    elif loss_type == "lovasz":
        from utils.lovasz import LovaszLoss
        return LovaszLoss(
            mode="multiclass",
            loss_weight=float(cfg_entry.get("loss_weight", 1.0)),
            ignore_index=None,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_loss(cfg_loss, device):
    if "losses" in cfg_loss:
        losses, weights = [], []
        for entry in cfg_loss["losses"]:
            weights.append(float(entry.get("weight", 1.0)))
            losses.append(_build_single_loss(entry))
        return CombinedLoss(losses, weights).to(device)
    else:
        return _build_single_loss(cfg_loss).to(device)


def build_model_from_cfg(cfg: DictConfig, num_channels: int) -> PointTransformerV3:
    m = OmegaConf.to_container(cfg.model, resolve=True)
    class_subset = cfg.class_subset
    r = m.get("refinement", {})

    # embedding_dim: from refinement block, then top-level fallback, then 1
    embedding_dim = r.get("embedding_dim", m.get("embedding_dim", 1))
    dec_channels = [class_subset * embedding_dim] + list(m.get("dec_channels_rest", []))

    # Validate decoder channel shape: non-decreasing from output to encoder,
    # with constant runs only permitted at the embed channel count.
    embed_ch = class_subset * embedding_dim
    dec_ch_full = dec_channels + [m["enc_channels"][-1]]
    for i in range(len(dec_ch_full) - 1):
        a, b = dec_ch_full[i], dec_ch_full[i + 1]
        assert a <= b, (
            f"Decoder uprojection at stage {i}: {a} → {b}. "
            f"dec_channels must be non-decreasing from output ({embed_ch}) to encoder "
            f"({m['enc_channels'][-1]})."
        )
        assert a < b or a == embed_ch, (
            f"Constant decoder channels at stage {i} ({a}) above embed size ({embed_ch}). "
            f"Channels must strictly increase above the embed size — "
            f"use distinct values in dec_channels_rest."
        )

    # Validate each dec_channels[i] is divisible by dec_num_head[i]
    dec_num_head = m["dec_num_head"]
    for i, (ch, nh) in enumerate(zip(dec_channels, dec_num_head)):
        assert ch % nh == 0, (
            f"dec_channels[{i}]={ch} not divisible by dec_num_head[{i}]={nh}. "
            f"Choose dec_channels_rest values divisible by their corresponding dec_num_head."
        )

    kwargs = dict(
        in_channels=num_channels,
        num_classes=class_subset,
        stride=m["stride"],
        enc_depths=m["enc_depths"],
        enc_channels=m["enc_channels"],
        enc_num_head=m["enc_num_head"],
        enc_patch_size=m["enc_patch_size"],
        dec_depths=m["dec_depths"],
        dec_channels=dec_channels,
        dec_num_head=m["dec_num_head"],
        dec_patch_size=m["dec_patch_size"],
        enable_flash=m.get("enable_flash", True),
        enable_skip=m.get("enable_skip", False),
        # input_skip in refinement block maps to skip_to_output; top-level fallback for compat
        skip_to_output=r.get("input_skip", m.get("skip_to_output", False)),
    )
    if "mlp_ratio" in m:
        kwargs["mlp_ratio"] = m["mlp_ratio"]

    # --- Bottleneck class aggregation ---
    if "bottleneck" in r:
        enc_last = m["enc_channels"][-1]
        assert enc_last % class_subset == 0, (
            f"refinement.bottleneck requires enc_channels[-1] ({enc_last}) "
            f"to be divisible by class_subset ({class_subset}). "
            f"Set enc_channels[-1] = class_subset * D (e.g. {class_subset * 8} for D=8)."
        )
        kwargs["class_aggregator"] = r["bottleneck"]

    # --- Fine-resolution refinement (spatial and/or class) ---
    sp = r.get("spatial") or {}
    cl = r.get("class") or {}
    spatial_layers = sp.get("num_layers", 0) if sp else 0
    class_layers   = cl.get("num_layers", 0) if cl else 0
    d_head = r.get("d_head", 16)

    if spatial_layers > 0 or class_layers > 0:
        kwargs["fine_refinement"] = dict(
            d_head=d_head,
            embedding_dim=embedding_dim,
            spatial_layers=spatial_layers,
            class_layers=class_layers,
            num_heads=r.get("num_heads", 1),
            spatial_mlp_ratio=sp.get("mlp_ratio", 2),
            class_mlp_ratio=cl.get("mlp_ratio", 4),
            patch_size=sp.get("patch_size", m["dec_patch_size"][0]),
            enable_flash=sp.get("enable_flash", m.get("enable_flash", True)),
            random_order=sp.get("random_order", False),
        )
    elif "neighbor_transformer" in m:
        # Backward compatibility: old neighbor_transformer key → fine_refinement with spatial only
        nt = m["neighbor_transformer"]
        _embedding_dim = dec_channels[0] // class_subset
        kwargs["fine_refinement"] = dict(
            d_head=nt.get("d_head", 16),
            embedding_dim=_embedding_dim,
            spatial_layers=nt.get("num_layers", 1),
            class_layers=0,
            num_heads=nt.get("num_heads", 1),
            spatial_mlp_ratio=nt.get("mlp_ratio", 2),
            class_mlp_ratio=4,
            patch_size=nt.get("patch_size", m["dec_patch_size"][0]),
            enable_flash=nt.get("enable_flash", m.get("enable_flash", True)),
            random_order=False,
        )

    return PointTransformerV3(**kwargs)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    torch.multiprocessing.set_start_method('spawn')
    mp.set_start_method("spawn", force=True)

    local_rank = setup_ddp()
    is_main_process = dist.get_rank() == 0
    device = torch.device("cuda", local_rank)
    print(f"Using device: {device}")

    class_subset = cfg.class_subset
    os.makedirs(cfg.paths.subset_cache_root, exist_ok=True)

    common_ds_kwargs = dict(
        scene_root=cfg.paths.scene_root,
        metadata_root=cfg.paths.metadata_root,
        device=device,
        subset=cfg.training.subset,
        class_subset=class_subset,
        subset_cache_root=cfg.paths.subset_cache_root,
    )

    if cfg.training.preprocess:
        _ = ScanNetPPV2DataSet(**common_ds_kwargs, mode="train")
        _ = ScanNetPPV2DataSet(**common_ds_kwargs, mode="validation")
        cleanup_ddp()
        return

    batch_size = cfg.model.batch_size

    train_dataset = None
    train_dataloader = None
    train_sampler = None
    if not cfg.training.validate_only:
        train_dataset = ScanNetPPV2DataSet(**common_ds_kwargs, mode="train")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            pin_memory=True, drop_last=True, persistent_workers=True,
            collate_fn=train_dataset.collate_fn, num_workers=4, prefetch_factor=4,
        )

    val_dataset = ScanNetPPV2DataSet(**common_ds_kwargs, mode="validation")
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        pin_memory=True, drop_last=False, persistent_workers=True,
        collate_fn=val_dataset.collate_fn, num_workers=4, prefetch_factor=4,
    )

    # Use whichever dataset is available to get num_channels
    reference_dataset = train_dataset if train_dataset is not None else val_dataset
    num_channels = reference_dataset.num_channels

    # Build model name: <arch>_<class_subset>_classes[_<run_tag>]
    suffix = f"_{class_subset}_classes" if class_subset != 100 else "_fullclasses"
    run_tag = cfg.get("run_tag", "")
    model_name = cfg.model.name + suffix + (f"_{run_tag}" if run_tag else "")

    # Each unique training config (optimizer, lr, scheduler, …) gets its own run_NNNN subdir
    checkpoint_dir = get_or_create_run_dir(model_name, cfg.paths.checkpoint_dir, cfg, is_main_process)

    model = build_model_from_cfg(cfg, num_channels).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    N_EPOCHS = cfg.training.N_EPOCHS
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr*batch_size,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # Checkpoint loading
    epoch = 0
    global_step = 0
    wandb_run_id = None
    loss_window_init      = []
    pred_mIoU_window_init = []
    base_mIoU_window_init = []
    pred_acc_window_init  = []
    base_acc_window_init  = []
    load = cfg.training.load_chkpt or cfg.training.chkpt_newest
    chkpt_path = None
    if cfg.training.chkpt_newest:
        chkpt_path, wandb_run_id = get_newest(checkpoint_dir)
    elif cfg.training.load_chkpt:
        chkpt_path = cfg.training.chkpt_path

    # Validate the wandb run on rank 0 and broadcast the result to all ranks.
    # If the run no longer exists on wandb (e.g. was deleted), start from scratch.
    run_is_valid = [chkpt_path is None or wandb_run_id is None]
    if is_main_process and wandb_run_id is not None:
        try:
            wandb.Api().run(f"{cfg.training.wandb_project}/{wandb_run_id}")
            run_is_valid[0] = True
        except Exception:
            print(f"wandb run {wandb_run_id} not found on server, starting from scratch")
            run_is_valid[0] = False
    dist.broadcast_object_list(run_is_valid, src=0)
    if not run_is_valid[0]:
        chkpt_path = None
        wandb_run_id = None

    if chkpt_path is not None:
        d: dict = torch.load(chkpt_path, map_location=device)
        objects = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
        for key, obj in objects.items():
            state_key = key + "_state"
            if state_key in d:
                obj.load_state_dict(d[state_key])
        epoch = d.get("epoch", 0)
        N_EPOCHS = d.get("N_EPOCHS", N_EPOCHS)
        global_step = d.get("global_step", 0)
        loss_window_init       = list(d.get("loss_window", []))
        pred_mIoU_window_init  = list(d.get("pred_mIoU_window", []))
        base_mIoU_window_init  = list(d.get("base_mIoU_window", []))
        pred_acc_window_init   = list(d.get("pred_acc_window", []))
        base_acc_window_init   = list(d.get("base_acc_window", []))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    with open(cfg.paths.label_map_file, "r") as f:
        label_map = [line.strip() for line in f]

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded model with {num_params} number of parameters")

    overrides = list(HydraConfig.get().overrides.task)
    full_cfg = OmegaConf.to_container(cfg, resolve=True)

    train_config = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=build_loss(cfg.loss, device),
        name=model_name,
        val_every=cfg.training.val_every,
        ckpt_every=cfg.training.ckpt_every,
        eval_every=cfg.training.eval_every,
        label_map=label_map,
        excluded_classes=list(cfg.training.excluded_classes),
        last_scheduler_step=N_EPOCHS,
        epoch=epoch,
        class_subset=class_subset,
        num_classes=cfg.num_classes,
        device=device,
        N_EPOCHS=N_EPOCHS,
        wandb_project=cfg.training.wandb_project,
        wandb_run_id=wandb_run_id,
        global_step=global_step,
        checkpoint_dir=checkpoint_dir,
        log_dir=cfg.paths.log_dir,
        wandb_tags=overrides,
        wandb_full_cfg=full_cfg,
        wandb_dir=cfg.paths.wandb_dir,
        loss_window_init=loss_window_init,
        pred_mIoU_window_init=pred_mIoU_window_init,
        base_mIoU_window_init=base_mIoU_window_init,
        pred_acc_window_init=pred_acc_window_init,
        base_acc_window_init=base_acc_window_init,
    )

    if cfg.training.validate_only:
        validate(model, val_dataloader, device, train_config["loss"],
                 num_classes=cfg.num_classes, class_subset=class_subset,
                 excluded_classes=list(cfg.training.excluded_classes))
    else:
        train(train_dataloader, train_sampler, val_dataloader, train_config)


def train(train_dataloader: DataLoader, train_sampler: DistributedSampler, val_dataloader: DataLoader, config: dict):
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    model_name = config["name"]
    val_every = config["val_every"]
    ckpt_every = config["ckpt_every"]
    label_map = config["label_map"]
    excluded_classes = config["excluded_classes"]
    last_scheduler_step = config["last_scheduler_step"]
    start_epoch = config["epoch"]
    class_subset = config["class_subset"]
    num_classes = config["num_classes"]
    device = config["device"]
    eval_every = config["eval_every"]
    N_EPOCHS = config["N_EPOCHS"]
    wandb_project = config.get("wandb_project", "PostProcess")
    wandb_run_id = config.get("wandb_run_id", None)
    wandb_tags = config.get("wandb_tags", [])
    wandb_full_cfg = config.get("wandb_full_cfg", {})
    checkpoint_dir = config["checkpoint_dir"]
    log_dir = Path(config["log_dir"])

    is_main_process = dist.get_rank() == 0

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"{timestamp}_{model_name}_epoch-{start_epoch}"

    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    epoch_train_iou = IoUMetric(num_classes, class_subset, excluded_classes)
    epoch_base_iou = IoUMetric(num_classes, class_subset, excluded_classes)
    loss_window      = deque(config.get("loss_window_init", []),      maxlen=500)
    pred_mIoU_window = deque(config.get("pred_mIoU_window_init", []), maxlen=500)
    base_mIoU_window = deque(config.get("base_mIoU_window_init", []), maxlen=500)
    pred_acc_window  = deque(config.get("pred_acc_window_init", []),  maxlen=500)
    base_acc_window  = deque(config.get("base_acc_window_init", []),  maxlen=500)

    if is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        timing_logger = logging.getLogger("timing_logger")
        timing_logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_dir / "timing.log", mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        timing_logger.addHandler(fh)
        timing_logger.propagate = False
        wandb_dir = Path(config.get("wandb_dir", "."))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=wandb_project,
            name=run_name if wandb_run_id is None else None,
            id=wandb_run_id,
            resume="allow",
            config=wandb_full_cfg,
            tags=wandb_tags,
            dir=str(wandb_dir),
        )
        wandb.config.update({"num_params": num_params})
        wandb_run_id = wandb.run.id

    DEBUG_TIMING = False

    global_step = config["global_step"]
    best_val_loss = float("inf")
    print(f"Start training with {num_params} number of parameters")
    for epoch in range(start_epoch, N_EPOCHS):
        epoch_loss = 0.0
        epoch_train_iou.reset()
        epoch_base_iou.reset()
        train_sampler.set_epoch(epoch)

        loader_iter = iter(train_dataloader)
        if is_main_process:
            pbar = tqdm(range(len(train_dataloader)), smoothing=0.9)
        else:
            pbar = range(len(train_dataloader))
        for step in pbar:
            eval_stats = global_step % eval_every == 0 and global_step > 0
            eval_stats = eval_stats and is_main_process
            t_load_start = time.time()
            data = next(loader_iter)
            t_load_end = time.time()

            if DEBUG_TIMING:
                torch.cuda.synchronize()
            t0 = time.time()
            B = data["grid_size"].shape[0]

            data["grid_size"] = data["grid_size"][0]
            data = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in data.items()}
            batch_train_iou = IoUMetric(num_classes, class_subset, excluded_classes)
            batch_base_iou = IoUMetric(num_classes, class_subset, excluded_classes)

            gs_valid_mask = data["valid_mask"]
            ignore_index = data["ignore_index"].to(gs_valid_mask.device)

            if DEBUG_TIMING:
                torch.cuda.synchronize()
            t0_base = time.time()
            base_vote = 0
            if eval_stats:
                with torch.no_grad():
                    base_logits = data["feat"][:, :class_subset]
                    _, base_pred = torch.max(torch.sigmoid(base_logits), dim=1)
                base_vote = evaluate_stats(data, base_pred, batch_base_iou, num_classes, ignore_index, B)
                epoch_base_iou.combine(batch_base_iou)
            t1_base = time.time()

            optimizer.zero_grad()
            if DEBUG_TIMING:
                torch.cuda.synchronize()
            fake_grad = False
            new_feat = model(data)
            valid_new_feat = new_feat.feat[gs_valid_mask]
            gs_labels = data["segment"]
            gs_valid_labels = gs_labels[gs_valid_mask]
            output = loss(valid_new_feat, gs_valid_labels)
            if gs_valid_mask.sum() == 0:
                output *= 0.0
                eval_stats = False
                fake_grad = True
                print(f"Rank {dist.get_rank()} had empty mask", flush=True)
            output.backward()
            optimizer.step()
            if DEBUG_TIMING:
                torch.cuda.synchronize()
            t1 = time.time()

            epoch_loss += output.item()
            if is_main_process and not fake_grad:
                loss_window.append(output.item())
                pbar.set_description(f"Epoch {epoch} | Loss {output.item():.4f}")
                wandb.log({"train/loss": output.item(), "epoch/loss": sum(loss_window) / len(loss_window)}, step=global_step)

            if DEBUG_TIMING:
                torch.cuda.synchronize()
            t0_pred = time.time()
            pred_vote = 0
            if eval_stats:
                new_pred = torch.argmax(new_feat.feat, dim=-1)
                pred_vote = evaluate_stats(data, new_pred, batch_train_iou, num_classes, ignore_index, B)
                if DEBUG_TIMING:
                    torch.cuda.synchronize()
            t1_pred = time.time()

            load_time = t_load_end - t_load_start
            base_time = t1_base - t0_base
            compute_time = t1 - t0 - base_time
            pred_time = t1_pred - t0_pred
            if is_main_process:
                timing_logger.info(
                    f"Step {step:05d} | Load wait: {load_time:.3f}s | Compute: {compute_time:.3f}s | Base Eval: {base_time:.3f}s | Pred Eval: {pred_time:.3f}s | Base Vote: {base_vote:.3f}s | Pred Vote: {pred_vote:.3f}s"
                )
            postfix_dict = {}
            if eval_stats:
                batch_train_mIoU_val = batch_train_iou.compute()
                batch_base_mIoU_val = batch_base_iou.compute()
                batch_delta_mIoU_val = batch_train_mIoU_val - batch_base_mIoU_val
                if batch_base_mIoU_val > 0:
                    batch_rel_mIoU_val = batch_train_mIoU_val / batch_base_mIoU_val
                else:
                    batch_rel_mIoU_val = 0
                wandb.log({
                    "train/mIoU": batch_train_mIoU_val,
                    "train/acc": batch_train_iou.compute_acc(),
                    "train/delta_mIoU": batch_delta_mIoU_val,
                    "train/base/mIoU": batch_base_mIoU_val,
                    "train/base/acc": batch_base_iou.compute_acc(),
                    "train/relative_mIoU": batch_rel_mIoU_val,
                }, step=global_step)
                epoch_train_iou.combine(batch_train_iou)
                postfix_dict.update({
                    "mIoU": f"{batch_train_mIoU_val:.3f}",
                    "base": f"{batch_base_mIoU_val:.3f}",
                    "ΔmIoU": f"{batch_delta_mIoU_val:.3f}",
                })
            checkpoint_saved = False
            if global_step % val_every == 0 and global_step > 0:
                val_loss, val_miou = validate(model, val_dataloader, device, loss, global_step, num_classes, class_subset, excluded_classes)
                if is_main_process:
                    postfix_dict.update({"val_loss": f"{val_loss:.4f}", "val_mIoU": f"{val_miou:.3f}"})
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, scheduler, epoch, global_step, val_loss,
                            f"{checkpoint_dir}/{wandb_run_id}/best.pt",
                            windows={"loss_window": loss_window, "pred_mIoU_window": pred_mIoU_window,
                                     "base_mIoU_window": base_mIoU_window, "pred_acc_window": pred_acc_window,
                                     "base_acc_window": base_acc_window},
                        )
                        checkpoint_saved = True
                        postfix_dict.update({"ckpt": f"saved @ {global_step}"})

            if is_main_process and global_step % ckpt_every == 0 and global_step > 0 and not checkpoint_saved:
                ckpt_path = f"{checkpoint_dir}/{wandb_run_id}/step_{global_step}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, val_loss=None,
                    path=ckpt_path,
                    windows={"loss_window": loss_window, "pred_mIoU_window": pred_mIoU_window,
                             "base_mIoU_window": base_mIoU_window, "pred_acc_window": pred_acc_window,
                             "base_acc_window": base_acc_window},
                )
                postfix_dict.update({"ckpt": f"saved @ {global_step}"})

            if eval_stats:
                pred_mIoU_window.append(epoch_train_iou.compute())
                epoch_train_mIoU_val = sum(pred_mIoU_window) / len(pred_mIoU_window)
                base_mIoU_window.append(epoch_base_iou.compute())
                epoch_base_mIoU_val = sum(base_mIoU_window) / len(base_mIoU_window)
                pred_acc_window.append(epoch_train_iou.compute_acc())
                epoch_train_acc_val = sum(pred_acc_window) / len(pred_acc_window)
                base_acc_window.append(epoch_base_iou.compute_acc())
                epoch_base_acc_val = sum(base_acc_window) / len(base_acc_window)

                if epoch_base_mIoU_val > 0:
                    epoch_rel_mIoU_val = epoch_train_mIoU_val / epoch_base_mIoU_val
                else:
                    epoch_rel_mIoU_val = 1. + epoch_train_mIoU_val
                epoch_log = {
                    "epoch/mIoU": epoch_train_mIoU_val,
                    "epoch/acc": epoch_train_acc_val,
                    "epoch/base/acc": epoch_base_acc_val,
                    "epoch/base/mIoU": epoch_base_mIoU_val,
                    "epoch/delta_mIoU": epoch_train_mIoU_val - epoch_base_mIoU_val,
                    "epoch/relative_mIoU": epoch_rel_mIoU_val,
                }
                epoch_log.update(epoch_train_iou.log_all("train_classIoU/", label_map))
                epoch_log.update(epoch_base_iou.log_all("base_classIoU/", label_map))
                wandb.log(epoch_log, step=global_step)
            if is_main_process:
                wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                pbar.set_postfix(postfix_dict)
            global_step += B
        scheduler.step()
        if is_main_process:
            ckpt_path = f"{checkpoint_dir}/{wandb_run_id}/step_{global_step}_epoch_{epoch}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                val_loss=None, path=ckpt_path,
                windows={"loss_window": loss_window, "pred_mIoU_window": pred_mIoU_window,
                         "base_mIoU_window": base_mIoU_window, "pred_acc_window": pred_acc_window,
                         "base_acc_window": base_acc_window},
            )
        gc.collect()
        torch.cuda.empty_cache()


def validate(model, val_loader, device, loss_fn, global_step=None, num_classes=100, class_subset=100, excluded_classes=None):
    model.eval()
    val_loss = 0.0
    iou_metric_post = IoUMetric(num_classes, class_subset, excluded_classes)
    iou_metric_base = IoUMetric(num_classes, class_subset, excluded_classes)
    is_main_process = dist.get_rank() == 0

    with torch.no_grad(), tqdm(total=len(val_loader), desc="Validating", leave=False) as pbar:
        for data in val_loader:
            B = data["grid_size"].shape[0]
            data["grid_size"] = data["grid_size"][0]
            data = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in data.items()}
            valid_mask = data["valid_mask"]
            labels = data["segment"]
            valid_labels = labels[valid_mask]
            ignore_index = data["ignore_index"].to(labels.device)
            gs_valid_feat_mask = data["valid_feat_mask"]
            full_sorted_mask = data["full_sorted_mask"]

            base_logits = data["feat"][:, :class_subset]
            base_pred = torch.argmax(base_logits, dim=-1)
            for b in range(B):
                pc_start = 0
                gs_start = 0
                if b > 0:
                    pc_start = int(data["pc_offset"][b - 1])
                    gs_start = int(data["offset"][b - 1])
                pc_end = int(data["pc_offset"][b])
                gs_end = int(data["offset"][b])
                pc_segment = data["pc_segment"][pc_start:pc_end]
                pc_valid_mask = pc_segment != ignore_index[b]
                pc_query = data["pc_coord"][pc_start:pc_end][pc_valid_mask]
                gs_coords = data["coord"][gs_start:gs_end]
                neighbours = data["neighbours"][pc_start:pc_end][pc_valid_mask]
                pc_base_pred = neighbor_voting(neighbours, base_pred[gs_start:gs_end][gs_valid_feat_mask[gs_start:gs_end]], pc_query)
                pc_segment_valid = pc_segment[pc_valid_mask]
                iou_metric_base.update(pc_base_pred, pc_segment_valid, full_sorted_mask[b * num_classes:(b + 1) * num_classes])

            new_feat = model(data)
            valid_new_feat = new_feat.feat[valid_mask]
            loss = loss_fn(valid_new_feat, valid_labels)
            val_loss += loss.item()
            new_pred = torch.argmax(new_feat.feat, dim=-1)
            for b in range(B):
                pc_start = 0
                gs_start = 0
                if b > 0:
                    pc_start = int(data["pc_offset"][b - 1])
                    gs_start = int(data["offset"][b - 1])
                pc_end = int(data["pc_offset"][b])
                gs_end = int(data["offset"][b])
                pc_segment = data["pc_segment"][pc_start:pc_end]
                pc_valid_mask = pc_segment != ignore_index[b]
                pc_query = data["pc_coord"][pc_start:pc_end][pc_valid_mask]
                gs_coords = data["coord"][gs_start:gs_end]
                pc_segment_valid = pc_segment[pc_valid_mask]
                neighbours = data["neighbours"][pc_start:pc_end][pc_valid_mask]
                pc_new_pred = neighbor_voting(neighbours, new_pred[gs_start:gs_end][gs_valid_feat_mask[gs_start:gs_end]], pc_query)
                iou_metric_post.update(pc_new_pred, pc_segment_valid, full_sorted_mask[b * num_classes:(b + 1) * num_classes])
            pbar.update(1)

    intersections_base = iou_metric_base.intersections.to(device)
    unions_base = iou_metric_base.unions.to(device)
    in_gts_base = iou_metric_base.in_gts.to(device).float()
    gt_base = iou_metric_base.gt.to(device)

    dist.all_reduce(intersections_base, op=dist.ReduceOp.SUM)
    dist.all_reduce(unions_base, op=dist.ReduceOp.SUM)
    dist.all_reduce(in_gts_base, op=dist.ReduceOp.SUM)
    dist.all_reduce(gt_base, op=dist.ReduceOp.SUM)

    iou_metric_base.intersections = intersections_base.cpu()
    iou_metric_base.unions = unions_base.cpu()
    iou_metric_base.in_gts = in_gts_base.bool().cpu()
    iou_metric_base.gt = gt_base.cpu()

    intersections_post = iou_metric_post.intersections.to(device)
    unions_post = iou_metric_post.unions.to(device)
    in_gts_post = iou_metric_post.in_gts.to(device).float()
    gt_post = iou_metric_post.gt.to(device)

    dist.all_reduce(intersections_post, op=dist.ReduceOp.SUM)
    dist.all_reduce(unions_post, op=dist.ReduceOp.SUM)
    dist.all_reduce(gt_post, op=dist.ReduceOp.SUM)
    dist.all_reduce(in_gts_post, op=dist.ReduceOp.SUM)

    iou_metric_post.intersections = intersections_post.cpu()
    iou_metric_post.unions = unions_post.cpu()
    iou_metric_post.in_gts = in_gts_post.bool().cpu()
    iou_metric_post.gt = gt_post.cpu()

    val_loss = torch.tensor([val_loss], device=device)
    loss_counts = torch.tensor([len(val_loader)], device=device)

    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)

    val_loss = float(val_loss) / float(loss_counts)
    val_miou = iou_metric_post.compute()
    base_miou = iou_metric_base.compute()
    delta_miou = val_miou - base_miou
    if is_main_process and global_step is not None:
        if base_miou > 0:
            rel_mIoU_val = val_miou / base_miou
        else:
            rel_mIoU_val = 0
        wandb.log({
            "val/loss": val_loss,
            "val/mIoU": val_miou,
            "val/acc": iou_metric_post.compute_acc(),
            "val/base/acc": iou_metric_base.compute_acc(),
            "val/base/mIoU": base_miou,
            "val/delta_mIoU": delta_miou,
            "val/relative_mIoU": rel_mIoU_val,
        }, step=global_step)

    model.train()
    return val_loss, val_miou


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, path, windows=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
    }
    if windows is not None:
        for k, v in windows.items():
            checkpoint[k] = list(v)
    torch.save(checkpoint, path)


def evaluate_stats(data, pred, batch_iou, num_classes, ignore_index, B):
    full_sorted_mask = data["full_sorted_mask"]
    gs_valid_feat_mask = data["valid_feat_mask"]
    vote_time = 0
    with torch.no_grad():
        for b in range(B):
            pc_start = 0
            gs_start = 0
            if b > 0:
                pc_start = int(data["pc_offset"][b - 1])
                gs_start = int(data["offset"][b - 1])
            pc_end = int(data["pc_offset"][b])
            gs_end = int(data["offset"][b])
            pc_segment = data["pc_segment"][pc_start:pc_end]
            pc_valid_mask = pc_segment != (ignore_index[b])
            pc_query = data["pc_coord"][pc_start:pc_end][pc_valid_mask]
            neighbours = data["neighbours"][pc_start:pc_end][pc_valid_mask]
            t0_vote = time.time()
            pc_pred = neighbor_voting(neighbours, pred[gs_start:gs_end][gs_valid_feat_mask[gs_start:gs_end]], pc_query)
            t1_vote = time.time()
            vote_time += t1_vote - t0_vote
            pc_segment_valid = pc_segment[pc_valid_mask]
            batch_iou.update(pc_pred, pc_segment_valid, full_sorted_mask[b * num_classes:(b + 1) * num_classes])
    return vote_time


def get_newest(checkpoint_dir: str):
    folder = Path(checkpoint_dir)
    if not folder.exists():
        return None, None
    best_file, best_run_id = None, None
    best_step = -1
    for subdir in folder.iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if not f.is_file() or f.suffix != ".pt":
                continue
            m = re.search(r"step_(\d+)", f.name)
            step = int(m.group(1)) if m else -1
            if step > best_step:
                best_step = step
                best_file = f
                best_run_id = subdir.name
    return best_file, best_run_id


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"Set local_rank: {local_rank} to device")
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

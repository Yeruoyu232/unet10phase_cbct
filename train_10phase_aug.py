"""10-phase E2E U-Net trained with case-specific physical-FOV L1 loss.

Run from the directory containing dataset.py and model.py:
    python train_10phase_fov_l1.py
"""
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_aug import DIRLab10PhaseDataset
from model import build_model


# Original training configuration (unchanged except SAVE_DIR).
DATA_ROOT = "/home/yry/ENV/unet_10phase/training/cbct_motion_10phase_physical_aug"


VAL_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/"
    "cbct_motion_10phase_samez_minus_1024_div_1024"
)
SAVE_DIR = "./checkpoints/unet_10phase_fov_physical_aug_50"

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2
SDD = 1200.0
NU = 512

BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
SEED = 42

class FOVL1Loss(nn.Module):
    """Mean per-image FOV MAE; each image contributes equal weight."""

    def forward(self, prediction, target, mask):
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={prediction.shape}, target={target.shape}, mask={mask.shape}"
            )
        # float32 loss calculation is more stable with autocast enabled.
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or (mask.sum((1,2,3)) <= 0).any():
            raise ValueError('Invalid binary mask or empty FOV')
        error = (prediction.float() - target.float()).abs()
        per_image = (error * mask).sum(dim=(1, 2, 3)) / mask.sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        return per_image.mean()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if any(os.path.exists(os.path.join(SAVE_DIR, name)) for name in ('best.pth', 'last.pth')):
        raise FileExistsError('Checkpoint directory already contains weights; choose a new SAVE_DIR')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 80)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    dataset_kwargs = dict(
    
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
        return_fov_mask=True,
    )
    train_dataset = DIRLab10PhaseDataset(
        root=DATA_ROOT,
        prompt_file="prompt_train.json",
        **dataset_kwargs,
    )

    val_dataset = DIRLab10PhaseDataset(
        root=VAL_ROOT,
        prompt_file="prompt_val.json",
        **dataset_kwargs,
    )
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    # Fail before training if the merged data are incomplete or incorrectly split.
    from collections import Counter
    if len(train_dataset) != 2721 or len(val_dataset) != 128:
        raise ValueError("Expected 2721 training pairs and 128 validation pairs")
    variants = Counter(r.get("variant") for r in train_dataset.records)
    if variants != {"original": 907, "flipx": 907, "shift_5_0_0": 907}:
        raise ValueError(f"Unexpected training variants: {variants}")
    print("Training variants:", dict(variants))
    for dataset, allowed_cases in [(train_dataset, set(range(1,9))), (val_dataset, {9})]:
        identities = set()
        for r in dataset.records:
            if int(r["case_id"]) not in allowed_cases:
                raise ValueError("Unexpected case in data split")
            if not np.isclose(float(r["SDD"]), SDD):
                raise ValueError("SDD mismatch")
            if dataset is val_dataset:
                variant = r.get("variant", "original")
                if (variant != "original" or r.get("augmentation")
                        or r.get("aug_id", 0) != 0 or r.get("flip_lr", False)
                        or float(r.get("dx", 0)) != 0 or float(r.get("dy", 0)) != 0):
                    raise ValueError(f"Validation must be original: case={r['case_id']} z={r['z_idx']} variant={variant}")
            identity = (int(r["case_id"]), int(r["z_idx"]), r.get("variant", "original"))
            if identity in identities:
                raise ValueError(f"Duplicate sample: {identity}")
            identities.add(identity)
            keys = ["source", "target"]
            if "mask" in r:
                keys.append("mask")
            elif dataset is train_dataset:
                raise ValueError("Training sample missing saved mask")
            for key in keys:
                path = os.path.join(dataset.root, r[key])
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
    print("All source/target/mask paths checked.")


    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    model = build_model().to(device)  # E2E: prediction=model(source), no residual addition.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params/1e6:.2f}M | trainable: {trainable_params/1e6:.2f}M")

    criterion = FOVL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")

    scan_condition = {
        "SDD": SDD,
        "scan_time": SCAN_TIME,
        "breathing_period": BREATHING_PERIOD,
        "start_phase": START_PHASE,
        "SOD": SOD,
        "du": DU,
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for step, batch in enumerate(train_loader, start=1):
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["fov_mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            if not torch.isfinite(source).all() or not torch.isfinite(target).all():
                raise RuntimeError(f"Non-finite input at epoch={epoch}, step={step}")
            if epoch == 1 and step == 1:
                print("[Smoke] source:", tuple(source.shape), "target:", tuple(target.shape))
                print("[Smoke] mask:", tuple(mask.shape), "ratios:",
                      [round(x, 4) for x in mask.mean(dim=(1, 2, 3)).tolist()])
                print("[Smoke] case_ids:", batch["case_id"].tolist())

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(source)
                loss = criterion(prediction, target, mask)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = source.shape[0]
            train_loss_sum += loss.item() * bs
            train_sample_count += bs
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(f"[Train] Epoch {epoch:03d}/{EPOCHS} | "
                      f"Step {step:04d}/{len(train_loader):04d} | FOV-L1 {loss.item():.6f}")

        train_loss = train_loss_sum / train_sample_count
        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for batch in val_loader:
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                mask = batch["fov_mask"].to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    prediction = model(source)
                    loss = criterion(prediction, target, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss at epoch={epoch}")
                bs = source.shape[0]
                val_loss_sum += loss.item() * bs
                val_sample_count += bs

        val_loss = val_loss_sum / val_sample_count
        print(f"\nEpoch {epoch:03d}/{EPOCHS} | Train FOV-L1 = {train_loss:.6f} | "
              f"Val FOV-L1 = {val_loss:.6f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "scan_condition": scan_condition,
                "loss_type": "saved_physical_fov_l1",
                "prediction_mode": "e2e",
                "augmentation": "flip_x_and_shift_5_0_0_before_projection",
                "data_root": DATA_ROOT,
                "mask_policy": "train_saved_nearest;val_saved_or_legacy_bilinear_threshold",
                "val_root": VAL_ROOT,
                "seed": SEED,
                "epochs_planned": EPOCHS,
            },
            os.path.join(SAVE_DIR, "last.pth"),
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "scan_condition": scan_condition,
                    "loss_type": "saved_physical_fov_l1",
                "prediction_mode": "e2e",
                "augmentation": "flip_x_and_shift_5_0_0_before_projection",
                "data_root": DATA_ROOT,
                "mask_policy": "train_saved_nearest;val_saved_or_legacy_bilinear_threshold",
                "val_root": VAL_ROOT,
                "seed": SEED,
                "epochs_planned": EPOCHS,
                },
                os.path.join(SAVE_DIR, "best.pth"),
            )
            print(f"✓ Best model saved | Val FOV-L1 = {best_val_loss:.6f}")
        print("-" * 80)

    print("\nTraining finished.")
    print("Best validation FOV-L1:", best_val_loss)
    print("Best checkpoint:", os.path.join(SAVE_DIR, "best.pth"))


if __name__ == "__main__":
    main()

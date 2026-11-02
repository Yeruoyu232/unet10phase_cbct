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

from dataset import DIRLab10PhaseDataset
from model import build_model


# Original training configuration (unchanged except SAVE_DIR).
DATA_ROOT = "/home/yry/ENV/unet_10phase/training/cbct_motion_10phase_scan30_period4_start0_sod720_du1p2_aug6"
SAVE_DIR = "./checkpoints/unet_10phase_fov_aug"

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
EPOCHS = 100
SEED = 42

# (original width, original height, spacing_x mm, spacing_y mm)
CASE_GEOMETRY = {
    1: (256, 256, 0.97, 0.97),
    2: (256, 256, 1.16, 1.16),
    3: (256, 256, 1.15, 1.15),
    4: (256, 256, 1.13, 1.13),
    5: (256, 256, 1.10, 1.10),
    6: (512, 512, 0.97, 0.97),
    7: (512, 512, 0.97, 0.97),
    8: (512, 512, 0.97, 0.97),
    9: (512, 512, 0.97, 0.97),
    10: (512, 512, 0.97, 0.97),
}

# Cache CPU masks so GPU memory is not permanently occupied.
_FOV_CACHE = {}


def case_fov_mask(case_id, height=256, width=256):
    """Return a [1,H,W] float32 binary mask aligned with target resize.

    Original FOV: (x_mm^2 + y_mm^2) <= (NU*DU*SOD/(2*SDD))^2.
    GT uses PIL bilinear resizing of the FOV-zeroed image; for the
    training ROI, resize the original binary mask the same way and
    threshold at >0.5. Boundary pixels are therefore an explicit choice.
    """
    case_id = int(case_id)
    key = (case_id, height, width)
    if key not in _FOV_CACHE:
        if case_id not in CASE_GEOMETRY:
            raise ValueError(f"Unknown case_id={case_id}")
        nx, ny, dx, dy = CASE_GEOMETRY[case_id]
        radius_mm = NU * DU * SOD / (2.0 * SDD)
        x = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * dx
        y = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * dy
        xx, yy = np.meshgrid(x, y)
        original_mask = ((xx**2 + yy**2) <= radius_mm**2).astype(np.float32)

        if original_mask.shape != (height, width):
            pil_mask = Image.fromarray(original_mask, mode="F")
            resized = pil_mask.resize((width, height), resample=Image.Resampling.BILINEAR)
            mask_np = np.asarray(resized, dtype=np.float32)
        else:
            mask_np = original_mask

        binary_mask = (mask_np > 0.5).astype(np.float32)
        if not binary_mask.any():
            raise RuntimeError(f"Empty FOV mask for Case{case_id}")
        _FOV_CACHE[key] = torch.from_numpy(binary_mask).unsqueeze(0)
    return _FOV_CACHE[key]


def batch_fov_mask(batch, target):
    """Return [B,1,H,W] masks on the same device as target."""
    height, width = target.shape[-2:]
    masks = [case_fov_mask(cid, height, width) for cid in batch["case_id"]]
    return torch.stack(masks, dim=0).to(device=target.device, non_blocking=True)


class FOVL1Loss(nn.Module):
    """Mean per-image FOV MAE; each image contributes equal weight."""

    def forward(self, prediction, target, mask):
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred={prediction.shape}, target={target.shape}, mask={mask.shape}"
            )
        # float32 loss calculation is more stable with autocast enabled.
        error = (prediction.float() - target.float()).abs()
        per_image = (error * mask).sum(dim=(1, 2, 3)) / mask.sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        return per_image.mean()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
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
        root=DATA_ROOT,
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
    )
    train_dataset = DIRLab10PhaseDataset(prompt_file="prompt_train.json", **dataset_kwargs)
    val_dataset = DIRLab10PhaseDataset(prompt_file="prompt_val.json", **dataset_kwargs)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

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
            mask = batch_fov_mask(batch, target)

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
                mask = batch_fov_mask(batch, target)
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
                "loss_type": "case_specific_fov_l1",
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
                    "loss_type": "case_specific_fov_l1",
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

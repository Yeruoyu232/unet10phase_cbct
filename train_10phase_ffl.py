import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from focal_frequency_loss import FocalFrequencyLoss

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# 1. Configuration
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
)

SAVE_DIR = "./checkpoints/unet_10phase_ffl"
LOG_DIR = "./logs/unet_10phase_ffl"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, "training_history.csv")


# ============================================================
# 2. Fixed scan conditions
# ============================================================

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0

SOD = 720.0
DU = 1.2


# ============================================================
# 3. Training configuration
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS = 100
SEED = 42

# Focal Frequency Loss
FFL_WEIGHT = 0.05
FFL_ALPHA = 1.0

# Whether to use mixed precision
USE_AMP = True


# ============================================================
# 4. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = USE_AMP and DEVICE.type == "cuda"

print("=" * 80)
print("10P E2E U-Net + Focal Frequency Loss")
print("=" * 80)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("FFL weight:", FFL_WEIGHT)
print("FFL alpha:", FFL_ALPHA)
print("AMP:", USE_AMP)

print("=" * 80)


# ============================================================
# 5. Random seed
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 6. Dataset
# ============================================================

# Train: Case1–8
# Val:   Case9
# Test:  Case10 (not used during training)

train_dataset = DIRLab10PhaseDataset(
    root=DATA_ROOT,
    prompt_file="prompt_train.json",
    scan_time=SCAN_TIME,
    breathing_period=BREATHING_PERIOD,
    start_phase=START_PHASE,
    sod=SOD,
    du=DU,
)

val_dataset = DIRLab10PhaseDataset(
    root=DATA_ROOT,
    prompt_file="prompt_val.json",
    scan_time=SCAN_TIME,
    breathing_period=BREATHING_PERIOD,
    start_phase=START_PHASE,
    sod=SOD,
    du=DU,
)

print()
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))


# ============================================================
# 7. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)


# ============================================================
# 8. Model
# ============================================================

# Input:  [B, 10, 256, 256]
# Output: [B, 1, 256, 256]

model = build_model().to(DEVICE)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("=" * 80)
print("Model")
print("=" * 80)

print(f"Total parameters: {total_params / 1e6:.2f} M")
print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

print("=" * 80)


# ============================================================
# 9. Loss functions
# ============================================================

criterion_l1 = nn.L1Loss()

criterion_ffl = FocalFrequencyLoss(
    loss_weight=1.0,
    alpha=FFL_ALPHA,
    patch_factor=1,
    ave_spectrum=False,
    log_matrix=False,
    batch_matrix=False,
).to(DEVICE)


# ============================================================
# 10. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 11. AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP,
)


# ============================================================
# 12. Checkpoint helper
# ============================================================

def save_checkpoint(
    path,
    epoch,
    train_l1,
    train_ffl,
    train_total,
    val_l1,
    include_optimizer=False,
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "train_loss": train_total,
        "train_l1": train_l1,
        "train_ffl": train_ffl,
        "train_total": train_total,

        "val_loss": val_l1,
        "val_l1": val_l1,

        "prediction_mode": "e2e",

        "loss_type": "L1+FFL",
        "ffl_weight": FFL_WEIGHT,
        "ffl_alpha": FFL_ALPHA,

        "scan_condition": {
            "scan_time": SCAN_TIME,
            "breathing_period": BREATHING_PERIOD,
            "start_phase": START_PHASE,
            "SOD": SOD,
            "du": DU,
        },
    }

    if include_optimizer:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

        checkpoint["scaler_state_dict"] = (
            scaler.state_dict()
        )

    torch.save(checkpoint, path)


# ============================================================
# 13. CSV history
# ============================================================

history = []

CSV_FIELDS = [
    "epoch",
    "train_l1",
    "train_ffl",
    "train_total",
    "val_l1",
]


def save_history():

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 14. Training
# ============================================================

best_val_loss = float("inf")
best_epoch = -1

for epoch in range(1, EPOCHS + 1):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_l1_sum = 0.0
    train_ffl_sum = 0.0
    train_total_sum = 0.0

    train_sample_count = 0

    for step, batch in enumerate(
        train_loader,
        start=1,
    ):

        source = batch["source"].to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        target = batch["target"].to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if not torch.isfinite(source).all():
            raise RuntimeError(
                "Non-finite values found in source."
            )

        if not torch.isfinite(target).all():
            raise RuntimeError(
                "Non-finite values found in target."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward: E2E prediction
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            prediction = model(source)

            loss_l1 = criterion_l1(
                prediction,
                target,
            )

        # ----------------------------------------------------
        # Focal Frequency Loss
        #
        # Disable AMP during FFT.
        # Use float32 for numerical stability.
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=False,
        ):

            loss_ffl = criterion_ffl(
                prediction.float(),
                target.float(),
            )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            loss_l1.float()
            + FFL_WEIGHT * loss_ffl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at "
                f"epoch={epoch}, step={step}"
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = source.shape[0]

        train_l1_sum += (
            loss_l1.item() * batch_size
        )

        train_ffl_sum += (
            loss_ffl.item() * batch_size
        )

        train_total_sum += (
            loss.item() * batch_size
        )

        train_sample_count += batch_size

        # ----------------------------------------------------
        # Step logging
        # ----------------------------------------------------

        if (
            step == 1
            or step % 20 == 0
            or step == len(train_loader)
        ):

            print(
                f"[Train] "
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Step {step:04d}/{len(train_loader):04d} | "
                f"L1={loss_l1.item():.6f} | "
                f"FFL={loss_ffl.item():.6f} | "
                f"Total={loss.item():.6f}"
            )

    train_l1 = (
        train_l1_sum / train_sample_count
    )

    train_ffl = (
        train_ffl_sum / train_sample_count
    )

    train_total = (
        train_total_sum / train_sample_count
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_l1_sum = 0.0
    val_sample_count = 0

    with torch.no_grad():

        for batch in val_loader:

            source = batch["source"].to(
                DEVICE,
                dtype=torch.float32,
                non_blocking=True,
            )

            target = batch["target"].to(
                DEVICE,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):

                prediction = model(source)

                # Validation uses L1 only.
                # This is consistent with the original E2E baseline.
                loss_val = criterion_l1(
                    prediction,
                    target,
                )

            batch_size = source.shape[0]

            val_l1_sum += (
                loss_val.item() * batch_size
            )

            val_sample_count += batch_size

    val_l1 = (
        val_l1_sum / val_sample_count
    )

    # ========================================================
    # Epoch summary
    # ========================================================

    print()
    print("=" * 80)

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train L1={train_l1:.6f} | "
        f"Train FFL={train_ffl:.6f} | "
        f"Train Total={train_total:.6f} | "
        f"Val L1={val_l1:.6f}"
    )

    print("=" * 80)

    # ========================================================
    # Save history
    # ========================================================

    history.append({
        "epoch": epoch,
        "train_l1": train_l1,
        "train_ffl": train_ffl,
        "train_total": train_total,
        "val_l1": val_l1,
    })

    save_history()

    # ========================================================
    # Save last checkpoint
    # ========================================================

    last_path = os.path.join(
        SAVE_DIR,
        "last.pth",
    )

    save_checkpoint(
        path=last_path,
        epoch=epoch,
        train_l1=train_l1,
        train_ffl=train_ffl,
        train_total=train_total,
        val_l1=val_l1,
        include_optimizer=True,
    )

    # ========================================================
    # Save best checkpoint
    #
    # Best model selected using validation L1.
    # ========================================================

    if val_l1 < best_val_loss:

        best_val_loss = val_l1
        best_epoch = epoch

        best_path = os.path.join(
            SAVE_DIR,
            "best.pth",
        )

        save_checkpoint(
            path=best_path,
            epoch=epoch,
            train_l1=train_l1,
            train_ffl=train_ffl,
            train_total=train_total,
            val_l1=val_l1,
            include_optimizer=False,
        )

        print(
            f"✓ Best model saved | "
            f"Epoch={best_epoch} | "
            f"Val L1={best_val_loss:.6f}"
        )

    print("-" * 80)


# ============================================================
# 15. Finish
# ============================================================

print()
print("=" * 80)
print("Training finished.")

print("Best epoch:", best_epoch)
print("Best validation L1:", best_val_loss)

print(
    "Best checkpoint:",
    os.path.join(SAVE_DIR, "best.pth"),
)

print("Training history:", CSV_PATH)

print("=" * 80)
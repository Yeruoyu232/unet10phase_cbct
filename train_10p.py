import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# 1. Config
# ============================================================

DATA_ROOT = (
"/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez"
)

SAVE_DIR = "./checkpoints/unet_10phase_fixed"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 固定扫描条件
# ============================================================

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2


# ============================================================
# 3. Training config
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS = 100

SEED = 42


# ============================================================
# 4. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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
#
# 你已经提前划分好了：
#
# train -> Case 1~8
# val   -> Case 9
# test  -> Case 10
#
# 训练过程中只使用 train + val
# ============================================================

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
print("=" * 80)
print("Dataset")
print("=" * 80)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

print("=" * 80)


# ============================================================
# 7. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)


# ============================================================
# 8. Model
#
# input:
# [B, 10, 256, 256]
#
# output:
# [B, 1, 256, 256]
# ============================================================

model = build_model()

model = model.to(DEVICE)


# ============================================================
# 9. 打印参数量
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
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

print(
    f"Total parameters: "
    f"{total_params / 1e6:.2f} M"
)

print(
    f"Trainable parameters: "
    f"{trainable_params / 1e6:.2f} M"
)

print("=" * 80)


# ============================================================
# 10. Loss
#
# 第一轮实验只使用 L1
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# 11. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 12. AMP
#
# 5090训练建议开启混合精度
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# 13. Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss_sum = 0.0
    train_sample_count = 0


    for step, batch in enumerate(
        train_loader,
        start=1,
    ):

        source = batch["source"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not torch.isfinite(source).all():
            raise RuntimeError(
                "Non-finite value found in source."
            )

        if not torch.isfinite(target).all():
            raise RuntimeError(
                "Non-finite value found in target."
            )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp,
        ):

            prediction = model(source)

            loss = criterion(
                prediction,
                target,
            )


        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at "
                f"epoch={epoch}, step={step}"
            )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = source.shape[0]

        train_loss_sum += (
            loss.item()
            * batch_size
        )

        train_sample_count += batch_size


        # ----------------------------------------------------
        # 每20 step打印一次
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
                f"L1 {loss.item():.6f}"
            )


    train_loss = (
        train_loss_sum
        / train_sample_count
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_sum = 0.0
    val_sample_count = 0


    with torch.no_grad():

        for batch in val_loader:

            source = batch["source"].to(
                DEVICE,
                non_blocking=True,
            )

            target = batch["target"].to(
                DEVICE,
                non_blocking=True,
            )


            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp,
            ):

                prediction = model(
                    source
                )

                loss = criterion(
                    prediction,
                    target,
                )


            batch_size = source.shape[0]

            val_loss_sum += (
                loss.item()
                * batch_size
            )

            val_sample_count += (
                batch_size
            )


    val_loss = (
        val_loss_sum
        / val_sample_count
    )


    # ========================================================
    # Epoch summary
    # ========================================================

    print()
    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train L1 = {train_loss:.6f} | "
        f"Val L1 = {val_loss:.6f}"
    )


    # ========================================================
    # Save last
    # ========================================================

    last_path = os.path.join(
        SAVE_DIR,
        "last.pth",
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "scan_condition": {
                "scan_time":
                    SCAN_TIME,

                "breathing_period":
                    BREATHING_PERIOD,

                "start_phase":
                    START_PHASE,

                "SOD":
                    SOD,

                "du":
                    DU,
            },
        },
        last_path,
    )


    # ========================================================
    # Save best
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_path = os.path.join(
            SAVE_DIR,
            "best.pth",
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "val_loss":
                    val_loss,

                "scan_condition": {
                    "scan_time":
                        SCAN_TIME,

                    "breathing_period":
                        BREATHING_PERIOD,

                    "start_phase":
                        START_PHASE,

                    "SOD":
                        SOD,

                    "du":
                        DU,
                },
            },
            best_path,
        )


        print(
            f"✓ Best model saved | "
            f"Val L1 = {best_val_loss:.6f}"
        )


    print("-" * 80)


# ============================================================
# 14. Finish
# ============================================================

print()
print("=" * 80)

print("Training finished.")

print(
    "Best validation L1:",
    best_val_loss,
)

print(
    "Best checkpoint:",
    os.path.join(
        SAVE_DIR,
        "best.pth",
    ),
)

print("=" * 80)
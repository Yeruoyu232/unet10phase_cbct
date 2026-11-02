import os
import random
import numpy as np
import torch

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from focal_frequency_loss import FocalFrequencyLoss

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
)

SAVE_DIR = "./checkpoints/unet_10phase_fov_l1_ffl"

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0

SOD = 720.0
SDD = 1200.0
NU = 512
DU = 1.2

BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
SEED = 42

FFL_WEIGHT = 0.1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FOV Geometry
# ============================================================

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


def build_fov_mask(case_id, output_size=(256, 256)):

    nx, ny, dx, dy = CASE_GEOMETRY[int(case_id)]

    radius_mm = NU * DU * SOD / (2.0 * SDD)

    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy

    xx, yy = np.meshgrid(x, y)

    mask = (
        xx**2 + yy**2 <= radius_mm**2
    ).astype(np.float32)

    height, width = output_size

    if mask.shape != (height, width):

        mask = np.asarray(
            Image.fromarray(mask, mode="F").resize(
                (width, height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    mask = (mask > 0.5).astype(np.float32)

    return torch.from_numpy(mask).unsqueeze(0)


# ============================================================
# FOV Mask Cache
# ============================================================

class FOVMaskCache:

    def __init__(self):
        self.cache = {}

    def get_batch_masks(
        self,
        case_ids,
        height,
        width,
        device,
    ):

        masks = []

        for case_id in case_ids:

            case_id = int(case_id)
            key = (case_id, height, width)

            if key not in self.cache:
                self.cache[key] = build_fov_mask(
                    case_id,
                    output_size=(height, width),
                )

            masks.append(self.cache[key])

        return torch.stack(
            masks,
            dim=0,
        ).to(device)


# ============================================================
# FOV L1 + FFL
# ============================================================

class FOVL1FFLLoss(nn.Module):

    def __init__(self, ffl_weight=0.1):
        super().__init__()

        self.ffl_weight = ffl_weight

        self.ffl = FocalFrequencyLoss(
            loss_weight=1.0,
            alpha=1.0,
            patch_factor=1,
            ave_spectrum=False,
            log_matrix=False,
            batch_matrix=False,
        )

    def forward(self, pred, target, mask):

        pred = pred.float()
        target = target.float()
        mask = mask.float()

        # ----------------------------
        # FOV L1
        # ----------------------------

        error = torch.abs(pred - target)

        fov_l1 = (
            (error * mask).sum(dim=(1, 2, 3))
            /
            mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        ).mean()

        # ----------------------------
        # Masked FFL
        # ----------------------------

        masked_pred = pred * mask
        masked_target = target * mask

        ffl_loss = self.ffl(
            masked_pred,
            masked_target,
        )

        # ----------------------------
        # Combined loss
        # ----------------------------

        total_loss = (
            fov_l1
            + self.ffl_weight * ffl_loss
        )

        return total_loss, fov_l1, ffl_loss


# ============================================================
# Train / Validation
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    mask_cache,
    optimizer=None,
    scaler=None,
):

    is_train = optimizer is not None

    model.train(is_train)

    total_sum = 0.0
    l1_sum = 0.0
    ffl_sum = 0.0
    sample_count = 0

    for batch in loader:

        source = batch["source"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        masks = mask_cache.get_batch_masks(
            batch["case_id"],
            target.shape[-2],
            target.shape[-1],
            DEVICE,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):

            # Forward: AMP
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                prediction = model(source)

            # Loss: FP32 for FFT
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=False,
            ):
                loss, fov_l1, ffl_loss = criterion(
                    prediction.float(),
                    target.float(),
                    masks,
                )

            if is_train:

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = source.shape[0]

        total_sum += loss.item() * batch_size
        l1_sum += fov_l1.item() * batch_size
        ffl_sum += ffl_loss.item() * batch_size

        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Dataset is empty.")

    return {
        "total": total_sum / sample_count,
        "fov_l1": l1_sum / sample_count,
        "ffl": ffl_sum / sample_count,
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("Training: 10P E2E U-Net + FOV L1 + FFL")
    print("Device:", DEVICE)
    print("FFL weight:", FFL_WEIGHT)
    print("Save directory:", SAVE_DIR)
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    common_args = dict(
        root=DATA_ROOT,
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
    )

    train_dataset = DIRLab10PhaseDataset(
        prompt_file="prompt_train.json",
        **common_args,
    )

    val_dataset = DIRLab10PhaseDataset(
        prompt_file="prompt_val.json",
        **common_args,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model().to(DEVICE)

    criterion = FOVL1FFLLoss(
        ffl_weight=FFL_WEIGHT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    mask_cache = FOVMaskCache()

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            mask_cache=mask_cache,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            mask_cache=mask_cache,
        )

        val_loss = val_metrics["total"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["total"],
            "val_loss": val_loss,
            "train_fov_l1": train_metrics["fov_l1"],
            "val_fov_l1": val_metrics["fov_l1"],
            "train_ffl": train_metrics["ffl"],
            "val_ffl": val_metrics["ffl"],
            "ffl_weight": FFL_WEIGHT,
            "scan_condition": {
                "scan_time": SCAN_TIME,
                "breathing_period": BREATHING_PERIOD,
                "start_phase": START_PHASE,
                "sod": SOD,
                "du": DU,
            },
        }

        torch.save(
            checkpoint,
            os.path.join(SAVE_DIR, "last.pth"),
        )

        is_best = val_loss < best_val_loss

        if is_best:

            best_val_loss = val_loss

            torch.save(
                checkpoint,
                os.path.join(SAVE_DIR, "best.pth"),
            )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Total: {train_metrics['total']:.6f} | "
            f"Train FOV-L1: {train_metrics['fov_l1']:.6f} | "
            f"Train FFL: {train_metrics['ffl']:.6f} | "
            f"Val Total: {val_metrics['total']:.6f} | "
            f"Val FOV-L1: {val_metrics['fov_l1']:.6f} | "
            f"Val FFL: {val_metrics['ffl']:.6f} | "
            f"Best: {best_val_loss:.6f}"
            f"{' *' if is_best else ''}",
            flush=True,
        )

    print("=" * 70)
    print("Training completed.")
    print("Best checkpoint:", os.path.join(SAVE_DIR, "best.pth"))
    print("=" * 70)


if __name__ == "__main__":
    main()
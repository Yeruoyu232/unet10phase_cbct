# import torch
# import numpy as np

# from torch.utils.data import DataLoader
# from skimage.metrics import (
#     peak_signal_noise_ratio,
#     structural_similarity,
# )

# from dataset import DIRLab10PhaseDataset
# from model_5phase import build_model


# # ============================================================
# # Config
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/"
#     "training/cbct_motion_10phase_samez"
# )

# CKPT_PATH = "./checkpoints/unet_5phase_fixed/best.pth"

# BATCH_SIZE = 1
# NUM_WORKERS = 4

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # ============================================================
# # Fixed scan condition
# # ============================================================

# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0
# SOD = 720.0
# DU = 1.2


# # ============================================================
# # Dataset
# # ============================================================

# test_dataset = DIRLab10PhaseDataset(
#     root=DATA_ROOT,
#     prompt_file="prompt_test.json",
#     scan_time=SCAN_TIME,
#     breathing_period=BREATHING_PERIOD,
#     start_phase=START_PHASE,
#     sod=SOD,
#     du=DU,
#     input_mode="5phase",
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True,
# )

# print("=" * 80)
# print("5-Phase U-Net Test")
# print("=" * 80)

# print("Test samples:", len(test_dataset))

# # 检查第一条数据
# sample = test_dataset[0]

# print("Source shape:", sample["source"].shape)
# print("Target shape:", sample["target"].shape)

# assert sample["source"].shape[0] == 5, (
#     f'Expected 5 input channels, '
#     f'but got {sample["source"].shape}'
# )

# assert sample["target"].shape[0] == 1


# # ============================================================
# # Model
# # ============================================================

# model = build_model().to(DEVICE)

# checkpoint = torch.load(
#     CKPT_PATH,
#     map_location="cpu",
# )

# model.load_state_dict(
#     checkpoint["model_state_dict"]
# )

# model.eval()

# print("Checkpoint:", CKPT_PATH)

# if "epoch" in checkpoint:
#     print("Best epoch:", checkpoint["epoch"])

# if "val_loss" in checkpoint:
#     print("Best val L1:", checkpoint["val_loss"])

# print("=" * 80)


# # ============================================================
# # Evaluation
# # ============================================================

# mae_list = []
# psnr_list = []
# ssim_list = []

# z_list = []


# with torch.inference_mode():

#     for batch_idx, batch in enumerate(test_loader):

#         source = batch["source"].to(
#             DEVICE,
#             non_blocking=True,
#         )

#         target = batch["target"].to(
#             DEVICE,
#             non_blocking=True,
#         )

#         # ----------------------------------------------------
#         # Prediction
#         # ----------------------------------------------------

#         prediction = model(source)

#         # [1,1,H,W] -> [H,W]

#         pred_np = (
#             prediction[0, 0]
#             .float()
#             .cpu()
#             .numpy()
#         )

#         target_np = (
#             target[0, 0]
#             .float()
#             .cpu()
#             .numpy()
#         )

#         # ----------------------------------------------------
#         # MAE
#         # ----------------------------------------------------

#         mae = np.mean(
#             np.abs(
#                 pred_np - target_np
#             )
#         )

#         # ----------------------------------------------------
#         # PSNR
#         #
#         # 与之前 T20 / 10-phase 完全保持一致
#         # ----------------------------------------------------

#         psnr = peak_signal_noise_ratio(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )

#         # ----------------------------------------------------
#         # SSIM
#         # ----------------------------------------------------

#         ssim = structural_similarity(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )

#         mae_list.append(float(mae))
#         psnr_list.append(float(psnr))
#         ssim_list.append(float(ssim))

#         # ----------------------------------------------------
#         # Case / z information
#         # ----------------------------------------------------

#         case_id = batch["case_id"]

#         if torch.is_tensor(case_id):
#             case_id = int(case_id.item())
#         else:
#             case_id = int(case_id[0])

#         z_idx = batch["z_idx"]

#         if torch.is_tensor(z_idx):
#             z_idx = int(z_idx.item())
#         else:
#             z_idx = int(z_idx[0])

#         z_list.append(z_idx)

#         # 每张 slice 输出结果
#         print(
#             f"[{batch_idx + 1:03d}/{len(test_loader):03d}] "
#             f"Case={case_id} "
#             f"z={z_idx:03d} | "
#             f"MAE={mae:.6f} | "
#             f"PSNR={psnr:.3f} | "
#             f"SSIM={ssim:.4f}"
#         )


# # ============================================================
# # Overall results
# # ============================================================

# mean_mae = np.mean(mae_list)
# mean_psnr = np.mean(psnr_list)
# mean_ssim = np.mean(ssim_list)


# print()
# print("=" * 80)
# print("UNet 5-phase Test Results")
# print("=" * 80)

# print(f"Samples : {len(mae_list)}")
# print(f"MAE     : {mean_mae:.6f}")
# print(f"PSNR    : {mean_psnr:.4f} dB")
# print(f"SSIM    : {mean_ssim:.6f}")

# print("=" * 80)


# # ============================================================
# # Best / Median / Worst slice
# # ============================================================

# order = np.argsort(mae_list)

# best_idx = order[0]
# median_idx = order[len(order) // 2]
# worst_idx = order[-1]

# print()
# print("Representative slices:")
# print(
#     f"Best   : z={z_list[best_idx]:03d} | "
#     f"MAE={mae_list[best_idx]:.6f} | "
#     f"PSNR={psnr_list[best_idx]:.3f} | "
#     f"SSIM={ssim_list[best_idx]:.4f}"
# )

# print(
#     f"Median : z={z_list[median_idx]:03d} | "
#     f"MAE={mae_list[median_idx]:.6f} | "
#     f"PSNR={psnr_list[median_idx]:.3f} | "
#     f"SSIM={ssim_list[median_idx]:.4f}"
# )

# print(
#     f"Worst  : z={z_list[worst_idx]:03d} | "
#     f"MAE={mae_list[worst_idx]:.6f} | "
#     f"PSNR={psnr_list[worst_idx]:.3f} | "
#     f"SSIM={ssim_list[worst_idx]:.4f}"
# )

# print()
# print("Test finished.")

import os
import csv

import torch
import numpy as np

from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from dataset import DIRLab10PhaseDataset
from model_5phase import build_model


# ============================================================
# 1. Config
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/"
    "training/cbct_motion_10phase_samez"
)

CKPT_PATH = "./checkpoints/unet_5phase_fixed/best.pth"

OUTPUT_DIR = "./test_results/unet_5phase_fixed_fov"

BATCH_SIZE = 1
NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Fixed scan condition
# ============================================================

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0

SOD = 720.0
SDD = 1200.0

NU = 512
DU = 1.2

ORIGINAL_SPACING = 0.97
ORIGINAL_SIZE = 512
OUTPUT_SIZE = 256


# ============================================================
# 3. FOV configuration
# ============================================================

DATA_RANGE = 2.0

SSIM_WIN_SIZE = 7
SSIM_EROSION = 3

# 与 recon_fov0 的生成规则保持一致
FOV_RADIUS = (
    NU * DU * SOD
    / (2.0 * SDD)
    / ORIGINAL_SPACING
    * OUTPUT_SIZE
    / ORIGINAL_SIZE
)

print(f"FOV radius: {FOV_RADIUS:.6f} pixels")


def create_fov_mask(shape):
    """
    Circular FOV mask.

    Center:
        ((W-1)/2, (H-1)/2)
    """
    h, w = shape

    yy, xx = np.ogrid[:h, :w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= FOV_RADIUS ** 2
    )

    return mask


def calculate_metrics(target_np, pred_np):
    """
    Calculate Full and FOV metrics.

    Returns:
        Full_MAE
        Full_PSNR
        Full_SSIM
        FOV_MAE
        FOV_PSNR
        FOV_SSIM
    """

    # --------------------------------------------------------
    # Full image
    # --------------------------------------------------------

    abs_error = np.abs(pred_np - target_np)
    squared_error = (pred_np - target_np) ** 2

    full_mae = float(np.mean(abs_error))

    full_psnr = float(
        peak_signal_noise_ratio(
            target_np,
            pred_np,
            data_range=DATA_RANGE,
        )
    )

    full_ssim, ssim_map = structural_similarity(
        target_np,
        pred_np,
        data_range=DATA_RANGE,
        win_size=SSIM_WIN_SIZE,
        full=True,
    )

    full_ssim = float(full_ssim)

    # --------------------------------------------------------
    # Circular FOV
    # --------------------------------------------------------

    fov_mask = create_fov_mask(target_np.shape)

    fov_mae = float(
        np.mean(abs_error[fov_mask])
    )

    fov_mse = float(
        np.mean(squared_error[fov_mask])
    )

    if fov_mse == 0:
        fov_psnr = float("inf")
    else:
        fov_psnr = float(
            10.0 * np.log10(
                DATA_RANGE ** 2 / fov_mse
            )
        )

    # SSIM map requires a valid local neighborhood.
    # Erode the FOV mask by 3 pixels.
    ssim_mask = binary_erosion(
        fov_mask,
        iterations=SSIM_EROSION,
    )

    fov_ssim = float(
        np.mean(ssim_map[ssim_mask])
    )

    return {
        "Full_MAE": full_mae,
        "Full_PSNR": full_psnr,
        "Full_SSIM": full_ssim,
        "FOV_MAE": fov_mae,
        "FOV_PSNR": fov_psnr,
        "FOV_SSIM": fov_ssim,
    }


# ============================================================
# 4. Dataset
# ============================================================

test_dataset = DIRLab10PhaseDataset(
    root=DATA_ROOT,
    prompt_file="prompt_test.json",
    scan_time=SCAN_TIME,
    breathing_period=BREATHING_PERIOD,
    start_phase=START_PHASE,
    sod=SOD,
    du=DU,
    input_mode="5phase",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

print("=" * 80)
print("5-Phase E2E U-Net Test (Full + Circular FOV)")
print("=" * 80)

print("Device:", DEVICE)
print("Test samples:", len(test_dataset))

sample = test_dataset[0]

print("Source shape:", sample["source"].shape)
print("Target shape:", sample["target"].shape)

assert sample["source"].shape[0] == 5, (
    f'Expected 5 input channels, '
    f'but got {sample["source"].shape}'
)

assert sample["target"].shape[0] == 1

assert tuple(sample["target"].shape[-2:]) == (
    OUTPUT_SIZE,
    OUTPUT_SIZE,
), "Unexpected target image size"


# ============================================================
# 5. Model
# ============================================================

model = build_model().to(DEVICE)

checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=False,
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Checkpoint:", CKPT_PATH)

if isinstance(checkpoint, dict):
    if "epoch" in checkpoint:
        print("Best epoch:", checkpoint["epoch"])

    if "val_loss" in checkpoint:
        print("Best val L1:", checkpoint["val_loss"])

print("=" * 80)


# ============================================================
# 6. Evaluation
# ============================================================

metric_names = [
    "Full_MAE",
    "Full_PSNR",
    "Full_SSIM",
    "FOV_MAE",
    "FOV_PSNR",
    "FOV_SSIM",
]

metric_lists = {
    name: [] for name in metric_names
}

results = []


def get_scalar(value):
    if torch.is_tensor(value):
        return value.flatten()[0].item()

    if isinstance(value, (list, tuple)):
        return get_scalar(value[0])

    return value


with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

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
        # E2E prediction
        # ----------------------------------------------------

        prediction = model(source)

        # ----------------------------------------------------
        # Tensor -> NumPy
        # ----------------------------------------------------

        pred_np = (
            prediction[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        target_np = (
            target[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            target_np,
            pred_np,
        )

        for name in metric_names:
            metric_lists[name].append(metrics[name])

        # ----------------------------------------------------
        # Case / z information
        # ----------------------------------------------------

        case_id = int(
            get_scalar(batch["case_id"])
        )

        z_idx = int(
            get_scalar(batch["z_idx"])
        )

        results.append({
            "case_id": case_id,
            "z_idx": z_idx,
            **metrics,
        })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"[{batch_idx + 1:03d}/{len(test_loader):03d}] "
            f"Case={case_id} z={z_idx:03d} | "
            f"Full MAE={metrics['Full_MAE']:.6f}, "
            f"PSNR={metrics['Full_PSNR']:.3f}, "
            f"SSIM={metrics['Full_SSIM']:.4f} | "
            f"FOV MAE={metrics['FOV_MAE']:.6f}, "
            f"PSNR={metrics['FOV_PSNR']:.3f}, "
            f"SSIM={metrics['FOV_SSIM']:.4f}"
        )


# ============================================================
# 7. Overall results
# ============================================================

mean_metrics = {
    name: float(np.mean(metric_lists[name]))
    for name in metric_names
}

summary = (
    "\n"
    + "=" * 80 + "\n"
    + "5-Phase E2E U-Net Test Results (Full + FOV)\n"
    + "=" * 80 + "\n"
    + f"Samples: {len(results)}\n"
    + f"FOV radius: {FOV_RADIUS:.6f} pixels\n"
    + "FOV center: image center\n"
    + f"SSIM window: {SSIM_WIN_SIZE}\n"
    + f"FOV SSIM erosion: {SSIM_EROSION} pixels\n"
    + f"Data range: {DATA_RANGE}\n"
    + "\n"
    + f"Full_MAE  : {mean_metrics['Full_MAE']:.6f}\n"
    + f"Full_PSNR : {mean_metrics['Full_PSNR']:.4f} dB\n"
    + f"Full_SSIM : {mean_metrics['Full_SSIM']:.6f}\n"
    + "\n"
    + f"FOV_MAE   : {mean_metrics['FOV_MAE']:.6f}\n"
    + f"FOV_PSNR  : {mean_metrics['FOV_PSNR']:.4f} dB\n"
    + f"FOV_SSIM  : {mean_metrics['FOV_SSIM']:.6f}\n"
    + "=" * 80 + "\n"
)

print(summary)


# ============================================================
# 8. Representative slices
# ============================================================

# 根据 FOV MAE 排序
order = np.argsort(
    metric_lists["FOV_MAE"]
)

best_idx = int(order[0])
median_idx = int(order[len(order) // 2])
worst_idx = int(order[-1])

representative_lines = [
    "\nRepresentative slices (ranked by FOV MAE):"
]

for label, idx in [
    ("Best", best_idx),
    ("Median", median_idx),
    ("Worst", worst_idx),
]:
    row = results[idx]

    line = (
        f"{label:6s}: "
        f"Case={row['case_id']} "
        f"z={row['z_idx']:03d} | "
        f"FOV MAE={row['FOV_MAE']:.6f} | "
        f"FOV PSNR={row['FOV_PSNR']:.3f} | "
        f"FOV SSIM={row['FOV_SSIM']:.4f}"
    )

    representative_lines.append(line)
    print(line)


# ============================================================
# 9. Save CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "metrics_per_slice.csv",
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "z_idx",
            *metric_names,
        ],
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# 10. Save summary
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(summary)
    f.write(
        "\n".join(representative_lines) + "\n"
    )


print()
print("Metrics saved:", csv_path)
print("Summary saved:", summary_path)
print("Test finished.")
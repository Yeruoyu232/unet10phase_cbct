# import os
# import numpy as np
# import torch

# from torch.utils.data import DataLoader
# from skimage.metrics import (
#     peak_signal_noise_ratio,
#     structural_similarity,
# )

# from dataset import DIRLab10PhaseDataset
# from model import build_model


# # ============================================================
# # Config
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/"
#     "training/cbct_motion_10phase_samez"
# )

# CKPT_PATH = (
#     "./checkpoints/unet_10phase_fixed/best.pth"
# )

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# # 和训练完全相同的固定扫描条件
# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0
# SOD = 720.0
# DU = 1.2


# # ============================================================
# # Test dataset = Case10
# # ============================================================

# test_dataset = DIRLab10PhaseDataset(
#     root=DATA_ROOT,
#     prompt_file="prompt_test.json",
#     scan_time=SCAN_TIME,
#     breathing_period=BREATHING_PERIOD,
#     start_phase=START_PHASE,
#     sod=SOD,
#     du=DU,
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True,
# )

# print("Test samples:", len(test_dataset))


# # ============================================================
# # Load model
# # ============================================================

# model = build_model().to(DEVICE)

# ckpt = torch.load(
#     CKPT_PATH,
#     map_location="cpu",
# )

# model.load_state_dict(
#     ckpt["model_state_dict"]
# )

# model.eval()

# print("Checkpoint:", CKPT_PATH)
# print("Best epoch:", ckpt.get("epoch"))
# print("Best val L1:", ckpt.get("val_loss"))


# # ============================================================
# # Evaluation
# # ============================================================

# mae_list = []
# psnr_list = []
# ssim_list = []


# with torch.inference_mode():

#     for i, batch in enumerate(test_loader):

#         source = batch["source"].to(
#             DEVICE,
#             non_blocking=True,
#         )

#         target = batch["target"].to(
#             DEVICE,
#             non_blocking=True,
#         )

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


#         # ----------------------------------------
#         # MAE
#         # ----------------------------------------

#         mae = np.mean(
#             np.abs(
#                 pred_np - target_np
#             )
#         )


#         # ----------------------------------------
#         # PSNR / SSIM
#         #
#         # 这里与你之前 ControlNet evaluator 保持一致：
#         # normalized CT/FDK 使用 data_range = 2.0
#         # ----------------------------------------

#         psnr = peak_signal_noise_ratio(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )

#         ssim = structural_similarity(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )


#         mae_list.append(mae)
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)


#         if (
#             i % 20 == 0
#             or i == len(test_dataset) - 1
#         ):
#             print(
#                 f"[{i + 1:03d}/{len(test_dataset):03d}] "
#                 f"MAE={mae:.6f} "
#                 f"PSNR={psnr:.3f} "
#                 f"SSIM={ssim:.4f}"
#             )


# # ============================================================
# # Final
# # ============================================================

# print()
# print("=" * 80)
# print("UNet 10-phase Test Results")
# print("=" * 80)

# print(
#     f"Samples : {len(mae_list)}"
# )

# print(
#     f"MAE     : "
#     f"{np.mean(mae_list):.6f}"
# )

# print(
#     f"PSNR    : "
#     f"{np.mean(psnr_list):.4f} dB"
# )

# print(
#     f"SSIM    : "
#     f"{np.mean(ssim_list):.6f}"
# )

# print("=" * 80)

import os
import csv
import numpy as np
import torch

from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# 1. Config
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/"
    "training/cbct_motion_10phase_samez"
)

CKPT_PATH = (
    "./checkpoints/unet_10phase_fixed/best.pth"
)

OUTPUT_DIR = "./test_results/unet_10phase_fixed_fov"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 与训练完全相同的固定扫描条件
SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2


# ============================================================
# 2. FOV Config
# ============================================================

# Case10:
# 原始图像 512×512
# spacing = 0.97 mm
# resize 到 256×256
# 与 GT FOV Mask 的计算方式保持一致

FOV_RADIUS = 95.01030927835052

DATA_RANGE = 2.0
SSIM_WIN_SIZE = 7
SSIM_EROSION = 3


def create_fov_mask(shape):
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
    # FOV image
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

    # SSIM map 的局部窗口需要完整有效区域
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
# 3. Test dataset = Case10
# ============================================================

test_dataset = DIRLab10PhaseDataset(
    root=DATA_ROOT,
    prompt_file="prompt_test.json",
    scan_time=SCAN_TIME,
    breathing_period=BREATHING_PERIOD,
    start_phase=START_PHASE,
    sod=SOD,
    du=DU,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
)

print("=" * 80)
print("UNet 10-phase E2E Test (Full + FOV)")
print("=" * 80)

print("Device:", DEVICE)
print("Test samples:", len(test_dataset))


# ============================================================
# 4. Load model
# ============================================================

model = build_model().to(DEVICE)

ckpt = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print("Checkpoint:", CKPT_PATH)
print("Best epoch:", ckpt.get("epoch"))
print("Best val L1:", ckpt.get("val_loss"))

print("=" * 80)


# ============================================================
# 5. Evaluation
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

    for i, batch in enumerate(test_loader):

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
        # Full + FOV metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            target_np,
            pred_np,
        )

        for name in metric_names:
            metric_lists[name].append(
                metrics[name]
            )

        # ----------------------------------------------------
        # Case / Z information
        # ----------------------------------------------------

        case_id = get_scalar(
            batch["case_id"]
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

        if (
            i % 20 == 0
            or i == len(test_dataset) - 1
        ):
            print(
                f"[{i + 1:03d}/{len(test_dataset):03d}] "
                f"Case={case_id} z={z_idx:03d} | "
                f"Full MAE={metrics['Full_MAE']:.6f} "
                f"PSNR={metrics['Full_PSNR']:.3f} "
                f"SSIM={metrics['Full_SSIM']:.4f} | "
                f"FOV MAE={metrics['FOV_MAE']:.6f} "
                f"PSNR={metrics['FOV_PSNR']:.3f} "
                f"SSIM={metrics['FOV_SSIM']:.4f}"
            )


# ============================================================
# 6. Final results
# ============================================================

mean_metrics = {
    name: float(np.mean(metric_lists[name]))
    for name in metric_names
}

summary = (
    "\n"
    + "=" * 80 + "\n"
    + "UNet 10-phase E2E Test Results\n"
    + "=" * 80 + "\n"
    + f"Samples: {len(results)}\n"
    + f"FOV radius: {FOV_RADIUS:.6f} pixels; "
      "center: image center\n"
    + f"SSIM window: {SSIM_WIN_SIZE}; "
      f"FOV erosion: {SSIM_EROSION} pixels\n"
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
# 7. Save CSV
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
# 8. Save summary
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


print("Metrics saved:", csv_path)
print("Summary saved:", summary_path)
print("Test finished.")
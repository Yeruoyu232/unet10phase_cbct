# import os
# import csv
# import numpy as np
# import torch

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# from skimage.metrics import (
#     peak_signal_noise_ratio,
#     structural_similarity,
# )

# from dataset import DIRLab10PhaseDataset
# from model import build_model


# # ============================================================
# # 1. Configuration
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/"
#     "training/cbct_motion_10phase_samez"
# )

# PROMPT_FILE = os.path.join(
#     DATA_ROOT,
#     "prompt_test.json",
# )

# CHECKPOINT_PATH = (
#     "./checkpoints/unet_10phase_residual/best.pth"
# )

# OUTPUT_DIR = (
#     "./test_results/unet_10phase_residual"
# )

# # Scan parameters: keep consistent with training
# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0
# SOD = 720.0
# DU = 1.2

# BATCH_SIZE = 1
# NUM_WORKERS = 4

# # Save a figure every 10 slices
# VIS_INTERVAL = 10

# # Additionally save specified slices
# VIS_Z_INDICES = {71, 80}

# # Set True to save every slice
# SAVE_ALL_SLICES = False

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# VIS_DIR = os.path.join(
#     OUTPUT_DIR,
#     "visualizations",
# )

# os.makedirs(VIS_DIR, exist_ok=True)


# # ============================================================
# # 2. Visualization
# # ============================================================

# def save_visualization(
#     case_id,
#     z_idx,
#     phase_mean,
#     target,
#     prediction,
#     pred_residual,
#     save_path,
# ):
#     """
#     Six-panel visualization:

#     1. Phase Mean (10P)
#     2. GT Motion-FDK
#     3. Predicted Motion-FDK
#     4. GT Residual
#     5. Predicted Residual
#     6. Absolute Difference
#     """

#     gt_residual = target - phase_mean

#     difference = np.abs(
#         prediction - target
#     )

#     # Consistent grayscale range
#     image_vmin = min(
#         phase_mean.min(),
#         target.min(),
#         prediction.min(),
#     )

#     image_vmax = max(
#         phase_mean.max(),
#         target.max(),
#         prediction.max(),
#     )

#     # Symmetric residual range
#     residual_max = max(
#         np.abs(gt_residual).max(),
#         np.abs(pred_residual).max(),
#     )

#     residual_max = max(
#         float(residual_max),
#         1e-6,
#     )

#     fig, axes = plt.subplots(
#         2,
#         3,
#         figsize=(16, 10),
#     )

#     images = [
#         phase_mean,
#         target,
#         prediction,
#         gt_residual,
#         pred_residual,
#         difference,
#     ]

#     titles = [
#         "Phase Mean (10P)",
#         "GT Motion-FDK",
#         "Predicted Motion-FDK",
#         "GT Residual",
#         "Predicted Residual",
#         "Absolute Difference",
#     ]

#     for i, (ax, img, title) in enumerate(
#         zip(axes.flat, images, titles)
#     ):

#         if i < 3:
#             im = ax.imshow(
#                 img,
#                 cmap="gray",
#                 vmin=image_vmin,
#                 vmax=image_vmax,
#             )

#         elif i < 5:
#             im = ax.imshow(
#                 img,
#                 cmap="seismic",
#                 vmin=-residual_max,
#                 vmax=residual_max,
#             )

#         else:
#             im = ax.imshow(
#                 img,
#                 cmap="hot",
#                 vmin=0,
#                 vmax=max(
#                     float(difference.max()),
#                     1e-6,
#                 ),
#             )

#         ax.set_title(
#             title,
#             fontsize=12,
#         )

#         ax.axis("off")

#         fig.colorbar(
#             im,
#             ax=ax,
#             fraction=0.046,
#             pad=0.04,
#         )

#     fig.suptitle(
#         f"Case {case_id} | z={z_idx} | 10P Residual",
#         fontsize=15,
#     )

#     plt.tight_layout(
#         rect=[0, 0, 1, 0.96]
#     )

#     plt.savefig(
#         save_path,
#         dpi=200,
#         bbox_inches="tight",
#     )

#     plt.close(fig)


# # ============================================================
# # 3. Helper
# # ============================================================

# def to_scalar(value):
#     """Convert tensor/list/scalar to a Python scalar."""

#     if isinstance(value, torch.Tensor):
#         value = value.detach().cpu()

#         if value.numel() == 1:
#             return value.item()

#         return value.flatten()[0].item()

#     if isinstance(value, (list, tuple)):
#         return to_scalar(value[0])

#     return value


# def format_case_id(value):
#     """Return a stable case identifier for filenames."""

#     value = to_scalar(value)
#     value = str(value)

#     if value.lower().startswith("case"):
#         return value

#     return f"case{value}"


# # ============================================================
# # 4. Main test
# # ============================================================

# def main():

#     print("=" * 60)
#     print("10P Residual U-Net Test")
#     print("=" * 60)

#     print("Device:", DEVICE)
#     print("Checkpoint:", CHECKPOINT_PATH)
#     print("Output:", OUTPUT_DIR)

#     # --------------------------------------------------------
#     # Dataset
#     # --------------------------------------------------------

#     test_dataset = DIRLab10PhaseDataset(
#         root=DATA_ROOT,
#         prompt_file=PROMPT_FILE,
#         scan_time=SCAN_TIME,
#         breathing_period=BREATHING_PERIOD,
#         start_phase=START_PHASE,
#         sod=SOD,
#         du=DU,
#         input_mode="10phase",
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=(DEVICE.type == "cuda"),
#     )

#     print("Test samples:", len(test_dataset))

#     # --------------------------------------------------------
#     # Model
#     # --------------------------------------------------------

#     model = build_model().to(DEVICE)

#     if not os.path.isfile(CHECKPOINT_PATH):
#         raise FileNotFoundError(
#             f"Checkpoint not found: {CHECKPOINT_PATH}"
#         )

#     checkpoint = torch.load(
#         CHECKPOINT_PATH,
#         map_location=DEVICE,
#         weights_only=False,
#     )

#     if (
#         isinstance(checkpoint, dict)
#         and "model_state_dict" in checkpoint
#     ):
#         model.load_state_dict(
#             checkpoint["model_state_dict"]
#         )

#         print(
#             "Best epoch:",
#             checkpoint.get("epoch", "N/A"),
#         )

#         print(
#             "Best Val L1:",
#             checkpoint.get("val_loss", "N/A"),
#         )

#     else:
#         model.load_state_dict(checkpoint)

#     model.eval()

#     # --------------------------------------------------------
#     # Evaluation
#     # --------------------------------------------------------

#     results = []

#     mae_values = []
#     psnr_values = []
#     ssim_values = []

#     with torch.no_grad():

#         for idx, batch in enumerate(test_loader):

#             source = batch["source"].to(
#                 DEVICE,
#                 dtype=torch.float32,
#                 non_blocking=True,
#             )

#             target = batch["target"].to(
#                 DEVICE,
#                 dtype=torch.float32,
#                 non_blocking=True,
#             )

#             # =================================================
#             # IMPORTANT: Residual reconstruction
#             # =================================================

#             phase_mean = source.mean(
#                 dim=1,
#                 keepdim=True,
#             )

#             pred_residual = model(source)

#             prediction = (
#                 phase_mean + pred_residual
#             )

#             # =================================================
#             # Convert to numpy
#             # =================================================

#             phase_mean_np = (
#                 phase_mean[0, 0]
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#             pred_residual_np = (
#                 pred_residual[0, 0]
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#             pred_np = (
#                 prediction[0, 0]
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#             target_np = (
#                 target[0, 0]
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#             # =================================================
#             # Metrics
#             # =================================================

#             mae = float(
#                 np.mean(
#                     np.abs(pred_np - target_np)
#                 )
#             )

#             psnr = float(
#                 peak_signal_noise_ratio(
#                     target_np,
#                     pred_np,
#                     data_range=2.0,
#                 )
#             )

#             ssim = float(
#                 structural_similarity(
#                     target_np,
#                     pred_np,
#                     data_range=2.0,
#                 )
#             )

#             case_id = format_case_id(
#                 batch["case_id"]
#             )

#             z_idx = int(
#                 to_scalar(batch["z_idx"])
#             )

#             mae_values.append(mae)
#             psnr_values.append(psnr)
#             ssim_values.append(ssim)

#             results.append({
#                 "case_id": case_id,
#                 "z_idx": z_idx,
#                 "MAE": mae,
#                 "PSNR": psnr,
#                 "SSIM": ssim,
#             })

#             print(
#                 f"[{idx + 1:03d}/{len(test_loader):03d}] "
#                 f"{case_id} z={z_idx:03d} | "
#                 f"MAE={mae:.6f} | "
#                 f"PSNR={psnr:.4f} | "
#                 f"SSIM={ssim:.4f}"
#             )

#             # =================================================
#             # Save visualization
#             # =================================================

#             should_save = (
#                 SAVE_ALL_SLICES
#                 or z_idx % VIS_INTERVAL == 0
#                 or z_idx in VIS_Z_INDICES
#             )

#             if should_save:

#                 filename = (
#                     f"{case_id}_z{z_idx:03d}.png"
#                 )

#                 save_path = os.path.join(
#                     VIS_DIR,
#                     filename,
#                 )

#                 save_visualization(
#                     case_id=case_id,
#                     z_idx=z_idx,
#                     phase_mean=phase_mean_np,
#                     target=target_np,
#                     prediction=pred_np,
#                     pred_residual=pred_residual_np,
#                     save_path=save_path,
#                 )

#                 print(
#                     f"  [Saved visualization] {save_path}"
#                 )

#     # =========================================================
#     # 5. Save per-slice metrics
#     # =========================================================

#     csv_path = os.path.join(
#         OUTPUT_DIR,
#         "metrics_per_slice.csv",
#     )

#     with open(
#         csv_path,
#         "w",
#         newline="",
#         encoding="utf-8",
#     ) as f:

#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "case_id",
#                 "z_idx",
#                 "MAE",
#                 "PSNR",
#                 "SSIM",
#             ],
#         )

#         writer.writeheader()
#         writer.writerows(results)

#     # =========================================================
#     # 6. Summary
#     # =========================================================

#     mean_mae = float(
#         np.mean(mae_values)
#     )

#     mean_psnr = float(
#         np.mean(psnr_values)
#     )

#     mean_ssim = float(
#         np.mean(ssim_values)
#     )

#     summary = (
#         "\n"
#         "========================================\n"
#         "10P Residual Test Summary\n"
#         "========================================\n"
#         f"Samples : {len(results)}\n"
#         f"MAE     : {mean_mae:.6f}\n"
#         f"PSNR    : {mean_psnr:.4f}\n"
#         f"SSIM    : {mean_ssim:.4f}\n"
#         "========================================\n"
#     )

#     print(summary)

#     summary_path = os.path.join(
#         OUTPUT_DIR,
#         "summary.txt",
#     )

#     with open(
#         summary_path,
#         "w",
#         encoding="utf-8",
#     ) as f:
#         f.write(summary)

#     print("Saved metrics:", csv_path)
#     print("Saved summary:", summary_path)
#     print("Saved visualizations:", VIS_DIR)


# if __name__ == "__main__":
#     main()


import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)
from dataset import DIRLab10PhaseDataset
from model import build_model
# ============================================================
# 1. Configuration
# ============================================================
DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
)
PROMPT_FILE = os.path.join(
    DATA_ROOT,
    "prompt_test.json",
)
CHECKPOINT_PATH = (
    "./checkpoints/unet_10phase_residual/best.pth"
)
OUTPUT_DIR = (
    "./test_results/unet_10phase_residual_fov"
)
# Scan parameters: keep consistent with training
SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2
# Match the actual recon_fov0 preprocessing mask for Case10.
SDD = 1200.0
DETECTOR_PIXELS = 512
ORIGINAL_SPACING_MM = 0.97
ORIGINAL_SIZE = 512
OUTPUT_SIZE = 256
FOV_RADIUS = (DETECTOR_PIXELS * DU * SOD / (2.0 * SDD)
              / ORIGINAL_SPACING_MM * OUTPUT_SIZE / ORIGINAL_SIZE)
FOV_SSIM_EROSION = 3
SSIM_WIN_SIZE = 7
DATA_RANGE = 2.0
BATCH_SIZE = 1
NUM_WORKERS = 4
# Save a figure every 10 slices
VIS_INTERVAL = 10
# Additionally save specified slices
VIS_Z_INDICES = {71, 80}
# Set True to save every slice
SAVE_ALL_SLICES = False
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations",
)
os.makedirs(VIS_DIR, exist_ok=True)
# ============================================================
# 2. Visualization
# ============================================================
def save_visualization(
    case_id,
    z_idx,
    phase_mean,
    target,
    prediction,
    pred_residual,
    save_path,
):
    """
    Six-panel visualization:
    1. Phase Mean (10P)
    2. GT Motion-FDK
    3. Predicted Motion-FDK
    4. GT Residual
    5. Predicted Residual
    6. Absolute Difference
    """
    gt_residual = target - phase_mean
    difference = np.abs(
        prediction - target
    )
    # Consistent grayscale range
    image_vmin = min(
        phase_mean.min(),
        target.min(),
        prediction.min(),
    )
    image_vmax = max(
        phase_mean.max(),
        target.max(),
        prediction.max(),
    )
    # Symmetric residual range
    residual_max = max(
        np.abs(gt_residual).max(),
        np.abs(pred_residual).max(),
    )
    residual_max = max(
        float(residual_max),
        1e-6,
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
    )
    images = [
        phase_mean,
        target,
        prediction,
        gt_residual,
        pred_residual,
        difference,
    ]
    titles = [
        "Phase Mean (10P)",
        "GT Motion-FDK",
        "Predicted Motion-FDK",
        "GT Residual",
        "Predicted Residual",
        "Absolute Difference",
    ]
    for i, (ax, img, title) in enumerate(
        zip(axes.flat, images, titles)
    ):
        if i < 3:
            im = ax.imshow(
                img,
                cmap="gray",
                vmin=image_vmin,
                vmax=image_vmax,
            )
        elif i < 5:
            im = ax.imshow(
                img,
                cmap="seismic",
                vmin=-residual_max,
                vmax=residual_max,
            )
        else:
            im = ax.imshow(
                img,
                cmap="hot",
                vmin=0,
                vmax=max(
                    float(difference.max()),
                    1e-6,
                ),
            )
        ax.set_title(
            title,
            fontsize=12,
        )
        ax.axis("off")
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
    fig.suptitle(
        f"Case {case_id} | z={z_idx} | 10P Residual",
        fontsize=15,
    )
    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
# ============================================================
# 3. Helper
# ============================================================
def to_scalar(value):
    """Convert tensor/list/scalar to a Python scalar."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return to_scalar(value[0])
    return value
def format_case_id(value):
    """Return a stable case identifier for filenames."""
    value = to_scalar(value)
    value = str(value)
    if value.lower().startswith("case"):
        return value
    return f"case{value}"
# ============================================================
# 4. Main test
# ============================================================
def make_fov_mask(shape, radius=FOV_RADIUS):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def compute_metrics(target, prediction, fov_mask):
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    squared = difference ** 2
    full_mae = float(np.mean(np.abs(difference)))
    full_psnr = float(peak_signal_noise_ratio(target, prediction, data_range=DATA_RANGE))
    full_ssim, ssim_map = structural_similarity(
        target, prediction, data_range=DATA_RANGE,
        win_size=SSIM_WIN_SIZE, full=True,
    )
    fov_mae = float(np.mean(np.abs(difference[fov_mask])))
    fov_mse = float(np.mean(squared[fov_mask]))
    fov_psnr = float('inf') if fov_mse == 0 else float(
        10.0 * np.log10(DATA_RANGE ** 2 / fov_mse)
    )
    ssim_mask = binary_erosion(fov_mask, iterations=FOV_SSIM_EROSION)
    fov_ssim = float(np.mean(ssim_map[ssim_mask]))
    return {
        'Full_MAE': full_mae, 'Full_PSNR': full_psnr,
        'Full_SSIM': float(full_ssim),
        'FOV_MAE': fov_mae, 'FOV_PSNR': fov_psnr,
        'FOV_SSIM': fov_ssim,
    }


def main():
    print("=" * 60)
    print("10P Residual U-Net Test")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output:", OUTPUT_DIR)
    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    test_dataset = DIRLab10PhaseDataset(
        root=DATA_ROOT,
        prompt_file=PROMPT_FILE,
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print("Test samples:", len(test_dataset))
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = build_model().to(DEVICE)
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        print(
            "Best epoch:",
            checkpoint.get("epoch", "N/A"),
        )
        print(
            "Best Val L1:",
            checkpoint.get("val_loss", "N/A"),
        )
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    results = []
    metric_keys = ('Full_MAE', 'Full_PSNR', 'Full_SSIM',
                   'FOV_MAE', 'FOV_PSNR', 'FOV_SSIM')
    metric_values = {key: [] for key in metric_keys}
    print(f'FOV radius: {FOV_RADIUS:.6f} pixels; center: image center')
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
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
            # =================================================
            # IMPORTANT: Residual reconstruction
            # =================================================
            phase_mean = source.mean(
                dim=1,
                keepdim=True,
            )
            pred_residual = model(source)
            prediction = (
                phase_mean + pred_residual
            )
            # =================================================
            # Convert to numpy
            # =================================================
            phase_mean_np = (
                phase_mean[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            pred_residual_np = (
                pred_residual[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            pred_np = (
                prediction[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            target_np = (
                target[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            # =================================================
            # Metrics
            # =================================================
            mask = make_fov_mask(target_np.shape)
            metrics = compute_metrics(target_np, pred_np, mask)
            case_id = format_case_id(batch['case_id'])
            z_idx = int(to_scalar(batch['z_idx']))
            for key in metric_keys:
                metric_values[key].append(metrics[key])
            results.append({'case_id': case_id, 'z_idx': z_idx, **metrics})
            print(
                f'[{idx + 1:03d}/{len(test_loader):03d}] '
                f'{case_id} z={z_idx:03d} | '
                f'Full MAE={metrics["Full_MAE"]:.6f}, '
                f'PSNR={metrics["Full_PSNR"]:.3f}, '
                f'SSIM={metrics["Full_SSIM"]:.4f} | '
                f'FOV MAE={metrics["FOV_MAE"]:.6f}, '
                f'PSNR={metrics["FOV_PSNR"]:.3f}, '
                f'SSIM={metrics["FOV_SSIM"]:.4f}'
            )
            # =================================================
            # Save visualization
            # =================================================
            should_save = (
                SAVE_ALL_SLICES
                or z_idx % VIS_INTERVAL == 0
                or z_idx in VIS_Z_INDICES
            )
            if should_save:
                filename = (
                    f"{case_id}_z{z_idx:03d}.png"
                )
                save_path = os.path.join(
                    VIS_DIR,
                    filename,
                )
                save_visualization(
                    case_id=case_id,
                    z_idx=z_idx,
                    phase_mean=phase_mean_np,
                    target=target_np,
                    prediction=pred_np,
                    pred_residual=pred_residual_np,
                    save_path=save_path,
                )
                print(
                    f"  [Saved visualization] {save_path}"
                )
    csv_path = os.path.join(OUTPUT_DIR, 'metrics_per_slice.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'z_idx', *metric_keys])
        writer.writeheader()
        writer.writerows(results)
    summary_lines = [
        '10P Residual Test Results (Full Image vs Circular FOV)',
        '=' * 64,
        f'Samples: {len(results)}',
        f'FOV radius: {FOV_RADIUS:.6f} pixels; center: image center',
        f'SSIM window: {SSIM_WIN_SIZE}; FOV SSIM erosion: {FOV_SSIM_EROSION} pixels',
        f'Data range: {DATA_RANGE}',
        '',
    ]
    for key in metric_keys:
        summary_lines.append(f'{key:12s}: {np.mean(metric_values[key]):.6f}')
    summary = '\n'.join(summary_lines) + '\n'
    print('\n' + summary)
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print('Saved metrics:', csv_path)
    print('Saved summary:', summary_path)
    print('Saved visualizations:', VIS_DIR)


if __name__ == "__main__":
    main()

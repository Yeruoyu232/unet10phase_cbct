# import os
# import csv

# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity

# from dataset import DIRLab10PhaseDataset
# from model_5phase import build_model


# # ============================================================
# # 1. Config
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/training/"
#     "cbct_motion_10phase_samez"
# )

# CHECKPOINT_PATH = "./checkpoints/unet_5phase_residual/best.pth"

# OUTPUT_DIR = "./test_results/unet_5phase_residual"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0
# SOD = 720.0
# DU = 1.2

# BATCH_SIZE = 1
# NUM_WORKERS = 4

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# # 每隔多少个 slice 保存一次可视化
# VIS_INTERVAL = 10


# # ============================================================
# # 2. Dataset
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
#     pin_memory=(DEVICE.type == "cuda"),
# )

# print("=" * 80)
# print("Device:", DEVICE)
# print("Test samples:", len(test_dataset))
# print("Checkpoint:", CHECKPOINT_PATH)
# print("=" * 80)


# # ============================================================
# # 3. Model
# # ============================================================

# model = build_model().to(DEVICE)

# checkpoint = torch.load(
#     CHECKPOINT_PATH,
#     map_location=DEVICE,
#     weights_only=False,
# )

# model.load_state_dict(
#     checkpoint["model_state_dict"]
# )

# model.eval()

# print("Checkpoint epoch:", checkpoint.get("epoch", "N/A"))
# print("Checkpoint val loss:", checkpoint.get("val_loss", "N/A"))


# # ============================================================
# # 4. Visualization
# # ============================================================

# def save_visualization(
#     phase_mean,
#     target,
#     prediction,
#     pred_residual,
#     case_id,
#     z_idx,
# ):
#     gt_residual = target - phase_mean
#     difference = np.abs(prediction - target)

#     # 所有原始图像使用同一显示范围
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

#     # GT residual 和 predicted residual 使用相同显示范围
#     residual_abs_max = max(
#         np.abs(gt_residual).max(),
#         np.abs(pred_residual).max(),
#         1e-8,
#     )

#     fig, axes = plt.subplots(
#         2,
#         3,
#         figsize=(15, 10),
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
#         "Phase Mean (5P)",
#         "GT Motion-FDK",
#         "Predicted Motion-FDK",
#         "GT Residual",
#         "Predicted Residual",
#         "Absolute Difference",
#     ]

#     for i, (ax, image, title) in enumerate(
#         zip(axes.flat, images, titles)
#     ):
#         if i < 3:
#             im = ax.imshow(
#                 image,
#                 cmap="gray",
#                 vmin=image_vmin,
#                 vmax=image_vmax,
#             )
#         elif i < 5:
#             im = ax.imshow(
#                 image,
#                 cmap="seismic",
#                 vmin=-residual_abs_max,
#                 vmax=residual_abs_max,
#             )
#         else:
#             im = ax.imshow(
#                 image,
#                 cmap="hot",
#                 vmin=0,
#                 vmax=max(difference.max(), 1e-8),
#             )

#         ax.set_title(title)
#         ax.axis("off")
#         fig.colorbar(
#             im,
#             ax=ax,
#             fraction=0.046,
#             pad=0.04,
#         )

#     fig.suptitle(
#         f"Case {case_id} | z={z_idx} | 5P Residual",
#         fontsize=15,
#     )

#     plt.tight_layout()

#     save_path = os.path.join(
#         OUTPUT_DIR,
#         f"case{case_id}_z{z_idx:03d}.png",
#     )

#     plt.savefig(
#         save_path,
#         dpi=160,
#         bbox_inches="tight",
#     )

#     plt.close(fig)


# # ============================================================
# # 5. Testing
# # ============================================================

# results = []

# mae_list = []
# psnr_list = []
# ssim_list = []

# with torch.inference_mode():

#     for index, batch in enumerate(test_loader):

#         source = batch["source"].to(DEVICE)
#         target = batch["target"].to(DEVICE)

#         # ----------------------------------------------------
#         # Phase-Mean Residual Reconstruction
#         # ----------------------------------------------------

#         # [1, 5, H, W] -> [1, 1, H, W]
#         phase_mean = source.mean(
#             dim=1,
#             keepdim=True,
#         )

#         # 网络输出的是 residual
#         pred_residual = model(source)

#         # 恢复完整 Motion-FDK
#         prediction = phase_mean + pred_residual

#         # ----------------------------------------------------
#         # Convert to NumPy
#         # ----------------------------------------------------

#         phase_mean_np = (
#             phase_mean[0, 0].cpu().numpy()
#         )

#         pred_residual_np = (
#             pred_residual[0, 0].cpu().numpy()
#         )

#         pred_np = (
#             prediction[0, 0].cpu().numpy()
#         )

#         target_np = (
#             target[0, 0].cpu().numpy()
#         )

#         # ----------------------------------------------------
#         # Metrics
#         # ----------------------------------------------------

#         mae = float(
#             np.mean(np.abs(pred_np - target_np))
#         )

#         psnr = float(
#             peak_signal_noise_ratio(
#                 target_np,
#                 pred_np,
#                 data_range=2.0,
#             )
#         )

#         ssim = float(
#             structural_similarity(
#                 target_np,
#                 pred_np,
#                 data_range=2.0,
#             )
#         )

#         mae_list.append(mae)
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)

#         case_id = int(
#             batch["case_id"][0].item()
#         )

#         z_idx = int(
#             batch["z_idx"][0].item()
#         )

#         results.append({
#             "case_id": case_id,
#             "z_idx": z_idx,
#             "mae": mae,
#             "psnr": psnr,
#             "ssim": ssim,
#         })

#         print(
#             f"[{index + 1:03d}/{len(test_loader):03d}] "
#             f"Case {case_id} | "
#             f"z={z_idx:03d} | "
#             f"MAE={mae:.6f} | "
#             f"PSNR={psnr:.4f} | "
#             f"SSIM={ssim:.6f}"
#         )

#         # ----------------------------------------------------
#         # Visualization
#         # ----------------------------------------------------

#         if index % VIS_INTERVAL == 0:

#             save_visualization(
#                 phase_mean=phase_mean_np,
#                 target=target_np,
#                 prediction=pred_np,
#                 pred_residual=pred_residual_np,
#                 case_id=case_id,
#                 z_idx=z_idx,
#             )


# # ============================================================
# # 6. Overall Metrics
# # ============================================================

# mean_mae = float(np.mean(mae_list))
# mean_psnr = float(np.mean(psnr_list))
# mean_ssim = float(np.mean(ssim_list))

# print()
# print("=" * 80)
# print("5-Phase Residual Test Results")
# print("=" * 80)

# print(f"MAE  : {mean_mae:.6f}")
# print(f"PSNR : {mean_psnr:.4f}")
# print(f"SSIM : {mean_ssim:.6f}")

# print("=" * 80)


# # ============================================================
# # 7. Save CSV
# # ============================================================

# csv_path = os.path.join(
#     OUTPUT_DIR,
#     "metrics_per_slice.csv",
# )

# with open(
#     csv_path,
#     "w",
#     newline="",
#     encoding="utf-8",
# ) as f:

#     writer = csv.DictWriter(
#         f,
#         fieldnames=[
#             "case_id",
#             "z_idx",
#             "mae",
#             "psnr",
#             "ssim",
#         ],
#     )

#     writer.writeheader()
#     writer.writerows(results)


# # ============================================================
# # 8. Save Summary
# # ============================================================

# summary_path = os.path.join(
#     OUTPUT_DIR,
#     "summary.txt",
# )

# with open(
#     summary_path,
#     "w",
#     encoding="utf-8",
# ) as f:

#     f.write("5-Phase Residual Test Results\n")
#     f.write("=" * 50 + "\n")

#     f.write(f"Samples: {len(results)}\n")
#     f.write(f"MAE: {mean_mae:.6f}\n")
#     f.write(f"PSNR: {mean_psnr:.4f}\n")
#     f.write(f"SSIM: {mean_ssim:.6f}\n")


# print("Metrics saved:", csv_path)
# print("Summary saved:", summary_path)
# print("Visualizations saved:", OUTPUT_DIR)
# print("Testing finished.")

"""Evaluate 5-phase residual UNet on Case10, full image and fixed circular FOV."""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model_5phase import build_model

DATA_ROOT = '/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez'
CHECKPOINT_PATH = './checkpoints/unet_5phase_residual/best.pth'
OUTPUT_DIR = Path('./test_results/unet_5phase_residual_fov')
VIS_DIR = OUTPUT_DIR / 'visualizations'
SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2
BATCH_SIZE = 1
NUM_WORKERS = 4
VIS_INTERVAL = 10
DATA_RANGE = 2.0

# IMPORTANT: Verify the actual reconstruction FOV on a GT image before reporting.
# Case10 geometric FOV
SOD = 720.0
SDD = 1200.0
NU = 512
DU = 1.2

ORIGINAL_SPACING = 0.97
ORIGINAL_SIZE = 512
OUTPUT_SIZE = 256

fov_radius_mm = NU * DU * SOD / (2 * SDD)

FOV_RADIUS = (
    fov_radius_mm
    / ORIGINAL_SPACING
    * OUTPUT_SIZE
    / ORIGINAL_SIZE
)

print(f"FOV radius: {FOV_RADIUS:.2f} pixels")

print(f"FOV radius: {FOV_RADIUS:.2f} pixels")
FOV_CENTER = None  # None => ((W-1)/2, (H-1)/2); or (cx, cy)
SSIM_WIN_SIZE = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return scalar(value[0])
    return value


def make_fov_mask(shape):
    h, w = shape
    cx, cy = FOV_CENTER if FOV_CENTER is not None else ((w - 1) / 2, (h - 1) / 2)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= FOV_RADIUS ** 2
    if not np.any(mask):
        raise ValueError('FOV mask is empty: check FOV_RADIUS and FOV_CENTER.')
    return mask


def calculate_metrics(target, pred, fov_mask):
    error = pred.astype(np.float64) - target.astype(np.float64)
    full_mae = float(np.mean(np.abs(error)))
    fov_mae = float(np.mean(np.abs(error[fov_mask])))
    full_psnr = float(peak_signal_noise_ratio(target, pred, data_range=DATA_RANGE))
    fov_mse = float(np.mean(error[fov_mask] ** 2))
    fov_psnr = float('inf') if fov_mse == 0 else float(10 * np.log10(DATA_RANGE ** 2 / fov_mse))

    # SSIM is local-window based: compute its 2D map, then average over valid centers.
    # Erode FOV by half the window size to prevent FOV-external pixels leaking in.
    full_ssim, ssim_map = structural_similarity(
        target, pred, data_range=DATA_RANGE,
        win_size=SSIM_WIN_SIZE, full=True,
    )
    inner_mask = binary_erosion(fov_mask, iterations=SSIM_WIN_SIZE // 2)
    if not np.any(inner_mask):
        raise ValueError('Eroded SSIM FOV mask is empty.')
    fov_ssim = float(np.mean(ssim_map[inner_mask]))
    return {
        'full_mae': full_mae, 'full_psnr': full_psnr, 'full_ssim': float(full_ssim),
        'fov_mae': fov_mae, 'fov_psnr': fov_psnr, 'fov_ssim': fov_ssim,
    }


def save_visualization(phase_mean, target, prediction, pred_residual, mask, case_id, z_idx):
    gt_residual = target - phase_mean
    difference = np.abs(prediction - target)
    image_vmin = min(phase_mean.min(), target.min(), prediction.min())
    image_vmax = max(phase_mean.max(), target.max(), prediction.max())
    residual_abs_max = max(np.abs(gt_residual).max(), np.abs(pred_residual).max(), 1e-8)
    images = [phase_mean, target, prediction, gt_residual, pred_residual, difference]
    titles = ['Phase Mean (5P)', 'GT Motion-FDK', 'Predicted Motion-FDK',
              'GT Residual', 'Predicted Residual', 'Absolute Difference']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
        if i < 3:
            im = ax.imshow(img, cmap='gray', vmin=image_vmin, vmax=image_vmax)
        elif i < 5:
            im = ax.imshow(img, cmap='seismic', vmin=-residual_abs_max, vmax=residual_abs_max)
        else:
            im = ax.imshow(img, cmap='hot', vmin=0, vmax=max(difference.max(), 1e-8))
        # Draw the evaluation boundary, without modifying the displayed pixels.
        ax.contour(mask.astype(float), levels=[0.5], colors=['lime'], linewidths=0.8)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Case {case_id} | z={z_idx} | 5P Residual | Green: FOV boundary', fontsize=14)
    fig.tight_layout()
    fig.savefig(VIS_DIR / f'case{case_id}_z{z_idx:03d}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    test_dataset = DIRLab10PhaseDataset(
        root=DATA_ROOT, prompt_file='prompt_test.json',
        scan_time=SCAN_TIME, breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE, sod=SOD, du=DU, input_mode='5phase',
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',
    )
    model = build_model().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Device: {DEVICE} | Test samples: {len(test_dataset)}')
    print(f'Checkpoint: {CHECKPOINT_PATH} | Epoch: {checkpoint.get("epoch", "N/A")}')
    print(f'FOV radius: {FOV_RADIUS} | FOV center: {FOV_CENTER}')

    results = []
    with torch.inference_mode():
        for index, batch in enumerate(test_loader):
            source = batch['source'].to(DEVICE, non_blocking=True)
            target = batch['target'].to(DEVICE, non_blocking=True)
            if source.shape[0] != 1:
                raise ValueError('This evaluation script expects BATCH_SIZE=1.')
            phase_mean = source.mean(dim=1, keepdim=True)
            pred_residual = model(source)
            prediction = phase_mean + pred_residual  # Preserve residual reconstruction.
            phase_mean_np = phase_mean[0, 0].float().cpu().numpy()
            pred_residual_np = pred_residual[0, 0].float().cpu().numpy()
            pred_np = prediction[0, 0].float().cpu().numpy()
            target_np = target[0, 0].float().cpu().numpy()
            # ============================================================
            # Image Intensity Statistics
            # ============================================================

            def print_stats(name, image):
                print(
                    f"{name:<20} | "
                    f"min={np.min(image):8.4f} | "
                    f"max={np.max(image):8.4f} | "
                    f"mean={np.mean(image):8.4f} | "
                    f"median={np.median(image):8.4f}"
                )


            case_id = int(batch["case_id"][0])
            z_idx = int(batch["z_idx"][0])

            if case_id == 10 and z_idx == 60:

                print("\n" + "=" * 85)
                print(f"Case {case_id} | z={z_idx}")
                print("=" * 85)

                print_stats("Phase Mean (5P)", phase_mean_np)
                print_stats("GT Motion-FDK", target_np)
                print_stats("Predicted Motion-FDK", pred_np)

                print("=" * 85)
            mask = make_fov_mask(target_np.shape)
            metrics = calculate_metrics(target_np, pred_np, mask)
            case_id = scalar(batch['case_id'])
            z_idx = int(scalar(batch['z_idx']))
            results.append({'case_id': case_id, 'z_idx': z_idx, **metrics})
            if index % 20 == 0 or index == len(test_loader) - 1:
                print(f'[{index + 1:03d}/{len(test_loader):03d}] Case {case_id} z={z_idx:03d} | '
                      f'Full MAE={metrics["full_mae"]:.6f}, PSNR={metrics["full_psnr"]:.3f}, SSIM={metrics["full_ssim"]:.4f} | '
                      f'FOV MAE={metrics["fov_mae"]:.6f}, PSNR={metrics["fov_psnr"]:.3f}, SSIM={metrics["fov_ssim"]:.4f}')
            if index % VIS_INTERVAL == 0:
                save_visualization(phase_mean_np, target_np, pred_np, pred_residual_np, mask, case_id, z_idx)

    metric_names = ['full_mae', 'full_psnr', 'full_ssim', 'fov_mae', 'fov_psnr', 'fov_ssim']
    summary = {key: float(np.mean([row[key] for row in results])) for key in metric_names}
    csv_path = OUTPUT_DIR / 'metrics_per_slice.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'z_idx', *metric_names])
        writer.writeheader()
        writer.writerows(results)
    lines = [
        '5-Phase Residual Test Results (Full Image vs Circular FOV)',
        '=' * 64,
        f'Samples: {len(results)}',
        f'FOV radius: {FOV_RADIUS} pixels; center: {FOV_CENTER or "image center"}',
        f'SSIM window: {SSIM_WIN_SIZE}; FOV SSIM mask erosion: {SSIM_WIN_SIZE // 2} pixels',
        f'Data range: {DATA_RANGE}',
        '',
        f'Full MAE : {summary["full_mae"]:.6f}',
        f'Full PSNR: {summary["full_psnr"]:.4f} dB',
        f'Full SSIM: {summary["full_ssim"]:.6f}',
        '',
        f'FOV MAE  : {summary["fov_mae"]:.6f}',
        f'FOV PSNR : {summary["fov_psnr"]:.4f} dB',
        f'FOV SSIM : {summary["fov_ssim"]:.6f}',
    ]
    text = '\n'.join(lines) + '\n'
    print('\n' + text)
    (OUTPUT_DIR / 'summary.txt').write_text(text, encoding='utf-8')
    print('Saved:', csv_path, OUTPUT_DIR / 'summary.txt', VIS_DIR)


if __name__ == '__main__':
    main()

# import os
# import csv
# import argparse

# import numpy as np
# import torch
# import matplotlib

# matplotlib.use("Agg")

# import matplotlib.pyplot as plt

# from PIL import Image
# from scipy.ndimage import binary_erosion
# from skimage.metrics import structural_similarity
# from torch.utils.data import DataLoader

# from dataset import DIRLab10PhaseDataset
# from model import build_model


# # ============================================================
# # Configuration
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
# )

# CHECKPOINT_DIRS = {
#     "Residual_Full_L1": "./checkpoints/unet_10phase_residual",
#     "Residual_FOV_L1": "./checkpoints/unet_10phase_residual_fov",
# }

# OUTPUT_DIR = "./eval_10phase_residual_fov"

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0

# SOD = 720.0
# SDD = 1200.0

# NU = 512
# DU = 1.2

# # Case10 geometry
# NATIVE_NX = 512
# NATIVE_NY = 512

# DX = 0.97
# DY = 0.97

# OUTPUT_SIZE = 256

# DATA_RANGE = 2.0

# METRIC_NAMES = [
#     "full_mae",
#     "full_psnr",
#     "full_ssim",
#     "fov_mae",
#     "fov_psnr",
#     "fov_ssim",
# ]


# # ============================================================
# # FOV Mask
# # ============================================================

# def build_case10_fov_mask():

#     radius_mm = (
#         NU * DU * SOD / (2.0 * SDD)
#     )

#     x = (
#         np.arange(NATIVE_NX)
#         - (NATIVE_NX - 1) / 2.0
#     ) * DX

#     y = (
#         np.arange(NATIVE_NY)
#         - (NATIVE_NY - 1) / 2.0
#     ) * DY

#     xx, yy = np.meshgrid(x, y)

#     mask = (
#         xx ** 2 + yy ** 2 <= radius_mm ** 2
#     ).astype(np.float32)

#     mask = np.asarray(
#         Image.fromarray(
#             mask,
#             mode="F"
#         ).resize(
#             (OUTPUT_SIZE, OUTPUT_SIZE),
#             resample=Image.Resampling.BILINEAR
#         ),
#         dtype=np.float32
#     )

#     mask = mask > 0.5

#     return mask, radius_mm


# # ============================================================
# # Metrics
# # ============================================================

# def calculate_metrics(pred, target, mask):

#     pred = np.asarray(
#         pred,
#         dtype=np.float64
#     )

#     target = np.asarray(
#         target,
#         dtype=np.float64
#     )

#     if pred.shape != target.shape:
#         raise ValueError(
#             f"Shape mismatch: "
#             f"pred={pred.shape}, "
#             f"target={target.shape}"
#         )

#     if mask.shape != target.shape:
#         raise ValueError(
#             f"Mask mismatch: "
#             f"mask={mask.shape}, "
#             f"target={target.shape}"
#         )

#     error = pred - target

#     # --------------------------------------------------------
#     # Full MAE
#     # --------------------------------------------------------

#     full_mae = np.mean(
#         np.abs(error)
#     )

#     # --------------------------------------------------------
#     # Full PSNR
#     # --------------------------------------------------------

#     full_mse = np.mean(
#         error ** 2
#     )

#     if full_mse == 0:
#         full_psnr = float("inf")
#     else:
#         full_psnr = (
#             10.0
#             * np.log10(
#                 DATA_RANGE ** 2 / full_mse
#             )
#         )

#     # --------------------------------------------------------
#     # Full SSIM
#     # --------------------------------------------------------

#     full_ssim, ssim_map = structural_similarity(
#         target,
#         pred,
#         data_range=DATA_RANGE,
#         win_size=7,
#         full=True
#     )

#     # --------------------------------------------------------
#     # FOV MAE
#     # --------------------------------------------------------

#     fov_mae = np.mean(
#         np.abs(error[mask])
#     )

#     # --------------------------------------------------------
#     # FOV PSNR
#     # --------------------------------------------------------

#     fov_mse = np.mean(
#         error[mask] ** 2
#     )

#     if fov_mse == 0:
#         fov_psnr = float("inf")
#     else:
#         fov_psnr = (
#             10.0
#             * np.log10(
#                 DATA_RANGE ** 2 / fov_mse
#             )
#         )

#     # --------------------------------------------------------
#     # FOV SSIM
#     # --------------------------------------------------------

#     eroded_mask = binary_erosion(
#         mask,
#         iterations=3
#     )

#     fov_ssim = np.mean(
#         ssim_map[eroded_mask]
#     )

#     return {
#         "full_mae": float(full_mae),
#         "full_psnr": float(full_psnr),
#         "full_ssim": float(full_ssim),
#         "fov_mae": float(fov_mae),
#         "fov_psnr": float(fov_psnr),
#         "fov_ssim": float(fov_ssim),
#     }


# # ============================================================
# # Load Model
# # ============================================================

# def load_model(checkpoint_path):

#     if not os.path.isfile(checkpoint_path):
#         raise FileNotFoundError(
#             f"Checkpoint not found: {checkpoint_path}"
#         )

#     model = build_model().to(DEVICE)

#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location=DEVICE,
#         weights_only=True
#     )

#     if (
#         isinstance(checkpoint, dict)
#         and "model_state_dict" in checkpoint
#     ):

#         state_dict = checkpoint["model_state_dict"]

#         epoch = checkpoint.get(
#             "epoch",
#             "unknown"
#         )

#         prediction_mode = checkpoint.get(
#             "prediction_mode",
#             "unknown"
#         )

#         if prediction_mode not in (
#             "phase_mean_residual",
#             "unknown",
#         ):
#             raise ValueError(
#                 f"Unexpected prediction mode: "
#                 f"{prediction_mode}"
#             )

#     else:

#         state_dict = checkpoint

#         epoch = "unknown"

#     model.load_state_dict(
#         state_dict,
#         strict=True
#     )

#     model.eval()

#     print(
#         f"Loaded checkpoint: {checkpoint_path}"
#     )

#     print(
#         f"Checkpoint epoch: {epoch}"
#     )

#     return model


# # ============================================================
# # Residual Prediction
# # ============================================================

# def predict_residual_model(model, source):

#     # Source: [B, 10, 256, 256]

#     phase_mean = source.mean(
#         dim=1,
#         keepdim=True
#     )

#     # Predict residual
#     pred_residual = model(source)

#     # Final reconstruction
#     prediction = (
#         phase_mean + pred_residual
#     )

#     return prediction


# # ============================================================
# # Visualization
# # ============================================================

# def save_visualization(
#     source,
#     target,
#     predictions,
#     mask,
#     case_id,
#     z_idx,
#     output_dir
# ):

#     os.makedirs(
#         output_dir,
#         exist_ok=True
#     )

#     num_models = len(predictions)

#     # Input, GT, model predictions
#     # + corresponding error maps
#     num_cols = 2 + num_models

#     fig, axes = plt.subplots(
#         2,
#         num_cols,
#         figsize=(4 * num_cols, 8),
#         squeeze=False
#     )

#     phase_mean = source.mean(axis=0)

#     # --------------------------------------------------------
#     # First row: images
#     # --------------------------------------------------------

#     axes[0, 0].imshow(
#         phase_mean,
#         cmap="gray",
#         vmin=-1,
#         vmax=1
#     )

#     axes[0, 0].set_title(
#         "Input Phase Mean"
#     )

#     axes[0, 1].imshow(
#         target,
#         cmap="gray",
#         vmin=-1,
#         vmax=1
#     )

#     axes[0, 1].set_title(
#         "Ground Truth"
#     )

#     for i, (name, pred) in enumerate(
#         predictions.items(),
#         start=2
#     ):

#         axes[0, i].imshow(
#             pred,
#             cmap="gray",
#             vmin=-1,
#             vmax=1
#         )

#         axes[0, i].set_title(
#             name
#         )

#     # --------------------------------------------------------
#     # Second row: FOV and error maps
#     # --------------------------------------------------------

#     axes[1, 0].imshow(
#         target,
#         cmap="gray",
#         vmin=-1,
#         vmax=1
#     )

#     axes[1, 0].contour(
#         mask.astype(np.float32),
#         levels=[0.5],
#         colors="red",
#         linewidths=1
#     )

#     axes[1, 0].set_title(
#         "GT + FOV Boundary"
#     )

#     axes[1, 1].imshow(
#         target * mask,
#         cmap="gray",
#         vmin=-1,
#         vmax=1
#     )

#     axes[1, 1].set_title(
#         "GT inside FOV"
#     )

#     for i, (name, pred) in enumerate(
#         predictions.items(),
#         start=2
#     ):

#         error = (
#             np.abs(pred - target)
#             * mask
#         )

#         axes[1, i].imshow(
#             error,
#             cmap="hot",
#             vmin=0,
#             vmax=0.3
#         )

#         axes[1, i].set_title(
#             f"{name} | FOV Error"
#         )

#     for ax in axes.flat:
#         ax.axis("off")

#     fig.suptitle(
#         f"Case{case_id} | z={z_idx}",
#         fontsize=14
#     )

#     plt.tight_layout()

#     save_path = os.path.join(
#         output_dir,
#         f"case{case_id}_z{z_idx:04d}.png"
#     )

#     plt.savefig(
#         save_path,
#         dpi=150,
#         bbox_inches="tight"
#     )

#     plt.close(fig)


# # ============================================================
# # CSV
# # ============================================================

# def save_csv(path, rows, fieldnames):

#     with open(
#         path,
#         "w",
#         newline="",
#         encoding="utf-8"
#     ) as f:

#         writer = csv.DictWriter(
#             f,
#             fieldnames=fieldnames
#         )

#         writer.writeheader()

#         writer.writerows(rows)


# # ============================================================
# # Main
# # ============================================================

# def main(args):

#     os.makedirs(
#         args.output_dir,
#         exist_ok=True
#     )

#     vis_dir = os.path.join(
#         args.output_dir,
#         "visualizations"
#     )

#     print("=" * 70)
#     print(
#         "10P Residual: Full L1 vs FOV L1"
#     )
#     print(
#         "Checkpoint type:",
#         args.checkpoint_type
#     )
#     print(
#         "Device:",
#         DEVICE
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Dataset
#     # --------------------------------------------------------

#     test_set = DIRLab10PhaseDataset(
#         root=DATA_ROOT,
#         prompt_file="prompt_test.json",
#         scan_time=SCAN_TIME,
#         breathing_period=BREATHING_PERIOD,
#         start_phase=START_PHASE,
#         sod=SOD,
#         du=DU,
#         input_mode="10phase"
#     )

#     test_loader = DataLoader(
#         test_set,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0
#     )

#     print(
#         "Test samples:",
#         len(test_set)
#     )

#     # --------------------------------------------------------
#     # FOV Mask
#     # --------------------------------------------------------

#     fov_mask, radius_mm = (
#         build_case10_fov_mask()
#     )

#     print(
#         f"FOV coverage: "
#         f"{fov_mask.mean() * 100:.2f}%"
#     )

#     print(
#         f"FOV radius: "
#         f"{radius_mm:.2f} mm"
#     )

#     # --------------------------------------------------------
#     # Load Models
#     # --------------------------------------------------------

#     models = {}

#     for name, checkpoint_dir in CHECKPOINT_DIRS.items():

#         checkpoint_path = os.path.join(
#             checkpoint_dir,
#             f"{args.checkpoint_type}.pth"
#         )

#         models[name] = load_model(
#             checkpoint_path
#         )

#     # --------------------------------------------------------
#     # Evaluation
#     # --------------------------------------------------------

#     results = []

#     with torch.inference_mode():

#         for index, batch in enumerate(test_loader):

#             source = batch["source"].to(
#                 DEVICE
#             )

#             target = (
#                 batch["target"][0, 0]
#                 .float()
#                 .cpu()
#                 .numpy()
#             )

#             case_id = int(
#                 batch["case_id"][0]
#             )

#             z_idx = int(
#                 batch["z_idx"][0]
#             )

#             if case_id != 10:
#                 raise RuntimeError(
#                     f"Expected Case10, "
#                     f"got Case{case_id}"
#                 )

#             predictions = {}

#             # -----------------------------------------------
#             # Predict each model
#             # -----------------------------------------------

#             for model_name, model in models.items():

#                 prediction = predict_residual_model(
#                     model,
#                     source
#                 )

#                 prediction = (
#                     prediction[0, 0]
#                     .float()
#                     .cpu()
#                     .numpy()
#                 )

#                 if not np.isfinite(
#                     prediction
#                 ).all():

#                     raise RuntimeError(
#                         f"Non-finite prediction: "
#                         f"{model_name}, "
#                         f"z={z_idx}"
#                     )

#                 predictions[model_name] = (
#                     prediction
#                 )

#                 metrics = calculate_metrics(
#                     prediction,
#                     target,
#                     fov_mask
#                 )

#                 results.append({
#                     "model": model_name,
#                     "case_id": case_id,
#                     "z_idx": z_idx,
#                     **metrics
#                 })

#             # -----------------------------------------------
#             # Visualization
#             # -----------------------------------------------

#             if (
#                 args.vis_interval > 0
#                 and index % args.vis_interval == 0
#             ):

#                 source_np = (
#                     source[0]
#                     .float()
#                     .cpu()
#                     .numpy()
#                 )

#                 save_visualization(
#                     source=source_np,
#                     target=target,
#                     predictions=predictions,
#                     mask=fov_mask,
#                     case_id=case_id,
#                     z_idx=z_idx,
#                     output_dir=vis_dir
#                 )

#             if (
#                 index == 0
#                 or (index + 1) % 20 == 0
#                 or index + 1 == len(test_loader)
#             ):

#                 print(
#                     f"Evaluated "
#                     f"{index + 1}/{len(test_loader)}"
#                 )

#     # --------------------------------------------------------
#     # Save Per-Slice Metrics
#     # --------------------------------------------------------

#     per_slice_path = os.path.join(
#         args.output_dir,
#         "per_slice_metrics.csv"
#     )

#     save_csv(
#         per_slice_path,
#         results,
#         [
#             "model",
#             "case_id",
#             "z_idx",
#             *METRIC_NAMES
#         ]
#     )

#     # --------------------------------------------------------
#     # Summary
#     # --------------------------------------------------------

#     summary_rows = []

#     print("\n" + "=" * 90)
#     print("FINAL RESULTS")
#     print("=" * 90)

#     for model_name in models:

#         rows = [
#             row
#             for row in results
#             if row["model"] == model_name
#         ]

#         summary = {
#             "model": model_name,
#             "num_slices": len(rows)
#         }

#         for metric in METRIC_NAMES:

#             summary[metric] = float(
#                 np.mean([
#                     row[metric]
#                     for row in rows
#                 ])
#             )

#         summary_rows.append(
#             summary
#         )

#         print(f"\n{model_name}")

#         for metric in METRIC_NAMES:

#             print(
#                 f"  {metric:12s}: "
#                 f"{summary[metric]:.6f}"
#             )

#     # --------------------------------------------------------
#     # Save Summary
#     # --------------------------------------------------------

#     summary_path = os.path.join(
#         args.output_dir,
#         "summary.csv"
#     )

#     save_csv(
#         summary_path,
#         summary_rows,
#         [
#             "model",
#             "num_slices",
#             *METRIC_NAMES
#         ]
#     )

#     # --------------------------------------------------------
#     # FOV Comparison
#     # --------------------------------------------------------

#     if len(summary_rows) == 2:

#         full = next(
#             row
#             for row in summary_rows
#             if row["model"] == "Residual_Full_L1"
#         )

#         fov = next(
#             row
#             for row in summary_rows
#             if row["model"] == "Residual_FOV_L1"
#         )

#         print("\n" + "=" * 90)
#         print("FOV COMPARISON")
#         print("=" * 90)

#         mae_improvement = (
#             (full["fov_mae"] - fov["fov_mae"])
#             / full["fov_mae"]
#             * 100
#         )

#         psnr_difference = (
#             fov["fov_psnr"]
#             - full["fov_psnr"]
#         )

#         ssim_difference = (
#             fov["fov_ssim"]
#             - full["fov_ssim"]
#         )

#         print(
#             f"FOV MAE improvement: "
#             f"{mae_improvement:+.2f}%"
#         )

#         print(
#             f"FOV PSNR difference: "
#             f"{psnr_difference:+.4f} dB"
#         )

#         print(
#             f"FOV SSIM difference: "
#             f"{ssim_difference:+.6f}"
#         )

#     # --------------------------------------------------------
#     # Finish
#     # --------------------------------------------------------

#     print("\n" + "=" * 70)
#     print("Evaluation finished.")
#     print("=" * 70)

#     print(
#         "Per-slice:",
#         per_slice_path
#     )

#     print(
#         "Summary:",
#         summary_path
#     )

#     print(
#         "Visualizations:",
#         vis_dir
#     )


# # ============================================================
# # Entry
# # ============================================================

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--checkpoint_type",
#         type=str,
#         default="last",
#         choices=["best", "last"],
#         help="Checkpoint to evaluate"
#     )

#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default=OUTPUT_DIR
#     )

#     parser.add_argument(
#         "--vis_interval",
#         type=int,
#         default=20,
#         help="Save visualization every N slices"
#     )

#     args = parser.parse_args()

#     main(args)


#!/usr/bin/env python3
"""Evaluate raw/center/zscore 10-phase residual U-Nets on Case10 FOV only."""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model import build_model

DATA_ROOT = '/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024'
DEFAULT_MODELS = {
    'raw': './checkpoints/unet_10phase_residual_fov/best.pth',
    'center': './checkpoints/unet_10phase_residual_center_fov/best.pth',
    'zscore': './checkpoints/unet_10phase_residual_zscore_fov/best.pth',
}
DATA_RANGE = 2.0  # Keep identical to the original evaluation script.


def fov_mask_case10(output_size=256):
    nx = ny = 512
    dx = dy = 0.97
    radius_mm = 512 * 1.2 * 720.0 / (2 * 1200.0)
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    xx, yy = np.meshgrid(x, y)
    native = (xx * xx + yy * yy <= radius_mm * radius_mm).astype(np.float32)
    resized = Image.fromarray(native, mode='F').resize(
        (output_size, output_size), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) > 0.5


def fov_metrics(pred, target, mask):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert pred.shape == target.shape == mask.shape
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    error = pred[mask] - target[mask]
    mae = float(np.abs(error).mean())
    mse = float(np.square(error).mean())
    psnr = float('inf') if mse == 0 else float(10 * np.log10(DATA_RANGE ** 2 / mse))
    # Same SSIM convention as the original script: full SSIM map, FOV eroded 3 pixels.
    _, ssim_map = structural_similarity(
        target, pred, data_range=DATA_RANGE, win_size=7, full=True
    )
    inner_mask = binary_erosion(mask, iterations=3)
    if not inner_mask.any():
        raise ValueError('Empty eroded FOV mask')
    ssim = float(ssim_map[inner_mask].mean())
    return {'fov_mae': mae, 'fov_psnr': psnr, 'fov_ssim': ssim}


def load_checkpoint(path, device, expected_mode):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f'Unexpected checkpoint type: {path}')
    state = ckpt.get('model_state_dict', ckpt)
    # Legacy raw checkpoints did not save normalization metadata.
    mode = ckpt.get('residual_mode')
    if mode is None:
        if expected_mode != 'raw':
            raise ValueError(f'{path} has no residual_mode metadata; cannot safely test {expected_mode}')
        mode = 'raw'
    if mode != expected_mode:
        raise ValueError(f'{path}: expected {expected_mode}, checkpoint says {mode}')
    mean = float(ckpt.get('res_mean', 0.0))
    std = float(ckpt.get('res_std', 1.0))
    if mode == 'zscore' and std <= 0:
        raise ValueError('res_std must be positive')
    if mode != 'raw' and 'res_mean' not in ckpt:
        raise ValueError(f'{path} missing res_mean')
    if mode == 'zscore' and 'res_std' not in ckpt:
        raise ValueError(f'{path} missing res_std')
    model = build_model().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Loaded {mode}: {path} | epoch={ckpt.get("epoch", "?")} mean={mean} std={std}')
    return model, mode, mean, std


def predict(model_spec, source):
    model, mode, mean, std = model_spec
    raw_output = model(source)
    if mode == 'raw':
        residual = raw_output
    elif mode == 'center':
        residual = raw_output + mean
    elif mode == 'zscore':
        residual = raw_output * std + mean
    else:
        raise ValueError(mode)
    return source.mean(dim=1, keepdim=True) + residual


def save_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visualize(path, source, target, preds, mask, case_id, z_idx):
    names = ['CT phase mean', 'FDK target'] + list(preds)
    images = [source.mean(axis=0), target] + list(preds.values())
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 7), squeeze=False)
    for j, (name, arr) in enumerate(zip(names, images)):
        axes[0, j].imshow(np.where(mask, arr, 0), cmap='gray', vmin=-1, vmax=1)
        axes[0, j].set_title(name)
        axes[0, j].axis('off')
        if j >= 2:
            err = np.abs(arr - target) * mask
            axes[1, j].imshow(err, cmap='hot', vmin=0, vmax=0.3)
            axes[1, j].set_title(f'{name} FOV error')
        elif j == 0:
            axes[1, j].imshow(mask, cmap='gray')
            axes[1, j].set_title('FOV mask')
        else:
            axes[1, j].imshow(np.where(mask, target, 0), cmap='gray', vmin=-1, vmax=1)
            axes[1, j].set_title('FDK inside FOV')
        axes[1, j].axis('off')
    fig.suptitle(f'Case{case_id} z={z_idx}')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--output-dir', default='./eval_10phase_residual_modes_fov')
    parser.add_argument('--checkpoint-type', choices=['best', 'last'], default='best')
    parser.add_argument('--modes', nargs='+', choices=['raw', 'center', 'zscore'],
                        default=['raw', 'center', 'zscore'])
    parser.add_argument('--raw-checkpoint', default=None)
    parser.add_argument('--center-checkpoint', default=None)
    parser.add_argument('--zscore-checkpoint', default=None)
    parser.add_argument('--scan-time', type=float, default=30.0)
    parser.add_argument('--breathing-period', type=float, default=4.0)
    parser.add_argument('--start-phase', type=float, default=0.0)
    parser.add_argument('--vis-interval', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    if args.vis_interval > 0:
        vis_dir.mkdir(exist_ok=True)

    test_set = DIRLab10PhaseDataset(
        root=args.data_root, prompt_file='prompt_test.json',
        scan_time=args.scan_time, breathing_period=args.breathing_period,
        start_phase=args.start_phase, sod=720.0, du=1.2, input_mode='10phase'
    )
    if not len(test_set):
        raise RuntimeError('No test samples. Check scan/breathing/start-phase filters.')
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    mask = fov_mask_case10()
    print(f'Test samples: {len(test_set)}; FOV coverage: {mask.mean() * 100:.2f}%')

    models = {}
    for mode in args.modes:
        override = getattr(args, f'{mode}_checkpoint')
        path = override or DEFAULT_MODELS[mode]
        if override is None:
            path = str(Path(path).with_name(f'{args.checkpoint_type}.pth'))
        models[mode] = load_checkpoint(path, device, mode)

    results = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            source = batch['source'].to(device)
            target_tensor = batch['target']
            assert source.ndim == 4 and source.shape[1] == 10
            assert target_tensor.ndim == 4 and target_tensor.shape[1] == 1
            target = target_tensor[0, 0].float().numpy()
            case_id = int(batch['case_id'][0])
            z_idx = int(batch['z_idx'][0])
            if case_id != 10:
                raise RuntimeError(f'Case10 FOV geometry only; encountered Case{case_id}')
            if target.shape != mask.shape:
                raise RuntimeError(f'Unexpected image size: {target.shape}')
            predictions = {}
            for name, spec in models.items():
                pred = predict(spec, source)[0, 0].float().cpu().numpy()
                predictions[name] = pred
                results.append({
                    'model': name, 'case_id': case_id, 'z_idx': z_idx,
                    **fov_metrics(pred, target, mask)
                })
            if args.vis_interval > 0 and index % args.vis_interval == 0:
                visualize(vis_dir / f'case{case_id}_z{z_idx:04d}.png',
                          source[0].float().cpu().numpy(), target,
                          predictions, mask, case_id, z_idx)
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(loader):
                print(f'Evaluated {index + 1}/{len(loader)}', flush=True)

    metrics = ['fov_mae', 'fov_psnr', 'fov_ssim']
    save_csv(output_dir / 'per_slice_metrics.csv', results,
             ['model', 'case_id', 'z_idx'] + metrics)
    summary = []
    for mode in args.modes:
        rows = [r for r in results if r['model'] == mode]
        item = {'model': mode, 'num_slices': len(rows)}
        for metric in metrics:
            item[metric] = float(np.mean([r[metric] for r in rows]))
        summary.append(item)
        print(f'{mode}: ' + ' | '.join(f'{k}={item[k]:.6f}' for k in metrics))
    save_csv(output_dir / 'summary.csv', summary, ['model', 'num_slices'] + metrics)
    print(f'Saved FOV-only results in {output_dir.resolve()}')


if __name__ == '__main__':
    main()

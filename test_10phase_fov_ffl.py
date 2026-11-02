import os
import csv
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
)

# CHECKPOINTS = {
#     "Full_L1": "./checkpoints/unet_10phase_fixed/best.pth",
#     "FOV_L1": "./checkpoints/unet_10phase_fov_l1/best.pth",
#     "FOV_L1_FFL": "./checkpoints/unet_10phase_fov_l1_ffl/best.pth",
# }
CHECKPOINTS = {
    "Full_L1": "./checkpoints/unet_10phase_fixed/best.pth",
    "FOV_L1_Best": "./checkpoints/unet_10phase_fov_l1_ffl/best.pth",
    "FOV_L1_Last": "./checkpoints/unet_10phase_fov_l1_ffl/last.pth",
}

OUTPUT_DIR = "./eval_10phase_fov_l1_ffl"

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0

SOD = 720.0
SDD = 1200.0
NU = 512
DU = 1.2

DATA_RANGE = 2.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

METRIC_NAMES = [
    "full_mae",
    "full_psnr",
    "full_ssim",
    "fov_mae",
    "fov_psnr",
    "fov_ssim",
]


# ============================================================
# Case10 FOV mask
# ============================================================

def build_case10_fov_mask():

    nx = 512
    ny = 512

    dx = 0.97
    dy = 0.97

    radius_mm = NU * DU * SOD / (2.0 * SDD)

    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy

    xx, yy = np.meshgrid(x, y)

    mask = (
        xx**2 + yy**2 <= radius_mm**2
    ).astype(np.float32)

    mask = np.asarray(
        Image.fromarray(mask, mode="F").resize(
            (256, 256),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )

    return mask > 0.5


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(pred, target, mask):

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    error = pred - target

    # Full MAE
    full_mae = np.mean(np.abs(error))

    # Full PSNR
    full_mse = np.mean(error**2)

    full_psnr = (
        float("inf")
        if full_mse == 0
        else 10 * np.log10(DATA_RANGE**2 / full_mse)
    )

    # SSIM map
    full_ssim, ssim_map = structural_similarity(
        target,
        pred,
        data_range=DATA_RANGE,
        win_size=7,
        full=True,
    )

    # FOV MAE
    fov_mae = np.mean(np.abs(error[mask]))

    # FOV PSNR
    fov_mse = np.mean(error[mask] ** 2)

    fov_psnr = (
        float("inf")
        if fov_mse == 0
        else 10 * np.log10(DATA_RANGE**2 / fov_mse)
    )

    # FOV SSIM
    eroded_mask = binary_erosion(
        mask,
        iterations=3,
    )

    fov_ssim = np.mean(
        ssim_map[eroded_mask]
    )

    return {
        "full_mae": float(full_mae),
        "full_psnr": float(full_psnr),
        "full_ssim": float(full_ssim),
        "fov_mae": float(fov_mae),
        "fov_psnr": float(fov_psnr),
        "fov_ssim": float(fov_ssim),
    }


# ============================================================
# Load checkpoint
# ============================================================

def load_model(checkpoint_path):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", "unknown")
    else:
        state_dict = checkpoint
        epoch = "unknown"
        val_loss = "unknown"

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(f"Loaded: {checkpoint_path}")
    print(f"  Epoch: {epoch}")
    print(f"  Val Loss: {val_loss}")

    return model


# ============================================================
# Visualization
# ============================================================

def save_visualization(
    target,
    predictions,
    mask,
    case_id,
    z_idx,
    output_dir,
):

    model_names = list(predictions.keys())

    ncols = len(model_names) + 1

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4 * ncols, 8),
    )

    axes[0, 0].imshow(
        target,
        cmap="gray",
        vmin=-1,
        vmax=2,
    )
    axes[0, 0].set_title("GT")

    axes[1, 0].imshow(
        mask,
        cmap="gray",
    )
    axes[1, 0].set_title("FOV Mask")

    all_errors = [
        np.abs(predictions[name] - target)
        for name in model_names
    ]

    vmax = max(
        max(
            np.percentile(error[mask], 99)
            for error in all_errors
        ),
        1e-6,
    )

    for col, name in enumerate(model_names, start=1):

        pred = predictions[name]
        error = np.abs(pred - target)

        axes[0, col].imshow(
            pred,
            cmap="gray",
            vmin=-1,
            vmax=2,
        )
        axes[0, col].set_title(name)

        axes[1, col].imshow(
            error * mask,
            cmap="hot",
            vmin=0,
            vmax=vmax,
        )
        axes[1, col].set_title(
            f"{name} Error (FOV)"
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    filename = f"case{case_id}_z{z_idx:03d}.png"

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Evaluation
# ============================================================

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    vis_dir = os.path.join(
        args.output_dir,
        "visualizations",
    )
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 75)
    print("10-Phase CBCT Reconstruction Evaluation")
    print("Device:", DEVICE)
    print("=" * 75)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = DIRLab10PhaseDataset(
        root=DATA_ROOT,
        prompt_file="prompt_test.json",
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty.")

    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    selected_models = {
        name: path
        for name, path in CHECKPOINTS.items()
        if name in args.models
    }

    models = {
        name: load_model(path)
        for name, path in selected_models.items()
    }

    # --------------------------------------------------------
    # FOV mask
    # --------------------------------------------------------

    fov_mask = build_case10_fov_mask()

    print(
        "FOV coverage:",
        f"{fov_mask.mean() * 100:.2f}%",
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    results = []

    # with torch.inference_mode():

    #     for index, batch in enumerate(test_loader):

    #         source = batch["source"].to(DEVICE)

    #         target = (
    #             batch["target"][0, 0]
    #             .float()
    #             .numpy()
    #         )

    #         case_id = int(batch["case_id"][0])
    #         z_idx = int(batch["z_idx"][0])

    #         if case_id != 10:
    #             raise RuntimeError(
    #                 f"Expected Case10, got Case{case_id}"
    #             )

    #         predictions = {}
    with torch.inference_mode():

        for index, batch in enumerate(test_loader):

            source = batch["source"].to(DEVICE)

            target = (
                batch["target"][0, 0]
                .float()
                .numpy()
            )

            case_id = int(batch["case_id"][0])
            z_idx = int(batch["z_idx"][0])

            if case_id != 10:
                raise RuntimeError(
                    f"Expected Case10, got Case{case_id}"
                )

            # ========================================
            # 新增：检查 GT 与 FOV Mask 是否对齐
            # ========================================
            if index % 20 == 0:
                check_fov_alignment(
                    gt=target,
                    mask=fov_mask,
                    case_id=case_id,
                    z_idx=z_idx,
                )

            # ========================================
            # 原来的模型推理代码
            # ========================================
            predictions = {}

            for name, model in models.items():
                prediction = (
                    model(source)[0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                predictions[name] = prediction

            # 后面的指标计算保持不变

            for name, model in models.items():

                prediction = (
                    model(source)[0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                if not np.isfinite(prediction).all():
                    raise RuntimeError(
                        f"Non-finite prediction: "
                        f"{name}, z={z_idx}"
                    )

                predictions[name] = prediction

                metrics = calculate_metrics(
                    prediction,
                    target,
                    fov_mask,
                )

                results.append({
                    "model": name,
                    "case_id": case_id,
                    "z_idx": z_idx,
                    **metrics,
                })

            if (index + 1) % 20 == 0:
                print(
                    f"Evaluated {index + 1}/"
                    f"{len(test_dataset)}",
                    flush=True,
                )

            if index % args.vis_interval == 0:

                save_visualization(
                    target=target,
                    predictions=predictions,
                    mask=fov_mask,
                    case_id=case_id,
                    z_idx=z_idx,
                    output_dir=vis_dir,
                )

    # --------------------------------------------------------
    # Per-slice CSV
    # --------------------------------------------------------

    per_slice_path = os.path.join(
        args.output_dir,
        "per_slice_metrics.csv",
    )

    with open(
        per_slice_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "case_id",
                "z_idx",
                *METRIC_NAMES,
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = []

    print("\n" + "=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    for name in models:

        rows = [
            r for r in results
            if r["model"] == name
        ]

        averages = {
            metric: float(
                np.mean([r[metric] for r in rows])
            )
            for metric in METRIC_NAMES
        }

        summary.append({
            "model": name,
            "num_slices": len(rows),
            **averages,
        })

        print(f"\n{name}")

        for metric, value in averages.items():
            print(f"  {metric:12s}: {value:.6f}")

    summary_path = os.path.join(
        args.output_dir,
        "summary.csv",
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "num_slices",
                *METRIC_NAMES,
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nEvaluation completed.")
    print("Summary:", summary_path)
    print("Per-slice:", per_slice_path)
    print("Visualizations:", vis_dir)

import os
import numpy as np
import matplotlib.pyplot as plt


def check_fov_alignment(gt, mask, case_id, z_idx, save_dir="./fov_check"):
    os.makedirs(save_dir, exist_ok=True)

    gt = np.asarray(gt).squeeze()
    mask = np.asarray(mask).squeeze().astype(bool)

    assert gt.shape == mask.shape, (
        f"Shape mismatch: GT={gt.shape}, Mask={mask.shape}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 原始 GT
    axes[0].imshow(gt, cmap="gray", vmin=-1, vmax=2)
    axes[0].set_title("GT")

    # 2. GT + FOV 边界
    axes[1].imshow(gt, cmap="gray", vmin=-1, vmax=2)
    axes[1].contour(
        mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=1.5,
    )
    axes[1].set_title("GT + FOV Boundary")

    # 3. FOV 内部 GT
    masked_gt = np.where(mask, gt, 0.0)

    axes[2].imshow(masked_gt, cmap="gray", vmin=-1, vmax=2)
    axes[2].set_title("GT Inside FOV")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        f"case{case_id}_z{z_idx:03d}.png",
    )
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"Saved: {path}")
    print(f"GT shape: {gt.shape}")
    print(f"Mask coverage: {mask.mean() * 100:.2f}%")
# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CHECKPOINTS.keys()),
        # default=[
        #     "Full_L1",
        #     "FOV_L1",
        #     "FOV_L1_FFL",
        # ],
        default=[
    "Full_L1",
    "FOV_L1_Best",
    "FOV_L1_Last",
],
    )

    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--vis_interval",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.vis_interval < 1:
        parser.error("--vis_interval must be >= 1")

    main(args)


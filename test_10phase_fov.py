import os
import csv
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# 1. Configuration
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"
)

BASELINE_CKPT = "./checkpoints/unet_10phase_fixed/last.pth"

FOV_CKPT = "./checkpoints/unet_10phase_fov_l1/last.pth"

OUTPUT_DIR = "./eval_10phase_fov"

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


# ============================================================
# 2. FOV mask
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

    mask = mask > 0.5

    return mask


# ============================================================
# 3. Metrics
# ============================================================

def calculate_metrics(pred, target, mask):

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    error = pred - target

    # Full MAE
    full_mae = np.mean(np.abs(error))

    # Full PSNR
    full_mse = np.mean(error**2)

    full_psnr = (
        float("inf")
        if full_mse == 0
        else 10.0 * np.log10(DATA_RANGE**2 / full_mse)
    )

    # Full SSIM
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
        else 10.0 * np.log10(DATA_RANGE**2 / fov_mse)
    )

    # FOV SSIM
    # Remove 3 pixels to avoid boundary-window contamination
    eroded_mask = binary_erosion(
        mask,
        iterations=3,
    )

    fov_ssim = np.mean(ssim_map[eroded_mask])

    return {
        "full_mae": float(full_mae),
        "full_psnr": float(full_psnr),
        "full_ssim": float(full_ssim),
        "fov_mae": float(fov_mae),
        "fov_psnr": float(fov_psnr),
        "fov_ssim": float(fov_ssim),
    }


# ============================================================
# 4. Load model
# ============================================================

def load_model(checkpoint_path):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    if isinstance(checkpoint, dict):
        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch", "unknown"),
        )

    return model


# ============================================================
# 5. Visualization
# ============================================================

def save_visualization(
    target,
    baseline_pred,
    fov_pred,
    mask,
    case_id,
    z_idx,
    output_dir,
):

    baseline_error = np.abs(
        baseline_pred - target
    )

    fov_error = np.abs(
        fov_pred - target
    )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 8),
    )

    axes[0, 0].imshow(
        target,
        cmap="gray",
        vmin=-1,
        vmax=2,
    )
    axes[0, 0].set_title("GT")

    axes[0, 1].imshow(
        baseline_pred,
        cmap="gray",
        vmin=-1,
        vmax=2,
    )
    axes[0, 1].set_title("Full L1")

    axes[0, 2].imshow(
        fov_pred,
        cmap="gray",
        vmin=-1,
        vmax=2,
    )
    axes[0, 2].set_title("FOV L1")

    axes[1, 0].imshow(
        mask,
        cmap="gray",
    )
    axes[1, 0].set_title("FOV Mask")

    vmax = max(
        np.percentile(baseline_error[mask], 99),
        np.percentile(fov_error[mask], 99),
        1e-6,
    )

    axes[1, 1].imshow(
        baseline_error * mask,
        cmap="hot",
        vmin=0,
        vmax=vmax,
    )
    axes[1, 1].set_title("Full L1 Error")

    axes[1, 2].imshow(
        fov_error * mask,
        cmap="hot",
        vmin=0,
        vmax=vmax,
    )
    axes[1, 2].set_title("FOV L1 Error")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    filename = (
        f"case{case_id}_z{z_idx:03d}.png"
    )

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# 6. Main evaluation
# ============================================================

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    vis_dir = os.path.join(
        args.output_dir,
        "visualizations",
    )

    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 70)
    print("10P E2E: Full L1 vs FOV L1")
    print("Device:", DEVICE)
    print("=" * 70)

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

    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    baseline_model = load_model(
        args.baseline_ckpt
    )

    fov_model = load_model(
        args.fov_ckpt
    )

    # --------------------------------------------------------
    # FOV
    # --------------------------------------------------------

    fov_mask = build_case10_fov_mask()

    print(
        "FOV coverage:",
        f"{fov_mask.mean() * 100:.2f}%",
    )

    print(
        "FOV radius:",
        f"{NU * DU * SOD / (2 * SDD):.2f} mm",
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    with torch.inference_mode():

        for index, batch in enumerate(test_loader):

            source = batch["source"].to(DEVICE)

            target = batch["target"][0, 0].numpy()

            case_id = int(
                batch["case_id"][0]
            )

            z_idx = int(
                batch["z_idx"][0]
            )

            # 防止错误地评估其他 Case
            if case_id != 10:
                raise RuntimeError(
                    f"Expected Case10, got Case{case_id}"
                )

            # E2E prediction
            baseline_pred = (
                baseline_model(source)[0, 0]
                .float()
                .cpu()
                .numpy()
            )

            fov_pred = (
                fov_model(source)[0, 0]
                .float()
                .cpu()
                .numpy()
            )

            if not (
                np.isfinite(target).all()
                and np.isfinite(baseline_pred).all()
                and np.isfinite(fov_pred).all()
            ):
                raise RuntimeError(
                    f"Non-finite values at z={z_idx}"
                )

            baseline_metrics = calculate_metrics(
                baseline_pred,
                target,
                fov_mask,
            )

            fov_metrics = calculate_metrics(
                fov_pred,
                target,
                fov_mask,
            )

            for model_name, metrics in [
                ("Full_L1", baseline_metrics),
                ("FOV_L1", fov_metrics),
            ]:

                results.append({
                    "model": model_name,
                    "case_id": case_id,
                    "z_idx": z_idx,
                    **metrics,
                })

            if index % 20 == 0:
                print(
                    f"Evaluated {index + 1}/"
                    f"{len(test_dataset)}"
                )

            # Save selected visualizations
            if index % args.vis_interval == 0:

                save_visualization(
                    target=target,
                    baseline_pred=baseline_pred,
                    fov_pred=fov_pred,
                    mask=fov_mask,
                    case_id=case_id,
                    z_idx=z_idx,
                    output_dir=vis_dir,
                )

    # --------------------------------------------------------
    # Save per-slice CSV
    # --------------------------------------------------------

    metric_names = [
        "full_mae",
        "full_psnr",
        "full_ssim",
        "fov_mae",
        "fov_psnr",
        "fov_ssim",
    ]

    csv_path = os.path.join(
        args.output_dir,
        "per_slice_metrics.csv",
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
                "model",
                "case_id",
                "z_idx",
                *metric_names,
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = []

    print()
    print("=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)

    for model_name in ["Full_L1", "FOV_L1"]:

        model_results = [
            r for r in results
            if r["model"] == model_name
        ]

        averages = {
            key: float(np.mean([
                r[key] for r in model_results
            ]))
            for key in metric_names
        }

        summary.append({
            "model": model_name,
            "num_slices": len(model_results),
            **averages,
        })

        print()
        print(model_name)

        for key, value in averages.items():
            print(
                f"  {key:12s}: {value:.6f}"
            )

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
                *metric_names,
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 70)
    print("Evaluation finished.")
    print("Per-slice:", csv_path)
    print("Summary:", summary_path)
    print("Visualizations:", vis_dir)
    print("=" * 70)


# ============================================================
# 7. Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default=BASELINE_CKPT,
    )

    parser.add_argument(
        "--fov_ckpt",
        type=str,
        default=FOV_CKPT,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
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
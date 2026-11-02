import os
import csv
import argparse

import numpy as np
import torch

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
    "/home/yry/ENV/ControlNet-main_10phase/"
    "training/cbct_motion_10phase_samez"
)

# CHECKPOINTS = {
#     "Full_L1": "./checkpoints/unet_10phase_fixed/best.pth",
#     "FOV_L1": "./checkpoints/unet_10phase_fov_l1/best.pth",
# }

# CHECKPOINTS = {
#     "best": "./checkpoints/unet_10phase_fov_l1/best.pth",
#     "last": "./checkpoints/unet_10phase_fov_l1/last.pth",
# }
CHECKPOINTS = {
    "Full_L1": "./checkpoints/unet_10phase_fixed/best.pth",
    "FOV_L1_Best": "./checkpoints/unet_10phase_fov_l1/best.pth",
    "FOV_L1_Last": "./checkpoints/unet_10phase_fov_l1/last.pth",
}

OUTPUT_DIR = "./eval_train_case678_bl"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset parameters
SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0

SOD = 720.0
SDD = 1200.0

NU = 512
DU = 1.2

# Case6, Case7, Case8 geometry
NATIVE_NX = 512
NATIVE_NY = 512

DX = 0.97
DY = 0.97

OUTPUT_SIZE = 256

DATA_RANGE = 2.0

TARGET_CASES = {6, 7, 8}

METRIC_NAMES = [
    "full_mae",
    "full_psnr",
    "full_ssim",
    "fov_mae",
    "fov_psnr",
    "fov_ssim",
]


# ============================================================
# FOV Mask
# ============================================================

def build_fov_mask():

    radius_mm = (
        NU * DU * SOD / (2.0 * SDD)
    )

    x = (
        np.arange(NATIVE_NX)
        - (NATIVE_NX - 1) / 2.0
    ) * DX

    y = (
        np.arange(NATIVE_NY)
        - (NATIVE_NY - 1) / 2.0
    ) * DY

    xx, yy = np.meshgrid(x, y)

    mask = (
        xx ** 2 + yy ** 2 <= radius_mm ** 2
    ).astype(np.float32)

    # Resize 512x512 -> 256x256
    mask = np.asarray(
        Image.fromarray(
            mask,
            mode="F",
        ).resize(
            (OUTPUT_SIZE, OUTPUT_SIZE),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )

    mask = mask > 0.5

    return mask


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(pred, target, mask):

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} "
            f"!= target shape {target.shape}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            f"Mask shape {mask.shape} "
            f"!= target shape {target.shape}"
        )

    error = pred - target

    # --------------------------------------------------------
    # Full MAE
    # --------------------------------------------------------

    full_mae = np.mean(
        np.abs(error)
    )

    # --------------------------------------------------------
    # Full PSNR
    # --------------------------------------------------------

    full_mse = np.mean(
        error ** 2
    )

    if full_mse == 0:
        full_psnr = float("inf")
    else:
        full_psnr = (
            10.0
            * np.log10(
                DATA_RANGE ** 2 / full_mse
            )
        )

    # --------------------------------------------------------
    # Full SSIM + SSIM Map
    # --------------------------------------------------------

    full_ssim, ssim_map = (
        structural_similarity(
            target,
            pred,
            data_range=DATA_RANGE,
            win_size=7,
            full=True,
        )
    )

    # --------------------------------------------------------
    # FOV MAE
    # --------------------------------------------------------

    fov_mae = np.mean(
        np.abs(error[mask])
    )

    # --------------------------------------------------------
    # FOV PSNR
    # --------------------------------------------------------

    fov_mse = np.mean(
        error[mask] ** 2
    )

    if fov_mse == 0:
        fov_psnr = float("inf")
    else:
        fov_psnr = (
            10.0
            * np.log10(
                DATA_RANGE ** 2 / fov_mse
            )
        )

    # --------------------------------------------------------
    # FOV SSIM
    # --------------------------------------------------------

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
# Load Model
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

        epoch = checkpoint.get(
            "epoch",
            "unknown",
        )

    else:

        state_dict = checkpoint
        epoch = "unknown"

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Loaded: {checkpoint_path}"
    )

    print(
        f"Best Epoch: {epoch}"
    )

    return model


# ============================================================
# CSV
# ============================================================

def save_csv(path, rows, fieldnames):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Summary
# ============================================================

def summarize_rows(rows):

    if not rows:
        return None

    summary = {
        "num_slices": len(rows)
    }

    for metric in METRIC_NAMES:

        values = [
            row[metric]
            for row in rows
        ]

        summary[metric] = float(
            np.mean(values)
        )

    return summary


# ============================================================
# Main
# ============================================================

def main(args):

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print("=" * 75)
    print("Training Set Evaluation: Case6-8")
    print("=" * 75)

    print("Device:", DEVICE)
    print("Cases:", sorted(TARGET_CASES))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DIRLab10PhaseDataset(
        root=DATA_ROOT,
        prompt_file="prompt_train.json",
        scan_time=SCAN_TIME,
        breathing_period=BREATHING_PERIOD,
        start_phase=START_PHASE,
        sod=SOD,
        du=DU,
        input_mode="10phase",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Total training samples:",
        len(dataset),
    )

    # --------------------------------------------------------
    # FOV Mask
    # --------------------------------------------------------

    fov_mask = build_fov_mask()

    print(
        "FOV coverage:",
        f"{fov_mask.mean() * 100:.2f}%",
    )

    print(
        "FOV pixels:",
        int(fov_mask.sum()),
    )

    # --------------------------------------------------------
    # Load Models
    # --------------------------------------------------------

    models = {}

    for name, path in CHECKPOINTS.items():

        models[name] = load_model(path)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    case_counts = {
        6: 0,
        7: 0,
        8: 0,
    }

    print("\nStarting evaluation...\n")

    with torch.inference_mode():

        for index, batch in enumerate(loader):

            case_id = int(
                batch["case_id"][0]
            )

            # Only evaluate Case6-8
            if case_id not in TARGET_CASES:
                continue

            source = batch["source"].to(
                DEVICE,
                non_blocking=True,
            )

            target = (
                batch["target"][0, 0]
                .float()
                .cpu()
                .numpy()
            )

            z_idx = int(
                batch["z_idx"][0]
            )

            case_counts[case_id] += 1

            # -----------------------------------------------
            # Evaluate each model
            # -----------------------------------------------

            for model_name, model in models.items():

                prediction = model(source)

                prediction = (
                    prediction[0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                if not np.isfinite(prediction).all():
                    raise RuntimeError(
                        f"Non-finite prediction: "
                        f"{model_name}, "
                        f"Case{case_id}, "
                        f"z={z_idx}"
                    )

                metrics = calculate_metrics(
                    prediction,
                    target,
                    fov_mask,
                )

                results.append({
                    "model": model_name,
                    "case_id": case_id,
                    "z_idx": z_idx,
                    **metrics,
                })

            if (
                sum(case_counts.values()) % 20
                == 0
            ):

                print(
                    f"Evaluated "
                    f"{sum(case_counts.values())} "
                    f"target slices",
                    flush=True,
                )

    # --------------------------------------------------------
    # Check counts
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DATASET COUNTS")
    print("=" * 75)

    for case_id in sorted(TARGET_CASES):

        print(
            f"Case{case_id}: "
            f"{case_counts[case_id]} slices"
        )

        if case_counts[case_id] == 0:
            raise RuntimeError(
                f"No samples found for Case{case_id}. "
                f"Check prompt_train.json and case_id."
            )

    # --------------------------------------------------------
    # Save per-slice results
    # --------------------------------------------------------

    per_slice_path = os.path.join(
        args.output_dir,
        "per_slice_metrics.csv",
    )

    save_csv(
        per_slice_path,
        results,
        [
            "model",
            "case_id",
            "z_idx",
            *METRIC_NAMES,
        ],
    )

    # --------------------------------------------------------
    # Per-case summary
    # --------------------------------------------------------

    per_case_summary = []

    print("\n" + "=" * 75)
    print("PER-CASE RESULTS")
    print("=" * 75)

    for case_id in sorted(TARGET_CASES):

        print(f"\nCase {case_id}")
        print("-" * 75)

        for model_name in models:

            rows = [
                row
                for row in results
                if row["case_id"] == case_id
                and row["model"] == model_name
            ]

            summary = summarize_rows(rows)

            if summary is None:
                continue

            per_case_summary.append({
                "case_id": case_id,
                "model": model_name,
                **summary,
            })

            print(
                f"{model_name:12s} | "
                f"FOV MAE: "
                f"{summary['fov_mae']:.6f} | "
                f"FOV PSNR: "
                f"{summary['fov_psnr']:.4f} | "
                f"FOV SSIM: "
                f"{summary['fov_ssim']:.4f}"
            )

    per_case_path = os.path.join(
        args.output_dir,
        "per_case_summary.csv",
    )

    save_csv(
        per_case_path,
        per_case_summary,
        [
            "case_id",
            "model",
            "num_slices",
            *METRIC_NAMES,
        ],
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall_summary = []

    print("\n" + "=" * 75)
    print("OVERALL RESULTS: CASE6-8")
    print("=" * 75)

    for model_name in models:

        rows = [
            row
            for row in results
            if row["model"] == model_name
        ]

        summary = summarize_rows(rows)

        if summary is None:
            continue

        overall_summary.append({
            "model": model_name,
            **summary,
        })

        print(f"\n{model_name}")

        for metric in METRIC_NAMES:

            print(
                f"  {metric:12s}: "
                f"{summary[metric]:.6f}"
            )

    overall_path = os.path.join(
        args.output_dir,
        "overall_summary.csv",
    )

    save_csv(
        overall_path,
        overall_summary,
        [
            "model",
            "num_slices",
            *METRIC_NAMES,
        ],
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    full_rows = [
        row
        for row in overall_summary
        if row["model"] == "Full_L1"
    ]

    fov_rows = [
        row
        for row in overall_summary
        if row["model"] == "FOV_L1"
    ]

    if full_rows and fov_rows:

        full_mae = full_rows[0]["fov_mae"]
        fov_mae = fov_rows[0]["fov_mae"]

        difference = fov_mae - full_mae

        relative_change = (
            difference / full_mae * 100
            if full_mae != 0
            else float("nan")
        )

        print("\n" + "=" * 75)
        print("FOV MAE COMPARISON")
        print("=" * 75)

        print(
            f"Full L1 FOV MAE : {full_mae:.6f}"
        )

        print(
            f"FOV L1 FOV MAE  : {fov_mae:.6f}"
        )

        print(
            f"Difference       : {difference:+.6f}"
        )

        print(
            f"Relative change  : {relative_change:+.2f}%"
        )

        if difference < 0:

            print(
                "\nResult: FOV L1 performs better "
                "on training Case6-8."
            )

        elif difference > 0:

            print(
                "\nResult: Full L1 performs better "
                "on training Case6-8."
            )

        else:

            print(
                "\nResult: Both models have "
                "identical FOV MAE."
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("Evaluation completed.")
    print("=" * 75)

    print(
        "Per-slice:",
        per_slice_path,
    )

    print(
        "Per-case:",
        per_case_path,
    )

    print(
        "Overall:",
        overall_path,
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    main(args)
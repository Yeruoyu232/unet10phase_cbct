import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import ndimage
from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model_t20 import build_model as build_model_t20
from model_5phase import build_model as build_model_5phase
from model import build_model as build_model_10phase


# ============================================================
# Config
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/"
    "training/cbct_motion_10phase_samez"
)

CKPT_T20 = "./checkpoints/unet_t20_fixed/best.pth"
CKPT_5P = "./checkpoints/unet_5phase_fixed/best.pth"
CKPT_10P = "./checkpoints/unet_10phase_fixed/best.pth"

OUTPUT_DIR = "./body_background_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 4

# ============================================================
# Fixed scan condition
# ============================================================

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2


# ============================================================
# Body mask config
#
# 先使用 -0.5。
# 正式分析前请一定检查生成的 body_mask 图是否合理。
# ============================================================

BODY_THRESHOLD = -0.5

# mask 向外扩张几个像素，把人体边界包含进去
BODY_DILATION = 3


# ============================================================
# Dataset
#
# 直接读取完整 10-phase：
#
# channel 0 = T00
# channel 1 = T10
# channel 2 = T20
# ...
# channel 9 = T90
#
# 后面从同一个 source 构造：
#
# T20 = channel 2
# 5P  = 0,2,4,6,8
# 10P = all
#
# 保证三个模型：
# Case / z / GT 完全一致
# ============================================================

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
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("=" * 90)
print("Body / Background Error Analysis")
print("=" * 90)
print("Device:", DEVICE)
print("Test samples:", len(test_dataset))
print("Body threshold:", BODY_THRESHOLD)
print("Body dilation:", BODY_DILATION)
print("=" * 90)


# ============================================================
# Load model
# ============================================================

def load_model(build_model, ckpt_path):

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("Loaded:", ckpt_path)
    print("Epoch:", checkpoint.get("epoch"))
    print("Val L1:", checkpoint.get("val_loss"))

    return model


model_t20 = load_model(
    build_model_t20,
    CKPT_T20,
)

model_5p = load_model(
    build_model_5phase,
    CKPT_5P,
)

model_10p = load_model(
    build_model_10phase,
    CKPT_10P,
)


# ============================================================
# Body mask
# ============================================================

def get_body_mask(
    gt,
    threshold=BODY_THRESHOLD,
    dilation=BODY_DILATION,
):
    """
    根据 GT Motion-FDK 生成 body mask。

    步骤：
    1. threshold
    2. binary closing
    3. fill holes
    4. 最大连通区域
    5. fill holes
    6. dilation

    输入:
        gt: [H, W]

    输出:
        body_mask: bool [H, W]
        bg_mask:   bool [H, W]
    """

    # --------------------------------------------------------
    # 1. Threshold
    # --------------------------------------------------------

    mask = gt > threshold

    # --------------------------------------------------------
    # 2. Closing
    #
    # 连接主体区域中的小间隙
    # --------------------------------------------------------

    structure = np.ones(
        (3, 3),
        dtype=bool,
    )

    mask = ndimage.binary_closing(
        mask,
        structure=structure,
        iterations=2,
    )

    # --------------------------------------------------------
    # 3. Fill holes
    # --------------------------------------------------------

    mask = ndimage.binary_fill_holes(
        mask
    )

    # --------------------------------------------------------
    # 4. Connected components
    # --------------------------------------------------------

    labeled, num_features = ndimage.label(
        mask
    )

    if num_features == 0:
        raise RuntimeError(
            "No body region found. "
            "Please check BODY_THRESHOLD."
        )

    component_sizes = ndimage.sum(
        mask,
        labeled,
        index=np.arange(
            1,
            num_features + 1,
        ),
    )

    largest_label = (
        np.argmax(component_sizes) + 1
    )

    body_mask = (
        labeled == largest_label
    )

    # --------------------------------------------------------
    # 5. Fill body interior
    # --------------------------------------------------------

    body_mask = ndimage.binary_fill_holes(
        body_mask
    )

    # --------------------------------------------------------
    # 6. Slight dilation
    #
    # 把人体边缘附近包含进 body
    # --------------------------------------------------------

    if dilation > 0:

        body_mask = ndimage.binary_dilation(
            body_mask,
            structure=structure,
            iterations=dilation,
        )

    body_mask = body_mask.astype(bool)

    bg_mask = ~body_mask
    print(
    "z =", z_idx,
    "GT min/max =",
    target_np.min(),
    target_np.max(),
    "body ratio =",
    body_mask.mean(),
)

    return body_mask, bg_mask


# ============================================================
# Helper
# ============================================================

def tensor_value_to_int(x):

    if torch.is_tensor(x):
        return int(x.item())

    if isinstance(x, (list, tuple)):
        return int(x[0])

    return int(x)


def safe_mean(x):

    if x.size == 0:
        return np.nan

    return float(
        np.mean(x)
    )


def safe_ratio(a, b):

    if b <= 1e-12:
        return 0.0

    return float(a / b)


# ============================================================
# Inference
# ============================================================

results = []

with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

        # ----------------------------------------------------
        # Complete 10-phase
        # [1,10,H,W]
        # ----------------------------------------------------

        source_10p = batch["source"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # T20
        # [1,1,H,W]
        # ----------------------------------------------------

        source_t20 = source_10p[
            :,
            2:3,
            :,
            :
        ]

        # ----------------------------------------------------
        # 5 phase
        #
        # T00 T20 T40 T60 T80
        # ----------------------------------------------------

        source_5p = source_10p[
            :,
            [0, 2, 4, 6, 8],
            :,
            :
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred_t20 = model_t20(
            source_t20
        )

        pred_5p = model_5p(
            source_5p
        )

        pred_10p = model_10p(
            source_10p
        )

        # ----------------------------------------------------
        # To numpy
        # ----------------------------------------------------

        target_np = (
            target[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        pred_t20_np = (
            pred_t20[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        pred_5p_np = (
            pred_5p[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        pred_10p_np = (
            pred_10p[0, 0]
            .float()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Absolute error
        # ----------------------------------------------------

        diff_t20 = np.abs(
            pred_t20_np - target_np
        )

        diff_5p = np.abs(
            pred_5p_np - target_np
        )

        diff_10p = np.abs(
            pred_10p_np - target_np
        )

        # ----------------------------------------------------
        # Generate SAME mask from GT
        # ----------------------------------------------------

        body_mask, bg_mask = get_body_mask(
            target_np
        )

        # ----------------------------------------------------
        # Whole MAE
        # ----------------------------------------------------

        whole_mae_t20 = safe_mean(
            diff_t20
        )

        whole_mae_5p = safe_mean(
            diff_5p
        )

        whole_mae_10p = safe_mean(
            diff_10p
        )

        # ----------------------------------------------------
        # Body MAE
        # ----------------------------------------------------

        body_mae_t20 = safe_mean(
            diff_t20[body_mask]
        )

        body_mae_5p = safe_mean(
            diff_5p[body_mask]
        )

        body_mae_10p = safe_mean(
            diff_10p[body_mask]
        )

        # ----------------------------------------------------
        # Background MAE
        # ----------------------------------------------------

        bg_mae_t20 = safe_mean(
            diff_t20[bg_mask]
        )

        bg_mae_5p = safe_mean(
            diff_5p[bg_mask]
        )

        bg_mae_10p = safe_mean(
            diff_10p[bg_mask]
        )

        # ----------------------------------------------------
        # Total absolute error
        # ----------------------------------------------------

        total_error_t20 = float(
            np.sum(diff_t20)
        )

        total_error_5p = float(
            np.sum(diff_5p)
        )

        total_error_10p = float(
            np.sum(diff_10p)
        )

        # ----------------------------------------------------
        # Background absolute error
        # ----------------------------------------------------

        bg_error_t20 = float(
            np.sum(
                diff_t20[bg_mask]
            )
        )

        bg_error_5p = float(
            np.sum(
                diff_5p[bg_mask]
            )
        )

        bg_error_10p = float(
            np.sum(
                diff_10p[bg_mask]
            )
        )

        # ----------------------------------------------------
        # Background contribution
        #
        # 背景误差占整张图绝对误差的比例
        # ----------------------------------------------------

        bg_contribution_t20 = safe_ratio(
            bg_error_t20,
            total_error_t20,
        )

        bg_contribution_5p = safe_ratio(
            bg_error_5p,
            total_error_5p,
        )

        bg_contribution_10p = safe_ratio(
            bg_error_10p,
            total_error_10p,
        )

        # ----------------------------------------------------
        # Body contribution
        # ----------------------------------------------------

        body_contribution_t20 = (
            1.0 - bg_contribution_t20
        )

        body_contribution_5p = (
            1.0 - bg_contribution_5p
        )

        body_contribution_10p = (
            1.0 - bg_contribution_10p
        )

        # ----------------------------------------------------
        # Number of pixels
        # ----------------------------------------------------

        body_pixels = int(
            np.sum(body_mask)
        )

        bg_pixels = int(
            np.sum(bg_mask)
        )

        total_pixels = (
            body_pixels + bg_pixels
        )

        body_ratio = (
            body_pixels / total_pixels
        )

        # ----------------------------------------------------
        # Case / z
        # ----------------------------------------------------

        case_id = tensor_value_to_int(
            batch["case_id"]
        )

        z_idx = tensor_value_to_int(
            batch["z_idx"]
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append(
            {
                "case_id": case_id,
                "z_idx": z_idx,

                "target": target_np,
                "body_mask": body_mask,
                "bg_mask": bg_mask,

                "whole_mae_t20": whole_mae_t20,
                "whole_mae_5p": whole_mae_5p,
                "whole_mae_10p": whole_mae_10p,

                "body_mae_t20": body_mae_t20,
                "body_mae_5p": body_mae_5p,
                "body_mae_10p": body_mae_10p,

                "bg_mae_t20": bg_mae_t20,
                "bg_mae_5p": bg_mae_5p,
                "bg_mae_10p": bg_mae_10p,

                "bg_contribution_t20":
                    bg_contribution_t20,

                "bg_contribution_5p":
                    bg_contribution_5p,

                "bg_contribution_10p":
                    bg_contribution_10p,

                "body_contribution_t20":
                    body_contribution_t20,

                "body_contribution_5p":
                    body_contribution_5p,

                "body_contribution_10p":
                    body_contribution_10p,

                "body_pixels": body_pixels,
                "bg_pixels": bg_pixels,
                "body_ratio": body_ratio,
            }
        )

        print(
            f"[{batch_idx + 1:03d}/"
            f"{len(test_loader):03d}] "
            f"z={z_idx:03d} | "
            f"T20 "
            f"W={whole_mae_t20:.4f} "
            f"B={body_mae_t20:.4f} "
            f"BG={bg_mae_t20:.4f} | "
            f"5P "
            f"W={whole_mae_5p:.4f} "
            f"B={body_mae_5p:.4f} "
            f"BG={bg_mae_5p:.4f} | "
            f"10P "
            f"W={whole_mae_10p:.4f} "
            f"B={body_mae_10p:.4f} "
            f"BG={bg_mae_10p:.4f}"
        )


# ============================================================
# Average helper
# ============================================================

def avg(key):

    values = np.array(
        [
            r[key]
            for r in results
        ],
        dtype=np.float64,
    )

    return float(
        np.nanmean(values)
    )


# ============================================================
# Overall statistics
# ============================================================

print()
print("=" * 100)
print("Case10 Body / Background Error Analysis")
print("=" * 100)

header = (
    f"{'Model':<10}"
    f"{'Whole MAE':>14}"
    f"{'Body MAE':>14}"
    f"{'BG MAE':>14}"
    f"{'BG Error %':>16}"
)

print(header)

print("-" * 100)


def print_model_row(
    model_name,
    whole_key,
    body_key,
    bg_key,
    contribution_key,
):

    print(
        f"{model_name:<10}"
        f"{avg(whole_key):>14.6f}"
        f"{avg(body_key):>14.6f}"
        f"{avg(bg_key):>14.6f}"
        f"{avg(contribution_key) * 100:>15.2f}%"
    )


print_model_row(
    "T20",
    "whole_mae_t20",
    "body_mae_t20",
    "bg_mae_t20",
    "bg_contribution_t20",
)

print_model_row(
    "5P",
    "whole_mae_5p",
    "body_mae_5p",
    "bg_mae_5p",
    "bg_contribution_5p",
)

print_model_row(
    "10P",
    "whole_mae_10p",
    "body_mae_10p",
    "bg_mae_10p",
    "bg_contribution_10p",
)

print("=" * 100)

print(
    f"Average body pixel ratio: "
    f"{avg('body_ratio') * 100:.2f}%"
)

print(
    f"Average background pixel ratio: "
    f"{(1.0 - avg('body_ratio')) * 100:.2f}%"
)


# ============================================================
# IMPORTANT:
# Global contribution over ALL pixels/slices
#
# 上面的 avg(bg_contribution) 是先逐 slice 算比例再平均。
# 这里再计算整个 Case10 的总体贡献。
# ============================================================

def global_error_contribution(
    model_name,
):

    whole_key = (
        f"whole_mae_{model_name}"
    )

    bg_key = (
        f"bg_mae_{model_name}"
    )

    total_error = 0.0
    total_bg_error = 0.0

    for r in results:

        h, w = r["target"].shape

        total_error += (
            r[whole_key] * h * w
        )

        total_bg_error += (
            r[bg_key] * r["bg_pixels"]
        )

    return (
        total_bg_error / total_error
    )


print()
print("=" * 100)
print("Global Background Error Contribution")
print("=" * 100)

print(
    f"T20 : "
    f"{global_error_contribution('t20') * 100:.2f}%"
)

print(
    f"5P  : "
    f"{global_error_contribution('5p') * 100:.2f}%"
)

print(
    f"10P : "
    f"{global_error_contribution('10p') * 100:.2f}%"
)

print("=" * 100)


# ============================================================
# Save CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "body_background_metrics.csv",
)

fieldnames = [
    "case_id",
    "z_idx",

    "whole_mae_t20",
    "body_mae_t20",
    "bg_mae_t20",
    "bg_contribution_t20",

    "whole_mae_5p",
    "body_mae_5p",
    "bg_mae_5p",
    "bg_contribution_5p",

    "whole_mae_10p",
    "body_mae_10p",
    "bg_mae_10p",
    "bg_contribution_10p",

    "body_pixels",
    "bg_pixels",
    "body_ratio",
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for r in results:

        row = {
            key: r[key]
            for key in fieldnames
        }

        writer.writerow(row)

print()
print("Saved CSV:", csv_path)


# ============================================================
# Visualize GT + Body Mask
#
# 自动保存 6 个均匀分布的 slice，
# 用来检查 mask 是否正确。
# ============================================================

mask_dir = os.path.join(
    OUTPUT_DIR,
    "mask_check",
)

os.makedirs(
    mask_dir,
    exist_ok=True,
)

sample_indices = np.linspace(
    0,
    len(results) - 1,
    6,
    dtype=int,
)

for idx in sample_indices:

    r = results[idx]

    target = r["target"]
    body_mask = r["body_mask"]

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = np.ma.masked_where(
        ~body_mask,
        body_mask,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    # GT
    axes[0].imshow(
        target,
        cmap="gray",
    )

    axes[0].set_title(
        "GT Motion-FDK"
    )

    # Body mask
    axes[1].imshow(
        body_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Body Mask"
    )

    # Overlay
    axes[2].imshow(
        target,
        cmap="gray",
    )

    axes[2].imshow(
        overlay,
        alpha=0.30,
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "GT + Body Mask"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        (
            f'Case {r["case_id"]} | '
            f'z={r["z_idx"]} | '
            f'Body={r["body_ratio"] * 100:.1f}%'
        )
    )

    plt.tight_layout()

    save_path = os.path.join(
        mask_dir,
        f"mask_z{r['z_idx']:03d}.png",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved mask:",
        save_path,
    )


# ============================================================
# Find z=71 specifically
#
# 因为你刚才展示的是 Case10 z=71
# ============================================================

for r in results:

    if r["z_idx"] == 71:

        target = r["target"]
        body_mask = r["body_mask"]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4),
        )

        axes[0].imshow(
            target,
            cmap="gray",
        )

        axes[0].set_title(
            "GT Motion-FDK"
        )

        axes[1].imshow(
            body_mask,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[1].set_title(
            "Body Mask"
        )

        axes[2].imshow(
            target,
            cmap="gray",
        )

        axes[2].contour(
            body_mask.astype(float),
            levels=[0.5],
            linewidths=1.0,
        )

        axes[2].set_title(
            "Body Boundary"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            "Case 10 | z=71"
        )

        plt.tight_layout()

        z71_path = os.path.join(
            OUTPUT_DIR,
            "body_mask_z71.png",
        )

        plt.savefig(
            z71_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            "Saved z=71 mask:",
            z71_path,
        )

        break


# ============================================================
# Per-slice Body MAE Curve
# ============================================================

z_values = [
    r["z_idx"]
    for r in results
]

body_t20 = [
    r["body_mae_t20"]
    for r in results
]

body_5p = [
    r["body_mae_5p"]
    for r in results
]

body_10p = [
    r["body_mae_10p"]
    for r in results
]


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    z_values,
    body_t20,
    label="T20",
    linewidth=1.8,
)

plt.plot(
    z_values,
    body_5p,
    label="5-Phase",
    linewidth=1.8,
)

plt.plot(
    z_values,
    body_10p,
    label="10-Phase",
    linewidth=1.8,
)

plt.xlabel("z Slice")
plt.ylabel("Body MAE")

plt.title(
    "Body-region MAE across Test Slices"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

body_curve_path = os.path.join(
    OUTPUT_DIR,
    "body_mae_by_slice.png",
)

plt.savefig(
    body_curve_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    body_curve_path,
)


# ============================================================
# Per-slice Background MAE Curve
# ============================================================

bg_t20 = [
    r["bg_mae_t20"]
    for r in results
]

bg_5p = [
    r["bg_mae_5p"]
    for r in results
]

bg_10p = [
    r["bg_mae_10p"]
    for r in results
]


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    z_values,
    bg_t20,
    label="T20",
    linewidth=1.8,
)

plt.plot(
    z_values,
    bg_5p,
    label="5-Phase",
    linewidth=1.8,
)

plt.plot(
    z_values,
    bg_10p,
    label="10-Phase",
    linewidth=1.8,
)

plt.xlabel("z Slice")
plt.ylabel("Background MAE")

plt.title(
    "Background-region MAE across Test Slices"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

bg_curve_path = os.path.join(
    OUTPUT_DIR,
    "background_mae_by_slice.png",
)

plt.savefig(
    bg_curve_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    bg_curve_path,
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 100)
print("Analysis finished.")
print("Output directory:")
print(OUTPUT_DIR)
print("=" * 100)
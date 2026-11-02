import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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

OUTPUT_DIR = "./difference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Fixed scan condition
# ============================================================

SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2


# ============================================================
# Dataset
#
# 这里直接读取完整 10-phase。
# 后面从同一个 source 中构造：
#
# T20 : channel 2
# 5P  : channel 0,2,4,6,8
# 10P : all channels
#
# 这样三个模型保证使用完全相同的 Case / z / GT
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
    num_workers=4,
    pin_memory=True,
)

print("Test samples:", len(test_dataset))


# ============================================================
# Load model function
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

    print("-" * 80)
    print("Loaded:", ckpt_path)
    print("Epoch:", checkpoint.get("epoch"))
    print("Val L1:", checkpoint.get("val_loss"))

    return model


# ============================================================
# Load three models
# ============================================================

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
# Inference
# ============================================================

results = []

with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

        # ----------------------------------------------------
        # Full 10-phase source
        #
        # shape:
        # [1,10,256,256]
        #
        # channel:
        # 0=T00
        # 1=T10
        # 2=T20
        # ...
        # 9=T90
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
        # Construct T20
        #
        # [B,10,H,W] -> [B,1,H,W]
        # ----------------------------------------------------

        source_t20 = source_10p[:, 2:3, :, :]

        # ----------------------------------------------------
        # Construct 5-phase
        #
        # T00,T20,T40,T60,T80
        #
        # [B,10,H,W] -> [B,5,H,W]
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

        pred_t20 = model_t20(source_t20)

        pred_5p = model_5p(source_5p)

        pred_10p = model_10p(source_10p)

        # ----------------------------------------------------
        # Tensor -> numpy
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
        # Absolute difference
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
        # MAE
        # ----------------------------------------------------

        mae_t20 = float(
            np.mean(diff_t20)
        )

        mae_5p = float(
            np.mean(diff_5p)
        )

        mae_10p = float(
            np.mean(diff_10p)
        )

        # ----------------------------------------------------
        # Case / z
        # ----------------------------------------------------

        case_id = batch["case_id"]

        if torch.is_tensor(case_id):
            case_id = int(case_id.item())
        else:
            case_id = int(case_id[0])

        z_idx = batch["z_idx"]

        if torch.is_tensor(z_idx):
            z_idx = int(z_idx.item())
        else:
            z_idx = int(z_idx[0])

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            {
                "case_id": case_id,
                "z_idx": z_idx,

                "target": target_np,

                "pred_t20": pred_t20_np,
                "pred_5p": pred_5p_np,
                "pred_10p": pred_10p_np,

                "diff_t20": diff_t20,
                "diff_5p": diff_5p,
                "diff_10p": diff_10p,

                "mae_t20": mae_t20,
                "mae_5p": mae_5p,
                "mae_10p": mae_10p,

                # 正数 = 10P 比 T20 更好
                "improvement": mae_t20 - mae_10p,
            }
        )

        print(
            f"[{batch_idx + 1:03d}/{len(test_loader):03d}] "
            f"Case={case_id} "
            f"z={z_idx:03d} | "
            f"T20={mae_t20:.6f} | "
            f"5P={mae_5p:.6f} | "
            f"10P={mae_10p:.6f}"
        )


# ============================================================
# Overall
# ============================================================

mean_t20 = np.mean(
    [r["mae_t20"] for r in results]
)

mean_5p = np.mean(
    [r["mae_5p"] for r in results]
)

mean_10p = np.mean(
    [r["mae_10p"] for r in results]
)


print()
print("=" * 80)
print("Overall MAE")
print("=" * 80)

print(f"T20 : {mean_t20:.6f}")
print(f"5P  : {mean_5p:.6f}")
print(f"10P : {mean_10p:.6f}")

print("=" * 80)


# ============================================================
# Visualization
# ============================================================

def save_difference_figure(result, filename):

    target = result["target"]

    pred_t20 = result["pred_t20"]
    pred_5p = result["pred_5p"]
    pred_10p = result["pred_10p"]

    diff_t20 = result["diff_t20"]
    diff_5p = result["diff_5p"]
    diff_10p = result["diff_10p"]

    # ========================================================
    # Prediction images:
    # GT + 三个 Prediction 使用完全相同的 grayscale range
    # ========================================================

    all_images = np.concatenate(
        [
            target.flatten(),
            pred_t20.flatten(),
            pred_5p.flatten(),
            pred_10p.flatten(),
        ]
    )

    image_vmin = np.percentile(
        all_images,
        1,
    )

    image_vmax = np.percentile(
        all_images,
        99,
    )

    # ========================================================
    # Difference:
    #
    # 三个模型必须使用同一个 vmax
    # 否则颜色深浅不能直接比较
    # ========================================================

    all_diff = np.concatenate(
        [
            diff_t20.flatten(),
            diff_5p.flatten(),
            diff_10p.flatten(),
        ]
    )

    diff_vmax = np.percentile(
        all_diff,
        99,
    )

    if diff_vmax <= 0:
        diff_vmax = 1.0

    # ========================================================
    # Figure
    #
    # 第一行:
    # GT | T20 Pred | 5P Pred | 10P Pred
    #
    # 第二行:
    # 空 | T20 Diff | 5P Diff | 10P Diff
    # ========================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    axes[0, 0].imshow(
        target,
        cmap="gray",
        vmin=image_vmin,
        vmax=image_vmax,
    )

    axes[0, 0].set_title(
        "GT Motion-FDK"
    )

    axes[0, 1].imshow(
        pred_t20,
        cmap="gray",
        vmin=image_vmin,
        vmax=image_vmax,
    )

    axes[0, 1].set_title(
        f'T20 Prediction\n'
        f'MAE={result["mae_t20"]:.4f}'
    )

    axes[0, 2].imshow(
        pred_5p,
        cmap="gray",
        vmin=image_vmin,
        vmax=image_vmax,
    )

    axes[0, 2].set_title(
        f'5P Prediction\n'
        f'MAE={result["mae_5p"]:.4f}'
    )

    axes[0, 3].imshow(
        pred_10p,
        cmap="gray",
        vmin=image_vmin,
        vmax=image_vmax,
    )

    axes[0, 3].set_title(
        f'10P Prediction\n'
        f'MAE={result["mae_10p"]:.4f}'
    )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    # 左下角不需要图
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(
        diff_t20,
        cmap="hot",
        vmin=0,
        vmax=diff_vmax,
    )

    axes[1, 1].set_title(
        "|T20 Prediction - GT|"
    )

    axes[1, 2].imshow(
        diff_5p,
        cmap="hot",
        vmin=0,
        vmax=diff_vmax,
    )

    axes[1, 2].set_title(
        "|5P Prediction - GT|"
    )

    axes[1, 3].imshow(
        diff_10p,
        cmap="hot",
        vmin=0,
        vmax=diff_vmax,
    )

    axes[1, 3].set_title(
        "|10P Prediction - GT|"
    )

    # --------------------------------------------------------
    # Remove axes
    # --------------------------------------------------------

    for ax in axes.flat:
        ax.axis("off")

    # --------------------------------------------------------
    # One shared colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=[
            axes[1, 1],
            axes[1, 2],
            axes[1, 3],
        ],
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Absolute Difference"
    )

    fig.suptitle(
        (
            f'Case {result["case_id"]} | '
            f'z={result["z_idx"]}'
        ),
        fontsize=16,
    )

    plt.subplots_adjust(
        left=0.03,
        right=0.92,
        top=0.90,
        bottom=0.05,
        wspace=0.08,
        hspace=0.15,
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", save_path)


# ============================================================
# 1. Save several uniformly sampled slices
# ============================================================

uniform_indices = np.linspace(
    0,
    len(results) - 1,
    6,
    dtype=int,
)

for i, idx in enumerate(uniform_indices):

    r = results[idx]

    save_difference_figure(
        r,
        (
            f"sample_{i + 1:02d}_"
            f"z{r['z_idx']:03d}.png"
        ),
    )


# ============================================================
# 2. Find slices where 10P improves most over T20
# ============================================================

sorted_by_improvement = sorted(
    results,
    key=lambda x: x["improvement"],
    reverse=True,
)

print()
print("=" * 80)
print("Top 5 slices: 10P improves most over T20")
print("=" * 80)

for rank, r in enumerate(
    sorted_by_improvement[:5],
    start=1,
):

    print(
        f"{rank}. "
        f"z={r['z_idx']:03d} | "
        f"T20={r['mae_t20']:.6f} | "
        f"5P={r['mae_5p']:.6f} | "
        f"10P={r['mae_10p']:.6f} | "
        f"Improvement={r['improvement']:.6f}"
    )

    save_difference_figure(
        r,
        (
            f"top_improvement_{rank}_"
            f"z{r['z_idx']:03d}.png"
        ),
    )


# ============================================================
# 3. Find slices where 10P is worse than T20
# ============================================================

sorted_worst = sorted(
    results,
    key=lambda x: x["improvement"],
)

print()
print("=" * 80)
print("Top 5 slices: 10P improves least / is worse")
print("=" * 80)

for rank, r in enumerate(
    sorted_worst[:5],
    start=1,
):

    print(
        f"{rank}. "
        f"z={r['z_idx']:03d} | "
        f"T20={r['mae_t20']:.6f} | "
        f"5P={r['mae_5p']:.6f} | "
        f"10P={r['mae_10p']:.6f} | "
        f"Improvement={r['improvement']:.6f}"
    )

    save_difference_figure(
        r,
        (
            f"worst_improvement_{rank}_"
            f"z{r['z_idx']:03d}.png"
        ),
    )


# ============================================================
# 4. MAE curve
# ============================================================

z_values = [
    r["z_idx"]
    for r in results
]

mae_t20_values = [
    r["mae_t20"]
    for r in results
]

mae_5p_values = [
    r["mae_5p"]
    for r in results
]

mae_10p_values = [
    r["mae_10p"]
    for r in results
]


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    z_values,
    mae_t20_values,
    label="T20",
    linewidth=1.8,
)

plt.plot(
    z_values,
    mae_5p_values,
    label="5-Phase",
    linewidth=1.8,
)

plt.plot(
    z_values,
    mae_10p_values,
    label="10-Phase",
    linewidth=1.8,
)

plt.xlabel("z Slice")
plt.ylabel("MAE")

plt.title(
    "Per-Slice MAE Comparison"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

mae_curve_path = os.path.join(
    OUTPUT_DIR,
    "mae_comparison.png",
)

plt.savefig(
    mae_curve_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print()
print("Saved:", mae_curve_path)

print()
print("=" * 80)
print("Difference analysis finished.")
print("Output:", OUTPUT_DIR)
print("=" * 80)
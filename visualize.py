import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset
from model import build_model


# ============================================================
# Config
# ============================================================

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/"
    "training/cbct_motion_10phase_samez"
)

CKPT_PATH = (
    "./checkpoints/unet_10phase_fixed/best.pth"
)

OUTPUT_DIR = "./visual_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

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
    pin_memory=True,
)

print("Test samples:", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = build_model().to(DEVICE)

checkpoint = torch.load(
    CKPT_PATH,
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint:", CKPT_PATH)
print("Best epoch:", checkpoint.get("epoch"))
print("Best val L1:", checkpoint.get("val_loss"))


# ============================================================
# Inference all test slices
# ============================================================

results = []


with torch.inference_mode():

    for index, batch in enumerate(test_loader):

        source = batch["source"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True,
        )

        prediction = model(source)


        # ----------------------------------------------------
        # [1,10,H,W] -> [10,H,W]
        # ----------------------------------------------------

        source_np = (
            source[0]
            .float()
            .cpu()
            .numpy()
        )


        # T20 = channel index 2
        # T00,T10,T20,...,T90
        t20_np = source_np[2]


        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        target_np = (
            target[0, 0]
            .float()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred_np = (
            prediction[0, 0]
            .float()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error_np = np.abs(
            pred_np - target_np
        )

        mae = float(
            np.mean(error_np)
        )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        case_id = batch["case_id"]

        if torch.is_tensor(case_id):
            case_id = int(
                case_id.item()
            )
        else:
            case_id = int(
                case_id[0]
            )


        z_idx = batch["z_idx"]

        if torch.is_tensor(z_idx):
            z_idx = int(
                z_idx.item()
            )
        else:
            z_idx = int(
                z_idx[0]
            )


        results.append(
            {
                "index": index,
                "case_id": case_id,
                "z_idx": z_idx,

                "t20": t20_np,
                "target": target_np,
                "prediction": pred_np,
                "error": error_np,

                "mae": mae,
            }
        )


# ============================================================
# Sort according to MAE
# ============================================================

sorted_results = sorted(
    results,
    key=lambda x: x["mae"],
)

best_result = sorted_results[0]

median_result = sorted_results[
    len(sorted_results) // 2
]

worst_result = sorted_results[-1]


print()
print("=" * 80)

print(
    "Best:",
    f'z={best_result["z_idx"]}',
    f'MAE={best_result["mae"]:.6f}',
)

print(
    "Median:",
    f'z={median_result["z_idx"]}',
    f'MAE={median_result["mae"]:.6f}',
)

print(
    "Worst:",
    f'z={worst_result["z_idx"]}',
    f'MAE={worst_result["mae"]:.6f}',
)

print("=" * 80)


# ============================================================
# Visualization function
# ============================================================

def save_comparison(
    result,
    name,
):

    t20 = result["t20"]

    target = result["target"]

    prediction = result["prediction"]

    error = result["error"]


    # --------------------------------------------------------
    # T20 / GT / Prediction 使用相同 window
    #
    # 不要每张图单独自动缩放，否则视觉比较不公平
    # --------------------------------------------------------

    vmin = min(
        np.percentile(t20, 1),
        np.percentile(target, 1),
    )

    vmax = max(
        np.percentile(t20, 99),
        np.percentile(target, 99),
    )


    # --------------------------------------------------------
    # Error map scale
    # --------------------------------------------------------

    error_vmax = np.percentile(
        error,
        99,
    )

    if error_vmax <= 0:
        error_vmax = 1.0


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )


    # --------------------------------------------------------
    # T20
    # --------------------------------------------------------

    axes[0].imshow(
        t20,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(
        "T20 Input"
    )

    axes[0].axis("off")


    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[1].imshow(
        target,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title(
        "GT Motion-FDK"
    )

    axes[1].axis("off")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[2].set_title(
        "U-Net Prediction"
    )

    axes[2].axis("off")


    # --------------------------------------------------------
    # Absolute Error
    # --------------------------------------------------------

    im = axes[3].imshow(
        error,
        cmap="hot",
        vmin=0,
        vmax=error_vmax,
    )

    axes[3].set_title(
        "Absolute Error"
    )

    axes[3].axis("off")


    fig.colorbar(
        im,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )


    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        (
            f'Case {result["case_id"]} | '
            f'z={result["z_idx"]} | '
            f'MAE={result["mae"]:.4f}'
        ),
        fontsize=14,
    )


    plt.tight_layout()


    save_path = os.path.join(
        OUTPUT_DIR,
        name,
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


    print(
        "Saved:",
        save_path
    )


# ============================================================
# Save Best / Median / Worst
# ============================================================

save_comparison(
    best_result,
    "best.png",
)

save_comparison(
    median_result,
    "median.png",
)

save_comparison(
    worst_result,
    "worst.png",
)


# ============================================================
# Uniformly sampled examples
# ============================================================

num_examples = 6

indices = np.linspace(
    0,
    len(results) - 1,
    num_examples,
    dtype=int,
)


for i, index in enumerate(indices):

    result = results[index]

    save_comparison(
        result,
        (
            f"sample_{i + 1:02d}_"
            f"z{result['z_idx']:03d}.png"
        ),
    )


# ============================================================
# Save MAE distribution
# ============================================================

mae_values = [
    r["mae"]
    for r in results
]

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    range(len(mae_values)),
    mae_values,
)

plt.xlabel(
    "Test Slice"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Case10 - MAE across axial slices"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

mae_path = os.path.join(
    OUTPUT_DIR,
    "mae_by_slice.png",
)

plt.savefig(
    mae_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    mae_path
)


print()
print("=" * 80)
print("Finished.")
print("Output:", OUTPUT_DIR)
print("=" * 80)
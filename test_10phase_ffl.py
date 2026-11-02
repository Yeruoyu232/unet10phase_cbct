# import os
# import csv
# import numpy as np
# import torch

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from scipy.ndimage import gaussian_filter

# from torch.utils.data import DataLoader
# from skimage.metrics import (
#     peak_signal_noise_ratio,
#     structural_similarity,
# )

# from dataset import DIRLab10PhaseDataset
# from model import build_model


# # ============================================================
# # 1. Config
# # ============================================================

# DATA_ROOT = (
#     "/home/yry/ENV/ControlNet-main_10phase/"
#     "training/cbct_motion_10phase_samez"
# )

# # FFL 模型路径
# CKPT_PATH = (
#     "./checkpoints/unet_10phase_ffl/best.pth"
# )

# # 测试结果保存路径
# OUTPUT_DIR = "./test_results/unet_10phase_ffl"

# VIS_DIR = os.path.join(
#     OUTPUT_DIR,
#     "visualizations"
# )

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(VIS_DIR, exist_ok=True)

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# NUM_WORKERS = 4

# # 每隔多少张保存一次可视化
# VIS_INTERVAL = 10

# # 额外保存指定 z 层
# VIS_Z_INDICES = {70, 80}

# # 是否保存所有切片
# SAVE_ALL_VIS = False

# # 高频可视化参数
# HIGH_PASS_SIGMA = 3.0


# # ============================================================
# # 2. Fixed scan conditions
# # ============================================================

# SCAN_TIME = 30.0
# BREATHING_PERIOD = 4.0
# START_PHASE = 0.0

# SOD = 720.0
# DU = 1.2


# # ============================================================
# # 3. Test dataset = Case10
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
#     num_workers=NUM_WORKERS,
#     pin_memory=(DEVICE.type == "cuda"),
# )

# print("=" * 80)
# print("UNet 10-phase + FFL Test")
# print("=" * 80)

# print("Device:", DEVICE)
# print("Test samples:", len(test_dataset))


# # ============================================================
# # 4. Load model
# # ============================================================

# model = build_model().to(DEVICE)

# ckpt = torch.load(
#     CKPT_PATH,
#     map_location="cpu",
#     weights_only=False,
# )

# model.load_state_dict(
#     ckpt["model_state_dict"]
# )

# model.eval()

# print("Checkpoint:", CKPT_PATH)
# print("Best epoch:", ckpt.get("epoch"))
# print("Best val L1:", ckpt.get("val_loss"))

# print("=" * 80)


# # ============================================================
# # 5. Visualization function
# # ============================================================

# def save_visualization(
#     phase_mean,
#     target,
#     prediction,
#     case_id,
#     z_idx,
#     save_path,
# ):

#     # ----------------------------------------
#     # 高频分量
#     # ----------------------------------------

#     gt_high = (
#         target
#         - gaussian_filter(
#             target,
#             sigma=HIGH_PASS_SIGMA,
#         )
#     )

#     pred_high = (
#         prediction
#         - gaussian_filter(
#             prediction,
#             sigma=HIGH_PASS_SIGMA,
#         )
#     )

#     # ----------------------------------------
#     # Absolute Difference
#     # ----------------------------------------

#     abs_diff = np.abs(
#         prediction - target
#     )

#     # ----------------------------------------
#     # 统一灰度范围
#     # ----------------------------------------

#     vmin = min(
#         phase_mean.min(),
#         target.min(),
#         prediction.min(),
#     )

#     vmax = max(
#         phase_mean.max(),
#         target.max(),
#         prediction.max(),
#     )

#     # 高频图统一颜色范围
#     high_limit = max(
#         np.abs(gt_high).max(),
#         np.abs(pred_high).max(),
#         1e-6,
#     )

#     # ----------------------------------------
#     # Plot
#     # ----------------------------------------

#     fig, axes = plt.subplots(
#         2,
#         3,
#         figsize=(16, 10),
#     )

#     # 第一行
#     im0 = axes[0, 0].imshow(
#         phase_mean,
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax,
#     )

#     axes[0, 0].set_title(
#         "Phase Mean (10P)"
#     )

#     im1 = axes[0, 1].imshow(
#         target,
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax,
#     )

#     axes[0, 1].set_title(
#         "GT Motion-FDK"
#     )

#     im2 = axes[0, 2].imshow(
#         prediction,
#         cmap="gray",
#         vmin=vmin,
#         vmax=vmax,
#     )

#     axes[0, 2].set_title(
#         "Predicted Motion-FDK (FFL)"
#     )

#     # 第二行
#     im3 = axes[1, 0].imshow(
#         gt_high,
#         cmap="seismic",
#         vmin=-high_limit,
#         vmax=high_limit,
#     )

#     axes[1, 0].set_title(
#         "GT High-Frequency"
#     )

#     im4 = axes[1, 1].imshow(
#         pred_high,
#         cmap="seismic",
#         vmin=-high_limit,
#         vmax=high_limit,
#     )

#     axes[1, 1].set_title(
#         "Predicted High-Frequency"
#     )

#     im5 = axes[1, 2].imshow(
#         abs_diff,
#         cmap="hot",
#         vmin=0,
#         vmax=max(abs_diff.max(), 1e-6),
#     )

#     axes[1, 2].set_title(
#         "Absolute Difference"
#     )

#     images = [
#         im0, im1, im2,
#         im3, im4, im5,
#     ]

#     for ax, im in zip(
#         axes.flat,
#         images,
#     ):

#         ax.axis("off")

#         fig.colorbar(
#             im,
#             ax=ax,
#             fraction=0.046,
#             pad=0.04,
#         )

#     fig.suptitle(
#         f"{case_id} | z={z_idx} | "
#         f"10P E2E + FFL",
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
# # 6. Helper
# # ============================================================

# def get_scalar(value):

#     if isinstance(value, torch.Tensor):
#         return value.flatten()[0].item()

#     if isinstance(value, (list, tuple)):
#         return get_scalar(value[0])

#     return value


# # ============================================================
# # 7. Evaluation
# # ============================================================

# mae_list = []
# psnr_list = []
# ssim_list = []

# results = []


# with torch.inference_mode():

#     for i, batch in enumerate(test_loader):

#         # ----------------------------------------
#         # Load data
#         # ----------------------------------------

#         source = batch["source"].to(
#             DEVICE,
#             dtype=torch.float32,
#             non_blocking=True,
#         )

#         target = batch["target"].to(
#             DEVICE,
#             dtype=torch.float32,
#             non_blocking=True,
#         )

#         # ----------------------------------------
#         # E2E prediction
#         #
#         # 注意：不使用 Phase Mean Residual
#         # ----------------------------------------

#         prediction = model(source)

#         # 仅用于可视化
#         phase_mean = source.mean(
#             dim=1,
#             keepdim=True,
#         )

#         # ----------------------------------------
#         # Tensor -> NumPy
#         # ----------------------------------------

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

#         phase_mean_np = (
#             phase_mean[0, 0]
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
#         # PSNR
#         # ----------------------------------------

#         psnr = peak_signal_noise_ratio(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )

#         # ----------------------------------------
#         # SSIM
#         # ----------------------------------------

#         ssim = structural_similarity(
#             target_np,
#             pred_np,
#             data_range=2.0,
#         )

#         mae_list.append(mae)
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)

#         # ----------------------------------------
#         # Case ID / Z index
#         # ----------------------------------------

#         case_id = str(
#             get_scalar(batch["case_id"])
#         )

#         z_idx = int(
#             get_scalar(batch["z_idx"])
#         )

#         # ----------------------------------------
#         # Save per-slice metrics
#         # ----------------------------------------

#         results.append({
#             "case_id": case_id,
#             "z_idx": z_idx,
#             "MAE": float(mae),
#             "PSNR": float(psnr),
#             "SSIM": float(ssim),
#         })

#         # ----------------------------------------
#         # Print
#         # ----------------------------------------

#         if (
#             i % 20 == 0
#             or i == len(test_dataset) - 1
#         ):

#             print(
#                 f"[{i + 1:03d}/{len(test_dataset):03d}] "
#                 f"{case_id} z={z_idx:03d} | "
#                 f"MAE={mae:.6f} "
#                 f"PSNR={psnr:.3f} "
#                 f"SSIM={ssim:.4f}"
#             )

#         # ----------------------------------------
#         # Save visualization
#         # ----------------------------------------

#         should_save = (
#             SAVE_ALL_VIS
#             or z_idx % VIS_INTERVAL == 0
#             or z_idx in VIS_Z_INDICES
#         )

#         if should_save:

#             save_path = os.path.join(
#                 VIS_DIR,
#                 f"{case_id}_z{z_idx:03d}.png",
#             )

#             save_visualization(
#                 phase_mean=phase_mean_np,
#                 target=target_np,
#                 prediction=pred_np,
#                 case_id=case_id,
#                 z_idx=z_idx,
#                 save_path=save_path,
#             )


# # ============================================================
# # 8. Save metrics
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
#             "MAE",
#             "PSNR",
#             "SSIM",
#         ],
#     )

#     writer.writeheader()
#     writer.writerows(results)


# # ============================================================
# # 9. Final results
# # ============================================================

# mean_mae = float(np.mean(mae_list))
# mean_psnr = float(np.mean(psnr_list))
# mean_ssim = float(np.mean(ssim_list))

# summary = (
#     "\n"
#     + "=" * 80
#     + "\n"
#     + "UNet 10-phase + FFL Test Results\n"
#     + "=" * 80
#     + "\n"
#     + f"Samples : {len(mae_list)}\n"
#     + f"MAE     : {mean_mae:.6f}\n"
#     + f"PSNR    : {mean_psnr:.4f} dB\n"
#     + f"SSIM    : {mean_ssim:.6f}\n"
#     + "=" * 80
# )

# print(summary)

# summary_path = os.path.join(
#     OUTPUT_DIR,
#     "summary.txt",
# )

# with open(
#     summary_path,
#     "w",
#     encoding="utf-8",
# ) as f:

#     f.write(summary)

# print()
# print("Metrics saved:", csv_path)
# print("Summary saved:", summary_path)
# print("Visualizations saved:", VIS_DIR)
import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion
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
# FFL 模型路径
CKPT_PATH = (
    "./checkpoints/unet_10phase_ffl/best.pth"
)
# 测试结果保存路径
OUTPUT_DIR = "./test_results/unet_10phase_ffl_fov"
VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
NUM_WORKERS = 4
# 每隔多少张保存一次可视化
VIS_INTERVAL = 10
# 额外保存指定 z 层
VIS_Z_INDICES = {70, 80}
# 是否保存所有切片
SAVE_ALL_VIS = False
# 高频可视化参数
HIGH_PASS_SIGMA = 3.0

# FOV matches the original recon_fov0 generation for Case10 (512 -> 256).
FOV_RADIUS = (512 * 1.2 * 720.0 / (2 * 1200.0)) / 0.97 * (256 / 512)
DATA_RANGE = 2.0
SSIM_WIN_SIZE = 7
SSIM_EROSION = 3

def get_fov_mask(shape):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - (w - 1) / 2.0) ** 2 +
            (yy - (h - 1) / 2.0) ** 2) <= FOV_RADIUS ** 2

# ============================================================
# 2. Fixed scan conditions
# ============================================================
SCAN_TIME = 30.0
BREATHING_PERIOD = 4.0
START_PHASE = 0.0
SOD = 720.0
DU = 1.2
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
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
print("=" * 80)
print("UNet 10-phase + FFL Test")
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
# 5. Visualization function
# ============================================================
def save_visualization(
    phase_mean,
    target,
    prediction,
    case_id,
    z_idx,
    save_path,
):
    # ----------------------------------------
    # 高频分量
    # ----------------------------------------
    gt_high = (
        target
        - gaussian_filter(
            target,
            sigma=HIGH_PASS_SIGMA,
        )
    )
    pred_high = (
        prediction
        - gaussian_filter(
            prediction,
            sigma=HIGH_PASS_SIGMA,
        )
    )
    # ----------------------------------------
    # Absolute Difference
    # ----------------------------------------
    abs_diff = np.abs(
        prediction - target
    )
    # ----------------------------------------
    # 统一灰度范围
    # ----------------------------------------
    vmin = min(
        phase_mean.min(),
        target.min(),
        prediction.min(),
    )
    vmax = max(
        phase_mean.max(),
        target.max(),
        prediction.max(),
    )
    # 高频图统一颜色范围
    high_limit = max(
        np.abs(gt_high).max(),
        np.abs(pred_high).max(),
        1e-6,
    )
    # ----------------------------------------
    # Plot
    # ----------------------------------------
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
    )
    # 第一行
    im0 = axes[0, 0].imshow(
        phase_mean,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title(
        "Phase Mean (10P)"
    )
    im1 = axes[0, 1].imshow(
        target,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title(
        "GT Motion-FDK"
    )
    im2 = axes[0, 2].imshow(
        prediction,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 2].set_title(
        "Predicted Motion-FDK (FFL)"
    )
    # 第二行
    im3 = axes[1, 0].imshow(
        gt_high,
        cmap="seismic",
        vmin=-high_limit,
        vmax=high_limit,
    )
    axes[1, 0].set_title(
        "GT High-Frequency"
    )
    im4 = axes[1, 1].imshow(
        pred_high,
        cmap="seismic",
        vmin=-high_limit,
        vmax=high_limit,
    )
    axes[1, 1].set_title(
        "Predicted High-Frequency"
    )
    im5 = axes[1, 2].imshow(
        abs_diff,
        cmap="hot",
        vmin=0,
        vmax=max(abs_diff.max(), 1e-6),
    )
    axes[1, 2].set_title(
        "Absolute Difference"
    )
    images = [
        im0, im1, im2,
        im3, im4, im5,
    ]
    for ax, im in zip(
        axes.flat,
        images,
    ):
        ax.axis("off")
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
    fig.suptitle(
        f"{case_id} | z={z_idx} | "
        f"10P E2E + FFL",
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
# 6. Helper
# ============================================================
def get_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return get_scalar(value[0])
    return value
# ============================================================
# 7. Evaluation
# ============================================================
mae_list = []
psnr_list = []
ssim_list = []
fov_mae_list = []
fov_psnr_list = []
fov_ssim_list = []
results = []
with torch.inference_mode():
    for i, batch in enumerate(test_loader):
        # ----------------------------------------
        # Load data
        # ----------------------------------------
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
        # ----------------------------------------
        # E2E prediction
        #
        # 注意：不使用 Phase Mean Residual
        # ----------------------------------------
        prediction = model(source)
        # 仅用于可视化
        phase_mean = source.mean(
            dim=1,
            keepdim=True,
        )
        # ----------------------------------------
        # Tensor -> NumPy
        # ----------------------------------------
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
        phase_mean_np = (
            phase_mean[0, 0]
            .float()
            .cpu()
            .numpy()
        )
        # ----------------------------------------
        # MAE
        # ----------------------------------------
        mae = np.mean(
            np.abs(
                pred_np - target_np
            )
        )
        # ----------------------------------------
        # PSNR
        # ----------------------------------------
        psnr = peak_signal_noise_ratio(
            target_np,
            pred_np,
            data_range=2.0,
        )
        # ----------------------------------------
        # SSIM
        # ----------------------------------------
        ssim = structural_similarity(
            target_np,
            pred_np,
            data_range=2.0,
        )
        mae_list.append(mae)
        psnr_list.append(psnr)
        # Masked FOV metrics; calculate PSNR from FOV-only MSE.
        fov_mask = get_fov_mask(target_np.shape)
        err = pred_np - target_np
        fov_mae = float(np.abs(err[fov_mask]).mean())
        fov_mse = float(np.square(err[fov_mask]).mean())
        fov_psnr = float(10.0 * np.log10(DATA_RANGE ** 2 / max(fov_mse, 1e-12)))
        _, ssim_map = structural_similarity(
            target_np, pred_np, data_range=DATA_RANGE,
            win_size=SSIM_WIN_SIZE, full=True,
        )
        ssim_mask = binary_erosion(fov_mask, iterations=SSIM_EROSION)
        fov_ssim = float(ssim_map[ssim_mask].mean())
        fov_mae_list.append(fov_mae)
        fov_psnr_list.append(fov_psnr)
        fov_ssim_list.append(fov_ssim)
        ssim_list.append(ssim)
        # ----------------------------------------
        # Case ID / Z index
        # ----------------------------------------
        case_id = str(
            get_scalar(batch["case_id"])
        )
        z_idx = int(
            get_scalar(batch["z_idx"])
        )
        # ----------------------------------------
        # Save per-slice metrics
        # ----------------------------------------
        results.append({
            "case_id": case_id,
            "z_idx": z_idx,
            "MAE": float(mae),
            "PSNR": float(psnr),
            "SSIM": float(ssim),
           "FOV_MAE": fov_mae,
           "FOV_PSNR": fov_psnr,
           "FOV_SSIM": fov_ssim,
        })
        # ----------------------------------------
        # Print
        # ----------------------------------------
        if (
            i % 20 == 0
            or i == len(test_dataset) - 1
        ):
            print(
                f"[{i + 1:03d}/{len(test_dataset):03d}] "
                f"{case_id} z={z_idx:03d} | "
                f"MAE={mae:.6f} "
                f"PSNR={psnr:.3f} "
                f"SSIM={ssim:.4f} | "
               f"FOV MAE={fov_mae:.6f} "
               f"PSNR={fov_psnr:.3f} "
               f"SSIM={fov_ssim:.4f}"
            )
        # ----------------------------------------
        # Save visualization
        # ----------------------------------------
        should_save = (
            SAVE_ALL_VIS
            or z_idx % VIS_INTERVAL == 0
            or z_idx in VIS_Z_INDICES
        )
        if should_save:
            save_path = os.path.join(
                VIS_DIR,
                f"{case_id}_z{z_idx:03d}.png",
            )
            save_visualization(
                phase_mean=phase_mean_np,
                target=target_np,
                prediction=pred_np,
                case_id=case_id,
                z_idx=z_idx,
                save_path=save_path,
            )
# ============================================================
# 8. Save metrics
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
        "MAE",
        "PSNR",
        "SSIM",
        "FOV_MAE",
        "FOV_PSNR",
        "FOV_SSIM",
    ],
)
    writer.writeheader()
    writer.writerows(results)
# ============================================================
# 9. Final results
# ============================================================
mean_mae = float(np.mean(mae_list))
mean_psnr = float(np.mean(psnr_list))
mean_ssim = float(np.mean(ssim_list))
mean_fov_mae = float(np.mean(fov_mae_list))
mean_fov_psnr = float(np.mean(fov_psnr_list))
mean_fov_ssim = float(np.mean(fov_ssim_list))
summary = (
    "\n"
    + "=" * 80
    + "\n"
    + "UNet 10-phase + FFL Test Results\n"
    + "=" * 80
    + "\n"
    + f"Samples : {len(mae_list)}\n"
    + f"MAE     : {mean_mae:.6f}\n"
    + f"PSNR    : {mean_psnr:.4f} dB\n"
    + f"SSIM    : {mean_ssim:.6f}\n"
   + f"FOV radius: {FOV_RADIUS:.6f} pixels; center: image center\n"
   + f"SSIM window: {SSIM_WIN_SIZE}; FOV erosion: {SSIM_EROSION} pixels\n"
   + f"Data range: {DATA_RANGE}\n"
   + f"FOV_MAE  : {mean_fov_mae:.6f}\n"
   + f"FOV_PSNR : {mean_fov_psnr:.6f} dB\n"
   + f"FOV_SSIM : {mean_fov_ssim:.6f}\n"
    + "=" * 80
)
print(summary)
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
print()
print("Metrics saved:", csv_path)
print("Summary saved:", summary_path)
print("Visualizations saved:", VIS_DIR)

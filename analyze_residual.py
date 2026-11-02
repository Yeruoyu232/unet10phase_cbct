import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import DIRLab10PhaseDataset

DATA_ROOT = (
    "/home/yry/ENV/ControlNet-main_10phase/training/"
    "cbct_motion_10phase_samez_minus_1024_div_1024"
)
PROMPT_FILE = os.path.join(DATA_ROOT, "prompt_train.json")
OUTPUT_DIR = "./residual_statistics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 这里需要与你训练脚本的 Dataset 初始化参数保持一致
dataset = DIRLab10PhaseDataset(
    root=DATA_ROOT,
    prompt_file=PROMPT_FILE,
    scan_time=30.0,
    breathing_period=4.0,
    start_phase=0.0,
    sod=720,
    du=1.2
)

# 如果你的 dataset.py 使用不同参数名，请按原训练脚本修改。

# 从 FOV 内残差抽样用于绘图与分位数估计。
# 均值、标准差、最小值和最大值使用全部有效像素精确计算。
rng = np.random.default_rng(42)
MAX_SAMPLES = 2_000_000
sampled = []
seen = 0

count = 0
total = 0.0
total_sq = 0.0
min_value = float("inf")
max_value = float("-inf")

# 注意：这里使用 Target 的非零区域作为 FOV 近似。
# 最严谨的做法是复用训练脚本中的 build_fov_mask(case_id)。
for i in range(len(dataset)):
    item = dataset[i]

    source = item["source"]
    target = item["target"]

    if hasattr(source, "numpy"):
        source = source.numpy()
    if hasattr(target, "numpy"):
        target = target.numpy()

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).squeeze()

    # source: [10, H, W]
    assert source.ndim == 3 and source.shape[0] == 10, source.shape

    ct_mean = source.mean(axis=0)
    residual = target - ct_mean

    mask = target != 0
    values = residual[mask].astype(np.float64)

    if values.size == 0:
        continue

    count += values.size
    total += values.sum()
    total_sq += np.square(values).sum()
    min_value = min(min_value, values.min())
    max_value = max(max_value, values.max())

    # 每张图随机抽取部分像素，控制内存占用
    n = min(256, values.size)
    selected = rng.choice(values, size=n, replace=False)
    sampled.append(selected)
    seen += n

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(dataset)}")

mean = total / count
std = np.sqrt(max(total_sq / count - mean ** 2, 0))

samples = np.concatenate(sampled)

# 如果抽样量较大，再缩减用于分位数估计的样本
if len(samples) > MAX_SAMPLES:
    samples = rng.choice(samples, MAX_SAMPLES, replace=False)

median, p1, p5, p95, p99 = (
    np.percentile(samples, [50, 1, 5, 95, 99])
)

print("\nResidual Statistics (Train, FOV Approximation)")
print(f"Pixels = {count:,}")
print(f"Mean   = {mean:.6f}")
print(f"Std    = {std:.6f}")
print(f"Median = {median:.6f}")
print(f"P1     = {p1:.6f}")
print(f"P5     = {p5:.6f}")
print(f"P95    = {p95:.6f}")
print(f"P99    = {p99:.6f}")
print(f"Min    = {min_value:.6f}")
print(f"Max    = {max_value:.6f}")

plt.figure(figsize=(9, 5))
plt.hist(samples, bins=150, density=True, alpha=0.75)
plt.axvline(0, color="black", linestyle="--", label="Zero")
plt.axvline(mean, color="red", linestyle="--",
            label=f"Mean = {mean:.4f}")
plt.xlabel("Residual (FDK - CT_mean)")
plt.ylabel("Density")
plt.title("Training Residual Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "residual_histogram.png"),
    dpi=200
)
plt.close()
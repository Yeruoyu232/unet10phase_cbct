import re
import matplotlib.pyplot as plt


LOG_FILE = "/home/yry/ENV/unet_10phase/logs/train_10phase_residual_fov.log"
SAVE_FILE = "train_10phase_residual_fov.png"


epochs = []
train_losses = []
val_losses = []

pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+"
    r"Train L1\s*=\s*([0-9.eE+-]+)\s+\|\s+"
    r"Val L1\s*=\s*([0-9.eE+-]+)"
)


with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))

            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)


print(f"Found {len(epochs)} epochs")

if len(epochs) == 0:
    raise RuntimeError("No loss records found in log file.")


# 找 validation 最优 epoch
best_idx = min(
    range(len(val_losses)),
    key=lambda i: val_losses[i]
)

best_epoch = epochs[best_idx]
best_val = val_losses[best_idx]

print("Best epoch:", best_epoch)
print("Best val L1:", best_val)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    label="Train L1",
    linewidth=2,
)

plt.plot(
    epochs,
    val_losses,
    label="Validation L1",
    linewidth=2,
)

# 标出最佳 epoch
plt.scatter(
    best_epoch,
    best_val,
    s=60,
    zorder=5,
    label=f"Best Val: {best_val:.4f} (Epoch {best_epoch})",
)

plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("U-Net Training and Validation Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    SAVE_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", SAVE_FILE)
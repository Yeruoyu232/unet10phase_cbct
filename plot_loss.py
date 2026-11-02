

# import re
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt

# LOG_PATH = Path(
#     "/home/yry/ENV/unet_10phase/logs/residual_center.log"
# )

# OUTPUT_DIR = LOG_PATH.parent / "loss_figures/center"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # 只匹配每个 Epoch 的汇总行，不匹配 Step 日志
# pattern = re.compile(
#     r"^Epoch\s+(\d+)/\d+\s*\|"
#     r"\s*Train FOV L1\s*=\s*([\d.eE+-]+)\s*\|"
#     r"\s*Val FOV L1\s*=\s*([\d.eE+-]+)\s*\|"
#     r"\s*Train Full L1\s*=\s*([\d.eE+-]+)\s*\|"
#     r"\s*Val Full L1\s*=\s*([\d.eE+-]+)"
# )

# records = {}

# with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
#     for line in f:
#         match = pattern.search(line.strip())
#         if match:
#             epoch = int(match.group(1))
#             records[epoch] = tuple(
#                 float(match.group(i)) for i in range(2, 6)
#             )

# if not records:
#     raise RuntimeError("No epoch summary found in log.")

# epochs = np.array(sorted(records))
# values = np.array([records[e] for e in epochs])

# train_fov = values[:, 0]
# val_fov = values[:, 1]
# train_full = values[:, 2]
# val_full = values[:, 3]

# best_idx = np.argmin(val_fov)
# best_epoch = epochs[best_idx]
# best_loss = val_fov[best_idx]

# print(f"Parsed {len(epochs)} epochs")
# print(f"Best Epoch: {best_epoch}")
# print(f"Best Val FOV L1: {best_loss:.6f}")

# # Figure 1: FOV Loss
# plt.figure(figsize=(9, 5))
# plt.plot(epochs, train_fov, label="Train FOV L1", linewidth=2)
# plt.plot(epochs, val_fov, label="Val FOV L1", linewidth=2)

# plt.scatter(
#     [best_epoch], [best_loss],
#     marker="*", s=140, color="red", zorder=5,
#     label=f"Best Epoch {best_epoch}"
# )

# plt.xlabel("Epoch")
# plt.ylabel("FOV L1 Loss")
# plt.title("Residual U-Net: FOV Loss")
# plt.legend()
# plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "fov_loss.png", dpi=300)
# plt.close()

# # Figure 2: Full-image Loss
# plt.figure(figsize=(9, 5))
# plt.plot(epochs, train_full, label="Train Full L1", linewidth=2)
# plt.plot(epochs, val_full, label="Val Full L1", linewidth=2)

# plt.xlabel("Epoch")
# plt.ylabel("Full L1 Loss")
# plt.title("Residual U-Net: Full-image Loss")
# plt.legend()
# plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "full_loss.png", dpi=300)
# plt.close()

# # 保存 CSV，方便后续比较 Raw / Center / Z-score
# data = np.column_stack(
#     [epochs, train_fov, val_fov, train_full, val_full]
# )
# np.savetxt(
#     OUTPUT_DIR / "loss.csv",
#     data,
#     delimiter=",",
#     header="epoch,train_fov,val_fov,train_full,val_full",
#     comments="",
#     fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"],
# )

# print("Saved to:", OUTPUT_DIR)

# #!/usr/bin/env python3
# """Parse epoch summaries only; support FOV-L1 and FOV L1 log formats."""
# import argparse
# import csv
# import re
# from pathlib import Path


# def main():
#     p = argparse.ArgumentParser(description=__doc__)
#     p.add_argument('log', type=Path)
#     p.add_argument('--out-dir', type=Path, default=None)
#     args = p.parse_args()
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

#     number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
#     pattern = re.compile(
#         r'^Epoch\s+(\d+)/\d+\s*\|\s*Train FOV[- ]L1\s*=\s*'
#         + number + r'\s*\|\s*Val FOV[- ]L1\s*=\s*' + number)
#     rows = {}
#     duplicates = []
#     for line in args.log.read_text(encoding='utf-8', errors='replace').splitlines():
#         match = pattern.search(line.strip())
#         if match:
#             epoch = int(match[1])
#             if epoch in rows:
#                 duplicates.append(epoch)
#             rows[epoch] = (float(match[2]), float(match[3]))
#     if not rows:
#         raise SystemExit('No completed epoch summaries found. Step losses are intentionally excluded.')
#     if duplicates:
#         raise SystemExit('Repeated epoch numbers detected (possibly multiple runs in one log). '
#                          'Split the log by run before plotting: ' + str(sorted(set(duplicates))))
#     out = args.out_dir or args.log.parent / (args.log.stem + '_figures')
#     out.mkdir(parents=True, exist_ok=True)
#     epochs = sorted(rows)
#     train = [rows[e][0] for e in epochs]
#     val = [rows[e][1] for e in epochs]
#     best = min(epochs, key=lambda e: rows[e][1])
#     with (out / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['epoch', 'train_fov_l1', 'val_fov_l1'])
#         writer.writerows((e, *rows[e]) for e in epochs)
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.plot(epochs, train, label='Train FOV L1', linewidth=1.5)
#     ax.plot(epochs, val, label='Validation FOV L1', linewidth=1.5)
#     ax.scatter([best], [rows[best][1]], color='red', zorder=5,
#                label=f'Best val: {rows[best][1]:.6f} (epoch {best})')
#     ax.set(xlabel='Epoch', ylabel='FOV L1', title='Training and validation loss')
#     ax.grid(alpha=.25)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(out / 'fov_loss.png', dpi=300)
#     fig.savefig(out / 'fov_loss.pdf')
#     plt.close(fig)
#     print(f'Completed epochs: {len(epochs)}; best epoch: {best}; best val: {rows[best][1]:.6f}')
#     print('Outputs:', out.resolve())


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
"""Parse epoch summaries only; support FOV-L1 and FOV L1 log formats."""
import argparse
import csv
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    pattern = re.compile(
        r'^Epoch\s+(\d+)/\d+\s*\|\s*Train FOV[- ]L1\s*=\s*'
        + number + r'\s*\|\s*Val FOV[- ]L1\s*=\s*' + number)
    rows = {}
    duplicates = []
    for line in args.log.read_text(encoding='utf-8', errors='replace').splitlines():
        match = pattern.search(line.strip())
        if match:
            epoch = int(match[1])
            if epoch in rows:
                duplicates.append(epoch)
            rows[epoch] = (float(match[2]), float(match[3]))
    if not rows:
        raise SystemExit('No completed epoch summaries found. Step losses are intentionally excluded.')
    if duplicates:
        raise SystemExit('Repeated epoch numbers detected (possibly multiple runs in one log). '
                         'Split the log by run before plotting: ' + str(sorted(set(duplicates))))
    out = args.out_dir or args.log.parent / (args.log.stem + '_figures')
    out.mkdir(parents=True, exist_ok=True)
    epochs = sorted(rows)
    train = [rows[e][0] for e in epochs]
    val = [rows[e][1] for e in epochs]
    best = min(epochs, key=lambda e: rows[e][1])
    with (out / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_fov_l1', 'val_fov_l1'])
        writer.writerows((e, *rows[e]) for e in epochs)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train, label='Train FOV L1', linewidth=1.5)
    ax.plot(epochs, val, label='Validation FOV L1', linewidth=1.5)
    ax.scatter([best], [rows[best][1]], color='red', zorder=5,
               label=f'Best val: {rows[best][1]:.6f} (epoch {best})')
    ax.set(xlabel='Epoch', ylabel='FOV L1', title='Training and validation loss')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'fov_loss.png', dpi=300)
    fig.savefig(out / 'fov_loss.pdf')
    plt.close(fig)
    print(f'Completed epochs: {len(epochs)}; best epoch: {best}; best val: {rows[best][1]:.6f}')
    print('Outputs:', out.resolve())


if __name__ == '__main__':
    main()

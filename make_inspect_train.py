import json
import math
from pathlib import Path

root = Path(
    "/home/yry/ENV/ControlNet-main_10phase/training/"
    "cbct_motion_10phase_samez_minus_1024_div_1024"
)

conditions = {
    "scan_time": 30,
    "breathing_period": 4,
    "start_phase": 0,
    "SOD": 720,
    "SDD": 1200,
    "du": 1.2,
}

selected = []

for line in (root / "prompt_train.json").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)

    if int(r["case_id"]) not in range(1, 9):
        continue

    if all(
        math.isclose(float(r[k]), v, rel_tol=0, abs_tol=1e-6)
        for k, v in conditions.items()
    ):
        selected.append(r)

with (root / "prompt_inspect_train.json").open("w") as f:
    for r in selected:
        f.write(json.dumps(r) + "\n")

print("原始训练切片：", len(selected))
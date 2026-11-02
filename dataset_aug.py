#fov训练
import os
from PIL import Image
import json

import numpy as np
import torch
from torch.utils.data import Dataset
CASE_GEOMETRY = {
    1:  (256, 256, 0.97, 0.97),
    2:  (256, 256, 1.16, 1.16),
    3:  (256, 256, 1.15, 1.15),
    4:  (256, 256, 1.13, 1.13),
    5:  (256, 256, 1.10, 1.10),
    6:  (512, 512, 0.97, 0.97),
    7:  (512, 512, 0.97, 0.97),
    8:  (512, 512, 0.97, 0.97),
    9:  (512, 512, 0.97, 0.97),
    10: (512, 512, 0.97, 0.97),
}

class DIRLab10PhaseDataset(Dataset):

    # def __init__(
    #     self,
    #     root,
    #     prompt_file,
    #     scan_time,
    #     breathing_period,
    #     start_phase,
    #     sod,
    #     du,
    # ):

    #     self.root = root
    #     self.records = []
    def __init__(
    self,
    root,
    prompt_file,
    scan_time,
    breathing_period,
    start_phase,
    sod,
    du,
    input_mode="10phase",
    return_fov_mask=False,
):
        self.root = root
        self.input_mode = input_mode
        self.records = []

        self.return_fov_mask = return_fov_mask

        self.sod = float(sod)
        self.du = float(du)

        self.sdd = 1200.0
        self.nu = 512

        self.fov_mask_cache = {}

        prompt_path = os.path.join(
            root,
            prompt_file,
        )

        with open(
            prompt_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                r = json.loads(line)

                # ==========================================
                # 只保留一种固定扫描条件
                # ==========================================

                if not np.isclose(
                    float(r["scan_time"]),
                    scan_time,
                ):
                    continue

                if not np.isclose(
                    float(r["breathing_period"]),
                    breathing_period,
                ):
                    continue

                if not np.isclose(
                    float(r["start_phase"]),
                    start_phase,
                ):
                    continue

                if not np.isclose(
                    float(r["SOD"]),
                    sod,
                ):
                    continue

                if not np.isclose(
                    float(r["du"]),
                    du,
                ):
                    continue

                if not np.isclose(float(r['SDD']), self.sdd, rtol=0, atol=1e-6):
                    continue
                self.records.append(r)

        print(
            f"[{prompt_file}] "
            f"Loaded {len(self.records)} samples."
        )

        if len(self.records) == 0:
            raise RuntimeError(
                f"No samples found in {prompt_file}. "
                "Please check the fixed scan condition."
            )
    def _build_fov_mask(
        self,
        case_id,
        output_height=256,
        output_width=256,
    ):

        cache_key = (
            case_id,
            output_height,
            output_width,
        )

        if cache_key in self.fov_mask_cache:
            return self.fov_mask_cache[cache_key]

        if case_id not in CASE_GEOMETRY:
            raise ValueError(
                f"Unknown case_id: {case_id}"
            )

        nx, ny, dx, dy = CASE_GEOMETRY[case_id]

        # 与 GT 生成代码保持一致
        fov_radius_mm = (
            self.nu
            * self.du
            * self.sod
            / (2.0 * self.sdd)
        )

        # 原始分辨率的物理坐标
        x = (
            np.arange(nx) - (nx - 1) / 2.0
        ) * dx

        y = (
            np.arange(ny) - (ny - 1) / 2.0
        ) * dy

        xx, yy = np.meshgrid(x, y)

        # 原始分辨率 FOV Mask
        original_mask = (
            xx ** 2 + yy ** 2
            <= fov_radius_mm ** 2
        ).astype(np.float32)

        # 与 Target 使用相同的 PIL BILINEAR
        if original_mask.shape != (
            output_height,
            output_width,
        ):

            mask_image = Image.fromarray(
                original_mask,
                mode="F",
            )

            mask_image = mask_image.resize(
                (output_width, output_height),
                resample=Image.Resampling.BILINEAR,
            )

            resized_mask = np.asarray(
                mask_image,
                dtype=np.float32,
            )

        else:
            resized_mask = original_mask

        # 二值化
        binary_mask = (
            resized_mask > 0.5
        ).astype(np.float32)

        fov_mask = torch.from_numpy(
            binary_mask
        ).unsqueeze(0)

        self.fov_mask_cache[cache_key] = fov_mask

        print(
            f"[FOV] Case{case_id}, "
            f"radius_mm={fov_radius_mm:.2f}, "
            f"mask_ratio={binary_mask.mean():.4f}"
        )

        return fov_mask


    def __len__(self):

        return len(self.records)


    def __getitem__(self, index):

        r = self.records[index]

        # ==========================================
        # Source
        # [H,W,10]
        # ==========================================

        source_path = os.path.join(
            self.root,
            r["source"],
        )

        source = np.load(
            source_path, allow_pickle=False
        ).astype(np.float32)

        if source.shape != (256, 256, 10):
            raise RuntimeError(
                f"Bad source shape: {source.shape}"
            )

        # ============================================================
        # Input mode
        # ============================================================

        if self.input_mode == "10phase":

            # T00,T10,T20,T30,T40,T50,T60,T70,T80,T90
            # [H,W,10]
            pass


        elif self.input_mode == "5phase":

            # T00,T20,T40,T60,T80
            # channel index = 0,2,4,6,8
            # [H,W,10] -> [H,W,5]

            source = source[:, :, [0, 2, 4, 6, 8]]


        elif self.input_mode == "T20":

            # T20 only
            # [H,W,10] -> [H,W,1]

            source = source[:, :, 2:3]


        else:

            raise ValueError(
                f"Unknown input_mode: {self.input_mode}"
            )
        # ==========================================
        # Target
        # ==========================================

        target_path = os.path.join(
            self.root,
            r["target"],
        )

        target = np.load(
            target_path, allow_pickle=False
        ).astype(np.float32)

        # 原来的 target 是三个重复灰度通道
        if (
            target.ndim == 3
            and target.shape[-1] == 3
        ):
            target = target[:, :, 0]

        elif (
            target.ndim == 3
            and target.shape[-1] == 1
        ):
            target = target[:, :, 0]

        elif target.ndim != 2:
            raise RuntimeError(
                f"Bad target shape: {target.shape}"
            )


        if target.shape != (256, 256):
            raise RuntimeError(f'Bad target shape: {target.shape}')
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise RuntimeError(f'Nonfinite source/target at index {index}')
        # ==========================================
        # HWC -> CHW
        # ==========================================

        source = torch.from_numpy(
            source
        ).permute(
            2, 0, 1
        ).contiguous()

        target = torch.from_numpy(
            target
        ).unsqueeze(
            0
        ).contiguous()


        result = {
            "source": source,
            "target": target,
            "case_id": int(r["case_id"]),
            "z_idx": int(r["z_idx"]),
            "variant": r.get("variant", "original"),
        }

        if self.return_fov_mask:

            case_id = int(r["case_id"])

            if "mask" in r:
                mask_path = os.path.join(self.root, r["mask"])
                mask_np = np.load(mask_path, allow_pickle=False)
                if mask_np.shape != tuple(target.shape[-2:]):
                    raise RuntimeError(f"Bad mask shape: {mask_np.shape}")
                if not np.isfinite(mask_np).all() or not np.isin(mask_np, [0, 1]).all():
                    raise RuntimeError(f"Invalid binary mask: {mask_path}")
                if not mask_np.any():
                    raise RuntimeError(f"Empty mask: {mask_path}")
                fov_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
            else:
                if r.get("variant", "original") != "original" or r.get("augmentation") or r.get("aug_id", 0) != 0:
                    raise RuntimeError("Augmented sample missing saved mask")
                fov_mask = self._build_fov_mask(
                    case_id=case_id,
                    output_height=target.shape[-2],
                    output_width=target.shape[-1],
                )

            result["fov_mask"] = fov_mask

        return result

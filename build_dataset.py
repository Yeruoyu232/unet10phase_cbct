# import os
# import json
# import argparse
# from pathlib import Path
# from typing import Dict, List, Tuple
# import numpy as np
# from PIL import Image


# SOURCE_PHASES = [
#     "00",
#     "10",
#     "20",
#     "30",
#     "40",
#     "50",
#     "60",
#     "70",
#     "80",
#     "90",
# ]

# SOURCE_PHASE_NAMES = [
#     f"T{phase}"
#     for phase in SOURCE_PHASES
# ]

# SOURCE_MODE = (
#     "T00_to_T90_same_z_10ch"
# )

# OUTPUT_SIZE = 256

# CASE_CONFIGS: Dict[
#     int,
#     Dict[str, Tuple[float, ...]]
# ] = {

#     1: {
#         "dims_xyz": (
#             256,
#             256,
#             94,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },

#     2: {
#         "dims_xyz": (
#             256,
#             256,
#             112,
#         ),
#         "spacing_xyz": (
#             1.16,
#             1.16,
#             2.5,
#         ),
#     },

#     3: {
#         "dims_xyz": (
#             256,
#             256,
#             104,
#         ),
#         "spacing_xyz": (
#             1.15,
#             1.15,
#             2.5,
#         ),
#     },

#     4: {
#         "dims_xyz": (
#             256,
#             256,
#             99,
#         ),
#         "spacing_xyz": (
#             1.13,
#             1.13,
#             2.5,
#         ),
#     },

#     5: {
#         "dims_xyz": (
#             256,
#             256,
#             106,
#         ),
#         "spacing_xyz": (
#             1.10,
#             1.10,
#             2.5,
#         ),
#     },

#     6: {
#         "dims_xyz": (
#             512,
#             512,
#             128,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },

#     7: {
#         "dims_xyz": (
#             512,
#             512,
#             136,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },

#     8: {
#         "dims_xyz": (
#             512,
#             512,
#             128,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },

#     9: {
#         "dims_xyz": (
#             512,
#             512,
#             128,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },

#     10: {
#         "dims_xyz": (
#             512,
#             512,
#             120,
#         ),
#         "spacing_xyz": (
#             0.97,
#             0.97,
#             2.5,
#         ),
#     },
# }

# USE_CLIP = False

# CT_CLIP_MIN = -1024.0

# CT_CLIP_MAX = 2048.0

# RECON_CLIP_MIN = -1024.0

# RECON_CLIP_MAX = 2048.0

# def load_ct_int16(
#     path: str,
#     dims_xyz: Tuple[int, int, int],
# ) -> np.ndarray:

#     """
#     DIR-Lab int16 CT

#     input:
#         dims_xyz = (x,y,z)

#     output:
#         (z,y,x)
#     """

#     x, y, z = dims_xyz

#     array = np.fromfile(
#         path,
#         dtype=np.int16,
#     )

#     expected = (
#         x * y * z
#     )

#     if array.size != expected:

#         raise ValueError(
#             f"CT size mismatch: {path}\n"
#             f"got={array.size}, "
#             f"expected={expected}"
#         )

#     return array.reshape(
#         z,
#         y,
#         x,
#     )

# def load_recon_float32(
#     path: str,
#     dims_xyz: Tuple[int, int, int],
# ) -> np.ndarray:

#     """
#     FDK reconstruction float32

#     input:
#         dims_xyz=(x,y,z)

#     output:
#         (z,y,x)
#     """

#     x, y, z = dims_xyz

#     array = np.fromfile(
#         path,
#         dtype=np.float32,
#     )

#     expected = (
#         x * y * z
#     )

#     if array.size != expected:

#         raise ValueError(
#             f"Recon size mismatch: {path}\n"
#             f"got={array.size}, "
#             f"expected={expected}"
#         )

#     return array.reshape(
#         z,
#         y,
#         x,
#     )

# def convert_ct_to_float(
#     ct_slice: np.ndarray,
# ) -> np.ndarray:

#     """
#     DIR-Lab CT stored value -> training value

#     stored
#       ↓
#     stored - 1024
#       ↓
#     /1024
#     """

#     value = (
#         ct_slice.astype(np.float32)
#         - 1024.0
#     )

#     if USE_CLIP:

#         value = np.clip(
#             value,
#             CT_CLIP_MIN,
#             CT_CLIP_MAX,
#         )

#     return (
#         value / 1024.0
#     ).astype(
#         np.float32
#     )

# def make_fov_mask(
#     shape,
#     spacing_xy,
#     sod=720.0,
#     sdd=1200.0,
#     nu=512,
#     du=1.2,
# ):
#     ny, nx = shape
#     dx, dy = spacing_xy

#     radius = nu * du * sod / sdd / 2.0

#     x = (np.arange(nx) - (nx - 1) / 2.0) * dx
#     y = (np.arange(ny) - (ny - 1) / 2.0) * dy

#     xx, yy = np.meshgrid(x, y)

#     return (xx**2 + yy**2) <= radius**2

# def convert_recon_to_float(
#     recon_slice: np.ndarray,
#     fov_mask: np.ndarray,
# ) -> np.ndarray:

#     value = recon_slice.astype(np.float32)

#     # FDK 原始存储值转换到 CT 的归一化尺度
#     value = value - 1024.0

#     if USE_CLIP:
#         value = np.clip(
#             value,
#             RECON_CLIP_MIN,
#             RECON_CLIP_MAX,
#         )

#     value = value / 1024.0

#     # FOV 外仍然保持 0
#     value = np.where(fov_mask, value, 0.0)

#     return value.astype(np.float32)

# def resize_float_image(
#     image: np.ndarray,
#     size: int = OUTPUT_SIZE,
# ) -> np.ndarray:

#     if image.ndim != 2:

#         raise ValueError(
#             f"Expected 2D image, "
#             f"got shape={image.shape}"
#         )

#     if image.shape == (
#         size,
#         size,
#     ):

#         return image.astype(
#             np.float32
#         )

#     pil_image = Image.fromarray(
#         image.astype(
#             np.float32
#         ),
#         mode="F",
#     )

#     pil_image = pil_image.resize(
#         (
#             size,
#             size,
#         ),
#         resample=Image.BILINEAR,
#     )

#     return np.asarray(
#         pil_image,
#         dtype=np.float32,
#     )

# def make_preview_png(
#     image: np.ndarray,
#     out_path: str,
#     vmin: float = -1.0,
#     vmax: float = 2.0,
# ) -> None:

#     value = np.clip(
#         image,
#         vmin,
#         vmax,
#     )

#     value = (
#         value - vmin
#     ) / (
#         vmax - vmin + 1e-8
#     )

#     value = np.round(
#         value * 255.0
#     ).astype(
#         np.uint8
#     )

#     Image.fromarray(
#         value,
#         mode="L",
#     ).save(
#         out_path
#     )

# def phase_path(case_dir, case_id, phase, variant):
#     folder = case_dir / ('Images' if variant == 'original' else 'Images_flipx')
#     names = [f'case{case_id}_T{phase}.img', f'case{case_id}_T{phase}_s.img',
#              f'case{case_id}_T{phase}-ssm.img']
#     found = [folder / n for n in names if (folder / n).is_file()]
#     if len(found) != 1:
#         raise RuntimeError(f'Expected one phase file: {folder}, T{phase}; found {found}')
#     return found[0]


# def main():
#     parser = argparse.ArgumentParser(description='Merge original and pre-projection flipx paired data')
#     parser.add_argument('--dirlab-root', type=Path, default=Path('/home/yry/ENV/dirlab'))
#     parser.add_argument('--out-root', type=Path, required=True, help='New directory; must not exist')
#     parser.add_argument('--cases', type=int, nargs='+', default=list(range(1, 11)))
#     parser.add_argument('--previews', action='store_true')
#     args = parser.parse_args()
#     if len(set(args.cases)) != len(args.cases) or any(c not in CASE_CONFIGS for c in args.cases):
#         raise ValueError('Invalid/duplicate cases')
#     # Preflight all required inputs before creating outputs.
#     inputs = []
#     for cid in args.cases:
#         case_dir = args.dirlab_root / f'Case{cid}Pack'
#         variants = ['original', 'flipx'] if cid <= 8 else ['original']
#         for variant in variants:
#             paths = [phase_path(case_dir, cid, phase, variant) for phase in SOURCE_PHASES]
#             folder = 'recon_fov0' if variant == 'original' else 'recon_flipx_fov0'
#             candidates = []
#             for path in (case_dir / folder).glob('recon_*.raw'):
#                 try:
#                     numbers = [float(v) for v in path.stem.split('_')[1:]]
#                 except ValueError:
#                     continue
#                 if len(numbers) == 5 and np.allclose(numbers, [30,4,0,720,1.2], rtol=0, atol=1e-6):
#                     candidates.append(path)
#             if len(candidates) != 1:
#                 raise RuntimeError(f'Expected one fixed-condition FDK: {case_dir / folder}: {candidates}')
#             dims = CASE_CONFIGS[cid]['dims_xyz']
#             count = int(np.prod(dims))
#             for path in paths:
#                 if path.stat().st_size != count * 2:
#                     raise ValueError(f'CT int16 file size mismatch: {path}')
#             if candidates[0].stat().st_size != count * 4:
#                 raise ValueError(f'FDK float32 file size mismatch: {candidates[0]}')
#             inputs.append((cid, variant, paths, candidates[0]))
#     args.out_root.mkdir(parents=True, exist_ok=False)
#     for folder in ['source', 'target', 'mask'] + (['preview_source','preview_target'] if args.previews else []):
#         (args.out_root / folder).mkdir()
#     records = []
#     for cid, variant, paths, recon_path in inputs:
#         config = CASE_CONFIGS[cid]
#         dims = config['dims_xyz']; spacing = config['spacing_xyz']
#         # mmap avoids retaining ten complete volumes in RAM.
#         shape = (dims[2], dims[1], dims[0])
#         volumes = [np.memmap(path, dtype=np.int16, mode='r', shape=shape) for path in paths]
#         recon = np.memmap(recon_path, dtype=np.float32, mode='r', shape=shape)
#         native_mask = make_fov_mask(shape[1:], spacing[:2], sod=720, sdd=1200, nu=512, du=1.2)
#         mask = np.asarray(Image.fromarray(native_mask.astype(np.uint8)).resize(
#             (256,256), Image.Resampling.NEAREST), dtype=bool)
#         mask_rel = f'mask/case{cid:02d}_{variant}.npy'
#         np.save(args.out_root / mask_rel, mask.astype(np.uint8))
#         for z in range(dims[2]):
#             # flipx source files are ALREADY reflected. No second flip here.
#             source = np.stack([resize_float_image(convert_ct_to_float(v[z]), 256)
#                                for v in volumes], axis=-1).astype(np.float32)
#             target_z = dims[2] - 1 - z
#             target = resize_float_image(convert_recon_to_float(recon[target_z], native_mask), 256)
#             target = np.where(mask, target, 0).astype(np.float32)
#             if not np.isfinite(source).all() or not np.isfinite(target).all():
#                 raise ValueError(f'NaN/Inf: Case{cid}, {variant}, z={z}')
#             stem = f'case{cid:02d}_{variant}_z{z:03d}'
#             source_rel, target_rel = f'source/{stem}.npy', f'target/{stem}.npy'
#             np.save(args.out_root/source_rel, source)
#             np.save(args.out_root/target_rel, np.repeat(target[...,None],3,axis=-1))
#             record = dict(source=source_rel, target=target_rel, mask=mask_rel,
#                 prompt='FDK CBCT reconstructed from respiratory-motion cone-beam projections.',
#                 case_id=cid, z_idx=z, target_z_idx=target_z, z_mapping='reverse_z',
#                 source_mode=SOURCE_MODE, source_phases=SOURCE_PHASE_NAMES,
#                 source_channel_layout=[f'T{p}[z]' for p in SOURCE_PHASES],
#                 source_z_indices=[z]*10, source_channels=10, source_z_norm=z/(dims[2]-1),
#                 target_z_norm=target_z/(dims[2]-1), spacing=list(spacing),
#                 scan_time=30., breathing_period=4., start_phase=0., SOD=720., SDD=1200.,
#                 du=1.2, nu=512, n_angle=400, scan_range=360, output_size=256,
#                 variant=variant, flip_stage='before_projection' if variant=='flipx' else 'none',
#                 source_raw_paths={f'T{p}':str(path) for p,path in zip(SOURCE_PHASES,paths)},
#                 target_raw_path=str(recon_path),
#                 value_scaling='ct_minus_1024_div_1024__recon_minus_1024_div_1024_inside_fov')
#             records.append(record)
#             if args.previews:
#                 make_preview_png(source[:,:,2], str(args.out_root/'preview_source'/f'{stem}.png'))
#                 make_preview_png(target, str(args.out_root/'preview_target'/f'{stem}.png'))
#         print(f'Finished Case{cid} {variant}: {dims[2]} pairs', flush=True)
#     splits = {'prompt.json':records,
#               'prompt_train.json':[r for r in records if r['case_id']<=8],
#               'prompt_val.json':[r for r in records if r['case_id']==9],
#               'prompt_test.json':[r for r in records if r['case_id']==10]}
#     for name, subset in splits.items():
#         with (args.out_root/name).open('w',encoding='utf-8') as f:
#             for record in subset:
#                 f.write(json.dumps(record,ensure_ascii=False)+'\n')
#         print(name, len(subset))
#     print('Dataset:',args.out_root)


# if __name__ == '__main__':
#     main()


import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


SOURCE_PHASES = [
    "00",
    "10",
    "20",
    "30",
    "40",
    "50",
    "60",
    "70",
    "80",
    "90",
]

SOURCE_PHASE_NAMES = [
    f"T{phase}"
    for phase in SOURCE_PHASES
]

SOURCE_MODE = (
    "T00_to_T90_same_z_10ch"
)

OUTPUT_SIZE = 256

CASE_CONFIGS: Dict[
    int,
    Dict[str, Tuple[float, ...]]
] = {

    1: {
        "dims_xyz": (
            256,
            256,
            94,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },

    2: {
        "dims_xyz": (
            256,
            256,
            112,
        ),
        "spacing_xyz": (
            1.16,
            1.16,
            2.5,
        ),
    },

    3: {
        "dims_xyz": (
            256,
            256,
            104,
        ),
        "spacing_xyz": (
            1.15,
            1.15,
            2.5,
        ),
    },

    4: {
        "dims_xyz": (
            256,
            256,
            99,
        ),
        "spacing_xyz": (
            1.13,
            1.13,
            2.5,
        ),
    },

    5: {
        "dims_xyz": (
            256,
            256,
            106,
        ),
        "spacing_xyz": (
            1.10,
            1.10,
            2.5,
        ),
    },

    6: {
        "dims_xyz": (
            512,
            512,
            128,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },

    7: {
        "dims_xyz": (
            512,
            512,
            136,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },

    8: {
        "dims_xyz": (
            512,
            512,
            128,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },

    9: {
        "dims_xyz": (
            512,
            512,
            128,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },

    10: {
        "dims_xyz": (
            512,
            512,
            120,
        ),
        "spacing_xyz": (
            0.97,
            0.97,
            2.5,
        ),
    },
}

USE_CLIP = False

CT_CLIP_MIN = -1024.0

CT_CLIP_MAX = 2048.0

RECON_CLIP_MIN = -1024.0

RECON_CLIP_MAX = 2048.0

def load_ct_int16(
    path: str,
    dims_xyz: Tuple[int, int, int],
) -> np.ndarray:

    """
    DIR-Lab int16 CT

    input:
        dims_xyz = (x,y,z)

    output:
        (z,y,x)
    """

    x, y, z = dims_xyz

    array = np.fromfile(
        path,
        dtype=np.int16,
    )

    expected = (
        x * y * z
    )

    if array.size != expected:

        raise ValueError(
            f"CT size mismatch: {path}\n"
            f"got={array.size}, "
            f"expected={expected}"
        )

    return array.reshape(
        z,
        y,
        x,
    )

def load_recon_float32(
    path: str,
    dims_xyz: Tuple[int, int, int],
) -> np.ndarray:

    """
    FDK reconstruction float32

    input:
        dims_xyz=(x,y,z)

    output:
        (z,y,x)
    """

    x, y, z = dims_xyz

    array = np.fromfile(
        path,
        dtype=np.float32,
    )

    expected = (
        x * y * z
    )

    if array.size != expected:

        raise ValueError(
            f"Recon size mismatch: {path}\n"
            f"got={array.size}, "
            f"expected={expected}"
        )

    return array.reshape(
        z,
        y,
        x,
    )

def convert_ct_to_float(
    ct_slice: np.ndarray,
) -> np.ndarray:

    """
    DIR-Lab CT stored value -> training value

    stored
      ↓
    stored - 1024
      ↓
    /1024
    """

    value = (
        ct_slice.astype(np.float32)
        - 1024.0
    )

    if USE_CLIP:

        value = np.clip(
            value,
            CT_CLIP_MIN,
            CT_CLIP_MAX,
        )

    return (
        value / 1024.0
    ).astype(
        np.float32
    )

def make_fov_mask(
    shape,
    spacing_xy,
    sod=720.0,
    sdd=1200.0,
    nu=512,
    du=1.2,
):
    ny, nx = shape
    dx, dy = spacing_xy

    radius = nu * du * sod / sdd / 2.0

    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy

    xx, yy = np.meshgrid(x, y)

    return (xx**2 + yy**2) <= radius**2

def convert_recon_to_float(
    recon_slice: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:

    value = recon_slice.astype(np.float32)

    # FDK 原始存储值转换到 CT 的归一化尺度
    value = value - 1024.0

    if USE_CLIP:
        value = np.clip(
            value,
            RECON_CLIP_MIN,
            RECON_CLIP_MAX,
        )

    value = value / 1024.0

    # FOV 外仍然保持 0
    value = np.where(fov_mask, value, 0.0)

    return value.astype(np.float32)

def resize_float_image(
    image: np.ndarray,
    size: int = OUTPUT_SIZE,
) -> np.ndarray:

    if image.ndim != 2:

        raise ValueError(
            f"Expected 2D image, "
            f"got shape={image.shape}"
        )

    if image.shape == (
        size,
        size,
    ):

        return image.astype(
            np.float32
        )

    pil_image = Image.fromarray(
        image.astype(
            np.float32
        ),
        mode="F",
    )

    pil_image = pil_image.resize(
        (
            size,
            size,
        ),
        resample=Image.BILINEAR,
    )

    return np.asarray(
        pil_image,
        dtype=np.float32,
    )

def make_preview_png(
    image: np.ndarray,
    out_path: str,
    vmin: float = -1.0,
    vmax: float = 2.0,
) -> None:

    value = np.clip(
        image,
        vmin,
        vmax,
    )

    value = (
        value - vmin
    ) / (
        vmax - vmin + 1e-8
    )

    value = np.round(
        value * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        value,
        mode="L",
    ).save(
        out_path
    )

def phase_path(case_dir, case_id, phase, variant):
    folder = case_dir / {'original': 'Images', 'flipx': 'Images_flipx',
                         'shift_5_0_0': 'Images_shift_5_0_0'}[variant]
    names = [f'case{case_id}_T{phase}.img', f'case{case_id}_T{phase}_s.img',
             f'case{case_id}_T{phase}-ssm.img']
    found = [folder / n for n in names if (folder / n).is_file()]
    if len(found) != 1:
        raise RuntimeError(f'Expected one phase file: {folder}, T{phase}; found {found}')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description='Merge original, pre-projection flipx and shift_5_0_0 pairs')
    parser.add_argument('--dirlab-root', type=Path, default=Path('/home/yry/ENV/dirlab'))
    parser.add_argument('--out-root', type=Path, required=True, help='New directory; must not exist')
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--previews', action='store_true')
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or any(c not in CASE_CONFIGS for c in args.cases):
        raise ValueError('Invalid/duplicate cases')
    # Preflight all required inputs before creating outputs.
    inputs = []
    for cid in args.cases:
        case_dir = args.dirlab_root / f'Case{cid}Pack'
        variants = ['original', 'flipx', 'shift_5_0_0'] if cid <= 8 else ['original']
        for variant in variants:
            paths = [phase_path(case_dir, cid, phase, variant) for phase in SOURCE_PHASES]
            folder = {'original': 'recon_fov0', 'flipx': 'recon_flipx_fov0',
                      'shift_5_0_0': 'recon_shift_5_0_0_fov0'}[variant]
            candidates = []
            for path in (case_dir / folder).glob('recon_*.raw'):
                try:
                    numbers = [float(v) for v in path.stem.split('_')[1:]]
                except ValueError:
                    continue
                if len(numbers) == 5 and np.allclose(numbers, [30,4,0,720,1.2], rtol=0, atol=1e-6):
                    candidates.append(path)
            if len(candidates) != 1:
                raise RuntimeError(f'Expected one fixed-condition FDK: {case_dir / folder}: {candidates}')
            dims = CASE_CONFIGS[cid]['dims_xyz']
            count = int(np.prod(dims))
            for path in paths:
                if path.stat().st_size != count * 2:
                    raise ValueError(f'CT int16 file size mismatch: {path}')
            if candidates[0].stat().st_size != count * 4:
                raise ValueError(f'FDK float32 file size mismatch: {candidates[0]}')
            inputs.append((cid, variant, paths, candidates[0]))
    args.out_root.mkdir(parents=True, exist_ok=False)
    for folder in ['source', 'target', 'mask'] + (['preview_source','preview_target'] if args.previews else []):
        (args.out_root / folder).mkdir()
    records = []
    for cid, variant, paths, recon_path in inputs:
        config = CASE_CONFIGS[cid]
        dims = config['dims_xyz']; spacing = config['spacing_xyz']
        # mmap avoids retaining ten complete volumes in RAM.
        shape = (dims[2], dims[1], dims[0])
        volumes = [np.memmap(path, dtype=np.int16, mode='r', shape=shape) for path in paths]
        recon = np.memmap(recon_path, dtype=np.float32, mode='r', shape=shape)
        native_mask = make_fov_mask(shape[1:], spacing[:2], sod=720, sdd=1200, nu=512, du=1.2)
        mask = np.asarray(Image.fromarray(native_mask.astype(np.uint8)).resize(
            (256,256), Image.Resampling.NEAREST), dtype=bool)
        mask_rel = f'mask/case{cid:02d}_{variant}.npy'
        np.save(args.out_root / mask_rel, mask.astype(np.uint8))
        for z in range(dims[2]):
            # Source files are already transformed; do not flip or shift them again.
            source = np.stack([resize_float_image(convert_ct_to_float(v[z]), 256)
                               for v in volumes], axis=-1).astype(np.float32)
            target_z = dims[2] - 1 - z
            target = resize_float_image(convert_recon_to_float(recon[target_z], native_mask), 256)
            target = np.where(mask, target, 0).astype(np.float32)
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError(f'NaN/Inf: Case{cid}, {variant}, z={z}')
            stem = f'case{cid:02d}_{variant}_z{z:03d}'
            source_rel, target_rel = f'source/{stem}.npy', f'target/{stem}.npy'
            np.save(args.out_root/source_rel, source)
            np.save(args.out_root/target_rel, np.repeat(target[...,None],3,axis=-1))
            record = dict(source=source_rel, target=target_rel, mask=mask_rel,
                prompt='FDK CBCT reconstructed from respiratory-motion cone-beam projections.',
                case_id=cid, z_idx=z, target_z_idx=target_z, z_mapping='reverse_z',
                source_mode=SOURCE_MODE, source_phases=SOURCE_PHASE_NAMES,
                source_channel_layout=[f'T{p}[z]' for p in SOURCE_PHASES],
                source_z_indices=[z]*10, source_channels=10, source_z_norm=z/(dims[2]-1),
                target_z_norm=target_z/(dims[2]-1), spacing=list(spacing),
                scan_time=30., breathing_period=4., start_phase=0., SOD=720., SDD=1200.,
                du=1.2, nu=512, n_angle=400, scan_range=360, output_size=256,
                variant=variant,
                shift_requested_mm=[5.,0.,0.] if variant=='shift_5_0_0' else [0.,0.,0.],
                shift_actual_mm=[int(np.rint(5./spacing[0]))*spacing[0],0.,0.] if variant=='shift_5_0_0' else [0.,0.,0.],
                augmentation_stage='before_projection' if variant!='original' else 'none',
                flip_stage='before_projection' if variant=='flipx' else 'none',
                source_raw_paths={f'T{p}':str(path) for p,path in zip(SOURCE_PHASES,paths)},
                target_raw_path=str(recon_path),
                value_scaling='ct_minus_1024_div_1024__recon_minus_1024_div_1024_inside_fov')
            records.append(record)
            if args.previews:
                make_preview_png(source[:,:,2], str(args.out_root/'preview_source'/f'{stem}.png'))
                make_preview_png(target, str(args.out_root/'preview_target'/f'{stem}.png'))
        print(f'Finished Case{cid} {variant}: {dims[2]} pairs', flush=True)
    splits = {'prompt.json':records,
              'prompt_train.json':[r for r in records if r['case_id']<=8],
              'prompt_val.json':[r for r in records if r['case_id']==9],
              'prompt_test.json':[r for r in records if r['case_id']==10]}
    for name, subset in splits.items():
        with (args.out_root/name).open('w',encoding='utf-8') as f:
            for record in subset:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(name, len(subset))
    from collections import Counter
    summary = dict(per_split={name:len(subset) for name,subset in splits.items()},
                   train_variants=dict(Counter(r['variant'] for r in splits['prompt_train.json'])),
                   conditions=dict(scan_time=30,breathing_period=4,start_phase=0,SOD=720,SDD=1200,du=1.2))
    (args.out_root/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Dataset:',args.out_root)


if __name__ == '__main__':
    main()

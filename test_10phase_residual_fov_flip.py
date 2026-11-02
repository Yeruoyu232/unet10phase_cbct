"""Compare original vs physical-flip RAW residual models on unaugmented Case10.
Includes phase-mean baseline and fixed-window PNGs. No NPY predictions.
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset_flip import DIRLab10PhaseDataset
from model import build_model

DATA_ROOT = '/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024'
DEFAULT_MODELS = {
    'original': './checkpoints/unet_10phase_residual_fov/best.pth',
    'physical_flip': './checkpoints/unet_10phase_residual_physical_flip/best.pth',
}

DATA_RANGE = 2.0  # Keep identical to the original evaluation script.


def fov_mask_case10(output_size=256):
    nx = ny = 512
    dx = dy = 0.97
    radius_mm = 512 * 1.2 * 720.0 / (2 * 1200.0)
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    xx, yy = np.meshgrid(x, y)
    native = (xx * xx + yy * yy <= radius_mm * radius_mm).astype(np.float32)
    resized = Image.fromarray(native, mode='F').resize(
        (output_size, output_size), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) > 0.5


def fov_metrics(pred, target, mask):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert pred.shape == target.shape == mask.shape
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    error = pred[mask] - target[mask]
    mae = float(np.abs(error).mean())
    mse = float(np.square(error).mean())
    psnr = float('inf') if mse == 0 else float(10 * np.log10(DATA_RANGE ** 2 / mse))
    # Same SSIM convention as the original script: full SSIM map, FOV eroded 3 pixels.
    _, ssim_map = structural_similarity(
        target, pred, data_range=DATA_RANGE, win_size=7, full=True
    )
    inner_mask = binary_erosion(mask, iterations=3)
    if not inner_mask.any():
        raise ValueError('Empty eroded FOV mask')
    ssim = float(ssim_map[inner_mask].mean())
    return {'fov_mae': mae, 'fov_psnr': psnr, 'fov_ssim': ssim}


def load_checkpoint(path, device, expected_mode):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f'Unexpected checkpoint type: {path}')
    state = ckpt.get('model_state_dict', ckpt)
    # Legacy raw checkpoints did not save normalization metadata.
    if ckpt.get('prediction_mode') not in (None, 'phase_mean_residual'):
        raise ValueError(f'{path}: expected residual checkpoint')
    mode = ckpt.get('residual_mode')
    if mode is None:
        if expected_mode != 'raw':
            raise ValueError(f'{path} has no residual_mode metadata; cannot safely test {expected_mode}')
        mode = 'raw'
    if mode != expected_mode:
        raise ValueError(f'{path}: expected {expected_mode}, checkpoint says {mode}')
    mean = float(ckpt.get('res_mean', 0.0))
    std = float(ckpt.get('res_std', 1.0))
    if mode == 'zscore' and std <= 0:
        raise ValueError('res_std must be positive')
    if mode != 'raw' and 'res_mean' not in ckpt:
        raise ValueError(f'{path} missing res_mean')
    if mode == 'zscore' and 'res_std' not in ckpt:
        raise ValueError(f'{path} missing res_std')
    model = build_model().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Loaded {mode}: {path} | epoch={ckpt.get("epoch", "?")} mean={mean} std={std}')
    return model, mode, mean, std


def predict(model_spec, source):
    model, mode, mean, std = model_spec
    raw_output = model(source)
    if mode == 'raw':
        residual = raw_output
    elif mode == 'center':
        residual = raw_output + mean
    elif mode == 'zscore':
        residual = raw_output * std + mean
    else:
        raise ValueError(mode)
    return source.mean(dim=1, keepdim=True) + residual


def save_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def visualize(path, source, target, preds, mask, case_id, z_idx):
    names = ['CT phase mean', 'FDK target'] + list(preds)
    images = [source.mean(axis=0), target] + list(preds.values())
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 7), squeeze=False)
    for j, (name, arr) in enumerate(zip(names, images)):
        axes[0, j].imshow(np.where(mask, arr, 0), cmap='gray', vmin=-1, vmax=2)
        axes[0, j].set_title(name)
        axes[0, j].axis('off')
        if j >= 2:
            err = np.abs(arr - target) * mask
            axes[1, j].imshow(err, cmap='hot', vmin=0, vmax=0.3)
            axes[1, j].set_title(f'{name} FOV error')
        elif j == 0:
            axes[1, j].imshow(mask, cmap='gray')
            axes[1, j].set_title('FOV mask')
        else:
            axes[1, j].imshow(np.where(mask, target, 0), cmap='gray', vmin=-1, vmax=2)
            axes[1, j].set_title('FDK inside FOV')
        axes[1, j].axis('off')
    fig.suptitle(f'Case{case_id} z={z_idx}')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--output-dir', default='./test_results/residual_physical_flip')
    parser.add_argument('--checkpoint-type', choices=['best', 'last'], default='best')
    parser.add_argument('--models', nargs='+', choices=['original', 'physical_flip'],
                        default=['original', 'physical_flip'])
    parser.add_argument('--original-checkpoint', default=None)
    parser.add_argument('--physical-flip-checkpoint', default=None)
    parser.add_argument('--scan-time', type=float, default=30.0)
    parser.add_argument('--breathing-period', type=float, default=4.0)
    parser.add_argument('--start-phase', type=float, default=0.0)
    parser.add_argument('--vis-interval', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    if args.vis_interval > 0:
        vis_dir.mkdir(exist_ok=True)

    test_set = DIRLab10PhaseDataset(
        root=args.data_root, prompt_file='prompt_test.json',
        scan_time=args.scan_time, breathing_period=args.breathing_period,
        start_phase=args.start_phase, sod=720.0, du=1.2, input_mode='10phase'
    )
    if not len(test_set):
        raise RuntimeError('No test samples. Check scan/breathing/start-phase filters.')
    seen = set()
    for r in test_set.records:
        if int(r['case_id']) != 10 or not np.isclose(float(r['SDD']), 1200):
            raise ValueError('Expected Case10, SDD=1200')
        if r.get('variant', 'original') != 'original' or r.get('augmentation'):
            raise ValueError('Use unaugmented test data')
        key = (int(r['case_id']), int(r['z_idx']))
        if key in seen:
            raise ValueError(f'Duplicate test slice {key}')
        seen.add(key)
        for key in ('source', 'target'):
            path = Path(args.data_root) / r[key]
            if not path.is_file():
                raise FileNotFoundError(path)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    mask = fov_mask_case10()
    print(f'Test samples: {len(test_set)}; FOV coverage: {mask.mean() * 100:.2f}%')

    models = {}
    for mode in args.models:
        override = getattr(args, f'{mode}_checkpoint')
        path = override or DEFAULT_MODELS[mode]
        if override is None:
            path = str(Path(path).with_name(f'{args.checkpoint_type}.pth'))
        models[mode] = load_checkpoint(path, device, 'raw')

    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    for name in args.models:
        (pred_dir / name).mkdir(exist_ok=True)
    results = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            source = batch['source'].to(device)
            target_tensor = batch['target']
            assert source.ndim == 4 and source.shape[1] == 10
            assert target_tensor.ndim == 4 and target_tensor.shape[1] == 1
            target = target_tensor[0, 0].float().numpy()
            case_id = int(batch['case_id'][0])
            z_idx = int(batch['z_idx'][0])
            if case_id != 10:
                raise RuntimeError(f'Case10 FOV geometry only; encountered Case{case_id}')
            if target.shape != mask.shape:
                raise RuntimeError(f'Unexpected image size: {target.shape}')
            predictions = {}
            phase_mean = source[0].mean(0).float().cpu().numpy()
            results.append(dict(model='phase_mean', case_id=case_id, z_idx=z_idx,
                                **fov_metrics(phase_mean, target, mask)))
            for name, spec in models.items():
                pred = predict(spec, source)[0, 0].float().cpu().numpy()
                predictions[name] = pred
                # Fixed [-1,2] display window; metrics use original floating-point values.
                png = np.round(np.clip((pred+1)/3, 0, 1)*255).astype(np.uint8)
                Image.fromarray(png).save(pred_dir / name / f'case{case_id}_z{z_idx:04d}.png')
                results.append({
                    'model': name, 'case_id': case_id, 'z_idx': z_idx,
                    **fov_metrics(pred, target, mask)
                })
            if args.vis_interval > 0 and index % args.vis_interval == 0:
                visualize(vis_dir / f'case{case_id}_z{z_idx:04d}.png',
                          source[0].float().cpu().numpy(), target,
                          predictions, mask, case_id, z_idx)
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(loader):
                print(f'Evaluated {index + 1}/{len(loader)}', flush=True)

    metrics = ['fov_mae', 'fov_psnr', 'fov_ssim']
    save_csv(output_dir / 'per_slice_metrics.csv', results,
             ['model', 'case_id', 'z_idx'] + metrics)
    summary = []
    for mode in ['phase_mean'] + args.models:
        rows = [r for r in results if r['model'] == mode]
        item = {'model': mode, 'num_slices': len(rows)}
        for metric in metrics:
            item[metric] = float(np.mean([r[metric] for r in rows]))
        summary.append(item)
        print(f'{mode}: ' + ' | '.join(f'{k}={item[k]:.6f}' for k in metrics))
    save_csv(output_dir / 'summary.csv', summary, ['model', 'num_slices'] + metrics)
    print(f'Saved FOV-only results in {output_dir.resolve()}')


if __name__ == '__main__':
    main()

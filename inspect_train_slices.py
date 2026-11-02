#!/usr/bin/env python3
"""Evaluate E2E checkpoints on unaugmented data. Requires dataset_aug.py, model.py,
numpy, torch, Pillow, scikit-image. Metrics are per-slice means, normalized units.
"""
import argparse
import csv
import json
from pathlib import Path
from scipy.ndimage import binary_erosion

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from dataset_aug import DIRLab10PhaseDataset
from model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--prompt', default='prompt_test.json')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', type=Path, default=Path('test_results/e2e_aug'))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-range', type=float, default=2.0)
    p.add_argument('--ssim-erosion', type=int, default=3)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--baseline-checkpoint', default=None, help='Optional E2E checkpoint without augmentation')
    p.add_argument('--vis-interval', type=int, default=1)
    p.add_argument('--error-max', type=float, default=0.3)
    p.add_argument('--png-min', type=float, default=-1.0)
    p.add_argument('--png-max', type=float, default=2.0)
    a = p.parse_args()
    if a.data_range <= 0 or a.ssim_erosion < 0 or a.png_max <= a.png_min or a.vis_interval < 1 or a.error_max <= 0:
        raise ValueError('Invalid metric settings')
    dataset = DIRLab10PhaseDataset(
        root=a.root, prompt_file=a.prompt, scan_time=30.0,
        breathing_period=4.0, start_phase=0.0, sod=720.0, du=1.2,
        input_mode='10phase', return_fov_mask=True)
    # Check full conditions and prevent accidental evaluation on augmented data.
    for r in dataset.records:
        
        if r.get('augmentation') or r.get('aug_id', 0) != 0:
            raise ValueError('Test index contains augmented samples')
        if 'SDD' not in r or not np.isclose(float(r['SDD']), 1200):
            raise ValueError('Test requires SDD=1200 metadata')
        for key in ('source', 'target'):
            path = Path(a.root) / r[key]
            if not path.is_file():
                raise FileNotFoundError(path)
    keys = [(int(r['case_id']), int(r['z_idx'])) for r in dataset.records]
    if len(keys) != len(set(keys)):
        raise ValueError('Multiple targets for the same case/z; check index conditions')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('prediction_mode') == 'phase_mean_residual':
        raise ValueError('This script supports E2E only, not residual checkpoints')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    baseline_model = None
    if a.baseline_checkpoint:
        baseline_model = build_model().to(device)
        baseline_ckpt = torch.load(a.baseline_checkpoint, map_location='cpu', weights_only=True)
        if baseline_ckpt.get('prediction_mode') == 'phase_mean_residual':
            raise ValueError('Baseline must also be E2E')
        baseline_model.load_state_dict(baseline_ckpt['model_state_dict'], strict=True)
        baseline_model.eval()
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=device.type == 'cuda')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.save_predictions:
        (a.out / 'predictions').mkdir(exist_ok=True)
    (a.out / 'visualizations').mkdir(exist_ok=True)
    baseline_rows = []
    rows = []
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch['source'].to(device)).float().cpu()
            reference = (baseline_model(batch['source'].to(device)).float().cpu()
                         if baseline_model is not None else batch['source'].mean(1, keepdim=True))
            target = batch['target'].float()
            masks = batch['fov_mask'].float()
            if prediction.shape != target.shape or masks.shape != target.shape:
                raise ValueError('Prediction/target/mask shape mismatch')
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError('Nonfinite predictions or targets')
            radius = a.ssim_erosion
            # Square 7x7 erosion for radius=3, including outside-image pixels as invalid.
            if radius:
                kernel = 2 * radius + 1
                padded = F.pad(masks, (radius,) * 4, value=0)
                eroded = F.avg_pool2d(padded, kernel, stride=1) >= (1 - 1e-6)
            else:
                eroded = masks > .5
            for i in range(len(prediction)):
                pred, gt = prediction[i, 0].numpy(), target[i, 0].numpy()
                mask = masks[i, 0].numpy() > .5
                ssim_mask = binary_erosion(
                    mask,
                    iterations=a.ssim_erosion,
                ) if a.ssim_erosion > 0 else mask.copy()
                if not mask.any() or not ssim_mask.any():
                    raise ValueError('Empty evaluation ROI')
                error = pred.astype(np.float64) - gt.astype(np.float64)
                mae = float(np.abs(error[mask]).mean())
                mse = float(np.square(error[mask]).mean())
                psnr = float(10 * np.log10(a.data_range ** 2 / mse)) if mse else float('inf')
                _, ssim_map = structural_similarity(
                    gt, pred, data_range=a.data_range, full=True, win_size=7,
                    gaussian_weights=False, use_sample_covariance=True)
                ssim = float(ssim_map[ssim_mask].mean())
                cid, z = int(batch['case_id'][i]), int(batch['z_idx'][i])
                rows.append(dict(case_id=cid, z_idx=z, fov_mae=mae,
                                 fov_mse=mse, fov_psnr=psnr, fov_ssim=ssim))
                ref = reference[i, 0].numpy()
                if not np.isfinite(ref).all():
                    raise ValueError('Nonfinite reference')
                ref_error = ref.astype(np.float64) - gt.astype(np.float64)
                ref_mse = float(np.square(ref_error[mask]).mean())
                _, ref_ssim_map = structural_similarity(
                    gt, ref, data_range=a.data_range, full=True, win_size=7,
                    gaussian_weights=False, use_sample_covariance=True)
                baseline_rows.append(dict(case_id=cid, z_idx=z,
                    fov_mae=float(np.abs(ref_error[mask]).mean()), fov_mse=ref_mse,
                    fov_psnr=float(10*np.log10(a.data_range**2/ref_mse)) if ref_mse else float('inf'),
                    fov_ssim=float(ref_ssim_map[ssim_mask].mean())))
                if (len(rows) - 1) % a.vis_interval == 0:
                    ref_title = 'Without augmentation' if baseline_model is not None else 'CT phase mean'
                    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
                    for ax, array, title in zip(axes[0], [gt, ref, pred],
                                               ['FDK target', ref_title, 'Augmented-training E2E']):
                        ax.imshow(array, cmap='gray', vmin=a.png_min, vmax=a.png_max)
                        ax.set_title(title)
                    axes[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
                    axes[1, 0].set_title('FOV mask')
                    cmap = plt.get_cmap('hot').copy()
                    cmap.set_bad('black')
                    for ax, err, title in zip(axes[1, 1:], [np.abs(ref_error), np.abs(error)],
                                              [ref_title + ' error', 'Augmented E2E error']):
                        heat = ax.imshow(np.ma.masked_where(~mask, err), cmap=cmap,
                                         vmin=0, vmax=a.error_max)
                        ax.set_title(title)
                    fig.colorbar(heat, ax=list(axes[1, 1:]), label='Absolute error (normalized)', shrink=.8)
                    for ax in axes.flat:
                        ax.axis('off')
                    fig.suptitle(f'Case {cid}, z={z} | E2E FOV MAE={mae:.6f}, PSNR={psnr:.3f}, SSIM={ssim:.4f}')
                    fig.savefig(a.out / 'visualizations' / f'case{cid:02d}_z{z:03d}.png', dpi=180)
                    plt.close(fig)
                if a.save_predictions:
                    display = np.clip((pred - a.png_min) / (a.png_max - a.png_min), 0, 1)
                    png = np.round(display * 255).astype(np.uint8)
                    Image.fromarray(png).save(
                        a.out / 'predictions' / f'case{cid:02d}_z{z:03d}.png')
    def summarize(items):
        return dict(samples=len(items), **{
            name: float(np.mean([r[name] for r in items]))
            for name in ('fov_mae', 'fov_mse', 'fov_psnr', 'fov_ssim')})
    with (a.out / 'per_slice.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (a.out / 'reference_per_slice.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    summary = dict(reference_type='E2E checkpoint' if baseline_model is not None else 'CT phase mean',
                   reference_checkpoint=a.baseline_checkpoint, reference_overall=summarize(baseline_rows),
                   error_display_max=a.error_max, checkpoint=str(Path(a.checkpoint).resolve()),
                   checkpoint_epoch=checkpoint.get('epoch'), root=a.root, prompt=a.prompt,
                   png_window=[a.png_min, a.png_max],
                   data_range=a.data_range, intensity_units='normalized, not HU',
                   aggregation='arithmetic mean of per-slice metrics',
                   ssim_erosion_pixels=a.ssim_erosion,
                   ssim_erosion_shape='scipy default cross, repeated erosion',
                   mask_policy='Dataset saved mask or original bilinear-threshold physical FOV',
                   overall=summarize(rows),
                   per_case={str(c): summarize([r for r in rows if r['case_id'] == c])
                             for c in sorted(set(r['case_id'] for r in rows))})
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Results:', a.out.resolve())


if __name__ == '__main__':
    main()

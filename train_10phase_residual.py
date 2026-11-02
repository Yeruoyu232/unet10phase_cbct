import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DIRLab10PhaseDataset
from model import build_model

DATA_ROOT = '/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez'
SAVE_DIR = './checkpoints/unet_10phase_residual'
SCAN_TIME, BREATHING_PERIOD, START_PHASE, SOD, DU = 30.0, 4.0, 0.0, 720.0, 1.2
BATCH_SIZE, NUM_WORKERS = 8, 4
LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-4
EPOCHS, SEED = 100, 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print('Device:', DEVICE)
    if DEVICE.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    def make_dataset(prompt_file):
        return DIRLab10PhaseDataset(
            root=DATA_ROOT, prompt_file=prompt_file,
            scan_time=SCAN_TIME, breathing_period=BREATHING_PERIOD,
            start_phase=START_PHASE, sod=SOD, du=DU, input_mode='10phase',
        )

    train_set = make_dataset('prompt_train.json')
    val_set = make_dataset('prompt_val.json')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')
    print('Train samples:', len(train_set), 'Val samples:', len(val_set))
    model = build_model().to(DEVICE)
    print('Parameters (M):', sum(p.numel() for p in model.parameters()) / 1e6)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    use_amp = DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_val_loss = float('inf')
    history = []
    condition = dict(scan_time=SCAN_TIME, breathing_period=BREATHING_PERIOD,
                     start_phase=START_PHASE, SOD=SOD, du=DU)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            source = batch['source'].to(DEVICE, non_blocking=True)
            target = batch['target'].to(DEVICE, non_blocking=True)
            assert source.ndim == 4 and source.shape[1] == 10, f'Unexpected source shape: {source.shape}'
            if not torch.isfinite(source).all() or not torch.isfinite(target).all():
                raise RuntimeError('Non-finite source or target')
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                phase_mean = source.mean(dim=1, keepdim=True)
                pred_residual = model(source)
                prediction = phase_mean + pred_residual
                loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss epoch={epoch} step={step}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = source.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(f'[Train] Epoch {epoch:03d}/{EPOCHS} | Step {step:04d}/{len(train_loader):04d} | L1 {loss.item():.6f}')
        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                source = batch['source'].to(DEVICE, non_blocking=True)
                target = batch['target'].to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                    phase_mean = source.mean(dim=1, keepdim=True)
                    pred_residual = model(source)
                    prediction = phase_mean + pred_residual
                    loss = criterion(prediction, target)
                bs = source.shape[0]
                val_loss_sum += loss.item() * bs
                val_count += bs
        val_loss = val_loss_sum / val_count
        history.append((epoch, train_loss, val_loss))
        print(f'Epoch {epoch:03d}/{EPOCHS} | Train L1 = {train_loss:.6f} | Val L1 = {val_loss:.6f}')

        common = dict(epoch=epoch, model_state_dict=model.state_dict(),
                      train_loss=train_loss, val_loss=val_loss,
                      scan_condition=condition, input_mode='10phase',
                      prediction_mode='phase_mean_residual')
        torch.save({**common, 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(SAVE_DIR, 'last.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(common, os.path.join(SAVE_DIR, 'best.pth'))
            print(f'Best model saved | Val L1 = {best_val_loss:.6f}')
        np.savetxt(os.path.join(SAVE_DIR, 'loss_history.csv'),
                   np.asarray(history), delimiter=',',
                   header='epoch,train_l1,val_l1', comments='')
        print('-' * 80)
    print('Training finished. Best validation L1:', best_val_loss)
    print('Best checkpoint:', os.path.join(SAVE_DIR, 'best.pth'))

if __name__ == '__main__':
    main()

import os
import argparse

import random



import numpy as np

import torch

import torch.nn as nn



from PIL import Image

from torch.utils.data import DataLoader



from dataset import DIRLab10PhaseDataset

from model import build_model



# 三选一："raw", "center", "zscore"

RESIDUAL_MODE = "zscore"



RES_MEAN = -0.063434

RES_STD = 0.121587



SAVE_DIR = f"./checkpoints/unet_10phase_residual_{RESIDUAL_MODE}_fov"



# ============================================================

# Configuration

# ============================================================



DATA_ROOT = (

    "/home/yry/ENV/ControlNet-main_10phase/training/cbct_motion_10phase_samez_minus_1024_div_1024"

)



#SAVE_DIR = './checkpoints/unet_10phase_residual_fov'



SCAN_TIME = 30.0

BREATHING_PERIOD = 4.0

START_PHASE = 0.0



SOD = 720.0

SDD = 1200.0



NU = 512

DU = 1.2



BATCH_SIZE = 8

NUM_WORKERS = 4



LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4



EPOCHS = 100

SEED = 42



DEVICE = torch.device(

    'cuda' if torch.cuda.is_available() else 'cpu'

)





# ============================================================

# Case Geometry

# ============================================================



CASE_GEOMETRY = {

    1: (256, 256, 0.97, 0.97),

    2: (256, 256, 1.16, 1.16),

    3: (256, 256, 1.15, 1.15),

    4: (256, 256, 1.13, 1.13),

    5: (256, 256, 1.10, 1.10),

    6: (512, 512, 0.97, 0.97),

    7: (512, 512, 0.97, 0.97),

    8: (512, 512, 0.97, 0.97),

    9: (512, 512, 0.97, 0.97),

    10: (512, 512, 0.97, 0.97),

}





def predict_fdk(model, source):

    phase_mean = source.mean(dim=1, keepdim=True)



    # 模型输出对应模式下的残差

    pred = model(source)



    if RESIDUAL_MODE == "raw":

        pred_residual = pred



    elif RESIDUAL_MODE == "center":

        pred_residual = pred + RES_MEAN



    elif RESIDUAL_MODE == "zscore":

        pred_residual = pred * RES_STD + RES_MEAN



    else:

        raise ValueError(f"Unknown mode: {RESIDUAL_MODE}")



    prediction = phase_mean + pred_residual

    return prediction



# ============================================================

# Build FOV Mask

# ============================================================



def build_fov_mask(case_id, output_size=256):



    nx, ny, dx, dy = CASE_GEOMETRY[case_id]



    radius_mm = (

        NU * DU * SOD / (2.0 * SDD)

    )



    x = (

        np.arange(nx) - (nx - 1) / 2.0

    ) * dx



    y = (

        np.arange(ny) - (ny - 1) / 2.0

    ) * dy



    xx, yy = np.meshgrid(x, y)



    mask = (

        xx ** 2 + yy ** 2 <= radius_mm ** 2

    ).astype(np.float32)



    mask = np.asarray(

        Image.fromarray(

            mask,

            mode='F'

        ).resize(

            (output_size, output_size),

            resample=Image.Resampling.BILINEAR

        ),

        dtype=np.float32

    )



    mask = (mask > 0.5).astype(np.float32)



    return torch.from_numpy(mask)[None, :, :]





# ============================================================

# Precompute Masks

# ============================================================



def create_mask_cache():



    cache = {}



    for case_id in CASE_GEOMETRY:



        mask = build_fov_mask(case_id)



        cache[case_id] = mask.to(DEVICE)



        print(

            f'Case{case_id}: '

            f'FOV coverage = '

            f'{mask.mean().item() * 100:.2f}%'

        )



    return cache





# ============================================================

# Get Batch Mask

# ============================================================



def get_batch_mask(case_ids, mask_cache):



    masks = []



    for case_id in case_ids:



        cid = int(case_id)



        if cid not in mask_cache:

            raise ValueError(

                f'Unknown case_id: {cid}'

            )



        masks.append(mask_cache[cid])



    return torch.stack(masks, dim=0)





# ============================================================

# FOV L1 Loss

# ============================================================



def masked_l1_loss(prediction, target, mask):



    # Compute in float32 for stable reduction under AMP.

    error = torch.abs(

        prediction.float() - target.float()

    )



    mask = mask.float()



    numerator = (

        error * mask

    ).sum(dim=(1, 2, 3))



    denominator = (

        mask.sum(dim=(1, 2, 3))

        .clamp_min(1.0)

    )



    loss_per_sample = numerator / denominator



    return loss_per_sample.mean()





# ============================================================

# Main

# ============================================================



def main():
    global RESIDUAL_MODE, SAVE_DIR, BREATHING_PERIOD

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['raw', 'center', 'zscore'],
                        default=RESIDUAL_MODE)
    parser.add_argument('--breathing-period', type=float,
                        default=BREATHING_PERIOD)
    args = parser.parse_args()

    RESIDUAL_MODE = args.mode
    BREATHING_PERIOD = args.breathing_period
    SAVE_DIR = f'./checkpoints/unet_10phase_residual_{RESIDUAL_MODE}_fov'
    print(f'Residual mode: {RESIDUAL_MODE}')
    print(f'Breathing period: {BREATHING_PERIOD}')




    os.makedirs(

        SAVE_DIR,

        exist_ok=True

    )



    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)



    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)



    print('Device:', DEVICE)



    if DEVICE.type == 'cuda':

        print(

            'GPU:',

            torch.cuda.get_device_name(0)

        )



    # --------------------------------------------------------

    # Dataset

    # --------------------------------------------------------



    def make_dataset(prompt_file):



        return DIRLab10PhaseDataset(

            root=DATA_ROOT,

            prompt_file=prompt_file,

            scan_time=SCAN_TIME,

            breathing_period=BREATHING_PERIOD,

            start_phase=START_PHASE,

            sod=SOD,

            du=DU,

            input_mode='10phase',

        )



    train_set = make_dataset('prompt_train.json')

    val_set = make_dataset('prompt_val.json')



    train_loader = DataLoader(

        train_set,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=DEVICE.type == 'cuda',

    )



    val_loader = DataLoader(

        val_set,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=DEVICE.type == 'cuda',

    )



    print(

        'Train samples:',

        len(train_set)

    )



    print(

        'Val samples:',

        len(val_set)

    )



    # --------------------------------------------------------

    # Masks

    # --------------------------------------------------------



    mask_cache = create_mask_cache()



    # --------------------------------------------------------

    # Model

    # --------------------------------------------------------



    model = build_model().to(DEVICE)



    print(

        'Parameters (M):',

        sum(

            p.numel()

            for p in model.parameters()

        ) / 1e6

    )



    # Full L1 is only used for logging.

    full_l1_criterion = nn.L1Loss()



    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )



    use_amp = DEVICE.type == 'cuda'



    scaler = torch.amp.GradScaler(

        'cuda',

        enabled=use_amp

    )



    best_val_loss = float('inf')



    history = []



    condition = dict(

        scan_time=SCAN_TIME,

        breathing_period=BREATHING_PERIOD,

        start_phase=START_PHASE,

        SOD=SOD,

        du=DU

    )



    # ========================================================

    # Training

    # ========================================================



    for epoch in range(1, EPOCHS + 1):



        model.train()



        train_fov_sum = 0.0

        train_full_sum = 0.0

        train_count = 0



        for step, batch in enumerate(

            train_loader,

            start=1

        ):



            source = batch['source'].to(

                DEVICE,

                non_blocking=True

            )



            target = batch['target'].to(

                DEVICE,

                non_blocking=True

            )



            case_ids = batch['case_id']



            mask = get_batch_mask(

                case_ids,

                mask_cache

            )



            assert (

                source.ndim == 4

                and source.shape[1] == 10

            ), f'Unexpected source shape: {source.shape}'



            if (

                not torch.isfinite(source).all()

                or not torch.isfinite(target).all()

            ):

                raise RuntimeError(

                    'Non-finite source or target'

                )



            optimizer.zero_grad(

                set_to_none=True

            )



            with torch.amp.autocast(

                device_type=DEVICE.type,

                enabled=use_amp

            ):



                # --------------------------------------------

                # Residual Prediction

                # --------------------------------------------



                # phase_mean = source.mean(

                #     dim=1,

                #     keepdim=True

                # )



                # pred_residual = model(source)



                # prediction = (

                #     phase_mean + pred_residual

                # )



                # # --------------------------------------------

                # # FOV L1: Actual Training Loss

                # # --------------------------------------------



                # loss = masked_l1_loss(

                #     prediction,

                #     target,

                #     mask

                # )

                prediction = predict_fdk(model, source)



                loss = masked_l1_loss(prediction, target, mask)



                # Full L1: Logging Only

                full_loss = full_l1_criterion(

                    prediction,

                    target

                )



            if not torch.isfinite(loss):

                raise RuntimeError(

                    f'Non-finite loss '

                    f'epoch={epoch} '

                    f'step={step}'

                )



            scaler.scale(loss).backward()



            scaler.step(optimizer)



            scaler.update()



            bs = source.shape[0]



            train_fov_sum += (

                loss.item() * bs

            )



            train_full_sum += (

                full_loss.item() * bs

            )



            train_count += bs



            if (

                step == 1

                or step % 20 == 0

                or step == len(train_loader)

            ):



                print(

                    f'[Train] '

                    f'Epoch {epoch:03d}/{EPOCHS} | '

                    f'Step {step:04d}/{len(train_loader):04d} | '

                    f'FOV L1 {loss.item():.6f} | '

                    f'Full L1 {full_loss.item():.6f}'

                )



        train_fov_loss = (

            train_fov_sum / train_count

        )



        train_full_loss = (

            train_full_sum / train_count

        )



        # ====================================================

        # Validation

        # ====================================================



        model.eval()



        val_fov_sum = 0.0

        val_full_sum = 0.0

        val_count = 0



        with torch.inference_mode():



            for batch in val_loader:



                source = batch['source'].to(

                    DEVICE,

                    non_blocking=True

                )



                target = batch['target'].to(

                    DEVICE,

                    non_blocking=True

                )



                case_ids = batch['case_id']



                mask = get_batch_mask(

                    case_ids,

                    mask_cache

                )



                with torch.amp.autocast(

                    device_type=DEVICE.type,

                    enabled=use_amp

                ):



                    # ----------------------------------------

                    # Residual Prediction

                    # ----------------------------------------



                    prediction = predict_fdk(model, source)



                    fov_loss = masked_l1_loss(prediction, target, mask)



                    full_loss = full_l1_criterion(

                        prediction,

                        target

                    )



                bs = source.shape[0]



                val_fov_sum += (

                    fov_loss.item() * bs

                )



                val_full_sum += (

                    full_loss.item() * bs

                )



                val_count += bs



        val_fov_loss = (

            val_fov_sum / val_count

        )



        val_full_loss = (

            val_full_sum / val_count

        )



        # ====================================================

        # Logging

        # ====================================================



        history.append((

            epoch,

            train_fov_loss,

            val_fov_loss,

            train_full_loss,

            val_full_loss,

        ))



        print(

            f'\nEpoch {epoch:03d}/{EPOCHS} | '

            f'Train FOV L1 = {train_fov_loss:.6f} | '

            f'Val FOV L1 = {val_fov_loss:.6f} | '

            f'Train Full L1 = {train_full_loss:.6f} | '

            f'Val Full L1 = {val_full_loss:.6f}'

        )



        # ====================================================

        # Checkpoints

        # ====================================================



        common = dict(

            epoch=epoch,

            model_state_dict=model.state_dict(),



            train_loss=train_fov_loss,

            val_loss=val_fov_loss,



            train_fov_l1=train_fov_loss,

            val_fov_l1=val_fov_loss,



            train_full_l1=train_full_loss,

            val_full_l1=val_full_loss,



            scan_condition=condition,

            input_mode='10phase',



            prediction_mode='phase_mean_residual',

            loss_type='fov_l1',
           residual_mode=RESIDUAL_MODE,
           res_mean=RES_MEAN,
           res_std=RES_STD,

        )



        # Last checkpoint

        torch.save(

            {

                **common,

                'optimizer_state_dict':

                    optimizer.state_dict(),

            },

            os.path.join(

                SAVE_DIR,

                'last.pth'

            )

        )



        # Best checkpoint:

        # selected by validation FOV L1

        if val_fov_loss < best_val_loss:



            best_val_loss = val_fov_loss



            torch.save(

                common,

                os.path.join(

                    SAVE_DIR,

                    'best.pth'

                )

            )



            print(

                f'Best model saved | '

                f'Val FOV L1 = '

                f'{best_val_loss:.6f}'

            )



        # Save history

        np.savetxt(

            os.path.join(

                SAVE_DIR,

                'loss_history.csv'

            ),

            np.asarray(history),

            delimiter=',',

            header=(

                'epoch,'

                'train_fov_l1,'

                'val_fov_l1,'

                'train_full_l1,'

                'val_full_l1'

            ),

            comments=''

        )



        print('-' * 80)



    # ========================================================

    # Finished

    # ========================================================



    print(

        'Training finished.'

    )



    print(

        'Best validation FOV L1:',

        best_val_loss

    )



    print(

        'Best checkpoint:',

        os.path.join(

            SAVE_DIR,

            'best.pth'

        )

    )





if __name__ == '__main__':

    main()

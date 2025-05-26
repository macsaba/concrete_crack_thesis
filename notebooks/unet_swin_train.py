# general imports:
import os
import importlib
from torch import nn, cuda
from torch import optim
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# my imports (reload all modules):

from data.dataset import load_data_deep_crack
from models.unet_resnet34 import UNetResNet34
from models.unet_swin import UNetSwin
from models.unet import UNet
from training.metrics import DiceLoss
from training.train import train
from utils import save_model_files
from utils import log_training_result


import data.dataset
import models.unet_resnet34
import models.unet_swin
import models.unet
import training.metrics
import training.train
import utils
importlib.reload(data.dataset)
importlib.reload(models.unet_resnet34)
importlib.reload(models.unet_swin)
importlib.reload(models.unet)
importlib.reload(training.metrics)
importlib.reload(training.train)
importlib.reload(utils)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print(f'Number of visible CUDA devices: {cuda.device_count()}')
cuda.is_available()
# NOTE: when referring to a CUDA device, torch will only know about visible devices
#       so while currently we will be using gpu05
#       this will be device `cuda:0`

data_source ='../../DeepCrack/dataset/DeepCrack'
train_image_dir = data_source + '/train_img'
train_mask_dir = data_source + '/train_lab'
test_image_dir = data_source + '/test_img'
test_mask_dir = data_source + '/test_lab'

train_dl, val_dl, train_dataset, val_dataset = load_data_deep_crack(train_image_dir, train_mask_dir, [0.8, 0.2])

import gc
# Define experiments: (layers_to_unfreeze, learning_rate)
experiments = [
    (['layers.3'], 1e-4),                       # B: Unfreeze layer4
    (['layers.2', 'layers.3'], 1e-5),           # C: Unfreeze layers 3,4
    (['layers.1', 'layers.2', 'layers.3'], 1e-5),# D: Unfreeze layers 2,3,4
    (None, 1e-5),                                # E: Unfreeze all
]

for i, (layers, lr) in enumerate(experiments, start=1):
    folder = f"swin_6_{i}/"
    
    print(f"=== Training experiment {i} | Unfreezing: {layers if layers is not None else 'ALL'} | LR: {lr} ===")

    model = UNetSwin(
        img_channels=3,
        mask_channels=1,
        base_channel_size=64
    )

    loss = DiceLoss()
    model.freeze_encoder_layers()
    model.unfreeze_encoder_layers(layers)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # Load logs from previous training (optional)
    dice_idcs = list(np.load('../saved_models/swin_2/dice_idcs.npy'))
    epoch_dice_idcs = list(np.load('../saved_models/swin_2/epoch_dice_idcs.npy'))
    val_dice_idcs = list(np.load('../saved_models/swin_2/val_dice_idcs.npy'))
    train_loss = list(np.load('../saved_models/swin_2/train_loss.npy'))
    val_loss = list(np.load('../saved_models/swin_2/val_loss.npy'))
    epoch_durations = list(np.load('../saved_models/swin_2/epoch_durations.npy'))
    best_model_wts = {}

    # Load pretrained weights
    model.load_state_dict(torch.load('../saved_models/swin_2/model_state_epoch_100.pth', weights_only=True))

    # Log config
    log_training_result('../saved_models/training_log_2.csv', {
        "timestamp": pd.Timestamp.now(),
        "weights_file": folder,
        "epochs": 100,
        "learning_rate": lr,
        "batch_size": 4,
        "accum_scale": 4,
        "comment": "Unfreeze layer",
        "augmentation": "rotate+randomCrop",
        "unfrezed layers": layers if layers is not None else "ALL"
    })

    # Train
    train(
        model, loss, optimizer,
        train_dl, val_dl,
        num_epochs=2,
        accum_scale=4,
        dice_idcs=dice_idcs,
        epoch_dice_idcs=epoch_dice_idcs,
        val_dice_idcs=val_dice_idcs,
        best_model_wts=best_model_wts,
        train_loss=train_loss,
        val_loss=val_loss,
        epoch_durations=epoch_durations,
        save_path='../saved_models/' + folder,
        n_epoch_save=5
    )
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()  # Python garbage collection
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
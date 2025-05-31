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

#train_dl, val_dl, train_dataset, val_dataset = load_data_deep_crack(train_image_dir, train_mask_dir, [0.8, 0.2], limit_im_nr=3)
train_dl, val_dl, train_dataset, val_dataset = load_data_deep_crack(train_image_dir, train_mask_dir, [0.8, 0.2])

import gc
# Define experiments: (layers_to_unfreeze, learning_rate)
experiments = [
    #(['encoder4'], 1e-4),                       # B: Unfreeze layer4
    #(['encoder3', 'encoder4'], 1e-5),           # C: Unfreeze layers 3,4
    #(['encoder2', 'encoder3', 'encoder4'], 1e-5),# D: Unfreeze layers 2,3,4
    (None, 1e-5),                                # E: Unfreeze all
]
nr_of_epochs = 50
nr_of_epochs_save =5
load_from_folders = ['resnet_1', 'resnet_1', 'resnet_1', 'resnet_1']
#load_from_folder = 'resnet_1'
for i, (layers, lr) in enumerate(experiments, start=1):
    #folder = f"resnet_unfreeze_again2_{i}/"
    folder = f"resnet_unfreeze_again_4/"
    load_from_folder = load_from_folders[i-1]
    print(f"=== Training experiment {i} | Unfreezing: {layers if layers is not None else 'ALL'} | LR: {lr} ===")

    model = UNetResNet34(   img_channels = 3,
                mask_channels = 1,
                base_channel_size = 64)  

    loss = DiceLoss()
    model.freeze_encoder_layers()
    model.unfreeze_encoder_layers(layers)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # Load logs from previous training (optional)
    dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/dice_idcs.npy'))
    epoch_dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/epoch_dice_idcs.npy'))
    val_dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/val_dice_idcs.npy'))
    train_loss = list(np.load('../saved_models/'+load_from_folder+'/train_loss.npy'))
    val_loss = list(np.load('../saved_models/'+load_from_folder+'/val_loss.npy'))
    epoch_durations = list(np.load('../saved_models/'+load_from_folder+'/epoch_durations.npy'))

    best_model_wts = torch.load('../saved_models/'+load_from_folder+'/best_model_wts.pth')
    min_index = val_loss.index(min(val_loss))

    dice_idcs = dice_idcs[:min_index+1]
    epoch_dice_idcs = epoch_dice_idcs[:min_index+1]
    val_dice_idcs = val_dice_idcs[:min_index+1]
    train_loss = train_loss[:min_index+1]
    val_loss = val_loss[:min_index+1]
    epoch_durations = epoch_durations[:min_index+1]
    # Load pretrained weights
    model.load_state_dict(torch.load('../saved_models/'+load_from_folder+'/best_model_wts.pth', weights_only=True))

    # Log config
    log_training_result('../saved_models/training_log_2.csv', {
        "timestamp": pd.Timestamp.now(),
        "weights_file": folder,
        "epochs": nr_of_epochs,
        "learning_rate": lr,
        "batch_size": 4,
        "accum_scale": 4,
        "comment": "Unfreeze layer",
        "augmentation": "rotate+randomCrop",
        "unfrezed layers": layers if layers is not None else "ALL"
    })
    model.save_trainable_layers_to_file('../saved_models/' + folder + 'trainable_layers.txt')
    # Train
    train(
        model, loss, optimizer,
        train_dl, val_dl,
        num_epochs=nr_of_epochs,
        accum_scale=4,
        dice_idcs=dice_idcs,
        epoch_dice_idcs=epoch_dice_idcs,
        val_dice_idcs=val_dice_idcs,
        best_model_wts=best_model_wts,
        train_loss=train_loss,
        val_loss=val_loss,
        epoch_durations=epoch_durations,
        save_path='../saved_models/' + folder,
        n_epoch_save=nr_of_epochs_save
    )
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()  # Python garbage collection
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


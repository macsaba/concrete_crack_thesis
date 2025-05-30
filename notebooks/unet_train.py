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
#train_dl, val_dl, train_dataset, val_dataset = load_data_deep_crack(train_image_dir, train_mask_dir, [0.8, 0.2], limit_im_nr=3)


lr = 0.01
nr_of_epochs = 100
nr_of_epochs_save = 10

model = UNet(   img_channels = 3,
                mask_channels = 1,
                base_channel_size = 32,
                depth=5)  
loss = DiceLoss()
optimizer = optim.Adam(params = model.parameters(), lr = lr)

dice_idcs = []
epoch_dice_idcs = []
val_dice_idcs = []
train_loss = []
val_loss = []
epoch_durations = []
best_model_wts = {}

log_training_result('../saved_models/training_log_2.csv', {
    "timestamp": pd.Timestamp.now(),
    "weights_file": "unet_4_lre-2/",
    "epochs": nr_of_epochs,
    "learning_rate": lr,
    "batch_size": 4,
    "accum_scale": 4,
    "depth": 5,
    "base_channel_size": 32,    
    "comment": "",
    "augmentation": "rotate+randomCrop"
})

train(model, loss, optimizer, train_dl, val_dl,
        num_epochs = nr_of_epochs,
        accum_scale = 4,
        dice_idcs = dice_idcs,
        epoch_dice_idcs = epoch_dice_idcs,
        val_dice_idcs = val_dice_idcs,
        best_model_wts = best_model_wts,
        train_loss=train_loss,
        val_loss=val_loss,
        epoch_durations=epoch_durations,
        save_path='../saved_models/unet_4_lr1e-2/',
        n_epoch_save=nr_of_epochs_save)

import gc
del model
del optimizer
torch.cuda.empty_cache()
gc.collect()  # Python garbage collection
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

folder = 'unet_4_lr2e-4/'
load_from_folder = 'unet_4'
lr = 2e-4
nr_of_epochs = 300
nr_of_epochs_save = 10
model = UNet(   img_channels = 3,
                mask_channels = 1,
                base_channel_size = 32,
                depth=5)  

loss = DiceLoss()

optimizer = optim.Adam(params=model.parameters(), lr=lr)

# Load logs from previous training (optional)
dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/dice_idcs.npy'))
epoch_dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/epoch_dice_idcs.npy'))
val_dice_idcs = list(np.load('../saved_models/'+load_from_folder+'/val_dice_idcs.npy'))
train_loss = list(np.load('../saved_models/'+load_from_folder+'/train_loss.npy'))
val_loss = list(np.load('../saved_models/'+load_from_folder+'/val_loss.npy'))
epoch_durations = list(np.load('../saved_models/'+load_from_folder+'/epoch_durations.npy'))
best_model_wts = {}

# Load pretrained weights
model.load_state_dict(torch.load('../saved_models/'+load_from_folder+'/model_state_epoch_100.pth', weights_only=True))

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
    "started at" : load_from_folder
})
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
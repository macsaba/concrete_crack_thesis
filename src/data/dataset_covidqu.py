from albumentations import augmentations, Compose
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
import os
import torch
from PIL import Image


def load_data_covidqu(data_src):
    imgs, masks = np.load(os.path.join(data_src, 'imgs.npy')), np.load(os.path.join(data_src, 'masks.npy'))
    
    print(f'Image array shape: {imgs.shape}')
    print(f'Mask array shape: {masks.shape}')
    # visualising first five images
    fig = plt.figure(figsize = (56, 20))
    for i in range(5):
        subplot = fig.add_subplot(2, 5, i + 1)
        subplot.set_frame_on(True)
        subplot.xaxis.set_ticks_position('none')
        subplot.yaxis.set_ticks_position('none')
        subplot.xaxis.set_ticklabels([])
        subplot.yaxis.set_ticklabels([])
        plt.imshow(imgs[i], cmap = 'gray')

        subplot = fig.add_subplot(2, 5, i + 6)
        subplot.set_frame_on(True)
        subplot.xaxis.set_ticks_position('none')
        subplot.yaxis.set_ticks_position('none')
        subplot.xaxis.set_ticklabels([])
        subplot.yaxis.set_ticklabels([])
        plt.imshow(masks[i], cmap = 'gray')

        # creating train-validation split

    # creating train-validation split
    sources = np.load(os.path.join(data_src, 'sources.npy')) # sources: whether the image comes from the train, validation or test set
    train_idcs = sources == 'Train'

    train_imgs, train_masks = imgs[train_idcs], masks[train_idcs]
    val_imgs, val_masks = imgs[~train_idcs], masks[~train_idcs]

    print(f'This gives us {len(train_imgs)} train and {len(val_imgs)} validation samples.')


    # keep only 10 images, for testing
    n_of_images = 10
    train_imgs = train_imgs[1:n_of_images, :,:]
    train_masks = train_masks[1:n_of_images, :,:]
    val_imgs = val_imgs[1:n_of_images, :,:]
    val_masks = val_masks[1:n_of_images, :,:]

    # augmentations
    train_trsfs = Compose([
                       #augmentations.transforms.PadIfNeeded(min_height = 288, min_width = 288, value = 1, border_mode = 0),
                       augmentations.crops.transforms.RandomCrop(256, 256),
                       augmentations.geometric.Rotate(limit = 15, p = 1, border_mode = 0),
                       augmentations.transforms.Normalize(mean = 0.5, std = 0.25),
                       ToTensorV2()
    ])
    val_trsfs = Compose([augmentations.transforms.Normalize(mean = 0.5, std = 0.25), ToTensorV2()])
    train_ds = CustomDataset(train_imgs, train_masks, train_trsfs)
    val_ds = CustomDataset(val_imgs, val_masks, val_trsfs)

    train_dl = data.DataLoader(train_ds, batch_size = 16, shuffle = True)
    val_dl = data.DataLoader(val_ds, batch_size = 16, shuffle = False)
    return train_dl, val_dl

class CustomDataset(data.Dataset):
    def __init__(self, imgs, masks, transforms = None):
        self.imgs = imgs
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, mask = self.imgs[idx], self.masks[idx]
        img = img.reshape((*img.shape, 1))
        bundle = dict(image = img, mask = mask)
        if self.transforms:
            bundle = self.transforms(**bundle)
            bundle['mask'] = torch.div(bundle['mask'].unsqueeze(0), 255)
        return bundle
    
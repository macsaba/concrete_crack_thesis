#from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    RandomCrop,
    Rotate,
    Normalize,
    Compose,
    ToTensorV2
)
import numpy as np
from torch.utils import data
from torch.utils.data import random_split
import os
import torch
from PIL import Image



def load_data_deep_crack(image_dir, mask_dir, train_test_lengths, limit_im_nr = None, shuffle_train = True):
    # Get image list from the given train folder
    images_list = sorted(os.listdir(image_dir))
    # do random train test cut on this list
    train_images_list, val_images_list = random_split(images_list, train_test_lengths, generator=torch.Generator().manual_seed(42))
    #train_images_list, val_images_list = random_split(images_list, train_test_lengths)

    print(f'This gives us {len(train_images_list)} train and {len(val_images_list)} validation samples.')

    if limit_im_nr:
        # keep only limit_im_nr images, for testing
        train_images_list = list(train_images_list)[0:limit_im_nr]
        val_images_list = list(val_images_list)[0:limit_im_nr]

    # augmentations
    train_trsfs = Compose([
                       #augmentations.transforms.PadIfNeeded(min_height = 288, min_width = 288, value = 1, border_mode = 0),
                       RandomCrop(384, 544),
                       Rotate(limit = 15, p = 1, border_mode = 0),
                       Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                       ToTensorV2()
    ])
    val_trsfs = Compose([Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), ToTensorV2()])

    train_dataset = DeepCrackDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_list = train_images_list,
    transform=train_trsfs
    )

    val_dataset = DeepCrackDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_list = val_images_list,
    transform=val_trsfs
    )


    train_dl = data.DataLoader(train_dataset, batch_size = 4, shuffle = shuffle_train)
    val_dl = data.DataLoader(val_dataset, batch_size = 4, shuffle = False)
    return train_dl, val_dl, train_dataset, val_dataset

class DeepCrackDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, image_list = None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        if image_list:
            # if we have an image list set it
            self.images = image_list
        else:
            # else set all image as image list
            self.images = sorted(os.listdir(image_dir))
        if transform:
            self.transform = transform
        else:
            self.transform = Compose(ToTensorV2())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale mask

        # Rotate if orientation is wrong (width > height)
        if img.shape[0] > img.shape[1]:  # shape = (H, W, C)
            img = np.rot90(img)
            mask = np.rot90(mask)

        bundle = dict(image = img, mask = mask)
        if self.transform:
            bundle = self.transform(**bundle)
            bundle['mask'] = torch.div(bundle['mask'].unsqueeze(0), 255)
        return bundle
    
    def get_image_name(self, idx):
        return self.images[idx]
    
    def get_item_by_name(self, name):
        try:
            idx = self.images.index(name)
        except ValueError:
            raise KeyError(f"Image '{name}' not found in dataset.")
        return self[idx]
    
    def get_image_and_mask(self, index):
        bundle = self[index]
        image = np.transpose(np.array(bundle['image']), (1, 2, 0))
        mask = np.transpose(np.array(bundle['mask']), (1, 2, 0))
        return image, mask
    
    def get_image_and_mask_by_name(self, name):
        bundle = self.get_item_by_name(name)
        image = np.transpose(np.array(bundle['image']), (1, 2, 0))
        mask = np.transpose(np.array(bundle['mask']), (1, 2, 0))
        return image, mask

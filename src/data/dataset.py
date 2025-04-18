from albumentations import augmentations, Compose
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from torch.utils import data
from torch.utils.data import random_split
import os
import torch
from PIL import Image



def load_data_deep_crack(image_dir, mask_dir, train_test_lengths, limit_im_nr = None):
    # Get image list from the given train folder
    images_list = sorted(os.listdir(image_dir))
    # do random train test cut on this list
    train_images_list, val_images_list = random_split(images_list, train_test_lengths)

    print(f'This gives us {len(train_images_list)} train and {len(val_images_list)} validation samples.')

    if limit_im_nr:
        # keep only limit_im_nr images, for testing
        train_images_list = list(train_images_list)[0:limit_im_nr]
        val_images_list = list(val_images_list)[0:limit_im_nr]

    # augmentations
    train_trsfs = Compose([
                       #augmentations.transforms.PadIfNeeded(min_height = 288, min_width = 288, value = 1, border_mode = 0),
                       augmentations.crops.transforms.RandomCrop(384, 544),
                       augmentations.geometric.Rotate(limit = 15, p = 1, border_mode = 0),
                       augmentations.transforms.Normalize(mean = 0.5, std = 0.25),
                       ToTensorV2()
    ])
    val_trsfs = Compose([augmentations.transforms.Normalize(mean = 0.5, std = 0.25), ToTensorV2()])

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


    train_dl = data.DataLoader(train_dataset, batch_size = 4, shuffle = True)
    val_dl = data.DataLoader(val_dataset, batch_size = 4, shuffle = False)
    return train_dl, val_dl, train_dataset, val_dataset

class DeepCrackDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))


        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale mask

        bundle = dict(image = img, mask = mask)
        if self.transform:
            bundle = self.transform(**bundle)
            bundle['mask'] = torch.div(bundle['mask'].unsqueeze(0), 255)
        return bundle
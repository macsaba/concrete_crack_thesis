import torch
from torch import nn
from torch.nn.functional import relu
from torchvision import models
import torch.nn as nn
import timm
import os

# pre-trained swin encoder
class SwinEncoder(nn.Module):
    def __init__(self, model_name="swin_base_patch4_window7_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Corresponding to resolutions: H/4, H/8, H/16, H/32
            img_size=(384, 544)
        )


    def forward(self, x):
        # Swin returns 4 feature maps with downsampling rates: H/4, H/8, H/16, H/32
        feats = self.backbone(x)
        # Transpose features from [B, H, W, C] to [B, C, H, W]
        feats = [f.permute(0, 3, 1, 2) for f in feats]
        return feats


# U-Net convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same')
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same')
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)

        return x


# U-Net up-convolution block
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor = 2)
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 1, padding = 'same')

        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1: coming through the upward path
        # x2: from downward pass, concatenate
        # upsample
        x1 = self.upsample(x1)
        x1 = self.up_conv(x1)
        x1 = relu(x1)
        # concatenate from the corresponding part in the down path
        x = torch.cat([x1, x2], axis = 1)
        # perform convolutional operations
        x = self.conv_block(x)
        return x

# Complete model architecture
class UNetSwin(nn.Module):
    def __init__(self,
                 img_channels = 1,
                 mask_channels = 1,
                 base_channel_size = 64
                 ):
        # TODO: implement automatic channel size calculations
        super().__init__()

        # ZEROth SKIP PART, to get the top level skip connection
        skip0_ch_size = 32
        self.stem1 = nn.Sequential(
            nn.Conv2d(img_channels, skip0_ch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip0_ch_size),
            nn.ReLU(inplace=True),
        )
    
        self.stem2 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Downsample to H/2 × W/2
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )

        # ENCODER PART
        self.encoder = SwinEncoder()
        
        # CENTER PART
        self.bottom_block = ConvBlock(1024, 1024)

        # DECODER PART
        up_in_channels = [1024, 512, 256, 128, 64]
        up_out_channels = [512, 256, 128, 64, 32]
        up_skip_channels = [512, 256, 128, 64, 32]

        self.up_blocks = nn.ModuleList(UpConvBlock(in_channels, skip_channel, out_channels) for in_channels, out_channels, skip_channel in zip (up_in_channels, up_out_channels, up_skip_channels))

        # OUTPUT PART
        self.segmentation_layer = nn.Conv2d(in_channels = skip0_ch_size, out_channels = mask_channels, kernel_size = 1)
        self.activation = torch.sigmoid if mask_channels == 1 else nn.Softmax(dim = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # INIT PART, to get the uppest skip connection
        x_stem1 = self.stem1(x)   # (B, 64, H, W) ← use this as the top skip connection
        x_stem2 = self.stem2(x) 
        # ENCODER PART
        lateral_skip_vals = self.encoder(x)  # lateral_skip_vals =  x0, x1, x2, x3, x4
        
        lateral_skip_vals.insert(0, x_stem2)
        lateral_skip_vals.insert(0, x_stem1)
        # CENTER PART
        #print('Shape: ', lateral_skip_vals.pop().shape())
        x = self.bottom_block(lateral_skip_vals.pop())  # list.pop() returns the last element by default
        # DECODER PART
        for up_block in self.up_blocks:
            skip_x = lateral_skip_vals.pop()
            x = up_block(x, skip_x) # list.pop() returns the last element by default
        
        # OUTPUT PART
        x = self.segmentation_layer(x)

        return self.activation(x)
    

    # Freeze encoder layers for training
    # If there are no specific layers, freeze all
    def freeze_encoder_layers(self, layers_to_freeze=None):
        if layers_to_freeze is None:
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Freeze only selected layers based on substring match
            for name, param in self.encoder.named_parameters():
                for freeze_name in layers_to_freeze:
                    if freeze_name in name:
                        param.requires_grad = False

    # unfreeze encoder layers for training
    # if there are no specific layers, unfreeze all
    def unfreeze_encoder_layers(self, layers_to_unfreeze=None):
        if layers_to_unfreeze is None:
            # Unfreeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only selected layers based on substring match
            for name, param in self.encoder.named_parameters():
                for unfreeze_name in layers_to_unfreeze:
                    if unfreeze_name in name:
                        param.requires_grad = True


    def print_trainable_layers(self):
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

    def save_trainable_layers_to_file(self, filepath="trainable_layers.txt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure the directory exists

        with open(filepath, "w") as f:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}: requires_grad={param.requires_grad}\n")

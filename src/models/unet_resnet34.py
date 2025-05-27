# This implementation is from the deep learning practice

import torch
from torch import nn
from torch.nn.functional import relu
from torchvision import models
import os

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        self.initial = nn.Sequential(
            resnet.conv1,  # H/2
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool           # H/4
        self.encoder1 = resnet.layer1           # H/4
        self.encoder2 = resnet.layer2           # H/8
        self.encoder3 = resnet.layer3           # H/16
        self.encoder4 = resnet.layer4           # H/32

    def forward(self, x):
        x0 = self.initial(x)        # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)       # (B, 64, H/4, W/4)
        x2 = self.encoder1(x1)      # (B, 64, H/4, W/4)
        x3 = self.encoder2(x2)      # (B, 128, H/8, W/8)
        x4 = self.encoder3(x3)      # (B, 256, H/16, W/16)
        x5 = self.encoder4(x4)      # (B, 512, H/32, W/32)
        return [x0, x2, x3, x4, x5]  # Return skip features


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
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # if in_channels != 2 * out_channels:
            # raise ValueError(f'In channel size should be twice out channel size, but got {in_channels} and {out_channels}.')
            # NOTE: technically it would be sufficient that the in channels of the convolutional block is equal to
            #       the out channels of the up convolution + the channel size of the lateral skip connection

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
    
class UNetResNet34(nn.Module):
    def __init__(self,
                 img_channels = 1,
                 mask_channels = 1,
                 base_channel_size = 64
                 ):
        # TODO: 
        # normalizálás a resnet-nek megfelelően
        # mi lesz a nem 2 hatvány képekkel
        # batch norm-ok kikapcsolása érdemes lehet a kis batch méret miatt
        
        depth = 5
        super().__init__()

        #channel_sizes = [base_channel_size * 2 ** i for i in range(depth)]

        # ZEROth SKIP PART, to get the uppest skip connection
        skip0_ch_size = 24
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, skip0_ch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip0_ch_size),
            nn.ReLU(inplace=True),
        )

        # ENCODER PART
        self.encoder = ResNetEncoder()
        
        # CENTER PART
        #self.bottom_block = ConvBlock(*channel_sizes[-2:])
        self.bottom_block = ConvBlock(512, 512)

        # DECODER PART
        #up_in_channels = channel_sizes[-1:0:-1]
        #up_out_channels = channel_sizes[-2::-1]
        #print('Up in channels: ',up_in_channels)
        #print('Up out channels: ',up_out_channels)
        up_in_channels = [512, 256, 128, 64, 48]
        up_out_channels = [256, 128, 64, 48, 24]
        up_skip_channels = [256, 128, 64, 64, 24]

        self.up_blocks = nn.ModuleList(UpConvBlock(in_channels, skip_channel, out_channels) for in_channels, out_channels, skip_channel in zip (up_in_channels, up_out_channels, up_skip_channels))

        # OUTPUT PART
        self.segmentation_layer = nn.Conv2d(in_channels = skip0_ch_size, out_channels = mask_channels, kernel_size = 1)
        self.activation = torch.sigmoid if mask_channels == 1 else nn.Softmax(dim = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # INIT PART, to get the uppest skip connection
        x_stem = self.stem(x)   # (B, 64, H, W) ← use this as the top skip connection
        # ENCODER PART
        lateral_skip_vals = self.encoder(x)  # lateral_skip_vals =  x0, x1, x2, x3, x4
        
        lateral_skip_vals.insert(0, x_stem)
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
        with open(filepath, "w") as f:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}: requires_grad={param.requires_grad}\n")

    def save_trainable_layers_to_file(self, filepath="trainable_layers.txt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure the directory exists

        with open(filepath, "w") as f:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}: requires_grad={param.requires_grad}\n")

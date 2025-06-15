# This implementation is from the deep learning practice

import torch
from torch import nn
from torch.nn.functional import relu

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
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != 2 * out_channels:
            raise ValueError(f'In channel size should be twice out channel size, but got {in_channels} and {out_channels}.')
            # NOTE: technically it would be sufficient that the in channels of the convolutional block is equal to
            #       the out channels of the up convolution + the channel size of the lateral skip connection

        self.upsample = nn.Upsample(scale_factor = 2)
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 1, padding = 'same')

        self.conv_block = ConvBlock(in_channels, out_channels)

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

# Complete U-Net model architecture
class UNet(nn.Module):
    def __init__(self,
                 img_channels = 1,
                 mask_channels = 1,
                 base_channel_size = 64,
                 depth = 5 # number of levels (including bottom block)
                 ):

        super().__init__()

        channel_sizes = [base_channel_size * 2 ** i for i in range(depth)]

        down_in_channels = (img_channels, *channel_sizes[:-2])
        down_out_channels = channel_sizes[:-1]

        self.down_blocks = nn.ModuleList(ConvBlock(in_channels, out_channels) for in_channels, out_channels in zip (down_in_channels, down_out_channels))
        self.bottom_block = ConvBlock(*channel_sizes[-2:])

        up_in_channels = channel_sizes[-1:0:-1]
        up_out_channels = channel_sizes[-2::-1]

        self.up_blocks = nn.ModuleList(UpConvBlock(in_channels, out_channels) for in_channels, out_channels in zip (up_in_channels, up_out_channels))

        self.segmentation_layer = nn.Conv2d(in_channels = base_channel_size, out_channels = mask_channels, kernel_size = 1)

        self.activation = torch.sigmoid if mask_channels == 1 else nn.Softmax(dim = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):

        lateral_skip_vals = []

        for down_block in self.down_blocks:
            x = down_block(x)
            lateral_skip_vals.append(x) # gather outputs of blocks to be passed through the lateral skip connections
            x = self.max_pool(x)

        x = self.bottom_block(x)

        for up_block in self.up_blocks:
            x = up_block(x, lateral_skip_vals.pop()) # list.pop() returns the last element by default

        x = self.segmentation_layer(x)

        return self.activation(x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act2 = nn.ELU()

    def forward(self, x):
        residual = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x + residual

class SimpleSpectrogramEncoder(nn.Module):
    """
    Encodes a spectrogram by downsampling frequency and time, increasing channels.
    """
    def __init__(self, input_size, dimension=128, n_residual_layers=2):
        super().__init__()
        self.input_size = input_size
        self.dimension = dimension
        
        # Initial convolution to increase channel dimension
        self.in_conv = nn.Conv2d(input_size, 8, kernel_size=5, padding=2)
        self.big_rf = nn.Conv2d(8, 8, kernel_size=(31, 5), padding=(15, 2))
        self.in_residual = nn.Sequential(*[ResidualBlock(8) for _ in range(n_residual_layers)])
        
        # Downsampling stages
        self.down_block1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=6, stride=(4, 4), padding=1), # F/2, T/2
            nn.GroupNorm(1, 16),
            nn.ELU()
        )
        self.down_block2 = nn.Sequential(
            nn.Conv2d(16, dimension, kernel_size=6, stride=(4, 4), padding=1), # F/4, T/4
            nn.GroupNorm(1, dimension),
            nn.ELU()
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(dimension) for _ in range(n_residual_layers)]
        )

    def forward(self, x):
        # x shape: (B, C_in, F, T)
        x = self.in_conv(x)
        y = self.big_rf(x)
        x = self.in_residual(x) + y
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.residuals(x)
        return x

class SimpleSpectrogramDecoder(nn.Module):
    """
    Decodes a latent representation back to a spectrogram.
    """
    def __init__(self, input_size, channels, n_residual_layers=2):
        super().__init__()
        self.input_size = input_size
        self.channels = channels

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(input_size) for _ in range(n_residual_layers)]
        )

        # Upsampling stages
        self.up_block1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 16, kernel_size=6, stride=(4, 4), padding=1), # F*2, T*2
            nn.GroupNorm(1, 16),
            nn.ELU()
        )
        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=(4, 4), padding=1, output_padding=(1,0)), # F*4, T*4
            nn.GroupNorm(1, 8),
            nn.ELU()
        )
        self.residual_out = nn.Sequential(
            *[ResidualBlock(8) for _ in range(n_residual_layers)]
        )
        # Final convolution to match output channels
        self.big_rf = nn.Conv2d(8, 8, kernel_size=(31, 5), padding=(15, 2))
        self.out_conv = nn.Conv2d(8, channels, kernel_size=5, padding=2)

    def forward(self, x):
        # x shape: (B, C_latent, F_latent, T_latent)
        x = self.residuals(x)
        x = self.up_block1(x)
        x = self.up_block2(x)
        x = self.residual_out(x)
        y = self.big_rf(x)  + x
        x = self.out_conv(x)
        return x
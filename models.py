# This file will have the class definitions for all of our model classes

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys


# Segmentation Mask - Unet model:

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=49, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Frame Prediction model:


class CNN_Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        k_size = 3
        p_len = 1
        self.input_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=k_size, stride=1, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=2, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=1, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=k_size, stride=2, padding=p_len),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.input_encoder(x)
        x_enc = x.clone()
        x = self.encoder(x)
        return x, x_enc


class GroupConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConvolution, self).__init__()
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.a_normalization = act_norm
        if input_channels % groups != 0:
            groups = 1
        self.convolution = nn.Conv2d(input_channels, output_channels, kernel_size=self.k, stride=self.s,
                                     padding=self.p, groups=groups)
        self.normalization = nn.GroupNorm(groups, output_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        y_label = self.convolution(x)
        if self.a_normalization:
            y_norm = self.normalization(y_label)
            y_label = self.activation(y_norm)
        return y_label


class InceptionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.k_size = 1
        self.s_size = 1
        list_layers = []
        self.convolution = nn.Conv2d(input_dim, hidden_dim, kernel_size=self.k_size, stride=self.s_size, padding=0)
        for k in inception_kernel:
            list_layers.append(
                GroupConvolution(hidden_dim, output_dim, kernel_size=k, stride=self.s_size, padding=k // 2,
                                 groups=groups, act_norm=True))
        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.convolution(x)
        y_label = 0
        for layer in self.layers:
            y_label += layer(x)
        return y_label


class InceptionBridge(nn.Module):
    def __init__(self, input_channels, hidden_channels, N_T, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.N_T = N_T
        encoder_layers = [
            InceptionModule(input_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups)]
        for i in range(1, N_T - 1):
            encoder_layers.append(InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels,
                                                  inception_kernel=inception_kernel, groups=groups))
        encoder_layers.append(
            InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups))
        decoder_layers = [
            InceptionModule(hidden_channels, hidden_channels // 2, hidden_channels, inception_kernel=inception_kernel,
                            groups=groups)]
        for i in range(1, N_T - 1):
            decoder_layers.append(InceptionModule(2 * hidden_channels, hidden_channels // 2, hidden_channels,
                                                  inception_kernel=inception_kernel, groups=groups))
        decoder_layers.append(InceptionModule(2 * hidden_channels, hidden_channels // 2, input_channels,
                                              inception_kernel=inception_kernel, groups=groups))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        list_pass = []
        z_hid = x
        for i in range(self.N_T):
            z_hid = self.encoder[i](z_hid)
            if (i < self.N_T - 1):
                list_pass.append(z_hid)
        z_hid = self.decoder[0](z_hid)
        for i in range(1, self.N_T):
            z_hid = self.decoder[i](torch.cat([z_hid, list_pass[-i]], dim=1))
        y_label = z_hid.reshape(B, T, C, H, W)
        return y_label


class CNN_Decoder(nn.Module):
    def __init__(self, hidden_channels, output_channels):
        super().__init__()
        self.k_size = 3
        self.p_size = 1
        self.output_p = 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=2, padding=self.p_size,
                               output_padding=self.output_p),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=1,
                               padding=self.p_size),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=self.k_size, stride=2, padding=self.p_size,
                               output_padding=self.output_p),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))
        self.output_decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=self.k_size, stride=1,
                               padding=self.p_size),
            nn.GroupNorm(2, hidden_channels),
            nn.LeakyReLU(0.2))

        self.output = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x, encoding):
        x = self.decoder(x)
        y_label = self.output_decoder(torch.cat([x, encoding], dim=1))
        y_label = self.output(y_label)
        return y_label


class DLModelVideoPrediction(nn.Module):
    def __init__(self, input_dim, hidden_size=16, translator_size=256, inception_kernel=[3, 5, 7, 11], groups=8):
        super().__init__()
        T, C, H, W = input_dim
        self.encoding = CNN_Encoder(C, hidden_size)
        self.hidden = InceptionBridge(T * hidden_size, translator_size, 8, inception_kernel, groups)
        self.decoding = CNN_Decoder(hidden_size, C)

    def forward(self, x_orig):
        B, T, C, H, W = x_orig.shape
        x = x_orig.view(B * T, C, H, W)

        embed, pass_ = self.encoding(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hidden_el = self.hidden(z)
        hidden_el = hidden_el.reshape(B * T, C_, H_, W_)

        Y = self.decoding(hidden_el, pass_)
        Y = Y.reshape(B, T, C, H, W)
        return Y

    
    
    
# COMBINED MODEL
    
class combined_model(nn.Module):
    def __init__(self, device):
        super(combined_model, self).__init__()
        self.frame_prediction_model = DLModelVideoPrediction((11, 3, 160, 240), 64, 512, groups=4)
        self.frame_prediction_model = self.frame_prediction_model.to(device)
        self.frame_prediction_model = nn.DataParallel(self.frame_prediction_model)
        self.image_segmentation_model = UNet(bilinear=True)
        self.image_segmentation_model = self.image_segmentation_model.to(device)
        self.image_segmentation_model = nn.DataParallel(self.image_segmentation_model)

    def forward(self, x):
        x = self.frame_prediction_model(x)
        x = x[:, -1]
        x = self.image_segmentation_model(x)
        return x

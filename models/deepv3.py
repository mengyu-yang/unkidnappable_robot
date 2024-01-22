"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch import nn
import functools
import pdb


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


class NonAtrousModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=256, num_layers=3):
        super(NonAtrousModule, self).__init__()

        self.features = []
        # 1x1
        for _ in range(num_layers + 1):
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )

        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(
        self,
        in_dim,
        reduction_dim,
        *rates,
    ):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        # if output_stride == 8:
        #     rates = [2 * r for r in rates]
        # elif output_stride == 16:
        #     pass
        # else:
        #     raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r, rr in rates[0]:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=(r, rr),
                        padding=(r, rr),
                        bias=False,
                    ),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def create_conv(
    input_channels,
    output_channels,
    kernel,
    paddings,
    batch_norm=True,
    Relu=True,
    stride=1,
):
    model = [
        nn.Conv2d(
            input_channels, output_channels, kernel, stride=stride, padding=paddings
        )
    ]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def unet_conv(input_nc, output_nc, kernel_size=4, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(
        input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1
    )
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


class DeepWV3PlusMultilossDepth(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(
        self,
        args,
        rates,
        ngf=64,
        input_nc=2,
        encoder_kernel_size=4,
        no_dilation=False,
        spec_feat_nc=256,
    ):
        super(DeepWV3PlusMultilossDepth, self).__init__()
        self.args = args

        if self.args.activation == "tanh":
            self.tanh = nn.Tanh()

        self.audionet_convlayer1 = unet_conv(
            input_nc, ngf, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer2 = unet_conv(
            ngf, ngf * 2, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer3 = unet_conv(
            ngf * 2, ngf * 4, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer4 = unet_conv(
            ngf * 4, ngf * 8, kernel_size=encoder_kernel_size
        )

        self.bot_aud1 = nn.Conv2d(
            (ngf * 8),
            spec_feat_nc,
            kernel_size=1,
            bias=False,
        )

        total_num_channels = spec_feat_nc * len(self.args.mic_channels)
        self.bot_multiaud = nn.Conv2d(
            total_num_channels, total_num_channels, kernel_size=1, bias=False
        )

        if no_dilation:
            self.aspp = NonAtrousModule(
                total_num_channels,
                total_num_channels // 8,
                num_layers=len(rates),
            )
        else:
            self.aspp = _AtrousSpatialPyramidPoolingModule(
                total_num_channels, total_num_channels // 8, rates
            )

        bot_aspp_in = (len(rates) + 2) * (total_num_channels // 8)
        self.bot_aspp = nn.Conv2d(bot_aspp_in, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(240 * 480, 1),
        )

        self.point_sin = nn.Sequential(nn.Linear(240 * 480, 1))
        self.point_cos = nn.Sequential(nn.Linear(240 * 480, 1))

        if self.args.condition_depth:
            self.point_depth = nn.Sequential(nn.Linear(240 * 480 + 5, 1))
        else:
            self.point_depth = nn.Sequential(nn.Linear(240 * 480, 1))
        self.point_class = nn.Sequential(nn.Linear(240 * 480, 1))

        initialize_weights(self.final)
        initialize_weights(self.bot_aud1)
        initialize_weights(self.bot_multiaud)

    def forward_Seg(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)

        return audio_conv4feature

    def forward(
        self,
        spectrograms,
        classes,
    ):
        out = []
        for spec in spectrograms:
            out_aud = self.forward_Seg(spec)
            dec0_aud = Upsample(out_aud, [60, 120])
            dec0_aud = self.bot_aud1(dec0_aud)
            out.append(dec0_aud)
        dec0_aud = torch.cat(out, 1)

        dec0_aud = self.bot_multiaud(dec0_aud)
        dec0_aud = self.aspp(dec0_aud)
        dec0_up = self.bot_aspp(dec0_aud)
        dec0_up = Upsample(dec0_up, [240, 480])
        dec1 = self.final(dec0_up)
        dec1 = dec1.view(dec1.size(0), -1)

        dec1_sin = self.point_sin(dec1)
        dec1_cos = self.point_cos(dec1)

        if self.args.activation == "tanh":
            dec1_sin = self.tanh(dec1_sin)
            dec1_cos = self.tanh(dec1_cos)
        else:
            # Clip to [-1, 1]
            dec1_sin = torch.clamp(dec1_sin, min=-1, max=1)
            dec1_cos = torch.clamp(dec1_cos, min=-1, max=1)

        dec1_class = self.point_class(dec1)

        if self.args.condition_depth:
            dec1_depth = torch.cat((dec1, classes), dim=1)
            dec1_depth = self.point_depth(dec1_depth)
        else:
            dec1_depth = self.point_depth(dec1)

        return dec1_sin, dec1_cos, dec1_depth, dec1_class


class DeepWV3PlusMultilossLearnBacksubDepth(nn.Module):
    """
    Wide_resnet version of DeepLabV3
    mod1
    pool2
    mod2 str2
    pool3
    mod3-7

      structure: [3, 3, 6, 3, 1, 1]
      channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                  (1024, 2048, 4096)]
    """

    def __init__(
        self,
        args,
        rates,
        ngf=64,
        input_nc=2,
        encoder_kernel_size=4,
        no_dilation=False,
        spec_feat_nc=256,
    ):
        super(DeepWV3PlusMultilossLearnBacksubDepth, self).__init__()
        self.args = args

        if self.args.activation == "tanh":
            self.tanh = nn.Tanh()

        self.audionet_convlayer1 = unet_conv(
            input_nc, ngf, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer2 = unet_conv(
            ngf, ngf * 2, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer3 = unet_conv(
            ngf * 2, ngf * 4, kernel_size=encoder_kernel_size
        )
        self.audionet_convlayer4 = unet_conv(
            ngf * 4, ngf * 8, kernel_size=encoder_kernel_size
        )

        self.bot_aud1 = nn.Conv2d(
            (ngf * 8),
            spec_feat_nc,
            kernel_size=1,
            bias=False,
        )

        total_num_channels = spec_feat_nc * len(self.args.mic_channels)
        self.bot_multiaud = nn.Conv2d(
            total_num_channels, total_num_channels, kernel_size=1, bias=False
        )

        if no_dilation:
            self.aspp = NonAtrousModule(
                total_num_channels,
                total_num_channels // 8,
                num_layers=len(rates),
            )
        else:
            self.aspp = _AtrousSpatialPyramidPoolingModule(
                total_num_channels, total_num_channels // 8, rates
            )

        bot_aspp_in = (len(rates) + 2) * (total_num_channels // 8)
        self.bot_aspp = nn.Conv2d(bot_aspp_in, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(240 * 480, 1),
        )

        self.backsub_conv = nn.Sequential(
            unet_conv(input_nc, ngf, kernel_size=encoder_kernel_size),
            unet_conv(ngf, 1, kernel_size=encoder_kernel_size),
            nn.ReLU(),
        )

        self.point_sin = nn.Sequential(nn.Linear(240 * 480, 1))
        self.point_cos = nn.Sequential(nn.Linear(240 * 480, 1))
        if self.args.multi_class:
            self.point_class = nn.Sequential(nn.Linear(240 * 480, 5))
        else:
            self.point_class = nn.Sequential(nn.Linear(240 * 480, 1))

        if self.args.condition_depth:
            self.point_depth = nn.Sequential(nn.Linear(240 * 480 + 5, 1))
        elif self.args.multiclass_depth:
            self.point_depth = nn.Sequential(
                nn.Linear(240 * 480, self.args.num_depth_bins)
            )
        else:
            self.point_depth = nn.Sequential(nn.Linear(240 * 480, 1))
        self.point_backsub = nn.Sequential(nn.Linear(60 * 80, 1))

        initialize_weights(self.final)
        initialize_weights(self.bot_aud1)
        initialize_weights(self.bot_multiaud)

    def forward_Seg(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)

        return audio_conv4feature

    def forward(
        self,
        spectrograms,
        empty_spectrograms,
        classes,
    ):
        # for i, empty_spec in enumerate(empty_spectrograms):
        #     empty_spec_feat = self.backsub_conv(empty_spec)
        #     empty_spec_feat = Upsample(empty_spec_feat, [60, 80])
        #     empty_spec_feat = empty_spec_feat.view(empty_spec_feat.size(0), -1)
        #     backsub_w = self.point_backsub(empty_spec_feat)
        #     backsub_w = torch.clamp(backsub_w, min=-0.01, max=1)
        #     spectrograms[i] = spectrograms[i] - backsub_w[i] * empty_spec

        for i, empty_spec in enumerate(empty_spectrograms):
            empty_spec_feat = self.backsub_conv(empty_spec)
            empty_spec_feat = Upsample(empty_spec_feat, [60, 80])
            empty_spec_feat = empty_spec_feat.view(empty_spec_feat.size(0), -1)
            backsub_w = self.point_backsub(empty_spec_feat)
            backsub_w = torch.clamp(backsub_w, min=-0.01, max=1)
            spectrograms[i] = (
                spectrograms[i] - backsub_w.unsqueeze(-1).unsqueeze(-1) * empty_spec
            )

        out = []
        for spec in spectrograms:
            out_aud = self.forward_Seg(spec)
            dec0_aud = Upsample(out_aud, [60, 120])
            dec0_aud = self.bot_aud1(dec0_aud)
            out.append(dec0_aud)
        dec0_aud = torch.cat(out, 1)

        dec0_aud = self.bot_multiaud(dec0_aud)
        dec0_aud = self.aspp(dec0_aud)
        dec0_up = self.bot_aspp(dec0_aud)
        dec0_up = Upsample(dec0_up, [240, 480])
        dec1 = self.final(dec0_up)
        dec1 = dec1.view(dec1.size(0), -1)

        dec1_sin = self.point_sin(dec1)
        dec1_cos = self.point_cos(dec1)

        if self.args.condition_depth:
            dec1_depth = torch.cat((dec1, classes), dim=1)
            dec1_depth = self.point_depth(dec1_depth)
        else:
            dec1_depth = self.point_depth(dec1)

        if not self.args.binary_depth:
            dec1_depth = torch.clamp(dec1_depth, min=0)

        if self.args.activation == "tanh":
            dec1_sin = self.tanh(dec1_sin)
            dec1_cos = self.tanh(dec1_cos)
        else:
            # Clip to [-1, 1]
            dec1_sin = torch.clamp(dec1_sin, min=-1, max=1)
            dec1_cos = torch.clamp(dec1_cos, min=-1, max=1)

        dec1_class = self.point_class(dec1)
        return dec1_sin, dec1_cos, dec1_depth, dec1_class

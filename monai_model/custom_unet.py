# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.layers.convutils import same_padding
from monai.utils import export, alias



@export("monai.networks.nets")
@alias("Unet")
class UNet(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        up_mode='transpose',
    ):
        super().__init__()

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.up_mode = up_mode

        def _create_block(inc, outc, channels, strides, is_top):
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_down_layer(self, in_channels, out_channels, strides, is_top):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides,
                self.kernel_size,
                self.num_res_units,
                self.act,
                self.norm,
                self.dropout,
            )
        else:
            return Convolution(
                self.dimensions, in_channels, out_channels, strides, self.kernel_size, self.act, self.norm, self.dropout
            )

    def _get_bottom_layer(self, in_channels, out_channels):
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        assert self.up_mode in ['transpose', 'resize']
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            self.up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=self.up_mode == 'transpose',
            is_resize_conv=self.up_mode == 'resize',
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                1,
                self.kernel_size,
                1,
                self.act,
                self.norm,
                self.dropout,
                last_conv_only=is_top,
            )
            return nn.Sequential(conv, ru)
        else:
            return conv

    def forward(self, x):
        x = self.model(x)
        return x




class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::
        -- (Conv|ConvTrans) -- Norm -- (Dropout) -- (Acti) --
    if ``conv_only`` set to ``True``::
        -- (Conv|ConvTrans) --
    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only:  whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
    See also:
        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.layers.Dropout`
        :py:class:`monai.networks.layers.Act`
        :py:class:`monai.networks.layers.Norm`
        :py:class:`monai.networks.layers.split_args`
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: int = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        is_resize_conv: bool = False,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert not (is_transposed and is_resize_conv), ('Two ways of doing the '
                                                'same thing. Can not do both.')
        self.is_transposed = is_transposed
        self.is_resize_conv = is_resize_conv

        padding = same_padding(kernel_size, dilation)
        self.padding = padding
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, dimensions]

        # define the normalisation type and the arguments to the constructor
        norm_name, norm_args = split_args(norm)
        norm_type = Norm[norm_name, dimensions]

        # define the activation type and the arguments to the constructor
        if act is not None:
            act_name, act_args = split_args(act)
            act_type = Act[act_name]
        else:
            act_type = act_args = None

        if dropout:
            # if dropout was specified simply as a p value, use default name and make a keyword map with the value
            if isinstance(dropout, (int, float)):
                drop_name = Dropout.DROPOUT
                drop_args = {"p": dropout}
            else:
                drop_name, drop_args = split_args(dropout)

            drop_type = Dropout[drop_name, dimensions]

        if is_transposed:
            conv = conv_type(in_channels, out_channels, kernel_size, strides, padding, strides - 1, 1, bias, dilation)
        elif is_resize_conv:
            conv = conv_type(in_channels, out_channels, kernel_size, 1      , padding, dilation, bias=bias)
        else:
            conv = conv_type(in_channels, out_channels, kernel_size, strides, padding, dilation, bias=bias)

        self.add_module("conv", conv)

        if not conv_only:
            self.add_module("norm", norm_type(out_channels, **norm_args))
            if dropout:
                self.add_module("dropout", drop_type(**drop_args))
            if act is not None:
                self.add_module("act", act_type(**act_args))

    def forward(self, x, *args, **kwargs):
        if self.is_resize_conv:
            # compute the output size if we had used a ConvTranspose layer
            out_padding = self.strides - 1
            out_shape = conv_transpose_out_shape(x.shape[2:],
                                self.strides, self.padding, self.dilation,
                                self.kernel_size, out_padding)
            x = F.interpolate(x, size=out_shape, mode='trilinear')

        return super().forward(x, *args, **kwargs)


def conv_transpose_out_shape(in_shape, strides, padding, dilation,
                             kernel_size, out_padding):
    out_shape = [None] * len(in_shape)
    for i, ind in enumerate(in_shape):
        outd = ((ind - 1) * strides
                - 2 * padding
                + dilation * (kernel_size - 1)
                + out_padding
                + 1)
        out_shape[i] = outd
    return tuple(out_shape)



Unet = unet = UNet

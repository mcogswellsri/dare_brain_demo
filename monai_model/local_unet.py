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

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import export, alias

from position_embedding import LearnedPositionalEmbedding

class UNet(nn.Module):
    """
    A UNet that does local segmentation.
    """
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
        assert up_mode == 'transpose', 'resize + conv not yet supported'

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
                subblock = Pass2nd(subblock)
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            if is_top:
                mask_attn_bottleneck = MaskAttentionBottleneck(self.dimensions,
                                            c, out_channels=c, key_channels=c)
                return nn.Sequential(Pass2nd(down),
                                     mask_attn_bottleneck,
                                     Pass2ndSkipConnection(subblock),
                                     Pass2nd(up))
            else:
                return nn.Sequential(Pass2nd(down),
                                     Pass2ndSkipConnection(subblock),
                                     Pass2nd(up))

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
            is_transposed=True,
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
        # x == (image, mask)
        x = self.model(x)
        logits, mask = x
        return logits


class Pass2nd(nn.Module):
    """
    Given an nn.Module that takes one input, this module takes the first
    of its two-tuple input and applies the nn.Module. It returns another
    two-tuple containing the result of that computation and the 2nd input.

    This is useful
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, tup2):
        x, passed = tup2
        return (self.module(x), passed)


class Pass2ndSkipConnection(nn.Module):
    """
    A Pass2nd layer that feeds both of its arguments to the skip connection's
    module.
    """
    def __init__(self, module, cat_dim: int = 1):
        super().__init__()
        self.module = module
        self.cat_dim = cat_dim

    def forward(self, tup2):
        x, _ = tup2
        y, passedy = self.module(tup2)
        skip_out = torch.cat([x, y], self.cat_dim)
        return (skip_out, passedy)


class MaskAttentionBottleneck(nn.Module):
    """
    This takes a feature map and a mask as input. Given that the output
    only needs to be relevant to the mask region, this layer returns
    an attended version of the feature map that only contains relevant regions.

    Args:
        max_width: The maximum size of any HWD dimension of an input feature
                   map. Position embeddings are computed for inputs up to this
                   size.
    """
    def __init__(self, dimensions, in_channels, out_channels=None,
                    key_channels=None, max_width=200):
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.max_width = max_width
        self.out_channels = in_channels if out_channels is None else out_channels
        self.key_channels = in_channels if key_channels is None else key_channels
        self.key_conv = Convolution(
                            self.dimensions,
                            self.in_channels,
                            self.key_channels,
                            kernel_size=1)
        self.val_conv = Convolution(
                            self.dimensions,
                            self.in_channels,
                            self.out_channels,
                            kernel_size=1)
        self.query_conv = Convolution(
                            self.dimensions,
                            self.in_channels,
                            self.key_channels,
                            kernel_size=1)
        self.h_emb = LearnedPositionalEmbedding(self.max_width,
                                                self.key_channels, None)
        self.w_emb = LearnedPositionalEmbedding(self.max_width,
                                                self.key_channels, None)
        self.d_emb = LearnedPositionalEmbedding(self.max_width,
                                                self.key_channels, None)


    def forward(self, tup2):
        x, mask = tup2
        B = x.shape[0]
        keys = self.key_conv(x)
        vals = self.val_conv(x)
        query_map = self.query_conv(x)

        # position features
        # NOTE: only the shape of the input tensor matters here
        h_emb = self.h_emb(keys[:, 0, :, 0, 0])
        h_emb = h_emb.transpose(1, 2)[:, :, :, None, None] # BNH11
        w_emb = self.w_emb(keys[:, 0, 0, :, 0])
        w_emb = w_emb.transpose(1, 2)[:, :, None, :, None] # BN1W1
        d_emb = self.d_emb(keys[:, 0, 0, 0, :])
        d_emb = d_emb.transpose(1, 2)[:, :, None, None, :] # BN11D
        # always adding to the large tensor avoids unnecessary memory allocation
        keys = (((keys + h_emb) + w_emb) + d_emb)
        query_map = (((query_map + h_emb) + w_emb) + d_emb)

        # pool the query to make it relevant only to the masked region
        mask = F.interpolate(mask, x.shape[2:])
        mask_npixels = mask.sum(dim=(2,3,4), keepdims=True)
        query = mask * query_map / mask_npixels
        query = query.sum(dim=(2,3,4), keepdims=True)

        # compute dot product attention
        logits = (query * keys).sum(dim=1, keepdims=True) / query.shape[1] ** 0.5
        mask_relevant_attn = F.softmax(logits.view(B, 1, -1), dim=2)
        mask_relevant_attn = mask_relevant_attn.view(logits.shape)

        # return attended values and resized mask
        # NOTE: The mask_npixels prevents feature map values from getting too
        # small due to the 3d spatial softmax.
        out_fmap = mask_npixels * mask_relevant_attn * vals
        return out_fmap, mask # the returned mask is the resized one

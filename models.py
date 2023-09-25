'''
VoxelMorph
Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Haiqiao Wang
2110246069@email.szu.edu.com
Shenzhen University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2*c, kernel_size=3, stride=2, padding=1),#80
            ResBlock(2*c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 4*c, kernel_size=3, stride=2, padding=1),#40
            ResBlock(4*c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4*c, 8*c, kernel_size=3, stride=2, padding=1),#20
            ResBlock(8*c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class RDP(nn.Module):
    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=1, channels=16):
        super(RDP, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.encoder_moving = Encoder(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))
            
        # bottleNeck
        self.cconv_4 = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )
        # warp scale 2
        self.defconv4 = nn.Conv3d(8*c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
        self.dconv4 = nn.Sequential(
            ConvInsBlock(3*8*c, 8*c),
            ConvInsBlock(8*c, 8*c)
        )
        
        self.upconv3 = UpConvBlock(8*c, 4*c, 4, 2)
        self.cconv_3 = CConv(3*4*c)

        # warp scale 1
        self.defconv3 = nn.Conv3d(3*4*c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = ConvInsBlock(3 * 4 * c, 4 * c)
        
        self.upconv2 = UpConvBlock(3*4*c, 2*c, 4, 2)
        self.cconv_2 = CConv(3*2*c)

        # warp scale 0
        self.defconv2 = nn.Conv3d(3*2*c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        self.dconv2 = ConvInsBlock(3 * 2 * c, 2 * c)
        
        self.upconv1 = UpConvBlock(3*2*c, c, 4, 2)
        self.cconv_1 = CConv(3*c)

        # decoder layers
        self.defconv1 = nn.Conv3d(3*c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))
        #self.dconv1 = ConvInsBlock(3 * c, c)

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)
        # c=16, 2c, 4c, 8c  # 160, 80, 40, 20

        # first dec layer
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)  # (1,128,20,24,20)
        flow = self.defconv4(C4)  # (1,3,20,24,20)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)  # (1,3,20,24,20)
        w = self.diff[3](v)


        D3 = self.upconv3(C4)   # (1, 64, 40, 48, 40)
        flow = self.upsample_trilin(2*(self.warp[3](flow, w)+w))
        warped = self.warp[2](M3, flow)  # (1, 64, 40, 48, 40)
        C3 = self.cconv_3(F3, warped, D3)  #  (1, 3 * 64, 40, 48, 40)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w)+w
        warped = self.warp[2](M3, flow)  # (1, 64, 40, 48, 40)
        D3 = self.dconv3(C3)
        C3 = self.cconv_3(F3, warped, D3)  #  (1, 3 * 64, 40, 48, 40)
        v = self.defconv3(C3)
        w = self.diff[2](v)

        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2*(self.warp[2](flow, w)+w))
        warped = self.warp[1](M2, flow)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w)+w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w)+w
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2(C2)
        C2 = self.cconv_2(F2, warped, D2)
        v = self.defconv2(C2)  # (1,3,80,96,80)
        w = self.diff[1](v)

        D1 = self.upconv1(C2)  # (1,16,160,196,160)
        flow = self.upsample_trilin(2*(self.warp[1](flow, w)+w))  # （1,3,160,196,160)
        warped = self.warp[0](M1, flow)  # （1,16,160,196,160)
        C1 = self.cconv_1(F1, warped, D1)  # （1,48,160,196,160)
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w)+w  # （1,3,160,196,160)

        y_moved = self.warp[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':
    size = (1, 1, 80, 96, 80)
    model = RDP(size[2:])
    # print(str(model))
    A = torch.ones(size)
    B = torch.ones(size)
    out, flow = model(A, B)
    print(out.shape, flow.shape)

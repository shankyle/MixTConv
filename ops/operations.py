# Code for "CFM"
# arXiv:TBD
# Kaiyu Shan
# shankyle@pku.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
# from arch import *
# from spatial_correlation_sampler import SpatialCorrelationSampler
import math
from IPython import embed


class GroupConv1d(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, kernel_size=(3, 1, 1)):
        super(GroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.kernel_size = kernel_size
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(net.in_channels, net.in_channels, kernel_size=kernel_size,
                                padding=((self.kernel_size[0] - 1) // 2, 0, 0), bias=False, groups=net.in_channels)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]
        self.net = net
    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes = self.conv1d.in_channels
        fold = planes // self.fold_div
        if self.kernel_size[0] == 3:
            weight = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, 0] = 1.0
            weight[fold: fold * 2, 0, 2] = 1.0
            weight[fold * 2:, 0, 1] = 1.0
        elif self.kernel_size[0] == 5:
            weight = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :2] = 1.0  # [11000]
            weight[fold: fold * 2, 0, 3:] = 1.0  # [00011]
            weight[fold * 2:, 0, 2] = 1.0  # [00100]

        elif self.kernel_size[0] == 7:
            weight = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :3] = 1.0  # [1110000]
            weight[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
            weight[fold * 2:, 0, 3] = 1.0  # [0001000]
        else:
            raise NotImplementedError(self.kernel_size)

        self.conv1d.weight = nn.Parameter(weight)

    def groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    def forward(self, x):
        x = self.groupconv1d(x)
        return self.net(x)


class MsGroupConv1d(nn.Module):
    def __init__(self, net, n_segment=8, n_div=4, inplace=True):
        super(MsGroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        if inplace:
            print('=> Using in-place multi-scale 1d conv...')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv13d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4)
        self.net = net
        self.weight_init()

    def ms_groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)

        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)

        x = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h, w)

        return x

    def weight_init(self):
        print('=> Using weight init of 4 parts for 4 multi-scale')
        planes = self.conv11d.in_channels
        fold = planes // self.fold_div # div = 4

        weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight1[:, 0, 0] = 1.0
        self.conv11d.weight = nn.Parameter(weight1)

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
        weight3[:fold, 0, 0] = 1.0
        weight3[fold: fold * 2, 0, 2] = 1.0
        weight3[fold * 2:, 0, 1] = 1.0
        self.conv13d.weight = nn.Parameter(weight3)

        weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight5[:fold, 0, :2] = 1.0  # [11000]
        weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
        weight5[fold * 2:, 0, 2] = 1.0 # [00100]
        self.conv15d.weight = nn.Parameter(weight5)

        weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight7[:fold, 0, :3] = 1.0  # [1110000]
        weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
        weight7[fold * 2:, 0, 3] = 1.0 # [0001000]
        self.conv17d.weight = nn.Parameter(weight7)

    def forward(self, x):
        x = self.ms_groupconv1d(x)
        return self.net(x)


class MsCorrBlock(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, mode='nearest', stride=1,
                 corr_size=7, dwise=False, corr_group=1, reduction=4, inplace=False):
        super(MsCorrBlock, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.reduction = reduction
        self.inplace = inplace
        if dwise:
            print('=> Using depth-wise correlation...')
            self.corr_group = net.conv1.in_channels // self.reduction
        else:
            self.corr_group = corr_group
            print('=> corr_group is {}...'.format(corr_group))

        if not self.inplace:
            c_in = net.conv1.in_channels
        else:
            c_in = net.conv2.in_channels
        c_corr_in = net.conv2.in_channels

        self.stride = stride
        self.mode = mode

        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=c_in // 4, stride=(1, 1, 1))
        self.conv13d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in // 4, stride=(1, 1, 1))
        self.conv15d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=c_in // 4, stride=(1, 1, 1))
        self.conv17d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=c_in // 4, stride=(1, 1, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # TBD: bias = False, weight init
        self.conv21 = nn.Conv3d(c_corr_in, c_corr_in // self.reduction, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False)
        self.bn21 = nn.BatchNorm3d(c_corr_in // self.reduction)

        self.kkgroup = corr_size * corr_size * self.corr_group
        self.corr = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=corr_size,
            stride=1,
            padding=0,
            dilation_patch=1,
        )
        self.bn22 = nn.BatchNorm2d(self.kkgroup)
        self.conv22 = nn.Conv2d(self.kkgroup, c_corr_in, kernel_size=(1, 1),
                                padding=(0, 0), bias=False)
        self.bn23 = nn.BatchNorm2d(c_corr_in)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv21.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv22.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn21.weight, 1)
        nn.init.constant_(self.bn21.bias, 0)
        nn.init.constant_(self.bn22.weight, 1)
        nn.init.constant_(self.bn22.bias, 0)
        nn.init.constant_(self.bn23.weight, 1)
        nn.init.constant_(self.bn23.bias, 0)

        self.net = net
        self.weight_init()

    def ms_group(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)
        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous()
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous()
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous()
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous()

        x = torch.cat((x1, x3, x5, x7), dim=2).view(nt, c, h, w)
        return x

    # TBD: init
    def corr_outer(self, x):
        residual = x

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu(x)

        index = list(range(1, self.n_segment))
        index.append(self.n_segment - 1)
        x2 = x[:, :, index, :, :]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)

        x = self.corr(x, x2).contiguous().view(nt, self.kkgroup, h, w) / (c // (self.reduction * self.corr_group))
        x = self.bn22(x)
        x = self.relu(x)

        x = self.conv22(x)
        x = self.bn23(x)
        x += residual
        x = self.relu(x)

        return x

    def weight_init(self):
        print('=> Using weight init of 4 parts for 4 multi-scale')
        planes = self.conv11d.in_channels
        fold = planes // self.fold_div  # div = 4

        weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight1[:, 0, 0] = 1.0
        self.conv11d.weight = nn.Parameter(weight1)

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
        weight3[:fold, 0, 0] = 1.0
        weight3[fold: fold * 2, 0, 2] = 1.0
        weight3[fold * 2:, 0, 1] = 1.0
        self.conv13d.weight = nn.Parameter(weight3)

        weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight5[:fold, 0, :2] = 1.0  # [11000]
        weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
        weight5[fold * 2:, 0, 2] = 1.0  # [00100]
        self.conv15d.weight = nn.Parameter(weight5)

        weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight7[:fold, 0, :3] = 1.0  # [1110000]
        weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
        weight7[fold * 2:, 0, 3] = 1.0  # [0001000]
        self.conv17d.weight = nn.Parameter(weight7)

    def forward(self, x):
        residual = x
        if self.net.downsample is not None:
            residual = self.net.downsample(x)

        if not self.inplace:
            out = self.ms_group(x)

        out = self.net.conv1(out)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        if self.inplace:
            out = self.ms_group(out)

        out2 = self.corr_outer(out)
        if self.stride != 1:
            out2 = self.avgpool2d(out2)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = out + out2
        out = self.net.conv3(out)
        out = self.net.bn3(out)

        out += residual

        out = self.net.relu(out)
        return out


class CorrBlock_cascade(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, mode='nearest', in_channels=64,
                 corr_size=7, dwise=False, corr_group=1, reduction=4):
        super(CorrBlock_cascade, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.mode = mode
        self.reduction = reduction

        # TBD: bias = False, weight init
        self.conv21 = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False)
        if dwise:
            print('=> Using depth-wise correlation...')
            self.corr_group = in_channels // self.reduction
        else:
            self.corr_group = corr_group
            print('=> corr_group is {}...'.format(corr_group))

        # kkgroup 要计算一下
        self.kkgroup = corr_size * corr_size * self.corr_group

        self.conv22 = nn.Conv2d(self.kkgroup, in_channels, kernel_size=(1, 1),
                                padding=(0, 0), bias=False)
        self.bn21 = nn.BatchNorm3d(in_channels // self.reduction)
        self.bn22 = nn.BatchNorm2d(self.kkgroup)
        self.bn23 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.corr = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=7,
            stride=1,
            padding=0,
            dilation_patch=1,
        )

        nn.init.kaiming_normal_(self.conv21.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.net = net

    # TBD: init
    def corr_outer(self, x):
        residual = x

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu(x)

        index = list(range(1, self.n_segment))
        index.append(self.n_segment - 1)
        x2 = x[:, :, index, :, :]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)

        x = self.corr(x, x2).contiguous().view(nt, self.kkgroup, h, w) / (c // (self.reduction * self.corr_group))

        x = self.bn22(x)
        x = self.relu(x)

        x = self.conv22(x)
        x = self.bn23(x)
        x += residual
        x = self.relu(x)

        return x

    def forward(self, x):
        out = self.corr_outer(x)
        return self.net(out)


class STM(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=4, inplace=True, reduction=16):
        super(STM, self).__init__()
        self.n_segment = n_segment
        self.reduction = reduction
        self.fold_div = n_div  # each shift part
        self.inplace = inplace

        c_in = net.conv1.out_channels

        self.stride = net.conv2.stride[0]
        if inplace:
            c_in = net.conv1.out_channels
            print('=> Using in-place group 1d conv...')
        else:
            c_in = net.conv1.in_channels
            print('=> Using out-place group 1d conv...')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(c_in, c_in, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in)

        self.conv1 = nn.Conv3d(c_in, c_in // self.reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_in // self.reduction)
        self.conv2 = nn.Conv3d(c_in // self.reduction, c_in // self.reduction, kernel_size=(1, 3, 3),
                               stride=1, padding=(0, 1, 1), bias=False, groups=c_in // self.reduction)

        self.conv3 = nn.Conv3d(c_in // self.reduction,  c_in, kernel_size=1, bias=False)

        if self.stride != 1:
            self.maxpooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.net = net
        self.weight_init()

    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes = self.conv1d.in_channels
        fold = planes // self.fold_div
        weight = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight[:fold, 0, 0] = 1.0
        weight[fold: fold * 2, 0, 2] = 1.0
        weight[fold * 2:, 0, 1] = 1.0
        self.conv1d.weight = nn.Parameter(weight)

    def groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    # TBD: init
    def cmm(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x = self.conv1(x)
        x = self.bn1(x)

        index = list(range(1, self.n_segment))
        index2 = list(range(0, self.n_segment - 1))
        x2 = x[:, :, index, :, :]
        x2 = self.conv2(x2)

        x2 = x2 - x[:, :, index2, :, :]
        x = F.pad(x2, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # t => (0, 1)
        x = self.conv3(x).permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    def forward(self, x):
        residual = x
        if self.net.downsample is not None:
            residual = self.net.downsample(x)

        if not self.inplace:
            x = self.groupconv1d(x)

        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out2 = self.cmm(out)

        if self.stride != 1:
            out2 = self.maxpooling(out2)

        if self.inplace:
            out = self.groupconv1d(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = out + out2
        out = self.net.conv3(out)
        out = self.net.bn3(out)

        out += residual
        out = self.net.relu(out)

        return out


class GroupConv1dSplit(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, kernel_size=(3, 1, 1)):
        super(GroupConv1dSplit, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.kernel_size = kernel_size[0]
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(net.in_channels, net.in_channels, kernel_size=kernel_size,
                                padding=((self.kernel_size - 1) // 2, 0, 0), bias=False, groups=net.in_channels)
        self.net = net
        self.stride = self.net.stride
        planes2d = self.net.in_channels
        self.conv21 = nn.Conv2d(planes2d // 2, planes2d // 2, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes2d // 2, planes2d // 2, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]


    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes = self.conv1d.in_channels
        fold = planes // self.fold_div
        if self.kernel_size == 3:
            weight = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, 0] = 1.0
            weight[fold: fold * 2, 0, 2] = 1.0
            weight[fold * 2:, 0, 1] = 1.0
        elif self.kernel_size == 5:
            weight = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :2] = 1.0  # [11000]
            weight[fold: fold * 2, 0, 3:] = 1.0  # [00011]
            weight[fold * 2:, 0, 2] = 1.0  # [00100]

        elif self.kernel_size == 7:
            weight = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :3] = 1.0  # [1110000]
            weight[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
            weight[fold * 2:, 0, 3] = 1.0  # [0001000]
        else:
            raise NotImplementedError(self.kernel_size)

        self.conv1d.weight = nn.Parameter(weight)
        size = self.net.weight.size()
        weight2d1 = self.net.weight[:size[0] // 2, : size[1] // 2, :, :]
        weight2d2 = self.net.weight[size[0] // 2:, size[1] // 2:, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)


    def groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    def forward(self, x):
        x = self.groupconv1d(x)
        nt, c, h, w = x.size()
        x1, x2 = x.split([c // 2, c // 2], dim=1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x = torch.cat((x1, x2), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


class MSGroupConv1dSplit(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(MSGroupConv1dSplit, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv13d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4)

        self.net = net
        self.stride = self.net.stride
        planes2d = self.net.in_channels
        self.conv21 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv23 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv24 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

    def weight_init(self):
        print('=> Using weight init of 4 parts for 4 multi-scale')
        planes = self.conv11d.in_channels
        fold = planes // self.fold_div # div = 4

        weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight1[:, 0, 0] = 1.0
        self.conv11d.weight = nn.Parameter(weight1)

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
        weight3[:fold, 0, 0] = 1.0
        weight3[fold: fold * 2, 0, 2] = 1.0
        weight3[fold * 2:, 0, 1] = 1.0
        self.conv13d.weight = nn.Parameter(weight3)

        weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight5[:fold, 0, :2] = 1.0  # [11000]
        weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
        weight5[fold * 2:, 0, 2] = 1.0 # [00100]
        self.conv15d.weight = nn.Parameter(weight5)

        weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight7[:fold, 0, :3] = 1.0  # [1110000]
        weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
        weight7[fold * 2:, 0, 3] = 1.0 # [0001000]
        self.conv17d.weight = nn.Parameter(weight7)

        size = self.net.weight.size()
        weight2d1 = self.net.weight[:size[0] // 4, :size[1] // 4, :, :]
        weight2d2 = self.net.weight[size[0] // 4: size[0] // 2, size[1] // 4: size[1] // 2, :, :]
        weight2d3 = self.net.weight[size[0] // 2: (size[0] // 4) * 3, size[1] // 2: (size[1] // 4) * 3, :, :]
        weight2d4 = self.net.weight[(size[0] // 4) * 3:, (size[0] // 4) * 3:, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)
        self.conv23.weight = nn.Parameter(weight2d3)
        self.conv24.weight = nn.Parameter(weight2d4)
        print("=> finish 3x3 init")

    def ms_groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)

        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)

        x = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h, w)

        return x

    def forward(self, x):
        x = self.ms_groupconv1d(x)

        nt, c, h, w = x.size()
        x1, x2, x3, x4 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x3 = self.conv23(x3)
        x4 = self.conv24(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


class Group1dCFMSplit(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(Group1dCFMSplit, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv13d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4)

        self.net = net
        self.stride = self.net.stride
        planes2d = self.net.in_channels
        self.conv21 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv23 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv24 = nn.Conv2d(planes2d // 4, planes2d // 4, kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

    def weight_init(self):
        print('=> Using weight init of 4 parts for 4 multi-scale')
        planes = self.conv11d.in_channels
        fold = planes // self.fold_div # div = 4

        weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight1[:, 0, 0] = 1.0
        self.conv11d.weight = nn.Parameter(weight1)

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
        weight3[:fold, 0, 0] = 1.0
        weight3[fold: fold * 2, 0, 2] = 1.0
        weight3[fold * 2:, 0, 1] = 1.0
        self.conv13d.weight = nn.Parameter(weight3)

        weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight5[:fold, 0, :2] = 1.0  # [11000]
        weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
        weight5[fold * 2:, 0, 2] = 1.0 # [00100]
        self.conv15d.weight = nn.Parameter(weight5)

        weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight7[:fold, 0, :3] = 1.0  # [1110000]
        weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
        weight7[fold * 2:, 0, 3] = 1.0 # [0001000]
        self.conv17d.weight = nn.Parameter(weight7)

        size = self.net.weight.size()
        weight2d1 = self.net.weight[:size[0] // 4, :size[1] // 4, :, :]
        weight2d2 = self.net.weight[size[0] // 4: size[0] // 2, size[1] // 4: size[1] // 2, :, :]
        weight2d3 = self.net.weight[size[0] // 2: (size[0] // 4) * 3, size[1] // 2: (size[1] // 4) * 3, :, :]
        weight2d4 = self.net.weight[(size[0] // 4) * 3:, (size[0] // 4) * 3:, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)
        self.conv23.weight = nn.Parameter(weight2d3)
        self.conv24.weight = nn.Parameter(weight2d4)
        print("=> finish 3x3 init")

    def ms_groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)

        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)

        x = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h, w)

        return x

    def forward(self, x):
        x = self.ms_groupconv1d(x)

        nt, c, h, w = x.size()
        x1, x2, x3, x4 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x3 = self.conv23(x3)
        x4 = self.conv24(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


class GST(nn.Module):
    def __init__(self, net, n_segment=8, n_div=4):
        super(GST, self).__init__()
        self.n_segment = n_segment
        self.stride = net.stride[0]
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv3d = nn.Conv3d(net.in_channels // 2, net.out_channels // self.fold_div, kernel_size=3,
                                padding=1, bias=False, stride=(1, self.stride, self.stride))
        self.conv2d = nn.Conv2d(net.in_channels // 2, net.out_channels - (net.out_channels // self.fold_div),
                                kernel_size=3, padding=1, bias=False, stride=self.stride)

        self.net = net
        self.weight_init()  # checkpoint[k] = checkpoint[k][:n_out//groups*(groups-1),:n_in//2,:,:,:]

    def weight_init(self):
        print('=> Using partial weight of conv2 in resnet')
        size = self.net.weight.size()
        weight2d = self.net.weight[:size[0] - (size[0] // self.fold_div), :size[1] // 2, :, :]
        self.conv2d.weight = nn.Parameter(weight2d)

    def gst(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x2 = x.split([c // 2, c // 2], dim=1)
        x1 = self.conv2d(x1.permute(0, 2, 1, 3, 4).contiguous().view(nt, c // 2, h, w))
        x2 = self.conv3d(x2)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt, -1, h // self.stride, w // self.stride)

        return torch.cat((x1, x2), dim=1)

    def forward(self, x):
        x = self.gst(x)
        return x


class PartialCorr(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, corr_size=3, corr_group=1):
        super(PartialCorr, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.corr_group = corr_group
        print('=> corr_group is {}...'.format(corr_group))

        # kkgroup 要计算一下
        self.kkgroup = corr_size * corr_size * self.corr_group

        self.corr = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=corr_size,
            stride=1,
            padding=0,
            dilation_patch=1,
        )

        nn.Parameter()
        nn.init.kaiming_normal_(self.conv21.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.net = net

    # TBD: init
    def corr_partial(self, x):
        residual = x

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu(x)

        index = list(range(1, self.n_segment))
        index.append(self.n_segment - 1)
        x2 = x[:, :, index, :, :]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt * self.corr_group, c // (self.reduction * self.corr_group), h, w)

        x = self.corr(x, x2).contiguous().view(nt, self.kkgroup, h, w) / (c // (self.reduction * self.corr_group))

        x = self.bn22(x)
        x = self.relu(x)

        x = self.conv22(x)
        x = self.bn23(x)
        x += residual
        x = self.relu(x)

        return x

    def forward(self, x):
        out = self.corr_outer(x)
        return self.net(out)


class GroupConv1dPcbam(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, kernel_size=(3, 1, 1)):
        super(GroupConv1dPcbam, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.kernel_size = kernel_size
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(net.in_channels, net.in_channels, kernel_size=kernel_size,
                                padding=((self.kernel_size[0] - 1) // 2, 0, 0), bias=False, groups=net.in_channels)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]
        self.net = net

        self.convat = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.softmax = nn.Softmax2d()
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)

    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes = self.conv1d.in_channels
        fold = planes // self.fold_div
        if self.kernel_size[0] == 3:
            weight = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, 0] = 1.0
            weight[fold: fold * 2, 0, 2] = 1.0
            weight[fold * 2:, 0, 1] = 1.0
        elif self.kernel_size[0] == 5:
            weight = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :2] = 1.0  # [11000]
            weight[fold: fold * 2, 0, 3:] = 1.0  # [00011]
            weight[fold * 2:, 0, 2] = 1.0  # [00100]

        elif self.kernel_size[0] == 7:
            weight = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :3] = 1.0  # [1110000]
            weight[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
            weight[fold * 2:, 0, 3] = 1.0  # [0001000]
        else:
            raise NotImplementedError(self.kernel_size)

        self.conv1d.weight = nn.Parameter(weight)

    def groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    # TBD: init
    def partial_cbam(self, x):
        nt, c, h, w = x.size()
        x1 = x[:, :c // self.fold_div, :, :]
        x1 = x1.view(nt, c // self.fold_div, h*w).permute(0, 2, 1).contiguous()
        attention_avg = self.avgpooling(x1)
        attention_max = self.maxpooling(x1)
        attention_avg = attention_avg.permute(0, 2, 1).contiguous().view(nt, 1, h, w)
        attention_max = attention_max.permute(0, 2, 1).contiguous().view(nt, 1, h, w)
        attention_spatial = self.softmax(self.convat(torch.cat((attention_avg, attention_max), dim=1)))
        return attention_spatial

    def forward(self, x):
        x = self.groupconv1d(x)
        x = self.partial_cbam(x) * x
        return self.net(x)


class GroupConv1dSplitdiv(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, kernel_size=(3, 1, 1)):
        super(GroupConv1dSplitdiv, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(net.in_channels, net.in_channels, kernel_size=kernel_size,
                                padding=((kernel_size[0] - 1) // 2, 0, 0), bias=False, groups=net.in_channels)
        self.kernel_size = kernel_size[0]
        self.weight2d = net.weight
        self.stride = net.stride
        planes22d = net.in_channels

        self.conv21 = nn.Conv2d(planes22d // (n_div // 2), planes22d // (n_div // 2), kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes22d - (planes22d // (n_div // 2)), planes22d - (planes22d // (n_div // 2)),
                                kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes = self.conv1d.in_channels
        fold = planes // self.fold_div
        if self.kernel_size == 3:
            weight = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, 0] = 1.0
            weight[fold: fold * 2, 0, 2] = 1.0
            weight[fold * 2:, 0, 1] = 1.0
        elif self.kernel_size == 5:
            weight = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :2] = 1.0  # [11000]
            weight[fold: fold * 2, 0, 3:] = 1.0  # [00011]
            weight[fold * 2:, 0, 2] = 1.0  # [00100]

        elif self.kernel_size == 7:
            weight = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
            weight[:fold, 0, :3] = 1.0  # [1110000]
            weight[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
            weight[fold * 2:, 0, 3] = 1.0  # [0001000]
        else:
            raise NotImplementedError(self.kernel_size)

        self.conv1d.weight = nn.Parameter(weight)
        size = self.weight2d.size()
        weight2d1 = self.weight2d[:size[0] // (self.fold_div // 2), : size[1] // (self.fold_div // 2), :, :]
        weight2d2 = self.weight2d[size[0] // (self.fold_div // 2):, size[1] // (self.fold_div // 2):, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)

    def groupconv1d(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv1d(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x.view(nt, c, h, w)

    def forward(self, x):
        x = self.groupconv1d(x)
        nt, c, h, w = x.size()
        x1, x2 = x.split([c // (self.fold_div // 2), c - (c // (self.fold_div // 2))], dim=1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x = torch.cat((x1, x2), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


class GroupConv1dSplit1x1(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(GroupConv1dSplit1x1, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.weight2d = net.weight
        self.stride = net.stride

        planes1 = net.in_channels
        planes2 = net.out_channels
        self.c_out = planes2
        self.conv11 = nn.Conv2d(planes1 // self.fold_div, planes2 // self.fold_div, kernel_size=1, bias=False)
        self.conv12 = nn.Conv2d(planes1 - (planes1 // self.fold_div), planes2 - (planes2 // self.fold_div), kernel_size=1, bias=False)

        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

    def weight_init(self):

        size = self.weight2d.size()
        weight2d1 = self.weight2d[:size[0] // self.fold_div, : size[1] // self.fold_div, :, :]
        weight2d2 = self.weight2d[size[0] // self.fold_div:, size[1] // self.fold_div:, :, :]
        self.conv11.weight = nn.Parameter(weight2d1)
        self.conv12.weight = nn.Parameter(weight2d2)

    def forward(self, x):
        nt, c, h, w = x.size()
        x1, x2 = x.split([c // self.fold_div, c - (c // self.fold_div)], dim=1)
        x1 = self.conv11(x1)
        x2 = self.conv12(x2)
        x = torch.cat((x1, x2), dim=1).view(nt, self.c_out, h, w)
        return x


class MsConv1dSplitdiv(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(MsConv1dSplitdiv, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))

        c_in_dy = (net.in_channels // 4) // (n_div // 2)
        c_in_sta = (net.in_channels // 4) - ((net.in_channels // 4) // (n_div // 2))

        self.c_dy = c_in_dy
        self.c_sta = c_in_sta

        self.conv1d11 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d13 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d15 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d17 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=c_in_dy)

        self.conv1d21 = nn.Conv3d(c_in_sta, c_in_sta, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=c_in_sta)
        self.conv1d23 = nn.Conv3d(c_in_sta, c_in_sta, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in_sta)
        self.conv1d25 = nn.Conv3d(c_in_sta, c_in_sta, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=c_in_sta)
        self.conv1d27 = nn.Conv3d(c_in_sta, c_in_sta, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=c_in_sta)

        self.weight2d = net.weight
        self.stride = net.stride
        planes22d = net.in_channels

        self.conv21 = nn.Conv2d(planes22d // (n_div // 2), planes22d // (n_div // 2), kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes22d - (planes22d // (n_div // 2)), planes22d - (planes22d // (n_div // 2)),
                                kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes_dy = self.conv1d11.in_channels
        planes_sta = self.conv1d21.in_channels

        weight13 = torch.zeros(planes_dy, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight13[:planes_dy // 2, 0, 0] = 1.0
        weight13[planes_dy // 2:, 0, 2] = 1.0

        weight15 = torch.zeros(planes_dy, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight15[:planes_dy // 2, 0, :2] = 1.0  # [11000]
        weight15[planes_dy // 2:, 0, 3:] = 1.0  # [00011]

        weight17 = torch.zeros(planes_dy, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight17[:planes_dy // 2, 0, :3] = 1.0  # [1110000]
        weight17[planes_dy // 2:, 0, 4:] = 1.0  # [0000111]

        weight23 = torch.zeros(planes_sta, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight23[:, 0, 1] = 1.0  # [010]
        weight25 = torch.zeros(planes_sta, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight25[:, 0, 2] = 1.0  # [00100]
        weight27 = torch.zeros(planes_sta, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight27[:, 0, 3] = 1.0  # [0001000]

        weight11 = torch.zeros(planes_dy, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight11[:, 0, 0] = 1.0
        weight21 = torch.zeros(planes_sta, 1, 1, 1, 1)
        weight21[:, 0, 0] = 1.0

        self.conv1d11.weight = nn.Parameter(weight11)
        self.conv1d13.weight = nn.Parameter(weight13)
        self.conv1d15.weight = nn.Parameter(weight15)
        self.conv1d17.weight = nn.Parameter(weight17)
        self.conv1d21.weight = nn.Parameter(weight21)
        self.conv1d23.weight = nn.Parameter(weight23)
        self.conv1d25.weight = nn.Parameter(weight25)
        self.conv1d27.weight = nn.Parameter(weight27)

        size = self.weight2d.size()
        weight2d1 = self.weight2d[:size[0] // (self.fold_div // 2), : size[1] // (self.fold_div // 2), :, :]
        weight2d2 = self.weight2d[size[0] // (self.fold_div // 2):, size[1] // (self.fold_div // 2):, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)

    def msgroupconv1d(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x2, x3, x4, x5, x6, x7, x8 = x.split([self.c_dy, self.c_dy, self.c_dy, self.c_dy,
                                                  self.c_sta, self.c_sta, self.c_sta, self.c_sta], dim=1)
        x1 = self.conv1d11(x1)
        x2 = self.conv1d13(x2)
        x3 = self.conv1d15(x3)
        x4 = self.conv1d17(x4)
        x5 = self.conv1d21(x5)
        x6 = self.conv1d23(x6)
        x7 = self.conv1d25(x7)
        x8 = self.conv1d27(x8)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(nt, c, h, w)

    def forward(self, x):
        x = self.msgroupconv1d(x)
        nt, c, h, w = x.size()
        x1, x2 = x.split([c // (self.fold_div // 2), c - (c // (self.fold_div // 2))], dim=1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x = torch.cat((x1, x2), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


class MsConv1dPartialSplitdiv(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8):
        super(MsConv1dPartialSplitdiv, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))

        c_in_dy = (net.in_channels // 4) // (n_div // 2)

        self.c_dy = c_in_dy

        self.conv1d11 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d13 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d15 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=c_in_dy)
        self.conv1d17 = nn.Conv3d(c_in_dy, c_in_dy, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=c_in_dy)

        self.weight2d = net.weight
        self.stride = net.stride
        planes22d = net.in_channels

        self.conv21 = nn.Conv2d(planes22d // (n_div // 2), planes22d // (n_div // 2), kernel_size=3, stride=self.stride,
                               padding=1, bias=False)
        self.conv22 = nn.Conv2d(planes22d - (planes22d // (n_div // 2)), planes22d - (planes22d // (n_div // 2)),
                                kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.weight_init()  #

    def weight_init(self):
        print('=> Using weight init of 3 parts')
        planes_dy = self.conv1d11.in_channels

        weight13 = torch.zeros(planes_dy, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight13[:planes_dy // 2, 0, 0] = 1.0
        weight13[planes_dy // 2:, 0, 2] = 1.0

        weight15 = torch.zeros(planes_dy, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight15[:planes_dy // 2, 0, :2] = 1.0  # [11000]
        weight15[planes_dy // 2:, 0, 3:] = 1.0  # [00011]

        weight17 = torch.zeros(planes_dy, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
        weight17[:planes_dy // 2, 0, :3] = 1.0  # [1110000]
        weight17[planes_dy // 2:, 0, 4:] = 1.0  # [0000111]

        weight11 = torch.zeros(planes_dy, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
        weight11[:, 0, 0] = 1.0

        self.conv1d11.weight = nn.Parameter(weight11)
        self.conv1d13.weight = nn.Parameter(weight13)
        self.conv1d15.weight = nn.Parameter(weight15)
        self.conv1d17.weight = nn.Parameter(weight17)

        size = self.weight2d.size()
        weight2d1 = self.weight2d[:size[0] // (self.fold_div // 2), : size[1] // (self.fold_div // 2), :, :]
        weight2d2 = self.weight2d[size[0] // (self.fold_div // 2):, size[1] // (self.fold_div // 2):, :, :]
        self.conv21.weight = nn.Parameter(weight2d1)
        self.conv22.weight = nn.Parameter(weight2d2)

    def msgroupconv1d(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x2, x3, x4 = x.split([self.c_dy, self.c_dy, self.c_dy, self.c_dy], dim=1)
        x1 = self.conv1d11(x1)
        x2 = self.conv1d13(x2)
        x3 = self.conv1d15(x3)
        x4 = self.conv1d17(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(nt, c, h, w)

    def forward(self, x):
        nt, c, h, w = x.size()
        x1, x2 = x.split([c // (self.fold_div // 2), c - (c // (self.fold_div // 2))], dim=1)
        x1 = self.msgroupconv1d(x1)
        x1 = self.conv21(x1)
        x2 = self.conv22(x2)
        x = torch.cat((x1, x2), dim=1).view(nt, c, h // self.stride[0], w // self.stride[0])
        return x


def make_operations(net, n_segment, n_div=8, operations='baseline', dwise=False, corr_group=1, inplace=False):
    if operations == 'baseline':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = GroupConv1d(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_group1d':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MsGroupConv1d(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_group1douter':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, inplace=False)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_corr':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment, corr=False):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if corr:
                        stride = b.conv2.stride[0]
                        blocks[i] = MsCorrBlock(b, n_segment=this_segment, n_div=n_div, stride=stride,
                                                corr_group=corr_group, reduction=4)
                    else:
                        blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment, corr=True)
            net.layer3 = make_block_temporal(net.layer3, n_segment, corr=True)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_corr_cascade':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment, corr=False):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i == 0 and corr:
                        in_channels = blocks[i].conv1.in_channels
                        blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div)
                        blocks[i] = CorrBlock_cascade(b, n_segment=this_segment, n_div=n_div,
                                                      in_channels=in_channels, dwise=dwise, corr_group=corr_group)
                    else:
                        blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment, corr=True)
            net.layer3 = make_block_temporal(net.layer3, n_segment, corr=True)
            net.layer4 = make_block_temporal(net.layer4, n_segment, corr=True)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_groupouter_nl':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'stm':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = STM(b, n_segment=this_segment, n_div=n_div, inplace=inplace)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_corrall_r16cs5':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment, corr=False):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if corr:
                        stride = b.conv2.stride[0]
                        blocks[i] = MsCorrBlock(b, n_segment=this_segment, n_div=n_div, stride=stride,
                                                corr_group=corr_group, reduction=16, corr_size=5)
                    else:
                        blocks[i].conv1 = MsGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div)

                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment, corr=True)
            net.layer2 = make_block_temporal(net.layer2, n_segment, corr=True)
            net.layer3 = make_block_temporal(net.layer3, n_segment, corr=True)
            net.layer4 = make_block_temporal(net.layer4, n_segment, corr=True)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dk5':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, kernel_size=(5, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dk7':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, kernel_size=(7, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dsplit':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = GroupConv1dSplit(b.conv2, n_segment=this_segment, n_div=n_div, kernel_size=(3, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'msgroup1dsplit':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MSGroupConv1dSplit(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'gst':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = GST(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dk3outer':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, kernel_size=(3, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dk3pcbam':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1dPcbam(b.conv1, n_segment=this_segment, n_div=n_div, kernel_size=(3, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dsplitdiv1x1':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1dSplit1x1(b.conv1, n_segment=this_segment, n_div=n_div)
                    blocks[i].conv2 = GroupConv1dSplitdiv(b.conv2, n_segment=this_segment, n_div=n_div, kernel_size=(3, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'group1dsplitdiv':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = GroupConv1dSplitdiv(b.conv2, n_segment=this_segment, n_div=n_div, kernel_size=(3, 1, 1))
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'msgroup1dsplitdiv1x1':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = GroupConv1dSplit1x1(b.conv1, n_segment=this_segment, n_div=n_div)
                    blocks[i].conv2 = MsConv1dSplitdiv(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'msgroup1dsplitdiv':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MsConv1dSplitdiv(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'msgroup1dpartalsplitdiv':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MsConv1dPartialSplitdiv(b.conv2, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    else:
        raise NotImplementedError(operations)


if __name__ == '__main__':
    import torchvision
    # from ops.non_local import make_non_local
    model = torchvision.models.resnet50(True)
    # import arch as ac
    # basemodel = 'resnet50'
    # model = getattr(ac,basemodel)(True)
    # model = resnet50(False)
    make_operations(model, 8, n_div=8, operations='msgroup1dpartalsplitdiv')
    data = torch.autograd.Variable(torch.ones(16, 3, 320, 256))

    # make_non_local(model, 8)
    # test weight init
    parm = {}
    for name,parameters in model.layer2.named_parameters():
        print(name, ':', parameters.size())
        parm[name] = parameters.detach()
    layer = '2.conv2.conv1d13.weight'

    # layer2 = '2.conv2.conv1d23.weight'
    size = parm[layer].size()
    # print(parm[layer][: size[0] // 4, :, :, :, :])
    # print(parm[layer][size[0] // 4: size[0] // 2, :, :, :, :])
    # print(parm[layer][size[0] // 2:, :, :, :, :])
    print(parm[layer][: size[0] // 2, :, :, :, :])
    print(parm[layer][size[0] // 2:, :, :, :, :])
    # print(parm[layer2][:, :, :, :, :])

    out = model(data)
    out.mean().backward()
    print(model)
    print(out.size())





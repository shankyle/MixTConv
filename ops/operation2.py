# Code for "CFM"
# arXiv:TBD
# Kaiyu Shan
# shankyle@pku.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import *
from spatial_correlation_sampler import SpatialCorrelationSampler

class GroupConv1d(nn.Module):
    def __init__(self, net, n_segment=8, n_div=8, inplace=True):
        super(GroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        if inplace:
            print('=> Using in-place group 1d conv...')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(net.in_channels, net.in_channels, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]
        self.net = net

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

    def forward(self, x):
        x = self.groupconv1d(x)
        return self.net(x)


class MsCfm(nn.Module):
    def __init__(self, net, n_segment=8, n_div=4, inplace=True):
        super(MsCfm, self).__init__()
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
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.in_channels // 4)

        self.conv21d = nn.Conv2d(net.in_channels // 4, net.in_channels // 4,
                                 kernel_size=3, padding=1, stride=net.stride, bias=False)
        self.conv23d = nn.Conv2d(net.in_channels // 4, net.in_channels // 4,
                                 kernel_size=3, padding=1, stride=net.stride, bias=False)
        self.conv25d = nn.Conv2d(net.in_channels // 4, net.in_channels // 4,
                                 kernel_size=3, padding=1, stride=net.stride, bias=False)
        self.conv27d = nn.Conv2d(net.in_channels // 4, net.in_channels // 4,
                                 kernel_size=3, padding=1, stride=net.stride, bias=False)
        self.stride = net.stride[0]

    def ms_cfm(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x1, x3, x5, x7 = x.split([c // 4, c // 4, c // 4, c // 4], dim=1)

        x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)

        # x => [N, T, C, H, W]
        x1 = self.conv21d(x1).view(n_batch, -1, c // 4, h // self.stride, w // self.stride).contiguous()
        x3 = self.conv23d(x3).view(n_batch, -1, c // 4, h // self.stride, w // self.stride).contiguous()
        x5 = self.conv25d(x5).view(n_batch, -1, c // 4, h // self.stride, w // self.stride).contiguous()
        x7 = self.conv27d(x7).view(n_batch, -1, c // 4, h // self.stride, w // self.stride).contiguous()

        index5 = [0, 4, 1, 5, 2, 6, 3, 7]
        index7 = [0, 2, 4, 6, 1, 3, 5, 7]
        x5 = x5.repeat(1, 2, 1, 1, 1)[:, index5, :, :, :]
        x7 = x7.repeat(1, 4, 1, 1, 1)[:, index7, :, :, :]
        x = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h // self.stride, w // self.stride)

        return x

    # TBD: weight init and transform

    def forward(self, x):
        x = self.ms_cfm(x)
        return x


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


class MsCEGroupConv1d(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, inplace=True, stride=(1, 1, 1), upsample=False, mode='nearest'):
        super(MsCEGroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        self.stride = stride
        self.upsample = upsample
        self.mode = mode
        if inplace:
            print('=> Using in-place multi-scale 1d CE conv...')
        print('=> Using fold div: {}'.format(self.fold_div))

        self.conv13d = nn.Conv3d(net.in_channels // 2, net.in_channels // 2, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 2, stride=stride)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4, stride=stride)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4, stride=stride)
        self.net = net
        self.weight_init()

    def ms_ce(self, x):
        if self.upsample:
            nt, c, h, w = x.size()
            n_batch = nt // (self.n_segment // 2)
            x = x.view(n_batch, self.n_segment // 2, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
            x = nn.functional.interpolate(x, scale_factor=(2, 1, 1), mode=self.mode)  # x => [N, C, 2 * T, H, W]
            nt = nt * 2
        else:
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment
            x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x3, x5, x7 = x.split([c // 2, c // 4, c // 4], dim=1)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous()
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous()
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous()

        x = torch.cat((x3, x5, x7), dim=2).view(nt // self.stride[0], c, h, w)

        return x

    # TBD: weight init and transform
    def weight_init(self):
        print('=> Using weight init of 3 parts for 3 multi-scale')
        planes = self.conv15d.in_channels
        fold = planes // self.fold_div  # div = 4

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes * 2, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]1/2 [100]1/4 [110]1/4
        weight3[:2 * fold, 0, 0] = 1.0
        weight3[2 * fold: fold * 4, 0, 2] = 1.0
        weight3[fold * 4:, 0, 1] = 1.0
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
        x = self.ms_ce(x)
        return self.net(x)


class MsCERes(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, inplace=True, mode='nearest', stride=1):
        super(MsCERes, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        self.stride = stride
        self.mode = mode
        if inplace:
            print('=> Using in-place multi-scale 1d CE conv...')
        print('=> Using fold div: {}'.format(self.fold_div))

        self.conv13d = nn.Conv3d(net.in_channels // 2, net.in_channels // 2, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.in_channels // 2, stride=1)
        self.conv15d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.in_channels // 4, stride=1)
        self.conv17d = nn.Conv3d(net.in_channels // 4, net.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.in_channels // 4, stride=1)
        self.avgpool3d = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.net = net
        self.weight_init()

    def ms_ce(self, x, residual):

        nt, c, h, w = x.size()
        n_batch = nt // (self.n_segment // 2)
        x = x.view(n_batch, self.n_segment // 2, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = nn.functional.interpolate(x, scale_factor=(2, 1, 1), mode=self.mode)  # x => [N, C, 2 * T, H, W]
        nt = nt * 2
        if self.stride != 1:
            residual = self.avgpool3d(residual.view(n_batch, self.n_segment, c, h*2, w*2).permute(0, 2, 1, 3, 4).contiguous())
            x = x + residual
        else:
            x = x + residual.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        x3, x5, x7 = x.split([c // 2, c // 4, c // 4], dim=1)
        x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous()
        x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous()
        x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous()

        x = torch.cat((x3, x5, x7), dim=2).view(nt, c, h, w)

        return x

    # TBD: weight init and transform
    def weight_init(self):
        print('=> Using weight init of 3 parts for 3 multi-scale')
        planes = self.conv15d.in_channels
        fold = planes // self.fold_div  # div = 4

        # diff 1357 = shift + stride 0 2 4 6
        weight3 = torch.zeros(planes * 2, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]1/2 [100]1/4 [110]1/4
        weight3[:2 * fold, 0, 0] = 1.0
        weight3[2 * fold: fold * 4, 0, 2] = 1.0
        weight3[fold * 4:, 0, 1] = 1.0
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

    def forward(self, x, residual):
        x = self.ms_ce(x, residual)
        return self.net(x)


class MsCorrBlock(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, inplace=True, mode='nearest', stride=1,
                 corr_size=7, corr_group=1):
        super(MsCorrBlock, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        self.stride = stride
        self.mode = mode
        if inplace:
            print('=> Using in-place correlation')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv13d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv15d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv17d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # TBD: bias = False, weight init
        self.conv21 = nn.Conv3d(net.conv1.in_channels, net.conv1.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=True)
        # kkgroup 要计算一下
        self.kkgroup = corr_size * corr_size * corr_group
        self.corr = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=7,
            stride=1,
            padding=0,
            dilation_patch=1,
        )

        self.conv22 = nn.Conv2d(self.kkgroup, net.conv3.out_channels, kernel_size=(1, 1),
                                padding=(0, 0), bias=True)
        self.bn21 = nn.BatchNorm3d(net.conv1.in_channels // 4)
        self.bn22 = nn.BatchNorm2d(net.conv3.out_channels)
        self.relu21 = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv21.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv21(x)
        x = self.bn21(x)
        # x = self.relu21(x)
        index= list(range(1, self.n_segment))
        index.append(self.n_segment - 1)
        x2 = x[:, :, index, :, :]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt, c // 4, h, w)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt, c // 4, h, w)
        x = self.corr(x, x2).contiguous().view(nt, self.kkgroup, h, w) / c

        x = self.conv22(x)
        x = self.bn22(x)
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

        out = self.ms_group(x)

        out_corr = self.corr_outer(x)
        if self.stride != 1:
            out_corr = self.avgpool2d(out_corr)

        out = self.net.conv1(out)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        out += residual

        out += out_corr

        out = self.net.relu(out)
        return out


class MsCorrBlock2(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=8, inplace=True, mode='nearest', stride=(1, 1, 1),
                 corr_size=7, corr_group=1):
        super(MsCorrBlock2, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.inplace = inplace
        self.stride = stride
        self.mode = mode
        if inplace:
            print('=> Using in-place correlation')
        print('=> Using fold div: {}'.format(self.fold_div))

        self.conv11d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv13d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv15d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.conv17d = nn.Conv3d(net.conv1.in_channels // 4, net.conv1.in_channels // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=net.conv1.in_channels // 4, stride=(1, 1, 1))
        self.avgpool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # TBD: bias = False, weight init
        self.conv21 = nn.Conv3d(net.conv2.in_channels, net.conv2.in_channels // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False)
        # kkgroup 要计算一下
        self.kkgroup = corr_size * corr_size * corr_group

        self.conv22 = nn.Conv2d(self.kkgroup, net.conv2.out_channels, kernel_size=(1, 1),
                                padding=(0, 0), bias=False)
        self.bn21 = nn.BatchNorm3d(net.conv2.in_channels // 4)
        self.bn22 = nn.BatchNorm2d(net.conv2.out_channels)
        self.relu21 = nn.ReLU(inplace=True)

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
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)
        index= list(range(1, self.n_segment))
        index.append(self.n_segment - 1)
        x2 = x[:, :, index, :, :]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt, c // 4, h, w)
        x2 = x2.permute(0, 2, 1, 3, 4).contiguous().view(nt, c // 4, h, w)
        x = self.corr(x, x2).contiguous().view(nt, self.kkgroup, h, w) / c

        x = self.conv22(x)
        x = self.bn22(x)
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
        residual = x
        if self.net.downsample is not None:
            residual = self.net.downsample(x)

        out = self.ms_group(x)

        out_corr = self.corr_outer(x)

        out = self.net.conv1(out)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        out += residual

        out += out_corr
        out = self.net.relu(out)

        return out


class UpSample(nn.Module):  # TBD
    def __init__(self, net, n_segment=8, n_div=4, mode='nearest'):
        super(UpSample, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        self.net = net
        self.mode = mode
        print("Using {} upsample".format(self.mode))

    def forward(self, x):
        x = self.net(x)
        nt, c, h, w = x.size()
        n_batch = nt // (self.n_segment // 2)
        x = x.view(n_batch, -1, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
        x = nn.functional.interpolate(x, scale_factor=(2, 1, 1), mode=self.mode)  # x => [N, C, 2*T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(nt * 2, c, h, w)
        return x


def make_operations(net, n_segment, n_div=8, operations='baseline'):
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

    elif operations == 'ms_cfm':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MsCfm(b.conv2, n_segment=this_segment, n_div=n_div)
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

    elif operations == 'ms_group1d2x':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv2 = MsGroupConv1d(b.conv2, n_segment=this_segment, n_div=n_div)
                    blocks[i].conv3 = MsGroupConv1d(b.conv3, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_ceneargroup1d2x':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = MsCEGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, stride=(2, 1, 1))
                    blocks[i].bn3 = UpSample(b.bn3, n_segment=this_segment, n_div=n_div, mode='nearest')
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment)
            net.layer2 = make_block_temporal(net.layer2, n_segment)
            net.layer3 = make_block_temporal(net.layer3, n_segment)
            net.layer4 = make_block_temporal(net.layer4, n_segment)
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    elif operations == 'ms_ceneargroup1diner':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        if isinstance(net, torchvision.models.ResNet):
        # if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = MsCEGroupConv1d(b.conv1, n_segment=this_segment, n_div=n_div, stride=(2, 1, 1))
                    blocks[i].conv3 = MsCEGroupConv1d(b.conv3, n_segment=this_segment, n_div=n_div, upsample=True, mode='nearest')

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

    elif operations == 'ms_cenearres':
        print('=> Using Operations as {}'.format(operations))
        import torchvision
        # if isinstance(net, torchvision.models.ResNet):
        if isinstance(net, ResNet):
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    stride = b.conv2.stride[0]
                    blocks[i].conv2 = MsCEGroupConv1d(b.conv2, n_segment=this_segment, n_div=n_div, stride=(2, 1, 1))
                    blocks[i].conv3 = MsCERes(b.conv3, n_segment=this_segment, n_div=n_div, mode='nearest', stride=stride)

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
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    stride = b.conv2.stride[0]
                    blocks[i] = MsCorrBlock(b, n_segment=this_segment, n_div=n_div, stride=stride)

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
    model = torchvision.models.resnet50(True)
    # import arch as ac
    # basemodel = 'resnet50'
    # model = getattr(ac,basemodel)(True)
    # model = resnet50(False)
    make_operations(model, 8, n_div=8, operations='ms_corr')
    data = torch.autograd.Variable(torch.ones(16, 3, 320, 256))

    # test weight init
    # parm = {}
    # for name,parameters in model.layer1.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach()
    # layer = '2.conv2.conv17d.weight'
    # size = parm[layer].size()
    # print(parm[layer][: size[0] // 4, :, :, :, :])
    # print(parm[layer][size[0] // 4: size[0] // 2, :, :, :, :])
    # print(parm[layer][size[0] // 2:, :, :, :, :])
    out = model(data)
    print(model)
    print(out.size())





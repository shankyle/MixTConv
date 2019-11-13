# Code for "CFM"
# arXiv:TBD
# Kaiyu Shan
# shankyle@pku.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
# from arch import *
from spatial_correlation_sampler import SpatialCorrelationSampler
import math


class GroupConv1d(nn.Module):
    def __init__(self, c_in, n_segment=8, n_div=8):
        super(GroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv1d = nn.Conv3d(c_in, c_in, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in)
        self.weight_init()  # init as [0,0,1], [1,0,0], [0,1,0]

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


class MsGroupConv1d(nn.Module):
    def __init__(self, c_in, n_segment=8, n_div=4):
        super(MsGroupConv1d, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div  # each shift part
        print('=> Using fold div: {}'.format(self.fold_div))
        self.conv11d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(1, 1, 1),
                                padding=(0, 0, 0), bias=False, groups=c_in // 4)
        self.conv13d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), bias=False, groups=c_in // 4)
        self.conv15d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(5, 1, 1),
                                padding=(2, 0, 0), bias=False, groups=c_in // 4)
        self.conv17d = nn.Conv3d(c_in // 4, c_in // 4, kernel_size=(7, 1, 1),
                                padding=(3, 0, 0), bias=False, groups=c_in // 4)
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
            print('=> Using in-place group 1d conv...')
        print('=> Using fold div: {}'.format(self.fold_div))
        self.group1d = GroupConv1d(c_in, n_segment=8, n_div=8)
        # self.conv1d = nn.Conv3d(c_in, c_in, kernel_size=(3, 1, 1),
        #                         padding=(1, 0, 0), bias=False, groups=c_in)

        self.conv1 = nn.Conv3d(c_in, c_in // self.reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_in // self.reduction)
        self.conv2 = nn.Conv3d(c_in // self.reduction, c_in // self.reduction, kernel_size=(1, 3, 3),
                               stride=1, padding=(0, 1, 1), bias=False, groups=c_in // self.reduction)

        self.conv3 = nn.Conv3d(c_in // self.reduction,  c_in, kernel_size=1, bias=False)

        if self.stride != 1:
            self.maxpooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.net = net

    # TBD: init
    def cmm(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]

        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)

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
            x = self.group1d.groupconv1d(x)
        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)

        out2 = self.cmm(out)
        if self.inplace:
            out = self.group1d.groupconv1d(out)

        if self.stride != 1:
            out2 = self.maxpooling(out2)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = out + out2
        out = self.net.conv3(out)
        out = self.net.bn3(out)

        out += residual
        out = self.net.relu(out)

        return out


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
    make_operations(model, 8, n_div=8, operations='stm')
    data = torch.autograd.Variable(torch.ones(16, 3, 320, 256))

    # make_non_local(model, 8)
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
    out.mean().backward()
    print(model)
    print(out.size())





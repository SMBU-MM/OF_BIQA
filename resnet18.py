# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:26:23 2020

@author: Zhihua WANG
"""
import torch
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = model.state_dict()
        pretrained_state_dict = torch.load('resnet18-5c106cde.pth')
        for key, _ in pretrained_state_dict.items():
            if not 'fc' in key:
                state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data)
    elif classname.find('Gdn2d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
    elif classname.find('Gdn1d') != -1:
        init.eye_(m.gamma.data)
        init.constant_(m.beta.data, 1e-4)
        
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class net_quality(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self):
        super(net_quality, self).__init__()
        layers1 = [nn.Linear(512, 256, bias=True)]
        self.net1 = nn.Sequential(*layers1)
        self.net1.apply(weights_init)
        layers2 = [ 
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(256, 128, bias=True)
                ]
        self.net2 = nn.Sequential(*layers2)
        self.net2.apply(weights_init)

        layers3 = [ 
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(128, 2, bias=True)
                ]
        self.net3 = nn.Sequential(*layers3)
        self.net3.apply(weights_init)

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        mean = x3[:, 0].unsqueeze(dim=-1)
        #var = torch.ones(mean.shape[0], mean.shape[1]).cuda()
        var = nn.functional.softplus(x3[:,1]).unsqueeze(dim=-1)
        # var = torch.exp(r[:, 1]).unsqueeze(dim=-1)
        return mean, var, x1, x2

class net_annotators(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self):
        super(net_annotators, self).__init__()
        layers =[
                    nn.Linear(1024, 256, bias=True),
                    nn.ReLU(),
                    nn.Linear(256, 128, bias=True),
                    nn.ReLU(),
                    nn.Linear(128, 1, bias=True),
                    nn.Sigmoid()
                ]
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, x):
        annotators = self.net(x)
        return annotators

class net_domain(nn.Module):
    # end-to-end unsupervised image quality assessment model
    def __init__(self):
        super(net_domain, self).__init__()
        # Dimension Reduction - feature
        layers = [
                    nn.Linear(512, 256, bias=True),
                    nn.ReLU(),
                    nn.Linear(256, 128, bias=True),
                    nn.ReLU(),
                    nn.Linear(128, 1, bias=True),
                    nn.Sigmoid()
                ]
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)
    
    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        domain = self.net(x)
        return domain.view(-1,1).squeeze(1)

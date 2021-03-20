# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 02:45:17 2020

@author: Zhihua WANG
"""
import os
import torch
import torch.nn.functional as F

eps = 1e-8

class FocalBCELoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class Binary_Loss(torch.nn.Module):

    def __init__(self):
        super(Binary_Loss, self).__init__()

    def forward(self, p, g, alpha, beta):
        log_a = g * torch.log(alpha) + (1 - g) * torch.log(1 - alpha)
        log_a = torch.sum(log_a, dim=1, keepdim=True)
        a = torch.exp(log_a)

        log_b = (1 - g) * torch.log(beta) + g * torch.log(1 - beta)
        log_b = torch.sum(log_b, dim=1, keepdim=True)
        b = torch.exp(log_b)
        loss_val = torch.log(a * p + b * (1 - p)) / g.size()[1]
        return -torch.mean(loss_val), loss_val

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None


    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

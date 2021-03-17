# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 01:35:41 2021

@author: Zhihua WANG
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from resnet18 import resnet18 as netF
from resnet18 import net_quality as netQ
from Transformers import AdaptiveResize

test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
         
netF = nn.DataParallel(netF(pretrained=False).cuda())
netF.load_state_dict(torch.load("weights_adapting_SPAQ.pt")['netF_dict'])

netQ = nn.DataParallel(netQ().cuda())
netQ.load_state_dict(torch.load("weights_adapting_SPAQ.pt")['netQ_dict'])

netF.eval()
netQ.eval()

img_name = "test_img.jpg"
img = test_transform(Image.open(img_name).convert("RGB")).unsqueeze(0)
y_bar,_,_,_ = netQ(netF(img))
y_bar = y_bar.cpu().item()

print(y_bar)

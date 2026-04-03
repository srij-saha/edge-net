import torch
import torch.nn as nn
import math
from collections import OrderedDict
from .layers import Flatten, Unsqueeze, TransposeReshapeLayer

__all__ = ['VanillaEncoder512', 'VanillaEncoder510']


class VanillaEncoder512(nn.Sequential):
    def __init__(self, inplace=True, projection=False):
        
        layers = OrderedDict([
            ('init', nn.Sequential(
               nn.Conv2d(1, 32, kernel_size=(31, 31), stride=1, padding=15), # 512
               nn.ReLU(inplace=inplace))),
            ('l1', nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(7, 7), stride=2, padding=3),   # 256
                nn.ReLU(inplace=inplace))),
            ('l2', nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(5, 5), stride=2, padding=2),  # 128
                nn.ReLU(inplace=inplace))),
            ('l3', nn.Sequential(
                nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1),  # 64
                nn.ReLU(inplace=inplace))),
            ('l4', nn.Sequential(
                nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1),  # 32
                nn.ReLU(inplace=inplace))),
            ('l5', nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=2, padding=1),  # 16
                nn.ReLU(inplace=inplace))),
            ('l6', nn.Sequential(
                nn.Conv2d(384, 512, kernel_size=(3, 3), stride=2, padding=1),  # 8
                nn.ReLU(inplace=inplace))),
            ('lx1', nn.Sequential(
                nn.Conv2d(512, 768, kernel_size=(3, 3), stride=(1,1), padding=0),  # 6
                nn.ReLU(inplace=inplace),
                nn.Conv2d(768, 1024, kernel_size=(3, 3), stride=(1,1), padding=0),  # 4
                nn.ReLU(inplace=inplace)
            ))
        ])
        
        if projection:
            layers['projection'] = nn.Sequential(
                Flatten(),
                nn.Linear(4*4*1024, 128, bias=False),
                nn.BatchNorm1d(128, eps=1e-05, momentum=1-0.9, affine=True, track_running_stats=True)
            )
        
        super().__init__(layers)
        
class VanillaEncoder510(nn.Sequential):
    def __init__(self, inplace=True, projection=False):
        
        layers = OrderedDict([
            ('init', nn.Sequential(
               nn.Conv2d(1, 32, kernel_size=(31, 31), stride=1, padding=16), # 512
               nn.ReLU(inplace=inplace))),
            ('l1', nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(7, 7), stride=2, padding=3),   # 256
                nn.ReLU(inplace=inplace))),
            ('l2', nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(5, 5), stride=2, padding=2),  # 128
                nn.ReLU(inplace=inplace))),
            ('l3', nn.Sequential(
                nn.Conv2d(128, 192, kernel_size=(3, 3), stride=2, padding=1),  # 64
                nn.ReLU(inplace=inplace))),
            ('l4', nn.Sequential(
                nn.Conv2d(192, 256, kernel_size=(3, 3), stride=2, padding=1),  # 32
                nn.ReLU(inplace=inplace))),
            ('l5', nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=(3, 3), stride=2, padding=1),  # 16
                nn.ReLU(inplace=inplace))),
            ('l6', nn.Sequential(
                nn.Conv2d(384, 512, kernel_size=(3, 3), stride=2, padding=1),  # 8
                nn.ReLU(inplace=inplace))),
            ('lx1', nn.Sequential(
                nn.Conv2d(512, 768, kernel_size=(3, 3), stride=(1,1), padding=0),  # 6
                nn.ReLU(inplace=inplace),
                nn.Conv2d(768, 1024, kernel_size=(3, 3), stride=(1,1), padding=0),  # 4
                nn.ReLU(inplace=inplace)
            ))
        ])
        
        if projection:
            layers['projection'] = nn.Sequential(
                Flatten(),
                nn.Linear(4*4*1024, 128, bias=False),
                nn.BatchNorm1d(128, eps=1e-05, momentum=1-0.9, affine=True, track_running_stats=True)
            )
        
        super().__init__(layers)
        

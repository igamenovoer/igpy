import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.misc import ConvNormActivation
from ..helpers import exec_sequential_model_and_print

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=None):
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1)//2
            
        self.m_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding)
        self.m_norm = nn.BatchNorm2d(out_channels)
        self.m_activate = nn.ReLU6()
        
    def forward(self, x):
        y = self.m_conv(x)
        y = self.m_norm(y)
        y = self.m_activate(y)
        return y
    
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expantion_factor : int, stride):
        super().__init__()
        n_mid_channels = in_channels * expantion_factor
        
        self.m_conv_in = Conv_BN_ReLU(in_channels, n_mid_channels, 1)
        self.m_conv_dw = Conv_BN_ReLU(n_mid_channels, n_mid_channels, 3, stride=stride, groups=n_mid_channels)
        self.m_conv_out = nn.Conv2d(n_mid_channels, out_channels, 1)
        self.m_num_in = in_channels
        self.m_num_out = out_channels
        self.m_stride = stride
        
    def forward(self, x):
        y = self.m_conv_in(x)
        y = self.m_conv_dw(y)
        y = self.m_conv_out(y)
        
        if self.m_num_in == self.m_num_out and self.m_stride==1:
            return x+y
        else:
            return y

class MobilenetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_channel_width_multiplier = 1.0
        self.m_output_channel = 1
        self.m_input_channel = 3
        self.m_model : nn.Sequential = None
        
    def set_num_input_features(self, channel:int):
        self.m_input_size = channel
        
    def set_num_output_features(self, n:int):
        self.m_output_channel = n
        
    def set_channel_width_multiplier(self, s:float):
        self.m_channel_width_multiplier = s
        
    def build_model(self) -> nn.Sequential:
        e = self.m_channel_width_multiplier
        blocks = []
        
        # t, c, n, s, kern
        # t = block-channel-expansion
        # c = out channels
        # n = number of repeating blocks
        # s = stride
        # kern = kernel size
        # padding = padding size
        n_final_channel = self.m_output_channel
        layer_config = [
            (nn.Conv2d, -1, 32, 1, 2, 3),
            (InvertedResidual, 1,16,1,1, 3),
            (InvertedResidual, 6,24,2,2, 3),
            (InvertedResidual, 6,32,3,2, 3),
            (InvertedResidual, 6,64,4,2, 3),
            (InvertedResidual, 6,96,3,1, 3),
            (InvertedResidual, 6,160,3,2, 3),
            (InvertedResidual, 6,320,1,1, 3),
            (nn.Conv2d, -1, 1280, 1, 1, 1),
            (nn.AvgPool2d, -1, 1280, 1, 1, 7),
            (nn.Conv2d, -1, n_final_channel, 1, 1, 1)
        ]
        
        in_features = self.m_input_channel
        for i, cfg in enumerate(layer_config):
            layer, invres_chn_expansion, out_features, num_repeat, stride, kern = cfg
            if i < len(layer_config)-1:
                out_features = np.round(out_features * e).astype(int)
            
            for k in range(num_repeat):
                
                # when repeating, do not repeat stride
                if k==0:
                    s = stride
                else:
                    s = 1
                    
                if layer == nn.Conv2d:
                    obj = nn.Conv2d(in_features, out_features, kern, s, padding=(kern-1)//2)
                elif layer == InvertedResidual:
                    obj = InvertedResidual(in_features, out_features, invres_chn_expansion, s)
                elif layer == nn.AvgPool2d:
                    obj = nn.AvgPool2d(kern)
                    
                in_features = out_features
                blocks.append(obj)
                
        self.m_model = nn.Sequential(*blocks)
        return self.m_model

    def forward(self, x):
        out = exec_sequential_model_and_print(self.m_model, x)
        return out
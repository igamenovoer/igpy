import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import orthogonal
from typing import Union

class Conv2d_BN_Res(nn.Module):
    def __init__(self, conv_layer : nn.Conv2d):
        super().__init__()
        self.m_layer_conv2d : nn.Conv2d = conv_layer
        self.m_layer_bn : nn.BatchNorm2d = nn.BatchNorm2d(conv_layer.out_channels)
    
    def forward(self, x):
        y = self.m_layer_conv2d(x)
        y = self.m_layer_bn(y)
        return y+x
    
class MultivariableGaussianLayer(nn.Module):
    def __init__(self, ndim : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_mean = nn.Parameter(torch.randn(ndim))
        sigma_gen = nn.Parameter(torch.randn(ndim, ndim))
        self.m_sigma_gen = sigma_gen
        self.m_coeff = nn.Parameter(torch.tensor([1.0]))
        
    @property
    def sigma(self):
        return self.m_sigma_gen & self.m_sigma_gen.t()
        
    def forward(self, x):
        y = torch.distributions.MultivariateNormal(self.m_mean, self.sigma).log_prob(x) + torch.log(self.m_coeff)
        return torch.exp(y)
    
class SumOfGaussian(nn.Module):
    def __init__(self, n_component : int, n_dim :int, segment_size : int = 10000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m_mean = nn.Parameter(torch.randn(n_component, n_dim))
        self.m_coef = nn.Parameter(torch.randn(n_component))
        
        # sigma[i] = rot_layers[i] @ diag(singular_values[i]**2) @ rot_layers[i].t()
        self.m_rot_layers = nn.ModuleList([orthogonal(nn.Linear(n_dim, n_dim)) for _ in range(n_component)])
        sgvals = np.sqrt(np.abs(np.random.randn(n_component, n_dim)).clip(1.0))
        self.m_singular_values_sqrt = nn.Parameter(torch.tensor(sgvals).float())
        self.m_segment_size = segment_size # compute this many points per round
        
        # freezed sigma in inference mode
        self.m_is_training : bool = True
        self.m_infer_sigma = nn.Parameter(torch.randn(n_component,n_dim, n_dim))   # for inference only
        self.m_infer_sigma.requires_grad_(False)
        
    @property
    def sigma(self) -> torch.Tensor:
        if not self.m_is_training:
            return self.m_infer_sigma
        else:
            sigma_per_comp = []
            for i in range(self.num_component):
                rot = self.m_rot_layers[i].weight
                s = torch.diag(self.m_singular_values_sqrt[i] ** 2)
                sigma_per_comp.append(rot @ s @ rot.t())
            return torch.stack(sigma_per_comp)
    
    @property
    def num_component(self) -> int:
        return int(self.m_mean.shape[0])
    
    @property
    def num_dim(self) -> int:
        return int(self.m_mean.shape[1])
    
    @property
    def segment_size(self) -> int:
        return self.m_segment_size
    
    @property
    def is_training(self) -> bool:
        return self.m_is_training
    
    def set_segment_size(self, segment_size : int):
        ''' set the number of points to be computed per round in forward(), a smaller number will reduce memory usage but increase computation time.
        Segment size does not affect the result, only the speed and memory usage.
        '''
        self.m_segment_size = segment_size
    
    def set_mean(self, mean : np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(mean).float().to(self.m_mean.device)
            self.m_mean.data = x
        
    def set_coef(self, coef : np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(coef).float().to(self.m_coef.device)
            self.m_coef.data = x
            
    def set_training_mode(self, is_training : bool):
        ''' set whether we are in training mode or inference mode. In inference mode, the sigma is freezed and not updated.
        '''
        self.m_is_training = is_training
        if not is_training:
            self.m_infer_sigma.data = self.sigma.data
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ''' compute sum of gaussians over input x
        
        parameters
        --------------
        x: (batch_size, n_dim)
            points where sum of gaussians are computed
        
        return
        --------
        sum_of_gaussians: (batch_size, 1)
            sum of gaussians over input x
        '''
        
        # x: [batch_size, n_dim]
        # mean: [n_component, n_dim]
        # sigma: [n_component, n_dim, n_dim]
        # coeff: [n_component]
        # output: [batch_size, 1]
        with torch.set_grad_enabled(self.m_is_training):
            signs = torch.sign(self.m_coef)
            f = torch.distributions.MultivariateNormal(self.m_mean, self.sigma)
            output = []
            
            if self.m_segment_size > x.shape[0]:
                seg_step = x.shape[0]
            else:
                seg_step = self.m_segment_size
                
            for i in range(0, x.shape[0], seg_step):
                x_segment = x[i:i+seg_step]
                logprob_batch2comp = f.log_prob(x_segment[:,None,:]) + torch.log(self.m_coef[None,:].abs())
                prob_batch2comp = torch.exp(logprob_batch2comp)
                sum_of_probs = (prob_batch2comp * signs[None,:]).sum(axis=1)
                output.append(sum_of_probs)
            return torch.cat(output)
        
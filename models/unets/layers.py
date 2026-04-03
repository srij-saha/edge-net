import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn.init import kaiming_uniform_,uniform_,xavier_uniform_,normal_

import numpy as np
from collections import OrderedDict

def isflatish(x): 
    return len(x.shape) == 2 or x.shape[2:] == (1,1)

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full
        
    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)
    
class Unsqueeze(nn.Module):
    "Add `n` singleton dimensions, e.g., 2x2 => 2x2x1"
    def __init__(self, n=1):
        super(Unsqueeze, self).__init__()
        self.n = n

    def forward(self, x):
        for i in range(self.n): x = x.view(*x.shape,1)
        return x
    
class ReshapeMap(nn.Module):
    def __init__(self, *args):
        super(ReshapeMap, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
    
class Map1D2DLayer(nn.Module):
    '''Map 1D channels to 2D channels, each feature -> Grid of `out_shape`, 
    using conv1d hack to spoof Grouped Linear Layer, and reshaping the output'''
    def __init__(self, ni, out_shape):
        super(Map1D2DLayer, self).__init__()
        
        # number of filters per dimension
        nf = int(np.prod(out_shape))
        
        self.flatten = Flatten()
        # Idealy we'd use a grouped linear layer for this, treating each
        # input feature as a "group", and mapping 1-to-many, e.g., mapping
        # each 1D feature to a 4x4 set of features. But PyTorch does not
        # have a grouped linear layer, so we're using a conv1d layer hack.
        self.unsqueeze = Unsqueeze()
        self.features = nn.Sequential(OrderedDict([
            ('conv1d_1', nn.Conv1d(in_channels=ni,out_channels=ni*nf,kernel_size=1,groups=ni,bias=False)),
            ('relu', nn.LeakyReLU(0.2)),
        ]))
        
        # finally, reshape to [bs x ni x H x W]
        self.reshape = ReshapeMap(ni, *out_shape)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.unsqueeze(x)
        x = self.features(x)
        x = self.reshape(x)
        return x
    
class TransposeReshapeLayer(nn.Module):
    '''Map 1D channels to 2D channels, each feature -> Grid of `out_shape`, 
    using tranpose convolution. I think this is eqivalent to Map1D2DLayer'''
    def __init__(self, ni, out_shape):
        super(TransposeReshapeLayer, self).__init__()
        
        # number of filters per dimension
        nf = int(np.prod(out_shape))
        kernel_size = out_shape[0]
        
        self.flatten = Flatten()
        self.unsqueeze = Unsqueeze(2)
        
        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d( ni, ni, kernel_size=kernel_size, stride=1, padding=0, bias=False, groups=ni)),
            ('relu', nn.LeakyReLU(0.2)),
        ]))

    def forward(self, x):
        x = self.flatten(x)
        x = self.unsqueeze(x)
        x = self.features(x)
        return x
    
class LinearReshapeLayer(nn.Module):
    '''Many-to-Many remapping from 1D features to 2D output, using a fully 
    connected layer to map each input (`ni`) to each output (ni*H*W, where (H,W)=`out_shape`), 
    then reshaping to `out_shape`.'''
    def __init__(self, ni, out_shape):
        super(LinearReshapeLayer, self).__init__()
        
        # number of filters per dimension
        nf = int(np.prod(out_shape))
        
        self.flatten = Flatten()
        #self.unsqueeze = Unsqueeze(2)
        
        self.features = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(ni, ni*nf, bias=True)),
            ('relu', nn.LeakyReLU(0.2,inplace=True)),
        ]))
        
        # finally, reshape to [bs x ni x H x W]
        self.reshape = ReshapeMap(ni, *out_shape)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.reshape(x)
        return x
    
def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None: normal_(m.bias, 0, bias_std)
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)
        
def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type='Weight', act_cls=nn.ReLU):
        super().__init__()
        nf = ni if nf is None else nf
        conv = nn.Conv2d(ni, nf*(scale**2), 1)
        if norm_type == 'Weight': conv = weight_norm(conv)
        act = nn.ReLU(inplace=False)
        init_linear(conv, act, init='auto', bias_std=0)
        layers = [conv, act, nn.PixelShuffle(scale)]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)   
        
class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, device=None):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.device = device
        if device is not None:
            self.to(device)
            
    def _conv(self,n_in,n_out):
        conv = nn.Conv1d(n_in, n_out, kernel_size=1, stride=1, padding=0, bias=False)
        init_linear(conv, None, init='auto', bias_std=0.01)
        conv = spectral_norm(conv)
        return nn.Sequential(conv)
        
    def forward(self, x):
        #Notation from the paper.
        orig_device = x.data.device
        device = orig_device if self.device is None else self.device
        x = x.to(device)
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous().to(orig_device)

class PooledSelfAttention2d(nn.Module):
    "Pooled self attention layer for 2d."
    def __init__(self, n_channels):
        super(PooledSelfAttention2d, self).__init__()
        self.n_channels = n_channels
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels//2)]
        self.out   = self._conv(n_channels//2, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        conv = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0, bias=False)
        init_linear(conv, None, init='auto', bias_std=0.01)
        conv = spectral_norm(conv)
        return nn.Sequential(conv)
    
    def forward(self, x):
        n_ftrs = x.shape[2]*x.shape[3]
        f = self.query(x).view(-1, self.n_channels//8, n_ftrs)
        g = F.max_pool2d(self.key(x),   [2,2]).view(-1, self.n_channels//8, n_ftrs//4)
        h = F.max_pool2d(self.value(x), [2,2]).view(-1, self.n_channels//2, n_ftrs//4)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self.out(torch.bmm(h, beta.transpose(1,2)).view(-1, self.n_channels//2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in:int, ks=1, sym=False):
        super(SimpleSelfAttention, self).__init__()
        self.sym,self.n_in = sym,n_in        
        self.conv = nn.Conv1d(n_in, n_in, ks, stride=1, padding=ks//2, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = spectral_norm(self.conv)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self,x):
        if self.sym:
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        size = x.size()
        x = x.view(*size[:2],-1)

        convx = self.conv(x)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()        
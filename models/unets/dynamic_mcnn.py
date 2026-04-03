'''
    Create an MCNN dynamically from any arbitrary encoder.    
    https://www.nature.com/articles/s41598-020-62484-z.pdf
    https://github.com/fengwang/MCNN
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import kaiming_normal_,kaiming_uniform_,uniform_,xavier_uniform_,normal_
from torch.nn.utils import weight_norm, spectral_norm
from collections import OrderedDict
from functools import partial
from IPython.core.debugger import set_trace
from pprint import pprint
from addict import Dict
from copy import deepcopy
from .layers import PixelShuffle_ICNR, SelfAttention

__all__ = ['DynamicMCNN']

VERBOSE_MCNN = False

def model_sizes(ms, size):
    
    out = []
    def hook_fn(module, input, output):
        out.append(output)
        
    hooks = [m.register_forward_hook(hook_fn) for m in ms]
    dummy_eval(ms, size)
    sizes = [o.shape for o in out]
    for h in hooks: h.remove()
    
    return sizes

def _get_sz_change_idxs(sizes):
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sz_chg_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    
    # double check first layer
    if feature_szs[0] != feature_szs[1] and sz_chg_idxs[0] != 0: 
        sz_chg_idxs = [0] + sz_chg_idxs
    
    # double check last layer
    if feature_szs[-1] != feature_szs[-2]: 
        sz_chg_idxs = sz_chg_idxs + [len(feature_szs)-1]
        
    return sz_chg_idxs

def vleaky_relu(input, inplace=True):
    "`F.leaky_relu` with 0.3 slope"
    return F.leaky_relu(input, negative_slope=0.3, inplace=inplace)

def sigmoid(input, eps=1e-7):
    "Same as `torch.sigmoid`, plus clamping to `(eps,1-eps)"
    return input.sigmoid().clamp(eps,1-eps)

def sigmoid_(input, eps=1e-7):
    "Same as `torch.sigmoid_`, plus clamping to `(eps,1-eps)"
    return input.sigmoid_().clamp_(eps,1-eps)

# Monkey Patch activation functions to use desired initializations
for o in F.relu,nn.ReLU,F.relu6,nn.ReLU6,F.leaky_relu,nn.LeakyReLU:
    o.__default_init__ = kaiming_uniform_

for o in F.sigmoid,nn.Sigmoid,F.tanh,nn.Tanh,sigmoid,sigmoid_:
    o.__default_init__ = xavier_uniform_
        
def apply_init(m, types, func):
    if isinstance(m, types):
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
                
def isflatish(x): 
    return len(x.shape) == 2 or x.shape[2:] == (1,1)

def dummy_eval(m, size):
    "Evaluate `m` on a dummy input of a certain `size`"
    for l in m.modules(): 
        if hasattr(l, 'weight') and getattr(l, 'weight', None) is not None and l.weight.ndim==4:
            break
    ch_in = l.weight.shape[1]
    x = torch.rand((1,ch_in,*size))
    with torch.no_grad(): return m.eval()(x)
    
def _get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2
    return padding

class Conv2d(nn.Sequential):
    def __init__(self, ni, nf, kernel_size=4, stride=2, padding=0, bn=False, out_size=None, activation='LeakyReLU'):
        #adjust padding to achieve out_size
        if out_size: padding = _get_padding(out_size, kernel_size=kernel_size, stride=1, dilation=1)
        conv = nn.Conv2d(ni, nf, kernel_size, stride=stride, padding=padding)  
        resize = [UpSample2D(size=out_size)] if out_size else []
        if activation.lower() == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=.20, inplace=False)
        elif activation.lower() == 'sigmoid':
            act = nn.Sigmoid()
        layers = [conv] + resize + [act]
        if bn:
            layers += [nn.BatchNorm2d(nf, momentum=0.8)]
        super().__init__(*layers)
        
class UpSample2D(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(UpSample2D, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        if self.size:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        elif self.scale_factor:
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x
    
class Deconv2d(nn.Module):
    def __init__(self, ni, nf, ks=4, stride=1, padding=0, bn=False, out_size=None):
        super(Deconv2d, self).__init__()
        if out_size: padding = _get_padding(out_size, kernel_size=ks, stride=1, dilation=1)
        self.upsample = UpSample2D(scale_factor=2)
        self.conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)
        self.resize = UpSample2D(size=out_size)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(nf, momentum=0.8)
    
    def forward(self, x, skip_input):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.resize(x)
        x = self.relu(x)
        x = self.bn(x)
        return torch.cat([x, skip_input], dim=1)  

class Bridge(nn.Module):
    '''Build bridge between encoder and decoder network'''
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1,1), padding=0, dilation=1):
        super(Bridge, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
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
    
class SkipLayer(nn.Module):
    #features = None
    valid_merge = ['add','concat']
    def __init__(self, module, clone=True, merge='add', name='', device=None, max_buffer=8):
        assert merge in self.valid_merge, f"`merge` should be in {self.valid_merge}, received {merge}"
        super(SkipLayer, self).__init__()
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.clone = clone
        self.merge = merge
        self.device = device
        self.max_buffer = max_buffer
        self.hook = module.register_forward_hook(self.hook_fn)
        
        # store features using a list in case using ModelParallel, 
        # to avoid encoder overriding features before decoder has chance to access them
        self.features = [None]*self.max_buffer
        # self.features = None
        
    def hook_fn(self, module, input, output): 
        # print(f"{self.name} hook", output.shape)
        # self.features.append(output.clone().to(self.device) if self.clone else output.to(self.device))
        # self.features = output.clone() if self.clone else output 
        device = 'cpu' if output.get_device() == -1 else self.device
        
        # keeping features in a buffer in case using ModelParallel, in case the encoder
        # processes items faster than the generator (with max_buffer limiting number of items cued)
        index = self.features.index(None)
        self.features[index] = output.clone().to(device) if self.clone else output.to(device)
        
    def remove(self): 
        self.hook.remove()
    
    def __repr__(self):
        main_str = self._get_name() + '('
        if len(self.name) > 0:
            main_str += self.name + ", "
        main_str += f'merge={self.merge})'
        return main_str
    
    def forward(self, x):

        #if self.merge == 'add':
        #    out = self.features + x
        #elif self.merge == 'concat':
        #    out = torch.cat([self.features, x], dim=1)

        # first on list, first off list   
        #print(self.name, self.device, self.features[0].data.device)
        features = self.features.pop(0).to(x.data.device)
        #print(features.data.device, x.data.device)
        self.features.append(None) # maintain buffer length
        if self.merge == 'add':
            out = features + x
        elif self.merge == 'concat':
            out = torch.cat([features, x], dim=1)

        return out
    
class RetainLayer(nn.Module):
    features = None
    def __init__(self, module, clone=False):
        super(RetainLayer, self).__init__()
        self.clone = clone
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output): 
        self.features = output.clone() if self.clone else output        
    
    def remove(self): 
        self.hook.remove()
        
    def forward(self):
        return self.features    
        
class PassThroughIdentityLayer(nn.Module):
    '''Runs a module, retains the features, but returns the input'''
    features = None
    def __init__(self, module, clone=False):
        super(PassThroughIdentityLayer, self).__init__()
        self.module = module
        self.clone = clone
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output): 
        self.features = output.clone() if self.clone else output        
    
    def remove(self): 
        self.hook.remove()
        
    def forward(self, x):
        # run module on data (hook will save fetures)
        self.module(x)
        
        # return original data
        return x
    
class GatherFeatures(nn.Module):
    def __init__(self, module_list):
        super(GatherFeatures, self).__init__()
        self.module_list = module_list

    def forward(self, x):
        out = [x] + [m.features for m in self.module_list]
        return out    
    
    
class Conv2d(nn.Sequential):
    def __init__(self, ni, nf, kernel_size=4, stride=2, padding=0, bn=False, out_size=None, activation='LeakyReLU'):
        #adjust padding to achieve out_size
        if out_size: padding = _get_padding(out_size, kernel_size=kernel_size, stride=1, dilation=1)
        conv = nn.Conv2d(ni, nf, kernel_size, stride=stride, padding=padding)  
        resize = [UpSample2D(size=out_size)] if out_size else []
        if activation.lower() == 'leakyrelu':
            act = nn.LeakyReLU(negative_slope=.20, inplace=False)
        elif activation.lower() == 'sigmoid':
            act = nn.Sigmoid()
        layers = [conv] + resize + [act]
        if bn:
            layers += [nn.BatchNorm2d(nf, momentum=0.8)]
                 
def config(**kwargs):
    return Dict(kwargs)

class UpsampleBlock(nn.Sequential):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        
        # name the layers
        layers = OrderedDict([
            ('up_out', up_out), 
            ('skip', skip), 
            ('gen' , gen)
        ])
        super().__init__(layers)
        
class UNetGenBlock(nn.Sequential):
    '''UNet Generator Block, upsamples input, merges with skip connection, then generates and stores an output'''
    
    sizes = np.array([4,8,16,32,64,128,256,512,1024])
    
    # each size has different params for the [ [upsample_convs], [generagor_convs] ]
    deconv_params = {
        4   : [config(nf=1024, ks=(3, 3), s=(1,1), p=0, output_pad=0)],
        8   : [config(nf=768, ks=(3, 3), s=(1,1), p=0, output_pad=0), 
               config(nf=512, ks=(3, 3), s=(1,1), p=0, output_pad=0)],
        16  : [config(nf=384, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        32  : [config(nf=256, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        64  : [config(nf=192, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        128 : [config(nf=128, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        256 : [config(nf=64, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        512 : [config(nf=32, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
        1024: [config(nf=16, ks=(3, 3), s=(2,2), p=1, output_pad=1)],
    }
    
    gen_params = {
        4   : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(3, 3), s=(1,1), p=1)],
        8   : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(5, 5), s=(1,1), p=2)],
        16  : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(5, 5), s=(1,1), p=2)],
        32  : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(7, 7), s=(1,1), p=3)],
        64  : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(9, 9), s=(1,1), p=4)],
        128 : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(11, 11), s=(1,1), p=5)],
        256 : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(17, 17), s=(1,1), p=8)],
        512 : [config(ks=(3, 3), s=(1,1), p=1), config(ks=(31, 31), s=(1,1), p=15)],
        1024: [config(ks=(3, 3), s=(1,1), p=1), config(ks=(61, 61), s=(1,1), p=30)],
    }
            
    def __init__(self, ni, from_sz, to_sz, skip_layer, skip_name, merge, gen_hidden=None, gen_out=1, islast=False,
                 upsample='ConvTranspose2d', self_attention=False, blur=False):
        super(UNetGenBlock, self).__init__()
        
        # params depend on output size
        near_from_sz = self.sizes[ np.argmin(np.abs(self.sizes - from_sz)) ]
        near_to_sz = self.sizes[ np.argmin(np.abs(self.sizes - to_sz)) ]
        
        # upsample (either simple Upscale2d, ConvTranspose2d, or PixelShuffle_ICNR)
        if upsample == 'ConvTranspose2d':
            # we have this loop because when using ConvTranspose2d, we upscale from 4x4 to 8x8 in 2 steps (4=>6=>8)
            # so configuration for all steps is a list, but in practice only 4=>8 uses more than one 
            modules = []
            ni_ = ni        
            for c in self.deconv_params[near_to_sz]:
                modules.append(nn.Sequential(nn.ConvTranspose2d(ni_, c.nf, c.ks, stride=c.s, padding=c.p, output_padding=c.output_pad),
                                             nn.ReLU(inplace=True)))
                ni_ = c.nf
            up_out = nn.Sequential(*modules)
        elif upsample == 'PixelShuffle':
            # PixelShuffle_ICNR can only upscale with integer scale_factors (so cannot do 4=>6, 6=>8)
            # so always just use deconv_params params from last step
            c = self.deconv_params[near_to_sz][-1] 
            up_out = PixelShuffle_ICNR(ni, c.nf, scale=2, blur=blur, norm_type='Weight', act_cls=nn.ReLU)
    
        # skip-connection (or "cross-connection")
        skip = SkipLayer(skip_layer, clone=True, merge=merge, name=skip_name)
        
        # processing
        if self_attention: attn = SelfAttention(c.nf)
        
        # branch that generates and stores an output, but returns it's input unaltered
        # allows us to pass image through a "Sequential" network, while generating 
        # outputs at each stage of the generator   
        gen_hidden = gen_hidden if gen_hidden is not None else nf//2
        c1,c2 = self.gen_params[near_to_sz]
        if islast:            
            gen = nn.Sequential(
                nn.Conv2d(c.nf, c.nf//2, kernel_size=c1['ks'], stride=c1['s'], padding=c1['p']),
                nn.ReLU(inplace=True),
                nn.Conv2d(c.nf//2, gen_hidden, kernel_size=c1['ks'], stride=c1['s'], padding=c1['p']),
                nn.ReLU(inplace=True),
                nn.Conv2d(gen_hidden, gen_out, kernel_size=c2['ks'], stride=c2['s'], padding=c2['p']),
                nn.Sigmoid()
            )
        else:
            gen = PassThroughIdentityLayer(
                nn.Sequential(
                    nn.Conv2d(c.nf, gen_hidden, kernel_size=c1['ks'], stride=c1['s'], padding=c1['p']),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(gen_hidden, gen_out, kernel_size=c2['ks'], stride=c2['s'], padding=c2['p']),
                    nn.Sigmoid()
                )
            )
        
        # name the layers
        layers = OrderedDict([
            ('up_out', up_out), 
            ('skip', skip), 
        ])
        if self_attention:
            layers['attn'] = attn
        layers['gen'] = gen
        
        super().__init__(layers)
    
    def get_gen_params(self, to_sz):
        near_idx = np.argmin(np.abs(self.sizes - to_sz))
        near_sz = self.sizes[near_idx]
        
class DynamicMCNN(nn.Sequential):
    '''Create an MCNN dynamically from any arbitrary encoder.    
        
        Multi-resolution convolutional neural networks:
        https://www.nature.com/articles/s41598-020-62484-z.pdf
        
        Args:
            encoder: (nn.Module) backbone encoder network, can be any neural network (e.g. vgg16 convolutional backbone)
            n_out: (int) the number of output dimensions (n_channels for image-to-image, n_classes if doing semantic segmentation)
            img_size:  (tuple) dimensions of input data (H, W, C)            
            blur: (bool) whether to use blur with PixelShuffle
            blur_final: (bool) whether blur with PixelShuffle should be applied to final layer 
            
            self_attention: (bool=False) whether to include a self-attention layer in the decoder
            last_cross: (bool=True) whether to include last_cross (input => output)
            
            bottle: bool = False,
            norm_type: Optional[NormType] = NormType.Batch,
            
            # Possible params to configure:            
            skip_connections: if 'default' is used take default skip connections,
                else provide a list of layer numbers or names starting from top of model,
                or set to 'auto' to automatically detect them
            upsample: (str) one of 'Upsample', 'ConvTranspose2d', or 'PixelShuffle'
            decoder_filters: (int) number of convolution layer filters in decoder blocks
            decoder_use_batchnorm: (bool) if True add batch normalization layer after `Conv2D` and `Activation` layers
            decoder_bn_first: (bool) place the batchnorm layer before the `Activation` layer
            n_upsample_blocks: (int) a number of upsampling blocks
            upsample_rates: (tuple of int) upsampling rates decoder blocks            
            activation: (str) one of keras activations for last model layer
            freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning. Can
                also call model.freeze_encoder() and model.unfreeze_encoder()

            **kwargs
        
        Returns:
            PyTorch nn.Sequential model
            
    '''
    def __init__(self, 
                 encoder, 
                 n_out, 
                 img_size, 
                 blur=False, 
                 blur_final=True, 
                 self_attention=False,
                 upsample='ConvTranspose2d', 
                 y_range=None, 
                 last_cross=True, 
                 bottle=False, 
                 act_cls=nn.ReLU,
                 init=nn.init.kaiming_normal_, 
                 norm_type=None, 
                 gen_hidden=8, 
                 **kwargs):
        super(DynamicMCNN, self).__init__()
        
        self.encoder = deepcopy(encoder)
        
        x = dummy_eval(self.encoder, img_size).detach()
        
        if VERBOSE_MCNN:
            print(x.shape)
        
        if isflatish(x):
            ni = x.shape[1]
            self.bridge = nn.Sequential(OrderedDict([
                ('reshape', TransposeReshapeLayer(ni, (2,2))),
                ('conv', nn.Conv2d(ni, 2048, kernel_size=(1,1),stride=(1,1), padding=0)),
                ('relu', nn.ReLU(inplace=True))
            ]))
        else:
            self.bridge = nn.Sequential(OrderedDict([
                ('lx2', nn.Sequential(
                    nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1,1), padding=0),  # 2
                    nn.ReLU(inplace=True)))
            ]))                
            
        x = self.bridge(x)
        
        if VERBOSE_MCNN:
            print(x.shape)
        
        # find crossover layers (blocks of the encoder show size changes) 
        sizes = model_sizes(encoder, size=img_size)        
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
        sz_chg_idxs = [idx for idx in sz_chg_idxs if len(sizes[idx]) == 4]
        self.skip_layers = [self.encoder[idx] for idx in sz_chg_idxs]
        
        self.generator = nn.Sequential()
        
        for i,idx in enumerate(sz_chg_idxs):
            islast = i == (len(sz_chg_idxs)-1)
            not_final = i != (len(sz_chg_idxs)-1)
            ni = x.shape[1]
            from_sz = x.shape[-1]
            to_sz = sizes[idx][-1]
            #sa = self_attention and (i==len(sz_chg_idxs)-3)
            sa = self_attention and (i==len(sz_chg_idxs)-4)
            do_blur = blur and (not_final or blur_final)
            
            block = UNetGenBlock(ni, from_sz=from_sz, to_sz=to_sz,
                                 skip_layer=self.encoder[idx], skip_name=f'encoder[{idx}]', merge='add', 
                                 gen_hidden=gen_hidden, gen_out=n_out, islast=islast,
                                 upsample=upsample, self_attention=sa, blur=do_blur)
            
            self.generator.add_module(f'UNetGenBlock{to_sz}', block)
        
            x = dummy_eval(self.encoder, img_size).detach()
            x = self.bridge(x)
            x = self.generator(x)
            #print(f'UNetGenBlock{from_sz}_{to_sz}', x.shape)
        
        # combine into a list the output of final generator block with previous generator blocks
        self.generator.add_module('GeneratorOutput', GatherFeatures([
            self.generator.UNetGenBlock256.gen,
            self.generator.UNetGenBlock128.gen,
            self.generator.UNetGenBlock64.gen,
            self.generator.UNetGenBlock32.gen,
            self.generator.UNetGenBlock16.gen,
            self.generator.UNetGenBlock8.gen,
            self.generator.UNetGenBlock4.gen,
        ]))
        
        # test pass
        x = dummy_eval(self.encoder, img_size).detach()
        
        if VERBOSE_MCNN:
            print(x.shape)
            
        x = self.bridge(x)
        
        if VERBOSE_MCNN:
            print(x.shape)
        x = self.generator(x)
        
        if VERBOSE_MCNN:
            print([o.shape for o in x])
        
        # apply kaiming_normal_ to Conv2d and ConvTranspose2d layers:
        # kaiming_normal_ equivalent to keras he_normal?
        self.apply(partial(apply_init, types=(nn.Conv2d, nn.ConvTranspose2d, nn.Linear), 
                           func=partial(kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu')))

    def forward(self, x):
        
        # encoder
        x = self.encoder(x)   # 512x512 -> 4x4
        
        if VERBOSE_MCNN:
            print(x.shape)
            
        # bridge: encoder -> decoder
        x = self.bridge(x)    # 2x2
        
        if VERBOSE_MCNN:
            print(x.shape)
            
        # generator # 2x2 => 512x512 (with mult-scale output)
        gen_out = self.generator(x)
        
        if VERBOSE_MCNN:
            print(x.shape)
            
        return gen_out
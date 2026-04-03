import torch
import torch.nn as nn
from .dynamic_mcnn import SkipLayer

__all__ = ['UNetModelParallel']

class UNetModelParallel(nn.Module):
    def __init__(self, model, split_size=8, dev1='cuda:0', dev2='cuda:1'):
        super(UNetModelParallel, self).__init__()
        self.encoder = model.encoder
        self.bridge = model.bridge
        self.generator = model.generator
        self.split_size = split_size
        self.dev1 = dev1
        self.dev2 = dev2
        
        self.encoder.to(self.dev1)        
        self.bridge.to(self.dev1)
        self.generator.to(self.dev2)
        for m in self.generator.modules():
            if isinstance(m, SkipLayer): 
                m.device = self.dev2
            
    def forward(self, x):
        
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.bridge(self.encoder(s_next)).to(self.dev2)
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.generator(s_prev)
            ret.append(s_prev)

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.bridge(self.encoder(s_next)).to(self.dev2)

        s_prev = self.generator(s_prev)
        ret.append(s_prev)
                
        return [torch.cat(r) for r in list(zip(*ret))]
        
# class SplitSequentialModel(nn.Module):
#     def __init__(self, model, cut):
#         super(SplitSequentialModel, self).__init__()
#         self.__class__.__name__ = 'Split' + encoder.__class__.__name__
#         children = list(model.named_children())
#         self.before = nn.Sequential(OrderedDict(children[:cut]))
#         self.bridge = nn.Sequential(OrderedDict([children[cut]]))
#         self.after = nn.Sequential(OrderedDict(children[(cut+1):]))
    
#     def forward(self, x):
#         x = self.before(x)
#         x = self.bridge(x)
#         x = self.after(x)
#         return x
    
# class SelfAttentionModelParallel(nn.Module):
#     def __init__(self, model, split_size=8, dev1='cuda:0', dev2='cuda:1'):
#         super(SelfAttentionModelParallel, self).__init__()
#         self.encoder = model.encoder
#         self.bridge = model.bridge
#         self.generator = SplitSequentialModel(model.generator)
#         self.split_size = split_size
#         self.dev1 = dev1
#         self.dev2 = dev2
        
#         self.encoder.to(self.dev1)        
#         self.bridge.to(self.dev1)
#         self.generator.before.to(self.dev1)
#         self.generator.bridge.to(self.dev2)
#         self.generator.after.to(self.dev1)
#         for m in self.generator.modules():
#             if isinstance(m, SkipLayer): 
#                 m.device = self.dev2
            
#     def forward(self, x):
        
#         splits = iter(x.split(self.split_size, dim=0))
#         s_next = next(splits)
#         s_prev = self.bridge(self.encoder(s_next)).to(self.dev2)
#         ret = []

#         for s_next in splits:
#             # A. s_prev runs on cuda:1
#             s_prev = self.generator(s_prev)
#             ret.append(s_prev)

#             # B. s_next runs on cuda:0, which can run concurrently with A
#             s_prev = self.bridge(self.encoder(s_next)).to(self.dev2)

#         s_prev = self.generator(s_prev)
#         ret.append(s_prev)
                
#         return [torch.cat(r) for r in list(zip(*ret))]    
    
        
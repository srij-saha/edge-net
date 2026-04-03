'''
    Custom version of PyTorch Weights calss that inferfaces with our custom model outputs.
'''
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, TypeVar, Union
from pathlib import Path 
from urllib.parse import urlparse
from pdb import set_trace
from functools import partial

import urllib
import urllib.request
import pandas as pd
from pathlib import Path
import json
import re

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

VERBOSE = False

@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.
    Args:
        url (str): The location where we find the weights.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory and thus we want to delay initialization until
            needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
            meta-data (for example the `classes` of a classification model) needed to use the model.
    """

    url: str
    # bucket: str
    transforms: Callable
    meta: Dict[str, Any]

    def __eq__(self, other: Any) -> bool:
        # We need this custom implementation for correct deep-copy and deserialization behavior.
        # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
        # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
        # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
        # for it, the check against the defined members would fail and effectively prevent the weights from being
        # deep-copied or deserialized.
        # See https://github.com/pytorch/vision/pull/7107 for details.
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False
        
        # if self.bucket != other.bucket:
        #     return False
        
        if self.meta != other.meta:
            return False
        
        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms
        
class WeightsEnum(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
    `Weights`.
    Args:
        value (Weights): The data class entry with the weight information.
    """
    
    @classmethod
    def verify(cls, obj: Any) -> Any:
        if obj is not None:
            if type(obj) is str:
                obj = cls[obj.replace(cls.__name__ + ".", "")]
            elif not isinstance(obj, cls):
                raise TypeError(
                    f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
                )
        return obj
    
    @classmethod
    def list_models(cls):
        print(list(cls.__members__.keys()))
        
    def get_checkpoint(self, model_dir=torch.hub.get_dir(), progress=True, check_hash=True) -> Mapping[str, Any]:
        
        cache_filename = os.path.basename(urlparse(self.url).path)
        
        checkpoint = load_state_dict_from_url(
            url = self.url,
            model_dir = model_dir,
            map_location = 'cpu',
            progress = progress,
            check_hash = check_hash,
            # file_name = cache_filename
        )
        
        if VERBOSE:
            print(f"local_filename: {os.path.join(model_dir,cache_filename)}")                
        
        return checkpoint
    
    def get_state_dict(self, model_dir=torch.hub.get_dir(), progress=True, check_hash=True) -> Mapping[str, Any]:
        
        cache_filename = os.path.basename(urlparse(self.url).path)
        
        
        checkpoint = load_state_dict_from_url(
            url = self.url,
            model_dir = model_dir,
            map_location = 'cpu',
            progress = progress,
            check_hash = check_hash,
            # file_name = cache_filename
        )
        
        if VERBOSE:
            print(f"local_filename: {os.path.join(model_dir,cache_filename)}")
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        pattern = re.compile(r'^module\.')
        state_dict = {pattern.sub('', k):v for k,v in state_dict.items()}
        
        return state_dict
    
    def get_train_log(self, cache_dir=None):
        target_url = self.meta['log_url']
        if target_url is None: return None
        arch = Path(target_url).name.split("_log-")[0]
        hash_id = Path(target_url).stem.split("_log-")[-1]
        
        if cache_dir is None:
            df = self._get_log_from_url(target_url, arch, hash_id)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            local_filename = os.path.join(cache_dir, Path(target_url).name)
            if not os.path.isfile(local_filename):
                print(f"downloading file from {target_url} to {local_filename}")
                urllib.request.urlretrieve(target_url, local_filename)
            df = self._get_log_from_file(local_filename, arch, hash_id)
        
        return df
    
    def get_train_params(self, cache_dir=None):
        target_url = self.meta['params_url']
        if target_url is None: return None
        hash_id = Path(target_url).stem.split("_params-")[-1]
        
        if cache_dir is None:
            params = self._get_params_from_url(target_url)
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            local_filename = os.path.join(cache_dir, Path(target_url).name)
            if not os.path.isfile(local_filename):
                print(f"downloading file from {target_url} to {local_filename}")
                urllib.request.urlretrieve(target_url, local_filename)
            params = self._get_params_from_file(local_filename)
        
        params['hash_id'] = hash_id
        
        return params
    
    def _process_log_file(self, f, arch, hash_id):
        lines = []
        for line in f.readlines():
            data = json.loads(line)
            data['arch'] = arch
            data['hash_id'] = hash_id
            lines.append(data)
        df = pd.DataFrame(lines)

        return df

    def _get_log_from_file(self, filename, arch, hash_id):
        with open(filename, 'r') as f:
            df = self._process_log_file(f, arch, hash_id)

        return df

    def _get_log_from_url(self, target_url, arch, hash_id):
        with urllib.request.urlopen(target_url) as f:
            df = self._process_log_file(f, arch, hash_id)

        return df
    
    def _get_params_from_url(self, target_url):
        with urllib.request.urlopen(target_url) as f:
            params = json.loads(f.read())
        return params
    
    def _get_params_from_file(self, filename):
        with open(filename, 'r') as f:
            params = json.loads(f.read())
        return params
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def url(self):
        return self.value.url
    
    # @property
    # def bucket(self):
    #     return self.value.bucket
    
    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta
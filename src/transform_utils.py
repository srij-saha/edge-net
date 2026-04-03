import torchvision.transforms as tv_transforms
from models.transforms import EdgeTransform, AddGaussianNoise, OpenImage, ToNumpyArray
from models.MUNet_v10 import transforms

def get_base_transform():
    """ Return default image -> tensor transform from MUNet_v10 transforms dict."""
    return transforms["default_transform"]

def make_eval_edge_transform(loaded = False):
    """ Base transform + EdgeTransform, for edge-based reconstruction models."""
    if loaded:
        return EdgeTransform()
    base = get_base_transform()
    
    return tv_transforms.Compose([
        *base.transforms,
        EdgeTransform(),
    ])

def make_eval_gaussian_transform(loaded = False, std_dev = 0.10, num_transforms = 10, start_seed = 42):
    """ Return a list of transforms: base + AddGaussianNoise with different seeds for denoising models"""
    base = get_base_transform()
    transforms = []
    for seed in range(start_seed, start_seed + num_transforms):
        if loaded:
            transforms.append(AddGaussianNoise(std_dev = std_dev, seed = seed))
        else:
            transforms.append(
                tv_transforms.Compose([
                    *base.transforms,
                    AddGaussianNoise(std_dev = std_dev, seed = seed),
                ])
            )
    return transforms
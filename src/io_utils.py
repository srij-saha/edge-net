import os
import numpy as np
from PIL import Image
import tifffile
import matplotlib.pyplot as plt

def get_mask(image_file, rotate90 = None, flip = None):
    if '.tif' in image_file:
        mask = tifffile.imread(image_file)
    else:
        mask = np.array(Image.open(image_file).convert('L'))
        
    mask = np.array(mask, dtype = float)/255
    
    if flip is not None:
        mask = np.fliplr(mask)
    if rotate90 is not None:
        mask = np.rot90(mask, rotate90,(1,0))
        
    mask = np.where(mask > 0.5, 1, 0)
    return mask

def createWhiteNoiseBg(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((512,512))*255).astype(np.uint8)


def get_files(root_folder, illusion_name, rotate90 = None, flip = None):
    path = os.path.join(root_folder, illusion_name)
    allowed_extensions = ('.tif', '.png', '.jpg')
    image_files = sorted(
        [os.path.join(path, file) for file in os.listdir(path)
         if file.lower().endswith(allowed_extensions)]
    )
    
    illusion_file = image_files[1]
    dark_mask = get_mask(image_files[0], rotate90, flip)
    light_mask = get_mask(image_files[2],rotate90, flip)
    
    return illusion_file, dark_mask, light_mask


def get_anderson_winawer_moon_files(root_folder, illusion_name, rotate90=None, flip=None):
    path = os.path.join(root_folder, illusion_name)
    allowed_extensions = ('.tif', '.png', '.jpg')
    image_files = sorted(
        [os.path.join(path, file) for file in os.listdir(path)
         if file.lower().endswith(allowed_extensions)]
    )

    dark_illusion_file = image_files[0]
    light_illusion_file = image_files[1]
    mask_file = image_files[2]
    
    mask = get_mask(mask_file, rotate90 = rotate90, flip = flip)
    
    return dark_illusion_file, light_illusion_file, mask


def get_haze_base(root_folder, illusion_name):
    path = os.path.join(root_folder, illusion_name, f"{illusion_name}_base.npy")
    return np.load(path)


def save_stim_images(stim_dict, out_root, illusion_name):
    out_dir = os.path.join(out_root, illusion_name)
    os.makedirs(out_dir, exist_ok = True)
    
    Image.fromarray(
        (stim_dict["img"] * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "illusion.png"))
    
    Image.fromarray(
        (stim_dict["light_mask"] * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "light_mask.png"))
    
    Image.fromarray(
        (stim_dict["dark_mask"] * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "dark_mask.png"))
    
    return out_dir


def save_image(image, out_dir, filename, vmin = 0, vmax = 1, cmap = "gray", dpi = 300):
    os.makedirs(out_dir, exist_ok = True)
    filepath = os.path.join(out_dir, filename)
    
    fig, ax = plt.subplots(figsize = (4,4))
    ax.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    ax.axis("off")
    
    fig.savefig(filepath, dpi = dpi, bbox_inches = "tight", pad_inches = 0)
    plt.close(fig)
    
    print(f"Image saved at {filepath}")
    
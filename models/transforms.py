import numpy as np
import tifffile
from PIL import Image
from scipy.signal import convolve
from scipy.fft import ifft2

__all__ = ['OpenImage', 'ToNumpyArray', 'IndividualNorm', 'DefaultNorm', 'DefaultNormOnly', 'EdgeTransform',
           'AddGaussianNoise', 'AddAndersonWinawerClouds', 'EdgeTransformTest']

class OpenImage:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, str) and img.startswith('http'):
            return Image.open(BytesIO(requests.get(img).content))
        elif isinstance(img, str) and '.tif' in img:
            return Image.fromarray(tifffile.imread(img))
        elif isinstance(img, str):
            return Image.open(img)
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
class ToNumpyArray:
    def __call__(self, img):
        return np.array(img)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
class IndividualNorm:
    def __call__(self, img):
        img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
class DefaultNorm:
    def __call__(self, img):
        return img/255
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
class DefaultNormOnly:
    def __call__(self, img):
        return img/255, img/255
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 
    
class EdgeTransform:
    def __init__(self):
        self.kernel = np.asarray( [[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype='float32' )
        
    def __call__(self, img):

        gradient = convolve( img, self.kernel, mode='same' )

        return gradient, img
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'     

class EdgeTransformTest:
    def __init__(self):
        self.kernel = np.asarray([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype='float32' )
    
    def __call__(self, img_file, new_shape = (514,514)):
        resized_img = Image.open(img_file).resize(new_shape)
        
        gradient = convolve(np.array(resized_img)/255, self.kernel, mode = 'valid')
        
        img = np.array(Image.open(img_file))/255
        
        return gradient, img
        
class AddGaussianNoise:
    def __init__ (self, std_dev, seed = 42):
        self.std_dev = std_dev
        self.seed = seed
        
    def __call__ (self, img):
        curr_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        img_rescale = img.copy() * 2 - 1
        noise = np.random.normal(0, self.std_dev, img_rescale.shape)
        img_noisy = np.clip(img_rescale + noise, -1, 1)
        np.random.set_state(curr_rng_state)
        return img_noisy, img
    
    def __repr__ (self):
        return f'{self.__class__.__name__}(std_dev={self.std_dev}, seed={self.seed})'
    
    
class AddAndersonWinawerClouds:
    def __init__ (self, sz = 512, seed = 42):
        self.seed = seed
        self.sz = sz
        
    def __call__ (self, img):
        curr_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        
        nx, ny = np.meshgrid(np.linspace(-1, 1, self.sz), np.linspace(-1, 1, self.sz))
        d_sq = nx**2 + ny**2 + 1e-10
        af = (1.0 / d_sq) * np.exp(2j * np.pi * np.random.rand(self.sz, self.sz))
    
        ax = ifft2(af)
        aa = np.real(ax * np.conj(ax))
        cloud = (aa - np.min(aa)) / (np.max(aa) - np.min(aa))
        
        img_noisy = img.copy() * 0.5
        img_noisy += (cloud * 0.5)
        img_noisy = np.clip(img_noisy,0,1)
        
        np.random.set_state(curr_rng_state)
        
        return img_noisy, img
    
    def __repr__ (self):
        return f'{self.__class__.__name__}(sz={self.sz}, seed={self.seed})'
    
        
        
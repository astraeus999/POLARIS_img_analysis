# Define astro img class
import os
import numpy as np
from astropy.io import fits #need to install the "astropy" package
import copy

class AstroImgs(object):
    _img_dir = ''
    _img_names = []
    _images = []
    
    def __init__(self, img_dir):
        self._img_dir = img_dir
        self._img_names = [name for name in os.listdir(self._img_dir)]
        self._images = np.concatenate([fits.getdata(self._img_dir + name) for name in os.listdir(self._img_dir) if 'parangs' not in name])
        
    def GetImgDir(self):
        return self._img_dir
    
    def SetImgDir(self, img_dir):
        self._img_dir = img_dir
        return True
    
    def GetImgNames(self):
        return self._img_names

    def LoadImg(self):
        self._images = np.concatenate([fits.getdata(self._img_dir + name) for name in os.listdir(self._img_dir) if 'parangs' not in name])
        return True
    
    def GetImg(self):
        imgs = np.copy(self._images)
        return imgs

def main():
    print("Import Load Astro Image as main")
    
if __name__ == '__main__':
    main()
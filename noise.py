import os
#import cv2
import numpy as np
import skimage
from skimage import io, util

count = 0
for filename in os.listdir(r"../dataset/lfw/lfw_sphere_224x192_V1"):
    for file in os.listdir(r"./filename"):
        img = io.imread(file)
        img = util.random_noise(img, mode='pepper', seed=None, clip=True, **kwargs)
        print (file)
        count = count + 1
print (count)

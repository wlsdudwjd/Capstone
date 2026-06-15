from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import os
import random
import cv2 as cv

input_dir = 'C:\Users\user\Desktop\cv\dalcomUoaaa_1768821004651.png'
target_dir = 'C:\Users\user\Desktop\cv'

img_siz=(160, 160)
n_class=3
batch_size=32

img_paths=sorted([os.path.join(input_dir,f) for f in os.listdir(input_dir) if f.endswith('.jpg')])
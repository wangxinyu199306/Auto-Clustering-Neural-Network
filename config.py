 
import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.DATA_ROOT_DIR = "/home/lzhpc/Dataset/cifar-10-batches-py"

__C.TRAIN_FOLDER = "train"

__C.TEST_FOLDER = "test"

# for a n-class problem, create n folders named 0~n-1, in both train and test folders
# put image into corresponding folders
__C.CLASSES = 10

__C.IMAGE_EXTENSION = ".jpg"

__C.TRAIN_NUM_FOR_EACH = 5000

__C.TEST_NUM_FOR_EACH = 1000
    

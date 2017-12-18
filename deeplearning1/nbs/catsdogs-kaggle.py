from __future__ import division,print_function

path = "comp_data/sample/"
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)

#import utils; reload(utils)
#from utils import plots

batch_size=64
import vgg16; reload(vgg16)
from vgg16 import Vgg16

vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
imgs,labels = next(batches)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
preds = vgg.predict(imgs, True)
with open("submission.csv", "w") as text_file:
    for i in range(len(preds[0])):
        text_file.write('{},{}\r\n'.format(i, preds[0][i]))



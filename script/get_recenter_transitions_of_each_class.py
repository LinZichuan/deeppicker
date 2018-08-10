# get recenter transitions of each class and save as transitions.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # plt 
import matplotlib.image as mpimg # mpimg
import numpy as np
import math
from skimage import morphology

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

import math
import sys, os

#particle_size = 180
#extract_size = 200

if len(sys.argv) != 3:
    print ("print get_recenterxxx.py [particle_size] [class_center.conf path] [classK]")
    exit()
particle_size = int(sys.argv[1])
extract_size = int(particle_size * 0.75) * 2
print ("EXTRACT_SIZE =====>>>>> ", extract_size)
class_center_file = sys.argv[2]
classK = int(sys.argv[3])

class_center = {}
with open(class_center_file, 'r') as fin:
    fin.readline()
    for l in fin:
        l = l.strip()
        t = l.split('=')
        class_center[int(t[0])] = map(int, t[1].split(','))
        #print (int(t[0]), class_center[int(t[0])])

transition_info = []
for idx in range(1, classK+1):
    print idx
    center = [extract_size/2, extract_size/2]
    mean_x, mean_y = class_center[idx]
    delta_x = center[0] - mean_x
    delta_y = center[1] - mean_y 
    delta_x_ = round(delta_x*particle_size/extract_size, 2)
    delta_y_ = round(delta_y*particle_size/extract_size, 2)
    transition_info.append((delta_x_, delta_y_))
    #TODO: write transition to file
with open('transitions.py', 'w') as fout:
    print>>fout, "transitions = " + str(transition_info)




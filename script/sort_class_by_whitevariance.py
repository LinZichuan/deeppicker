# sort class by their respective variance of white pixels, and write and show sorted result

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
if len(sys.argv) != 4:
    print ("python sort*.py [dir of classes photos, e.g. allclass] [particle_size] [classK]")
    exit()
allclass_path = sys.argv[1]
particle_size = int(sys.argv[2])
classK = int(sys.argv[3])
extract_size = int(particle_size * 0.75) * 2
print ("EXTRACT_SIZE =====>>>>> ", extract_size)
test = False

if not os.path.exists("analyze"):
    os.mkdir("analyze")

class_centers = {}
signal_var = []
transition_info = []
center = [extract_size/2, extract_size/2]
for idx in range(1, classK+1):
    print idx
    imgname = os.path.join(allclass_path, str(idx)+'.png')
    data = mpimg.imread(imgname) #
    shape = data.shape
    ori_data = np.copy(data)
    #center = [100, 100]
    #for i in range(200):
    #    for j in range(200):
    #        d = (i - center[0]) ** 2 + (j - center[1]) ** 2
    #        d = math.sqrt(d)
    #        if d >= 75:
    #            data[i][j] = 0
    for i in range(extract_size):
        for j in range(extract_size):
            d = (i - center[0]) ** 2 + (j - center[1]) ** 2
            d = math.sqrt(d)
            if d >= particle_size/2: #TODO
                data[i][j] = 0
    if test:
        fig = plt.figure()
        plt.subplot(1,5,1)
        plt.imshow(data, cmap='Greys_r')

#----------
    mythres = 0.35 #TODO:threshold
    data = data >= mythres
    pos = np.where(data == 1.0)
    if len(pos[0]) == 0:
        transition_info.append((0, 0))
        signal_var.append([idx, 10000, ori_data])
        continue
        
    if test:
        data[data> mythres] = 1.0
        data[data<=mythres] = 0.0
        plt.subplot(1,5,2)
        plt.imshow(data)

#----------
    mean_x = int(np.mean(pos[0]))
    mean_y = int(np.mean(pos[1]))
    class_centers[idx] = [mean_x, mean_y]
    for i in range(5):
        for j in range(5):
            data[mean_x+i][mean_y+i] = 0
    std_x = np.std(pos[0]) #TODO, nx = #column
    std_y = np.std(pos[1]) #TODO, ny = #row
    std_sum = std_x + std_y
    signal_var.append([idx, std_sum, ori_data])

    if test:
        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(data, selem)
        plt.subplot(1,5,3)
        plt.imshow(data, cmap=plt.cm.gray)

#-----------
    if test:
        #info.append((r, angle))
        delta_x = center[0] - mean_x
        delta_y = center[1] - mean_y
        delta_x_ = round(delta_x*particle_size/extract_size, 2) #TODO:18/20, 180/200, particle_size / classphoto_size
        delta_y_ = round(delta_y*particle_size/extract_size, 2)
        transition_info.append((delta_x_, delta_y_))
        #TODO: write transition to file
        plt.subplot(1,5,4)
        plt.imshow(data, cmap=plt.cm.gray)
#-----------
    if test:
        plt.subplot(1,5,5)
        data = morphology.binary_erosion(data, morphology.diamond(2)).astype(np.float32)
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.savefig('analyze/'+str(idx)+'.png')
        plt.close(0)

if test:
    exit()

signal_var = sorted(signal_var, key=lambda x:x[1])
info = [[item[0], item[1]] for item in signal_var]
print info


with open('sorted_class.txt', 'w+') as fout:
    ids = [str(sig[0]) for sig in signal_var]
    print >> fout, ','.join(ids)
with open('class_center.conf', 'w+') as fout:
    ids = [sig[0] for sig in signal_var]
    print >> fout, "[General]"
    print "valid class number:", len(class_centers.keys())
    for idx in ids:
        if idx in class_centers.keys():
            print >> fout, "%d=%d,%d" % (idx, class_centers[idx][0], class_centers[idx][1])
        else:
            print >> fout, "%d=%d,%d" % (idx, center[0], center[1])


#fig = plt.figure(figsize=(20,10))
for i in range(classK):
    plt.subplot(7, 8, i+1)
    plt.axis('off')
    plt.imshow(signal_var[i][2], cmap='Greys_r')
plt.close(0)

plt.savefig(os.path.join(allclass_path, 'all.png'))

#with open('transitions.py', 'w') as fout:
#    print>>fout, "transitions = " + str(transition_info)

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
basepath = "allclass/"
#import sys, os
#input = sys.argv[1]
#print ("inputpath:", input)

signal_var = []
transition_info = []
for idx in range(1, 51):
    print idx
    imgname = basepath + str(idx)+'.png'
    data = mpimg.imread(imgname) #
    shape = data.shape
    ori_data = np.copy(data)
    center = [100, 100]
    for i in range(200):
        for j in range(200):
            d = (i - center[0]) ** 2 + (j - center[1]) ** 2
            d = math.sqrt(d)
            if d >= 75:
                data[i][j] = 0

    fig = plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(data, cmap='Greys_r')

    #img = cv2.imread(imgname, 0)
    #img = cv2.medianBlur(img, 5)
    #th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                    cv2.THRESH_BINARY, 11, 2)
    #data[data> 0.2] = 1.0
    #data[data<=0.2] = 0.0
    plt.subplot(1,5,2)
    plt.imshow(data)

    radius = 15
    selem = disk(radius)
    local_otsu = rank.otsu(data, selem)
    plt.subplot(1,5,3)
    mythres = 0.2
    data = data >= mythres
    pos = np.where(data == 1.0)
    if len(pos[0]) == 0:
        transition_info.append((0, 0))
        signal_var.append((idx, 10000, ori_data))
        continue
        
    mean_x = int(np.mean(pos[0]))
    mean_y = int(np.mean(pos[1]))
    for i in range(5):
        for j in range(5):
            data[mean_x+i][mean_y+i] = 0
    std_x = np.std(pos[0]) #TODO, nx = #column
    std_y = np.std(pos[1]) #TODO, ny = #row
    std_sum = std_x + std_y
    signal_var.append((idx, std_sum, ori_data))
    plt.imshow(data, cmap=plt.cm.gray)
    #plt.imshow(data >= local_otsu, cmap=plt.cm.gray)
    #threshold_global_otsu = threshold_otsu(data)
    #global_otsu = data >= threshold_global_otsu
    #plt.imshow(global_otsu, cmap=plt.cm.gray)

    plt.subplot(1,5,4)
    pos = np.array(pos)
    delta_x = center[0] - mean_x
    delta_y = center[1] - mean_y 
    '''
    #r = math.sqrt(delta_x**2 + delta_y**2)
    if delta_y > 0 and delta_x > 0:
        angle = 1.5 * math.pi + math.atan(delta_x / delta_y)
    elif delta_y > 0 and delta_x < 0:
        angle = 1.5 * math.pi - math.atan(-delta_x / delta_y)
    elif delta_y < 0 and delta_x < 0:
        angle = math.pi - math.atan(delta_y / delta_x)
    elif delta_y < 0 and delta_x > 0:
        angle = math.atan(-delta_y / delta_x)
    '''

    #info.append((r, angle))
    delta_x_ = round(delta_x*18/20, 2)
    delta_y_ = round(delta_y*18/20, 2)
    transition_info.append((delta_x_, delta_y_))
    #TODO: write transition to file

    pos[0] = pos[0] + delta_x
    pos[1] = pos[1] + delta_y
    for i in range(200):
        for j in range(200):
            data[i][j] = 0
    for i in range(len(pos[0])):
        data[pos[0][i]][pos[1][i]] = 1
    plt.imshow(data, cmap=plt.cm.gray)
    '''
    data = morphology.binary_erosion(data, morphology.diamond(2)).astype(np.float32)
    plt.imshow(data, cmap='gray')
    '''

    plt.subplot(1,5,5)
    data = morphology.binary_erosion(data, morphology.diamond(2)).astype(np.float32)
    plt.imshow(data, cmap='gray')
    #plt.imshow(data)
    #plt.plot(dist.keys(), dist.values(), '-')
    #plt.show()
    plt.axis('off')
    #plt.savefig('bilation/'+str(idx)+'.png')
    plt.close(0)

signal_var = sorted(signal_var, key=lambda x:x[1])
info = [[item[0], item[1]] for item in signal_var]
print info

classes = [item[0] for item in signal_var]
print classes

with open('transitions.py', 'w') as fout:
    print>>fout, "transitions = " + str(transition_info)




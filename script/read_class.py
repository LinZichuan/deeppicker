# read class from run_it.mrcs file and save as png file

import math
import mrcfile
import numpy as np
import scipy.misc
from scipy import signal
import os, sys

from skimage.measure import compare_ssim as ssim
#filename = 'relion_zw18/Class2D/2dc_right_center/run_it025_classes.mrcs'
#filename = 'relion_zw18/Class2D/2dc_right_center/run_it025_data.star'
if len(sys.argv) != 4:
    print ("python read_class.py [path of run*classes.mrcs] [output of classes images] [classK]")
    exit()
input = sys.argv[1]
output = sys.argv[2] # output folder
classK = int(sys.argv[3])
if "classes.mrcs" not in input:
    print ("Please input run_*_classes.mrcs filepath.")
    exit()

filename = input
print ("filepath:", filename)
classnumber = [0 for _ in range(classK)]

#datafile = '/data00/UserHome/zlin/proteasome/OUTPUT/Class2D/default_job/run_it025_data.star'
#fin = open(datafile, 'r')
#tmp = 0
#for l in fin:
#    tmp += 1
#    if tmp <= 33:
#        continue
#    if len(l.split()) > 0:
#        classidx = int(l.split()[2]) - 1
#        classnumber[classidx] += 1

mrc = mrcfile.open(filename)
header = mrc.header
print header
body = mrc.data

#print header

def process(micrograph, idx):
    #micrograph = body[0]
    if classnumber[idx-1] != 0:
        #micrograph = 1.0 * micrograph / classnumber[idx-1]
        mean = micrograph.mean() / classnumber[idx-1]
        std = micrograph.std()
        micrograph = (micrograph - mean) / std
    '''
    truncate = 10
    min_ = mean - truncate * std
    max_ = mean + truncate * std
    sortmrc = np.sort(micrograph.reshape(-1))
    k = 50
    min_ = sortmrc[k]
    max_ = sortmrc[-k]
    leng = (max_ - min_) * 1.0
    micrograph = np.clip(micrograph, min_, max_) - min_
    micrograph = micrograph / leng
    '''
    #mean = micrograph.mean()
    #std = micrograph.std()
    #micrograph = (micrograph - mean) / std
    #mean = micrograph.mean()
    #std = micrograph.std()
    #print (mean, std)
    return micrograph

if not os.path.exists(output):
    os.makedirs(output)
print ("outputpath:", output)

idx = 1
post_micro = []
for micro in body:
    micrograph = process(micro, idx)
    outputpath = os.path.join(output, str(idx) + '.png')
    scipy.misc.imsave(outputpath, micrograph)
    print "%d,"%idx, classnumber[idx-1]
    idx += 1
    post_micro.append(micrograph)


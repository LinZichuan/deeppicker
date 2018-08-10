from dataLoader import DataLoader
import math
import mrcfile
import numpy as np
import scipy.misc
from deepModel import DeepModel
import tensorflow as tf
from scipy import signal

from skimage.measure import compare_ssim as ssim
filename = 'run_it025_classes.mrc'

mrc = mrcfile.open(filename)
header = mrc.header
body = mrc.data


#header, body = DataLoader.readMrcFile(filename)

#print header

def process(micrograph):
    #micrograph = body[0]
    mean = micrograph.mean()
    std = micrograph.std()
    truncate = 4.0
    min_ = mean - truncate * std
    max_ = mean + truncate * std
    sortmrc = np.sort(micrograph.reshape(-1))
    k = 100
    min_ = sortmrc[k]
    max_ = sortmrc[-k]
    leng = (max_ - min_) * 1.0
    micrograph = np.clip(micrograph, min_, max_) - min_
    micrograph = micrograph / leng
    return micrograph


idx = 1
post_micro = []
for micro in body:
    micrograph = process(micro)
    scipy.misc.imsave('allclass/' + str(idx) + '.png', micrograph)
    idx += 1
    post_micro.append(micrograph)

dist = [46, 29, 41, 18, 34, 28, 40, 17, 39, 38, 48, 7, 49, 35, 30, 11, 0, 27, 20, 4]
accRota = [0, 4, 30, 34, 27, 38, 49, 11, 28, 10, 17, 18, 20, 29, 35, 39, 40, 41, 46, 7]  #40 is houmian de
accTran = [34, 38, 0, 30, 49, 4, 18, 27, 11, 39, 20, 28, 7, 41, 46, 35, 10, 17, 29, 32]

score = {}
def addscore(l):
    for i in range(len(l)):
        if l[i] not in score.keys():
            score[l[i]] = 1.0 / (i + 1)
        else:
            score[l[i]] += 1.0 / (i + 1)

#addscore(dist)
addscore(accRota)
addscore(accTran)

score = score.items()
score = sorted(score, key=lambda x:x[1], reverse=True)
print score

base = [score[0][0]]
expand_base = True
maxsize = 20

while len(base) < maxsize:
    sim = []
    for p in range(len(post_micro)):
        #cor = signal.correlate2d(post_micro[p], post_micro[38])
        maxs = 0.0
        for b in base:
            s = ssim(post_micro[p], post_micro[b])
            #maxs = max(s, maxs)
            maxs += s
        #print p, maxs
        if math.isnan(maxs) or p in base:
            continue
        sim.append((p, maxs))
    sim = sorted(sim, key=lambda x:x[1], reverse=True)
    if expand_base == False:
        base.append(sim[0][0])

    start = 0
    term = True
    while start < len(sim) and expand_base:
        tar = sim[start][0]
        if tar in dist[:10] and tar in accRota[:10] and tar in accTran[:10]:
            base.append(tar)
            term = False
        start += 1
    if term:
        expand_base = False
    print base

base = [d + 1 for d in base]

print base[:6]
print base[-12:]

fout_base = open('filtered_class.txt', 'w')
print>>fout_base, ','.join(map(str, base[:6]))
print>>fout_base, ','.join(map(str, base[-12:]))
'''
model_input_size = [50, 64, 64, 1]

cand = []
for d in body:
    patch = np.copy(d)
    patch = DataLoader.preprocess_particle(patch, model_input_size)
    cand.append(patch)

cand = np.array(cand).reshape(50, 64, 64, 1)

deepModel = DeepModel(180, model_input_size, 2)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.all_variables())
    pretrained_model = 'trained_model/newmodel'
    saver.restore(sess, pretrained_model)
    pred = deepModel.evaluation(cand, sess)
    print pred
    pred_with_idx = [(idx+1, pred[idx][0]) for idx in range(len(pred))]
    pred_with_idx = sorted(pred_with_idx, key=lambda x:x[1])
    print pred_with_idx[:10]
'''

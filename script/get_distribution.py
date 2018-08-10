import os
import numpy as np
filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/run_it025_model.star'

fin = open(filename, 'r')

idx = 0
dist = []
accRota = []
accTran = []
item = 0
bad_class = []
for l in fin:
    idx += 1
    if idx <= 36:
        continue
    if idx >= 87:
        break
    t = l.split()
    if float(t[2]) == 999:
        bad_class.append(item)
        item += 1
        continue
    dist.append( (item, float(t[1])) )
    accRota.append( (item, float(t[2])) )
    accTran.append( (item, float(t[3])) )
    item += 1

print "filtered number:", len(dist)

dist = sorted(dist, key=lambda t:t[1], reverse=True)
accRota = sorted(accRota, key=lambda t:t[1])
accTran = sorted(accTran, key=lambda t:t[1])
print 'dist:'
print dist
print 'accRota:'
print accRota
print 'accTran:'
print accTran

print [item[0] for item in dist[:20]]
print [item[0] for item in accRota[:20]]
print [item[0] for item in accTran[:20]]

distRank = {}
accRotaRank = {}
accTranRank = {}
for i in range(len(dist)):
    distRank[dist[i][0]] = i
    accRotaRank[accRota[i][0]] = i
    accTranRank[accTran[i][0]] = i
print '>>>>>', distRank
print '>>>>>', accRotaRank
print '>>>>>', accTranRank

distRank = dict(distRank)
accRotaRank = dict(accRotaRank)
accTranRank = dict(accTranRank)
print distRank


jointScore = []
for k in distRank.keys():
    print k
    score = 0.5 * 1. / (distRank[k] + 1) + 1. / (accRotaRank[k] + 1) + 1. / (accTranRank[k] + 1)
    jointScore.append( (k, score) )

jointScore = sorted(jointScore, key=lambda t:t[1], reverse=True)
print jointScore
final_index = [item[0] for item in jointScore]
print '>>>>>>>>>>>'
final_index = [d+1 for d in final_index]
bad_class = [d+1 for d in bad_class]
bad_class = bad_class + final_index[10:]

good_res = sorted(final_index[:10])
bad_res = sorted(bad_class)
print good_res
print bad_res
'''
mean = np.mean(dist)
std = np.std(dist)
print mean, std
left = mean - 0.01
right = mean + 0.01

idxes = [i for i in range(len(dist)) if dist[i] >= left and dist[i] <= right]
print idxes
'''


fin1 = open('../../Select/select/good.class', 'r')
ground_truth = map(int, fin1.read().split(','))
print ground_truth
fin2 = open('../../Select/select_bad/bad.class', 'r')
ground_false = map(int, fin2.read().split(','))
print ground_false


cnt = len([1 for i in good_res if i in ground_truth])
print "precision:", 1.0 * cnt / len(good_res)

cnt = len([1 for i in bad_res if i in ground_truth])
print "error:", 1.0 * cnt / len(bad_res)





import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
num = sys.argv[1]
fin = open('wf_curve' + num + '.txt')
fin1 = open('lzc_curve' + num + '.txt')

wf = []
lzc = []
for l in fin:
    token = l.split()
    wf.append(token[2])
for l in fin1:
    token = l.split()
    lzc.append(token[2])

wf = map(float, wf)
lzc = map(float, lzc)

#plt.title('Splice training curve based on gammas-model')
plt.title('Training curve')
plt.ylabel('Error(%)')
plt.xlabel('Epoch')
axes = plt.gca()
axes.set_ylim([0, 50])
#plt.plot(t1, tr, label='training')
#plt.plot(t1, te, label='validation')
plt.plot(range(len(wf)), wf, label='default')
plt.plot(range(len(lzc)), lzc, label='finetune')
plt.legend(loc='upper right')

plt.show()
plt.savefig('error' + num + '.png')

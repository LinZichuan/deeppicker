import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fin = open('training_error.txt')
fin1 = open('training_error_splice.txt')

tr_err = []
te_err = []
tr = []
te = []
for l in fin:
    token = l.split()
    tr_err.append(token[1])
    te_err.append(token[2])
for l in fin1:
    token = l.split()
    tr.append(token[1])
    te.append(token[2])

tr_err = [float(x) for x in tr_err]
te_err = [float(x) for x in te_err]
t = range(0, len(tr_err))
tr = [float(x) for x in tr]
te = [float(x) for x in te]
t1 = range(0, len(tr))

#plt.title('Splice training curve based on gammas-model')
plt.title('Trpv1 training curve')
plt.ylabel('Error(%)')
plt.xlabel('Epoch')
axes = plt.gca()
axes.set_ylim([0, 50])
#plt.plot(t1, tr, label='training')
#plt.plot(t1, te, label='validation')
plt.plot(t, tr_err, label='training(continuous)')
plt.plot(t, te_err, label='validation(continuous)')
plt.legend(loc='upper right')

plt.show()
plt.savefig('error.png')

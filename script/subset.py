# seperate classes star, for pickle extract in deeppicker

import os
#filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/run_it025_data.star'
#filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job027/run_it025_data.star'
import sys


if len(sys.argv) < 3:
    print ("python subset*py [path of run_*_data.star]  [poslist]  [pickresult]  [symbol]")
    exit()
filename = sys.argv[1]
poslist = sys.argv[2].strip().split(',')
poslist = map(int, poslist)
print ("poslist", poslist)
pickresult = sys.argv[3]
symbol = sys.argv[4]

fin = open(filename, 'r')

#prefix = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2'
prefix = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnClassNumber #3\n_rlnAnglePsi #4\n_rlnAutopickFigureOfMerit #5'


starfile = {}

idx = 0
for l in fin:
    idx += 1
    if idx <= 33 or l.strip() == '':
        continue
    t = l.strip().split()  #28

    x = int(float(t[0]))
    y = int(float(t[1]))
    classid = int(t[2])
    imagename = t[6].split('/')[-1]
    if classid not in poslist:
        continue

    if imagename not in starfile.keys():
        starfile[imagename] = [(x, y)]
    else:
        starfile[imagename].append((x, y))

if not os.path.exists(pickresult):
    os.makedirs(pickresult)

for imagename, stars in starfile.items():
    print (imagename)
    filepath = os.path.join(pickresult, imagename.replace('.mrc', symbol+'.star'))
    with open(filepath, 'w+') as fout:
        print>>fout, prefix
        for star in stars:
            print>>fout, star[0], star[1], -999, -999, -999

print ("Subset Done.")

# seperate classes star, for pickle extract in deeppicker

import os
#filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/run_it025_data.star'
#filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job027/run_it025_data.star'
import sys


if len(sys.argv) < 3:
    print ("python seperate*py [path of run_*_data.star] [dir of output extracted pickle, e.g. class2d_seperated_star] [classK]")
    exit()
filename = sys.argv[1]
output = sys.argv[2] # class2d_seperated_star
classK = int(sys.argv[3])

fin = open(filename, 'r')

basepath = output
if not os.path.exists(basepath):
    os.makedirs(basepath)
prefix = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2'


all_star_dict = {}
for i in range(1, classK+1):
    all_star_dict[i] = {}

idx = 0
for l in fin:
    idx += 1
    if idx <= 33 or l.strip() == '':
        continue
    t = l.strip().split()  #28

    x = t[0]
    y = t[1]
    imagename = t[5].split('/')[-1]
    class_number = int(t[2])

    stars = all_star_dict[class_number]
    if imagename not in stars.keys():
        stars[imagename] = [(x, y)]
    else:
        stars[imagename].append((x, y))


opened = []
for c,stars in all_star_dict.items():
    for imagename, star_list in stars.items():
        print (imagename)
        #TODO: mrcs or mrc??? or both
        filepath = os.path.join(basepath, imagename.replace('.mrcs', '') + "_class%d.star"%c)
        if filepath not in opened:
            with open(filepath, 'w') as fout:
                print>>fout, prefix
            opened.append(filepath)
        with open(filepath, 'a+') as fout:
            for star in star_list:
                print>>fout, star[0], star[1]


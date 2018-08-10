# scan run_it_data line by line, and recenter each particle using info in transitions.py, generate _new.star

import os
#filename = '/data00/Programs/thuempicker/relion_zw18_retrain_2Dclass/Class2D/job005/run_it025_data.star'
#filename = '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/run_it025_data.star'
import sys
if len(sys.argv) < 4:
    print ("python sca*.py [path of run*data.star] [dir of _new.star] [path of transition.py]")
    exit()
input = sys.argv[1] # path of run_it*data.star
output = sys.argv[2] # dir of _new.star, e.g. class2dstar_recenter
trans = sys.argv[3]
print ("inputpath:", input)
print ("outputpath:", output)
print ("transition:", trans)

filename = input
fin = open(filename, 'r')

if not os.path.exists(output):
    os.makedirs(output)
prefix = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2'

all_star = {}
new_star = {}
for i in range(1, 51):
    all_star[i] = {}
    new_star[i] = {}

image_star = {}
new_image_star = {}

from transitions import transitions
import math
print transitions
print len(transitions)
from math import cos, sin, pi
#TODO: transitions size scale ???
idx = 0
for l in fin:
    idx += 1
    if idx <= 32 or l.strip() == '':
        continue
    t = l.strip().split()  #28

    print idx

    x = float(t[0]) #TODO: origin pos, or pos after transition and rotation ???
    y = float(t[1])
    psi = float(t[20]) / 180 * pi
    dx = float(t[21])
    dy = float(t[22])
    class_number = int(t[23])
    '''
    dx_sum = dx + transitions[class_number-1][0]
    dy_sum = dy + transitions[class_number-1][1]
    imagename = t[3].split('/')[-1]
    angle = -psi
    dx_new = cos(angle) * dx_sum - sin(angle) * dy_sum
    dy_new = cos(angle) * dy_sum + sin(angle) * dx_sum
    x_new = round(x - dx_new, 1)
    y_new = round(y - dy_new, 1)

    if x_new < 90 or x_new >= 3838-90 or y_new < 90 or y_new >= 3710-90:
        x_new = x
        y_new = y
    '''
    imagename = t[3].split('/')[-1]
    angle = -psi
    #dx_ = transitions[class_number-1][1]
    #dy_ = -transitions[class_number-1][0] #up
    #dx_rot = cos(angle) * dx_ - sin(angle) * dy_
    #dy_rot = cos(angle) * dy_ + sin(angle) * dx_ #up
    #dx_tra = dx - dx_rot
    #dy_tra = (-dy) - dy_rot #up
    #x_new = x + dx_tra
    #y_new = y + (-dy_tra) #down

    angle = psi
    dx_ = transitions[class_number-1][1]
    dy_ = transitions[class_number-1][0] #up
    dx_rot = cos(angle) * dx_ + sin(angle) * dy_
    dy_rot = cos(angle) * dy_ - sin(angle) * dx_ #up
    dx_tra = -dx - dx_rot
    dy_tra = -dy - dy_rot #up
    x_new = x + dx_tra
    y_new = y + dy_tra #down

    '''
    stars = all_star[class_number]
    stars_new = new_star[class_number]
    if imagename not in stars.keys():
        stars[imagename] = [(x, y)]
        stars_new[imagename] = [(x_new, y_new)]
    else:
        stars[imagename].append((x, y))
        stars_new[imagename].append((x_new, y_new))
    '''

    if imagename not in image_star.keys():
        image_star[imagename] = [(x, y, class_number)]
        new_image_star[imagename] = [(x_new, y_new, class_number)]
    else:
        image_star[imagename].append((x, y, class_number))
        new_image_star[imagename].append((x_new, y_new, class_number))

#print (image_star)
#print (stars[ks])
#print (stars_new[ks])

import sys
choice = sys.argv[1]
#Usage: python xx/scan_run_it_data.py writeclass


for imagename, stars in image_star.items():
    filepath = os.path.join(output, imagename[:-4] + ".star")
    with open(filepath, 'w') as fout:
        print>>fout, prefix
        for star in stars:
            if choice == 'writeclass':
                print>>fout, star[0], star[1], star[2]
            else:
                print>>fout, star[0], star[1]

for imagename, stars in new_image_star.items():
    filepath = os.path.join(output, imagename[:-4] + "_new.star")
    with open(filepath, 'w') as fout:
        print>>fout, prefix
        for star in stars:
            if choice == 'writeclass':
                print>>fout, star[0], star[1], star[2]
            else:
                print>>fout, star[0], star[1]

'''
opened = []
for c,stars in all_star.items():
    for imagename, star_list in stars.items():
        filepath = os.path.join(output, imagename[:-4] + "_class%d.star"%c)
        if filepath not in opened:
            with open(filepath, 'w') as fout:
                print>>fout, prefix
            opened.append(filepath)
        with open(filepath, 'a+') as fout:
            for star in star_list:
                print>>fout, star[0], star[1]
'''

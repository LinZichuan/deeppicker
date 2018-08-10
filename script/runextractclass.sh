#!/bin/bash

for ((i=1; i<=50; i++))
do
    echo '_class'$i''
    python extractData.py  --inputDir '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/class2Dstar' --mrc_number 400 --coordinate_symbol '_class'$i'' --particle_size 180 --save_dir './extracted_data/' --save_file 'zw18_class'$i'.pickle' --produce_negative True
done

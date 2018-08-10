#!/bin/bash
#python extractData.py  --inputDir '/data00/Data/gammas' --mrc_number 1081 --coordinate_symbol '_2dpick_ycy' --particle_size 320 --save_dir './extracted_data/' --save_file 'gammas_all.pickle'
#python extractData.py  --inputDir '/data00/Data/spliceosome' --mrc_number 1116 --coordinate_symbol '_manualpick' --particle_size 320 --save_dir './extracted_data/' --save_file 'splice_all.pickle'
#python extractData.py  --inputDir '../trpv1' --mrc_number 110 --coordinate_symbol '_manual_lgc' --particle_size 180 --save_dir './extracted_data/' --save_file 'trpv1_all.pickle'
#python extractData.py  --inputDir '/data00/Data/piezo' --mrc_number 2635 --coordinate_symbol '_DW_recentered' --particle_size 180 --save_dir './extracted_data/' --save_file 'piezo_all.pickle'

#python extractData.py  --inputDir '/data00/Data/gammas/train' --mrc_number 300 --coordinate_symbol '_2dpick_ycy' --particle_size 320 --save_dir './extracted_data/' --save_file 'gammas_01.pickle'
#python extractData.py  --inputDir '/data00/Data/spliceosome/train' --mrc_number 300 --coordinate_symbol '_manualpick' --particle_size 320 --save_dir './extracted_data/' --save_file 'splice_01.pickle'
#python extractData.py  --inputDir '/data00/Data/trpv1/train' --mrc_number 110 --coordinate_symbol '_manual_lgc' --particle_size 180 --save_dir './extracted_data/' --save_file 'trpv1_01.pickle'
#python extractData.py  --inputDir '/data00/Data/piezo/train' --mrc_number 300 --coordinate_symbol '_DW_recentered' --particle_size 180 --save_dir './extracted_data/' --save_file 'piezo_01.pickle'


for ((i=1; i<=50; i++))
do
    echo '_class'$i''
    /data00/Softwares/python/python27/bin/python extractData.py  --inputDir '/data00/Programs/thuempicker/relion_zw18/Class2D/job027/class2Dstar_recenter_class' --mrc_number 400 --coordinate_symbol '_class'$i'' --particle_size 180 --save_dir './extracted_data_recenter/' --save_file 'zw18_class'$i'.pickle' --produce_negative True
done

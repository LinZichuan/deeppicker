#python extractData.py  --inputDir '/data00/Data/gammas/train' --mrc_number 300 --coordinate_symbol '_2dpick_ycy' --particle_size 320 --save_dir './extracted_data/' --save_file 'gammas_all.pickle'
#python extractData.py  --inputDir '/data00/Data/spliceosome/train' --mrc_number 300 --coordinate_symbol '_manualpick' --particle_size 320 --save_dir './extracted_data/' --save_file 'splice_all.pickle'
#CUDA_VISIBLE_DEVICES=0 python extractData.py  --inputDir '/data00/Data/trpv1/train' --mrc_number 1010 --coordinate_symbol '_manual_lgc' --particle_size 180 --save_dir './extracted_data/' --save_file 'trpv1_all.pickle'
#250/1.32
#CUDA_VISIBLE_DEVICES=0 python extractData.py  --inputDir '/data00/Data/piezo/train' --mrc_number 2635 --coordinate_symbol '_DW_recentered' --particle_size 190 --save_dir './extracted_data/' --save_file 'piezo_all.pickle'
CUDA_VISIBLE_DEVICES=0 python extractData.py  --inputDir '/data00/Data/deeppicker_traindata/piezo/Micrographs' --mrc_number 10000 --coordinate_symbol '_pick' --particle_size 190 --save_dir './extracted_data/' --save_file 'piezo_all.pickle'

##120/1.32=91
#python extractData.py  --inputDir '/data00/Data/lzc_CNG/train' --mrc_number 1000 --coordinate_symbol '_zxy' --particle_size 100 --save_dir './extracted_data/' --save_file 'cng_all.pickle'
##140/1.338=105
#python extractData.py  --inputDir '/data00/Data/lzc_zxy/titan1_trpml3_micrographs' --mrc_number 1000 --coordinate_symbol '_zxy' --particle_size 110 --save_dir './extracted_data/' --save_file 'trpml3_all.pickle'
##140/1.338=105
#python extractData.py  --inputDir '/data00/Data/lzc_zxy/titan2_trpml3_H_micrographs' --mrc_number 1000 --coordinate_symbol '_zxy' --particle_size 110 --save_dir './extracted_data/' --save_file 'trpml3H_all.pickle'
##140/1.338=105
#python extractData.py  --inputDir '/data00/Data/lzc_zxy/titan2_trpml3_mlsa1_micrographs' --mrc_number 1000 --coordinate_symbol '_zxy' --particle_size 110 --save_dir './extracted_data/' --save_file 'trpml3mlsa1_all.pickle'


#python extractData.py  --inputDir '/data00/Data/gammas/train' --mrc_number 300 --coordinate_symbol '_2dpick_ycy' --particle_size 320 --save_dir './extracted_data/' --save_file 'gammas_01.pickle'
#python extractData.py  --inputDir '/data00/Data/spliceosome/train' --mrc_number 300 --coordinate_symbol '_manualpick' --particle_size 320 --save_dir './extracted_data/' --save_file 'splice_01.pickle'
#python extractData.py  --inputDir '/data00/Data/trpv1/train' --mrc_number 110 --coordinate_symbol '_manual_lgc' --particle_size 180 --save_dir './extracted_data/' --save_file 'trpv1_01.pickle'
#python extractData.py  --inputDir '/data00/Data/piezo/train' --mrc_number 300 --coordinate_symbol '_DW_recentered' --particle_size 180 --save_dir './extracted_data/' --save_file 'piezo_01.pickle'


#python extractData.py  --inputDir '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/class2Dstar' --mrc_number 300 --coordinate_symbol '_goodstar' --particle_size 180 --save_dir './extracted_data/' --save_file 'zw18_2dclass_goodstar.pickle' --produce_negative False
#
#python extractData.py  --inputDir '/data00/Programs/thuempicker/relion_zw18/Class2D/job019/class2Dstar' --mrc_number 300 --coordinate_symbol '_badstar' --particle_size 180 --save_dir './extracted_data/' --save_file 'zw18_2dclass_badstar.pickle' --produce_negative False

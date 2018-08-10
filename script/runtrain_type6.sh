# !/bin/sh

#python train.py --train_type 6 --train_inputDir './extracted_data' --train_inputFile '' --particle_number 20000 --model_save_dir './trained_model' --model_save_file 'newmodel_auto_filter' --model_load_file './trained_model/joint-4'
/data00/Softwares/python/python27/bin/python train.py --train_type 6 --train_inputDir './extracted_data_recenter' --train_inputFile '' --particle_number 20000 --model_save_dir './trained_model' --model_save_file 'newmodel_recenter'



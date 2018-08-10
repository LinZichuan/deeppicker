#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle' --particle_number 40000 --model_save_dir './trained_model' --model_save_file 'jointha_new_model'
python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'splice_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 's1'


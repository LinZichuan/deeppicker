#train on gammas data type 1
#python train.py --train_type 1 --train_inputDir '../gammas_mrc/' --particle_size 180 --particle_number 10000 --coordinate_symbol '_2dpick_ycy' --model_save_dir './trained_model' --model_save_file 'model-demo-gammas' --model_load_file './trained_model/model-demo'

#train on splice data type 1
#python train.py --train_type 1 --train_inputDir '../spliceosome1_mrc/' --particle_size 320 --particle_number 10000 --coordinate_symbol '_manualpick' --model_save_dir './trained_model' --model_save_file 'model-demo-splice' --model_load_file './trained_model/model-demo-gammas'

#train on gammas data type 2
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'gammas-model'

#train on splice data type 2
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'splice_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'splice-model'

#train on trpv1 data type 2
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'trpv1_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'trpv1-model'

#train on piezo data type 2
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'piezo_300mrc_0_1.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'piezo-model_0_1'

#joint train-4
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'piezo_01.pickle;trpv1_01.pickle;gammas_01.pickle;splice_01.pickle' --particle_number 40000 --model_save_dir './trained_model' --model_save_file 'joint-4'

#joint train-3
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'trpv1_01.pickle;gammas_01.pickle;splice_01.pickle' --particle_number 30000 --model_save_dir './trained_model' --model_save_file 'joint-3'

#train on splice data type 2 baseon gammas-model
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'splice_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'splice-model-baseon-gammas' --model_load_file './trained_model/gammas-model'

#train on trpv1 data type 2 baseon gammas-splice-model
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'trpv1_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'trpv1-model-baseon-gammas-splice' --model_load_file './trained_model/splice-model-baseon-gammas'

#train on splice & gammas
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle' --particle_number 20000 --model_save_dir './trained_model' --model_save_file 'splice-gammas-model'

#train on splice & gammas & trpv1
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle' --particle_number 30000 --model_save_dir './trained_model' --model_save_file 'joint-model'

#train on splice & gammas & trpv1 & piezo
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle' --particle_number 40000 --model_save_dir './trained_model' --model_save_file 'joint-4-fast'

#train on joint-8
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle;trpml3_all.pickle;trpml3H_all.pickle;trpml3mlsa1_all.pickle' --particle_number 100000 --model_save_dir './trained_model' --model_save_file 'joint-8-fast' --mixup 0

#train on joint-x-tmp
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle' --particle_number 40000 --model_save_dir './trained_model' --model_save_file 'joint-4-tmp' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle' --particle_number 50000 --model_save_dir './trained_model' --model_save_file 'joint-5-tmp' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle;trpml3_all.pickle' --particle_number 60000 --model_save_dir './trained_model' --model_save_file 'joint-6-tmp' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle;trpml3_all.pickle;trpml3H_all.pickle' --particle_number 70000 --model_save_dir './trained_model' --model_save_file 'joint-7-tmp' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle;trpml3_all.pickle;trpml3H_all.pickle;trpml3mlsa1_all.pickle' --particle_number 80000 --model_save_dir './trained_model' --model_save_file 'joint-8-tmp-deeper' --mixup 0
export PATH=/data00/Programs/thuempicker_copy_package/softwares/python/python27/bin/:$PATH
echo "Your python path:"$(which python)
echo "Your pip path:"$(which pip)

python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;cng_all.pickle;trpml3_all.pickle;trpml3H_all.pickle;trpml3mlsa1_all.pickle' --particle_number 80000 --model_save_dir './trained_model' --model_save_file 'joint-8' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle' --particle_number 40000 --model_save_dir './trained_model' --model_save_file 'joint-4-test-peace' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle;splice_all.pickle;trpv1_all.pickle;piezo_all.pickle;trpml3_all.pickle' --particle_number 50000 --model_save_dir './trained_model' --model_save_file 'joint-5-test' --mixup 0
#python train.py --train_type 2 --train_inputDir './extracted_data' --train_inputFile 'gammas_all.pickle' --particle_number 10000 --model_save_dir './trained_model' --model_save_file 'joint-1-tmp-resnet' --mixup 0

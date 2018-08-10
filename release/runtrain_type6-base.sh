# !/bin/sh
#class=$(ls | grep zw18_class)
#res=""
#for c in $class
#do
#    res+="$c;"
#done
#echo $res


#python train.py --train_type 6 --train_inputDir './extracted_data' --train_inputFile '' --particle_number 20000 --model_save_dir './trained_model' --model_save_file 'newmodel_auto_filter' --model_load_file './trained_model/joint-4'
extracted_data_dir=$1
model_save_dir=$2
new_model_name=$3
pos_list=$4
neg_list=$5

export PATH=DEEPPATH/softwares/python/python27/bin/:$PATH
echo "Your python path:"$(which python)
echo "Your pip path:"$(which pip)
# CUDA 8.0
#export PATH=/data00/Programs/thuempicker_copy_package/softwares/cuda/bin:$PATH
export LD_LIBRARY_PATH=DEEPPATH/softwares/cuda/lib64:$LD_LIBRARY_PATH


#/data00/Softwares/python/python27/bin/python train.py --train_type 6 --train_inputDir './extracted_data_recenter' --train_inputFile '' --particle_number 20000 --model_save_dir './trained_model' --model_save_file 'newmodel_recenter'

python train.py --train_type 6 --pos_list "$pos_list" --neg_list "$neg_list" --train_inputDir $extracted_data_dir --train_inputFile '' --particle_number 20000 --model_save_dir $model_save_dir --model_save_file $new_model_name




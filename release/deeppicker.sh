#!/bin/bash
################### Parameter Setting ####################
# picknumber:    the number of mrc that you want to do autopicking
# pickfilename:  the filename of mrc that you want to do autopicking.
#                       Note that if you specify this parameter, this program will only pick this one mrc photo. If you want to 
#                       pick specific number of mrc photos, please modify the picknumber parameter.
# model:         the deeppicker model path
# inputDir:      mrc photo folder path
# outputDir:     autopicking result folder path
# symbol:        autopicked result star filename symbol.
#                       e.g., if using default '_cnnpick', and your original mrc filename is 'cool.mrc', then the star 
#                       filename will be 'cool_cnnpick.star'
# particle_size: the particle diameter in pixel
# threshold:     default is 0.9, you can set this value between (0,1). 
#                       If you increase the value, the picked results will become less but more precise. If you decrease 
#                       the value, the picker results will become more but less precise.
##########################################################

# CUDA 8.0
export PATH=/data00/Softwares/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/data00/Softwares/cuda-8.0/lib64:$LD_LIBRARY_PATH

# You can use default settings for these three parameters below.
curr_dir=$(pwd)
echo "DeepPicker v0 is deprecated now, please run deeppickerdev."
exit
echo "Welcome to DeepPicker v0!"
echo "Your current path is: $curr_dir"

cd /data00/Programs/thuempicker/

# usage:
# sh deeppicker
# input: inputDir particle_size threshold picknumber pickfilename

inputDir=""
particle_size=""
threshold=""
picknumber=""
pickfilename=""

echo "Please put your mrc-photo folder in current directory, and input the folder name:"
while [[ -z "${inputDir// }" ]]; do 
    read inputDir
    inputDir="$curr_dir/$inputDir"
done

echo "Please input particle_size:"
while [[ -z "$particle_size" ]]; do 
    read particle_size 
done

echo "Please input threshold:"
while [[ -z "$threshold" ]]; do 
    read threshold 
done

echo "Please input picknumber:"
while [[ -z "$picknumber" ]]; do 
    read picknumber 
done

echo "Please input pickfilename(If you want to pick one specific image, please input filename. Otherwise, just press enter):"
read pickfilename

#model='/data00/Programs/thuempicker/trained_model/newmodel_auto_filter'
model='/data00/Programs/thuempicker/trained_model/joint-4'
#model='./trained_model/newmodel'
symbol='_pick'
outputDir=''$curr_dir'/pick-result/'

echo ">>>Your parameter settings are:"
echo "inputDir:$inputDir"
echo "particle_size:$particle_size"
echo "threshold:$threshold"
echo "picknumber:$picknumber"

echo "The pick results will be put in $outputDir."

# Specify these five parameter below.
#picknumber=84
#inputDir='/data00/Data/zw18/test/'
#particle_size=180
#threshold=0.95

# Uncomment next line to specify the mrc filename that you want to do autopicking.
#pickfilename='micro_circle_stack_2406_2x_SumCorr_movie_DW.mrc'


if [[ -z "$pickfilename" ]];
then
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
else
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --mrc_filename $pickfilename --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
fi

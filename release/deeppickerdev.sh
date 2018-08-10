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

export PATH=/data00/Programs/thuempicker_copy_package/softwares/python/python27/bin/:$PATH
echo "Your python path:"$(which python)
echo "Your pip path:"$(which pip)

# CUDA 8.0
export PATH=/data00/Softwares/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/data00/Softwares/cuda-8.0/lib64:$LD_LIBRARY_PATH

# You can use default settings for these three parameters below.
curr_dir=$(pwd)
echo "----------------------------------------------------------------"
echo ">>Welcome to DeepPicker v0.1!"
echo "* DeepPicker v0.1 is now consistent with its UI."
echo "* You can run 'picker' command to open the visualization."
echo "----------------------------------------------------------------"

echo "Your current path is: $curr_dir"

cd /data00/Programs/thuempicker_copy_package/

# usage:
# sh deeppicker
# input: inputDir particle_size threshold picknumber pickfilename

inputDir=""
outputDir=""
coorOutput=""
particle_size=""
threshold=""
picknumber=""
pickfilename=""

echo "Please input your import-mrc path[absolute path, e.g, /data00/Data/zw18/test/*.mrc]:"
while [[ -z "${inputDir// }" ]]; do 
    read inputDir
    inputDir="$inputDir"
done

echo "Please input your output path[absolute path, e.g, /data00/UserHome/$USER/OUTPUT/]:"
while [[ -z "${outputDir// }" ]]; do 
    read outputDir
    outputDir="$outputDir"
done

echo "Please input your coordinate output path[relative path, e.g, pick-result/]:"
while [[ -z "${coorOutput// }" ]]; do 
    read coorOutput
    coorOutput="$coorOutput"
done

echo "Please input particle_size[pixel size]:"
while [[ -z "$particle_size" ]]; do 
    read particle_size 
done

echo "Please input threshold[0~1.0]:"
while [[ -z "$threshold" ]]; do 
    read threshold 
done

#echo "Please input picknumber:"
#while [[ -z "$picknumber" ]]; do 
#    read picknumber 
#done

echo "Please input pickfilename(If you want to pick one specific image, please input filename. Otherwise, just press enter):"
read pickfilename

#model='/data00/Programs/thuempicker/trained_model/newmodel_auto_filter'
#model='/data00/Programs/thuempicker/trained_model/joint-4-fast'
model='/data00/Programs/thuempicker_copy_package/trained_model/joint-8'
#model='./trained_model/newmodel'
symbol='_pick'
#Output: outputDir/pick-result; outputDir/plot
#outputDir=''$curr_dir'/pick-result/'

echo ">>>Your parameter settings are:"
echo "inputDir:$inputDir"
echo "particle_size:$particle_size"
echo "threshold:$threshold"
#echo "picknumber:$picknumber"
picknumber=1

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
    python picker.py --inputDir "$inputDir" --outputDir $outputDir --coorOutput $coorOutput --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --coordinate_symbol $symbol --threshold $threshold --deeppickerRunDir $curr_dir
else
    python picker.py --inputDir "$inputDir" --outputDir $outputDir --coorOutput $coorOutput --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --mrc_filename $pickfilename --coordinate_symbol $symbol --threshold $threshold --deeppickerRunDir $curr_dir
fi

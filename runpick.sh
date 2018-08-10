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

# You can use default settings for these three parameters below.
model='./trained_model/joint-4'
#model='./trained_model/newmodel_auto_filter'
#model='./trained_model/newmodel_recenter'
#model='./trained_model/newmodel'
symbol='_pick'
outputDir='./pick-result/'


# Specify these five parameter below.
picknumber=50
inputDir='/data00/Data/zw18/test/'
particle_size=180
threshold=0.95

# Uncomment next line to specify the mrc filename that you want to do autopicking.
#pickfilename='micro_circle_stack_2406_2x_SumCorr_movie_DW.mrc'

if [[ -z "$pickfilename" ]];
then
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
else
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --mrc_filename $pickfilename --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
fi


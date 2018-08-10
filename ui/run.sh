cd ..
curr_dir=$(pwd)
model='/data00/Programs/thuempicker/trained_model/joint-4'
symbol='_pick'
outputDir=''$curr_dir'/pick-result/'

picknumber=1
inputDir='/data00/Data/zw18/test/'
particle_size=180
threshold=0.95

if [[ -z "$pickfilename" ]];
then
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
else
    python picker.py --inputDir $inputDir --pre_trained_model $model --particle_size $particle_size --mrc_number $picknumber --mrc_filename $pickfilename --outputDir $outputDir --coordinate_symbol $symbol --threshold $threshold
fi

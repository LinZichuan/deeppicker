#!/bin/bash
deeppathstr=$(pwd)
sed "s!DEEPPATH!$deeppathstr!" release/deeppickerdev-base.sh > deeppicker
sed "s!DEEPPATH!$deeppathstr!" release/deeppickerdev-base-ui.sh > deeppicker-ui
sed "s!DEEPPATH!$deeppathstr!" release/runtrain_type6-base.sh > runtrain_type6-base.sh

chmod +x deeppicker
chmod +x deeppicker-ui
chmod +x runtrain_type6-base.sh

#cd softwares/
#wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz
#tar xvf cudnn-8.0-linux-x64-v5.1.tgz

#sudo cp cuda/lib64/* /usr/local/cuda/lib64/
#sudo cp cuda/include/cudnn.h /usr/local/cuda/include/

#echo $PATH
#setenv PATH $deeppathstr:$PATH
#echo "-----------"
#echo $PATH

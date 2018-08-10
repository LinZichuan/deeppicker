#!/bin/bash
InSuffix=$1
OutSuffix=$2

if [ $# -lt 2 ] ; then
echo "usage: $0 InSuffix OutSuffix"
echo "Wrote by Heng Zhou in Dec.26 2016"
exit
fi

for i in *$InSuffix
do
  bname=`basename ${i} $InSuffix`
  OutFile="${bname}${OutSuffix}"
  mv $i $OutFile
 
  echo "rename ${i} $OutFile"
done

  echo "Wrote by Heng Zhou in Dec.26 2016"

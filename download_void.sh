#!/bin/bash

DOWNLOAD_LIST=("https://drive.google.com/open?id=1kZ6ALxCzhQP8Tq1enMyNhjclVNzG8ODA" "https://drive.google.com/open?id=1ys5EwYK6i8yvLcln6Av6GwxOhMGb068m" "https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI")
NPOINTS_LIST=(150 500 1500)

mkdir -p void
cd void

for ((j=0; j<1; j++))
do
    mkdir -p ${NPOINTS_LIST[$j]}/void_${NPOINTS_LIST[$j]}/data/
    cd ${NPOINTS_LIST[$j]}
    gdown ${DOWNLOAD_LIST[$j]}
    unzip void_${NPOINTS_LIST[$j]}.zip
    rm void_${NPOINTS_LIST[$j]}.zip
    unzip void_${NPOINTS_LIST[$j]}-0.zip
    rm void_${NPOINTS_LIST[$j]}-0.zip
    for ((i=1; i<=56; i++))
    do
        unzip void_150-${i}.zip -d void_${NPOINTS_LIST[$j]}/data/
        rm void_150-${i}.zip
    done
    cd ..
done

cd ..
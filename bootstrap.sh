#!/usr/bin/env bash

# MACH=k80
MACH=v100

# Prevent apt cache cleaning so that we can back up the files for future use
rm /etc/apt/apt.conf.d/docker-clean

debCache=/basedeps/deb-cache
if [[ -d $debCache ]]; then
  dpkg -i $debCache/*.deb
else
  apt-get update
  apt-get -y install libgirepository1.0-dev python3-gi gir1.2-pango fontconfig
  mkdir -p $debCache
  mv /var/cache/apt/archives/*.deb $debCache/
fi

pip3 install PyGobject

echo '== Installing precompiled deps'
tar -xzvf /basedeps/$MACH-precompiled/war-ctc.tar.gz -C /
tar -xzvf /basedeps/$MACH-precompiled/warpctc_pytorch-0.1.linux-x86_64.tar.gz -C /

echo '== Installing fonts'
mkdir -p $HOME/.local/share/fonts
cp /basedeps/fonts/*.ttf $HOME/.local/share/fonts/


echo '== Testing font installation'
fc-list :lang=ml

echo '== Starting training'

bash ./bin/pottan datagen \
  --batchSize 64 \
  --input /datadeps/train.txt.gz  \
  --count $(( 256* 512)) \
  --output /output/traindata_cache

bash ./bin/pottan datagen \
  --batchSize 64 \
  --input /datadeps/validate.txt.gz  \
  --count $(( 256* 16)) \
  --output /output/valdata_cache

bash ./bin/pottan train \
  --cuda \
  --traindata /datadeps/train.txt.gz  \
  --traindata_limit $(( 256* 512)) \
  --traindata_cache /output/traindata_cache \
  --valdata /datadeps/validate.txt.gz \
  --valdata_limit $(( 256* 16 )) \
  --valdata_cache /output/valdata_cache \
  --valInterval 512 \
  --batchSize 64 \
  --lr 0.0005 \
  --niter 50 \
  --outdir /output \
  --displayInterval 50 \
  --saveInterval 1024 \

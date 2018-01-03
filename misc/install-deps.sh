#!/usr/bin/env bash

thisDir=$( dirname $( readlink -f $0 ) )
warp_ctc_target="$thisDir/warp_ctc_install"

cd $thisDir

wget -c https://github.com/baidu-research/warp-ctc/archive/master.zip -O warp-ctc.zip

unzip warp-ctc.zip

cd warp-ctc-master
mkdir build
cd build

cmake ../
make install DESTDIR=$warp_ctc_target

echo "export WARP_CTC_PATH=\"$warp_ctc_target/usr/lib\"" >> $thisDir/../env.sh

. $thisDir/../env.sh

cd $thisDir

wget -c https://github.com/SeanNaren/warp-ctc/archive/pytorch_bindings.zip -O pytorch_bindings.zip
unzip  pytorch_bindings.zip   'warp-ctc-pytorch_bindings/pytorch_binding/*'
cd warp-ctc-pytorch_bindings/pytorch_binding

python3 ./setup.py install --user







#!/usr/bin/env bash
thisDir=$( dirname $( readlink -f $0 ) )

echo "Initializing submodules"
git submodule init
git submodule update



echo "Installing python dependecies ..."
pip3 install -r $thisDir/../requirements.txt



echo "Installing system libraries & utils ..."
apt-get update
apt-get -y install libgirepository1.0-dev
apt-get -y install python3-gi gir1.2-pango



echo "Installing gi ( PyGobject) package"
pip3 install PyGobject



echo "Downloading warpctc ..."
warp_ctc_target="$thisDir/warp_ctc_install"
cd $thisDir
wget -c https://github.com/SeanNaren/warp-ctc/archive/pytorch_bindings.zip -O pytorch_bindings.zip
unzip  pytorch_bindings.zip
cd warp-ctc-pytorch_bindings



echo "Compiling warpctc ..."
mkdir build
cd build
cmake ../
make install DESTDIR=$warp_ctc_target


echo "export WARP_CTC_PATH=\"$warp_ctc_target/usr/local/lib\"" >> $thisDir/../env.sh
. $thisDir/../env.sh



echo "Compiling and installing warpctc pytorch_bindings ..."
cd $thisDir/warp-ctc-pytorch_bindings/pytorch_binding
export CUDA_HOME=$( readlink -f $(dirname $( which nvcc ) )/../ )
python3 ./setup.py install --user


# tar -czvf /output/war-ctc.tar.gz $warp_ctc_target; python3 ./setup.py bdist; cp dist/warpctc_pytorch-0.1.linux-x86_64.tar.gz /output/






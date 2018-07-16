#!/usr/bin/env bash
thisDir=$( dirname $( readlink -f $0 ) )

init_pkgs_ubuntu(){

  pkgs=(
  # pip3 for managing dependecies of pottan-ocr
  python3-pip
  python
  perl
  # Python2 & pip2 for ocropy
  python-pip
  python-tk
  # Pango & Cairo
  python3-cairo
  libcairo2-dev
  libgirepository1.0-dev
  python3-gi
  gir1.2-pango
  )

  sudo apt-get update
  sudo apt-get install ${pkgs[@]}
}

init_pkgs_opensuse-15(){
  pkgs=(
  # pip3 for managing dependecies of pottan-ocr
  python3-pip
  # Python2 & pip2 for ocropy
  python2
  python2-pip
  python2-setuptools
  python2-tk
  python2-devel
  python2-numpy
  # Pango & Cairo
  python3-cairo
  cairo-devel
  gobject-introspection-devel
  python3-gobject
  pango
  )

  sudo zypper install ${pkgs[@]}
}

init_pkgs_fedora-28(){
  pkgs=(
  # pip3 for managing dependecies of pottan-ocr
  python3-pip
  prename
  redhat-rpm-config
  # Python2 & pip2 for ocropy
  python2
  python2-pip
  python2-setuptools
  python2-tkinter
  python2-devel
  python2-numpy
  # Pango & Cairo
  python3-cairo
  cairo-devel
  gobject-introspection-devel
  python3-gobject
  pango
  )

  sudo dnf install ${pkgs[@]}
}

init_pkgs(){
  distro=$1
  echo "Installing system packages for '$distro'"
  init_pkgs_$distro
}


init_repo(){
  echo "Initializing submodules"
  git submodule init
  git submodule update
}


init_python(){
  echo "Installing python dependecies ..."
  pip3 install --user -r $thisDir/../requirements.txt
  pip  install --user -r $thisDir/../ocropy/requirements.txt

  # Install pytorch as described in https://pytorch.org/
  pyMinor=$(python3 --version | sed 's/Python 3\.\(.\).*/\1/g')
  pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp3$pyMinor-cp3${pyMinor}m-linux_$(arch).whl
}


init_training_deps(){

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
}


distro=${DISTRO:-'ubuntu'}


init_repo
init_pkgs $distro
init_python
if [[ $1 == 'for_training' ]]; then
  init_training_deps
fi

# tar -czvf /output/war-ctc.tar.gz $warp_ctc_target; python3 ./setup.py bdist; cp dist/warpctc_pytorch-0.1.linux-x86_64.tar.gz /output/

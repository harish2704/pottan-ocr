#!/usr/bin/env bash
this_dir=$( dirname $( readlink -f $0 ) )
pkgs_common=(
  # GCC for installing pip packages
  gcc
  # pip3 for managing dependecies of pottan-ocr
  python3
  python3-pip
  python3-setuptools
  python3-wheel
  python3-cairo
)
pkgs_common_rpm=(
  # for rename command
  util-linux
  python3-devel
  # Python2 & pip2 for ocropy
  python2
  python2-pip
  python2-setuptools
  python2-wheel
  python2-tk
  python2-devel
  python2-numpy
  cairo-devel
  # gobject introspection
  python3-gobject
  gobject-introspection-devel
)


init_pkgs_debian(){

  pkgs+=(
    "${pkgs_common[@]}"
    # for rename command
    rename
    python3-dev
    # Python2 & pip2 for ocropy
    python
    python-dev
    python-pip
    python-setuptools
    python-wheel
    python-tk
    libcairo2-dev
    # gobject introspection
    python3-gi
    libgirepository1.0-dev
    gir1.2-pango
    python3-gi-cairo
  )

  sudo apt-get update
  sudo apt-get install --no-install-recommends ${pkgs[@]}
}

init_pkgs_ubuntu(){
  init_pkgs_debian
}

init_pkgs_opensuse(){
  pkgs+=(
    "${pkgs_common[@]}"
    "${pkgs_common_rpm[@]}"
    typelib-1_0-Pango-1_0
  )

  sudo zypper install ${pkgs[@]}
}


init_pkgs_fedora(){
  pkgs+=(
    "${pkgs_common[@]}"
    "${pkgs_common_rpm[@]}"
    redhat-rpm-config
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
  pip3 install --user -r $this_dir/../requirements.txt
  pip2  install --user -r $this_dir/../ocropy/requirements.txt

  # Install pytorch as described in https://pytorch.org/
  py_minor_ver=$(python3 --version | sed 's/Python 3\.\(.\).*/\1/g')
  pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp3${py_minor_ver}-cp3${py_minor_ver}m-linux_$(arch).whl
}


init_training_deps(){

  echo "Downloading warpctc ..."
  warp_ctc_target="$this_dir/warp_ctc_install"
  cd $this_dir
  wget -c https://github.com/SeanNaren/warp-ctc/archive/pytorch_bindings.zip -O pytorch_bindings.zip
  unzip  pytorch_bindings.zip
  cd warp-ctc-pytorch_bindings

  echo "Compiling warpctc ..."
  mkdir build
  cd build
  cmake ../
  make install DESTDIR=$warp_ctc_target

  echo "export WARP_CTC_PATH=\"$warp_ctc_target/usr/local/lib\"" >> $this_dir/../env.sh
  . $this_dir/../env.sh

  echo "Compiling and installing warpctc pytorch_bindings ..."
  cd $this_dir/warp-ctc-pytorch_bindings/pytorch_binding
  export CUDA_HOME=$( readlink -f $(dirname $( which nvcc ) )/../ )
  python3 ./setup.py install --user
}


main(){
  distro=${DISTRO:-'ubuntu'}

  init_repo
  init_pkgs $distro
  init_python
  if [[ $1 == 'for_training' ]]; then
    init_training_deps
  fi
}


main "$@"

# tar -czvf /output/war-ctc.tar.gz $warp_ctc_target; python3 ./setup.py bdist; cp dist/warpctc_pytorch-0.1.linux-x86_64.tar.gz /output/

#!/usr/bin/env bash
this_dir=$( dirname $( readlink -f $0 ) )
pkgs_common=(
  # GCC for installing pip packages
  gcc
  tesseract-ocr
  tesseract-ocr-traineddata-english
  # pip3 for managing dependecies of pottan-ocr
  python3
  python3-pip
  python3-setuptools
  python3-wheel
  python3-cairo
)
pkgs_common_rpm=(
  # for rename command
  tesseract-ocr
  tesseract-ocr-traineddata-english
  python3-devel
  cairo-devel
  # gobject introspection
  python3-gobject
  gobject-introspection-devel
)


init_pkgs_debian(){

  pkgs+=(
    "${pkgs_common[@]}"
    # for rename command
    python3-dev
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

init_python(){
  echo "Installing python dependecies ..."
  pip3 install --user -r $this_dir/../requirements.txt

}



main(){
  distro=${DISTRO:-'ubuntu'}
  init_pkgs $distro
  init_python
}


main "$@"

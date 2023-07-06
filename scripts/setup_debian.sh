#!/bin/bash

# Pull in the environment variables describing which deps we need
# These should match PHASM's CMake flags exactly
source $(dirname "$0")/collect_deps.sh

apt-get update 

# Prevent tzdata from hanging while attempting to prompt the user
export DEBIAN_FRONTEND=noninteractive
#export TZ=America/New_York

# Install basic dev environment
apt-get -y install --no-install-recommends \
    apt-transport-https \
    autoconf \
    bat \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    exa \
    gdb \
    git \
    gnupg \
    less \
    libasan6 \
    libtool \
    python3-tk \
    python3-venv \
    software-properties-common \
    unzip \
    valgrind \
    vim \
    wget \
    xz-utils \
    zlib1g-dev

# Fix certificates
wget -O /usr/local/share/ca-certificates/JLabCA.crt http://pki.jlab.org/JLabCA.crt
chmod 644 /usr/local/share/ca-certificates/JLabCA.crt
update-ca-certificates
echo "check_certificate = off" >> ~/.wgetrc

if [ $PHASM_USE_GEANT4 -eq 1 ]
then
    apt-get -y install \
        mysql-client \
        libmysqlclient-dev \
        libexpat1-dev \
        tcsh \
        scons \
        libx11-dev \
        libxext-dev \
        libglu1-mesa-dev \
        libxt-dev \
        libxmu-dev \
        libxrender-dev \
        libxft-dev \
        libafterimage-dev
fi

if [ $PHASM_USE_LLVM -eq 1 ]
then
    # Install clang static analyzer, etc, as needed1
    apt-get -y install \
        clang-format \
        clang-tidy \
        clang-tools \
        clang \
        clangd \
        libc++-dev \
        libc++1 \
        libc++abi-dev \
        libc++abi1 \
        libclang-dev \
        libclang1 \
        liblldb-dev \
        lld \
        lldb \
        llvm-dev \
        llvm-runtime \
        llvm python3-clang
fi

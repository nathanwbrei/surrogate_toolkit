#!/bin/bash

# Pull in the environment variables describing which deps we need
# These should match PHASM's CMake flags exactly
source $(dirname "$0")/collect_deps.sh

apt-get update 

# Install basic dev environment
apt-get -y install --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget \
    git \
    cmake \
    build-essential \
    libtool \
    autoconf \
    unzip \
    vim \
    curl \
    gdb \
    libasan6 \
    less \
    exa \
    bat \
    valgrind \
    python3-venv \
    python3-tk \
    xz-utils

# Fix certificates
wget -O /usr/local/share/ca-certificates/JLabCA.crt http://pki.jlab.org/JLabCA.crt
chmod 644 /usr/local/share/ca-certificates/JLabCA.crt
update-ca-certificates
echo "check_certificate = off" >> ~/.wgetrc

if [[ $USE_PIN ~= [oO][nN] ]]; then
    apt-get -y install \
        zlib1g-dev
fi

if [[ $USE_DWARF ~= [oO][nN] ]]; then
    apt-get -y install \
        zlib1g-dev
fi




    
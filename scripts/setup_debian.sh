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



FROM ubuntu:jammy
USER root
WORKDIR /deps

RUN apt-get update && \
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
        xz-utils \
        libexpat-dev \
        zlib1g-dev  # zlib for pin or libdwarf

ADD http://pki.jlab.org/JLabCA.crt /usr/local/share/ca-certificates/JLabCA.crt
RUN chmod 644 /usr/local/share/ca-certificates/JLabCA.crt && \
    update-ca-certificates && \
    echo "check_certificate = off" >> ~/.wgetrc

RUN cd /deps && \
    wget https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz && \
    tar xf geant4-v11.0.2.tar.gz && \
    cd geant4-v11.0.2 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/deps/geant4 && \
    make -j8 install


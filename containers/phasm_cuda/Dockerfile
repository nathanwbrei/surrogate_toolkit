# A Dockerfile based on nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#
# The building process takes 20 minutes.
# To see the building message, try `docker build progress=plain <PATH>`

# Ref:
# - MLPerf Dockerfile
# https://github.com/mlcommons/training_results_v2.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09/Dockerfile
# - JANA Dockerfile
# https://github.com/JeffersonLab/JANA2/blob/master/containers/Docker/Dockerfile

ARG FROM_IMAGE_NAME=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${FROM_IMAGE_NAME}
USER root

# Install basic development environment
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

# install clang static analyzer
RUN apt-get update && apt-get -y install clang-format clang-tidy clang-tools clang clangd libc++-dev libc++1 libc++abi-dev libc++abi1 libclang-dev libclang1 liblldb-dev libllvm-ocaml-dev libomp-dev libomp5 lld lldb llvm-dev llvm-runtime llvm python3-clang

# Updated on May/24/2023
ARG TORCH_CPU_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
# last updated on Mar/18/2023
ARG TORCH_CUDA_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip"
# TORCH_URL can switch between libtorch_cuda and libtorch_cpu
ARG PIN_URL="https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz"
ARG DWARF_URL="https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz"
ARG DEP_PATH=/deps

WORKDIR ${DEP_PATH}

# Fix certificates
ADD http://pki.jlab.org/JLabCA.crt /usr/local/share/ca-certificates/JLabCA.crt
RUN chmod 644 /usr/local/share/ca-certificates/JLabCA.crt && \
    update-ca-certificates && \
    echo "check_certificate = off" >> ~/.wgetrc

# Install libtorch
# Downloading the Torch zip appears to break due to mysterious checksum
# problems, so we download it externally
ADD libtorch libtorch


ENV LIBTORCH_INSTALL_PATH=${DEP_PATH}/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}

# Install pin
ADD ${PIN_URL} pin.tar.gz
RUN tar --no-same-owner -xf pin.tar.gz &&  \
    rm pin.tar.gz && \
    mv pin-3.22-98547-g7a303a835-gcc-linux pin
ENV PIN_ROOT=$DEP_PATH/pin

# Install libdwarf
ADD $DWARF_URL libdwarf.tar.xz
RUN tar -xf libdwarf.tar.xz && \
    rm libdwarf.tar.xz && \
    mv libdwarf-0.3.4 libdwarf && \
    mkdir libdwarf/build libdwarf/installdir && \
    cd libdwarf/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$DEP_PATH/libdwarf/installdir &&  \
    make -j8 install && \
    cd .. && \
    rm -rf build

# build and install JANA, locked to v2.0.6
# add cmake-specific configurations
RUN export CC=`which gcc` && \
    export CXX=`which g++` && cmake --version
RUN git clone https://github.com/JeffersonLab/JANA2.git --branch=v2.0.6 && \
    cd JANA2 && \
    mkdir build install && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    make -j8 install && \
    cd .. && \
    rm -rf build

# install Julia
RUN cd /deps && \
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.1-linux-x86_64.tar.gz && \
    tar xf julia-1.9.1-linux-x86_64.tar.gz


ENV JANA_HOME="${DEP_PATH}/JANA2/install"
ENV PATH=${JANA_HOME}/bin:/deps/julia-1.9.1/bin:${PATH}
ENV DEPS=$DEP_PATH

LABEL author="nbrei@jlab.org"
LABEL description="A CUDA+Ubuntu container that includes JANA2, libtorch, julia, Intel PIN, libdwarf, llvm/clang, and geant4"

WORKDIR /app
CMD /bin/bash

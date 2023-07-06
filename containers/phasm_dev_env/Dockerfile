# Note: Run me from PHASM's root directory like so:
# sudo docker build -f containers/phasm_dev_env/Dockerfile -t phasm_dev_env .

FROM --platform=linux/amd64 ubuntu:jammy

LABEL author="nbrei@jlab.org"
LABEL description="Basic development environment for PHASM."

ARG EXTERNAL_DOWNLOAD_DIR=downloads
ARG PHASM_USE_TORCH=1
ARG PHASM_USE_JULIA=1
ARG PHASM_USE_JANA=1
ARG PHASM_USE_LLVM=1
ARG PHASM_USE_GEANT4=1
ARG PHASM_USE_PIN=0
ARG PHASM_USE_DWARF=0

USER root
WORKDIR /app

RUN mkdir /scripts
ADD scripts /scripts/
RUN /scripts/setup_debian.sh

RUN mkdir /downloads
ADD $EXTERNAL_DOWNLOAD_DIR /downloads/
RUN /scripts/install_deps.sh /downloads /deps

ENV DEPS=/deps
ENV JANA_HOME=/deps/JANA
ENV PATH=${JANA_HOME}/bin:/deps/julia/bin:${PATH}
ENV LD_LIBRARY_PATH=/deps/libtorch/lib:${LD_LIBRARY_PATH}
#ENV CC=`which gcc`
#ENV CXX=`which g++`

CMD /bin/bash


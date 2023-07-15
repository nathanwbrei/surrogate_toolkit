# A Dockerfile based on nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#
# To see the building message, try `docker build progress=plain <PATH>`

#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# LibTorch bundles CuDNN, so we can use the smaller base image

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL author="nbrei@jlab.org"
LABEL description="A CUDA+Ubuntu container that includes JANA2, libtorch, julia, Intel PIN, libdwarf, llvm/clang, and geant4"

ARG EXTERNAL_DOWNLOAD_DIR=downloads
ARG PHASM_USE_CUDA=1
ARG PHASM_USE_TORCH=1
ARG PHASM_USE_JULIA=1
ARG PHASM_USE_JANA=1
ARG PHASM_USE_LLVM=1
ARG PHASM_USE_GEANT4=1
ARG PHASM_USE_PIN=1
ARG PHASM_USE_DWARF=1

USER root
WORKDIR /app

RUN mkdir /scripts
ADD scripts /scripts/
RUN /scripts/setup_debian.sh

# We used to obtain a pre-built Geant4, but turned this off because
# Docker really hates conditional file COPYs. 
#RUN mkdir -p /deps/geant4
#COPY --from=geant4_env:latest /deps/geant4 /deps/geant4

RUN mkdir /downloads
ADD $EXTERNAL_DOWNLOAD_DIR /downloads/
RUN /scripts/install_deps.sh /downloads /deps

ENV DEPS=/deps
ENV JANA_HOME=/deps/JANA
ENV PATH=${JANA_HOME}/bin:/deps/julia/bin:${PATH}
ENV LD_LIBRARY_PATH=/deps/libtorch/lib:${LD_LIBRARY_PATH}

CMD /bin/bash

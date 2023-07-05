# A Dockerfile based on nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#
# The building process takes 20 minutes.
# To see the building message, try `docker build progress=plain <PATH>`

# Ref:
# - MLPerf Dockerfile
# https://github.com/mlcommons/training_results_v2.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09/Dockerfile
# - JANA Dockerfile
# https://github.com/JeffersonLab/JANA2/blob/master/containers/Docker/Dockerfile

#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# LibTorch bundles CuDNN, so we can use the smaller base image

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL author="nbrei@jlab.org"
LABEL description="A CUDA+Ubuntu container that includes JANA2, libtorch, julia, Intel PIN, libdwarf, llvm/clang, and geant4"

USER root
WORKDIR /app

RUN mkdir /scripts
ADD scripts /scripts/
RUN /scripts/setup_debian.sh

RUN mkdir /deps
COPY --from=phasm_dev_env:latest /deps /deps

ENV JANA_HOME=/deps/JANA
ENV PATH=${JANA_HOME}/bin:/deps/julia/bin:${PATH}
ENV LD_LIBRARY_PATH=/deps/libtorch/lib:${LD_LIBRARY_PATH}
ENV DEPS=/deps
#ENV CC=`which gcc`
#ENV CXX=`which g++`

CMD /bin/bash

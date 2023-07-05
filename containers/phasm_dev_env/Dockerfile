# Note: Run me from PHASM's root directory like so:
# sudo docker build -f containers/phasm_dev_env/Dockerfile -t phasm_dev_env .

FROM ubuntu:jammy

LABEL author="nbrei@jlab.org"
LABEL description="Basic development environment for PHASM."

ARG EXTERNAL_DOWNLOAD_DIR=downloads

USER root
WORKDIR /app

RUN mkdir /scripts
ADD scripts /scripts/
RUN /scripts/setup_debian.sh

RUN mkdir /downloads
ADD $EXTERNAL_DOWNLOAD_DIR /downloads/
RUN /scripts/install_deps.sh /downloads /deps

ENV JANA_HOME=/deps/install
ENV PATH=${JANA_HOME}/bin:/deps/julia/bin:${PATH}
ENV LD_LIBRARY_PATH=/deps/libtorch/lib:${LD_LIBRARY_PATH}
ENV DEPS=/deps
#ENV CC=`which gcc`
#ENV CXX=`which g++`

CMD /bin/bash


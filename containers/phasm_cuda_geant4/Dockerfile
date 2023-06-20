
FROM phasm_cuda:latest
USER root
WORKDIR /deps

RUN cd /deps && \
    wget https://gitlab.cern.ch/geant4/geant4/-/archive/v11.0.2/geant4-v11.0.2.tar.gz && \
    tar xf geant4-v11.0.2.tar.gz && \
    cd geant4-v11.0.2 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/deps/geant4 && \
    make -j8 install


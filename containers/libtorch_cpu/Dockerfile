FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

USER root
RUN mkdir /app
WORKDIR /app

# Install basic development environment
# Last updated on July 11, 2023 by xmei@jlab.org. Add cpprestsdk.
RUN apt update -y \
 && apt install -y gdb valgrind cmake wget unzip vim libasan6 less exa bat git zlib1g-dev libcpprest-dev

# Fix certificates
ADD http://pki.jlab.org/JLabCA.crt /usr/local/share/ca-certificates/JLabCA.crt
RUN chmod 644 /usr/local/share/ca-certificates/JLabCA.crt && \
    update-ca-certificates && \
    echo "check_certificate = off" >> ~/.wgetrc

# Install libtorch
# This is still a cpu-only libtorch. Url updated on May 24 2023 by xmei@jlab.org.
ADD "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip" libtorch.zip
RUN unzip libtorch.zip

# Install libdwarf
ADD "https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz" libdwarf.tar.xz
RUN tar -xf libdwarf.tar.xz \
 && mv libdwarf-0.3.4 libdwarf \
 && mkdir libdwarf/build \
 && mkdir libdwarf/installdir \
 && cd libdwarf/build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/app/libdwarf/installdir \
 && make install

# Install JANA (Do this last because we are more likely to upgrade!)
RUN git clone -b v2.0.6 http://github.com/JeffersonLab/JANA2 \
 && mkdir JANA2/install \
 && mkdir JANA2/build \
 && cd JANA2/build \
 && cmake .. -DCMAKE_INSTALL_PREFIX=/app/JANA2/install \
 && make -j8 install

ENV JANA_HOME=/app/JANA2/install
ENV PATH="$PATH:/app/JANA2/install/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/JANA2/install/plugins"

CMD /bin/bash

FROM centos:centos7.7.1908

USER root
RUN mkdir /app
WORKDIR /app

# Install ssh so that we can remote from CLion

RUN yum install -y openssh-server
RUN echo 'root:password' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN ssh-keygen -A -f /etc/ssh/ssh_host
EXPOSE 22


# Add newer versions of GDB so that CLion lets us remote debug

RUN yum install -y centos-release-scl \
 && yum-config-manager --enable rhel-server-rhscl-7-rpms \
 && yum install -y devtoolset-11
# To use: `scl enable devtoolset-11 bash`


# Install development environment ala CUE

RUN yum install -y epel-release && \
    yum install -y \
		bzip2-devel cmake3 file gcc-c++ gcc-gfortran gdb git make python python-devel protobuf-c-devel \
		sudo czmq tree wget which vim libtool valgrind libasan.so.0.0.0 \
	&& ln -s /usr/bin/cmake3 /usr/local/bin/cmake \
	&& rm -rf /var/cache/yum

# Install PyTorch
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch-shared-with-deps-latest.zip

CMD /bin/bash




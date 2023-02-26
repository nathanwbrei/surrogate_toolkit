FROM nbrei_phasm_image:latest

USER root
WORKDIR /app


#ADD https://apt.llvm.org/llvm.sh llvm.sh
#RUN apt-get update \
# && apt-get install -y lsb-release wget software-properties-common gnupg

# Fix problem with JLab certificates
# TODO: Move this into outer Dockerfile
#ADD http://pki.jlab.org/JLabCA.crt /usr/local/share/ca-certificates/JLabCA.crt
#RUN update-ca-certificates
#RUN chmod +x llvm.sh
# && ./llvm.sh 15 all

RUN apt-get update \
 && apt-get install -y  \
    libllvm15 llvm-15 llvm-15-dev llvm-15-doc llvm-15-examples llvm-15-runtime  \
    clang-15 clang-tools-15 clang-15-doc libclang-common-15-dev libclang-15-dev libclang1-15 clang-format-15 python3-clang-15 clangd-15 clang-tidy-15 \
    libc++-15-dev libc++abi-15-dev

CMD /bin/bash




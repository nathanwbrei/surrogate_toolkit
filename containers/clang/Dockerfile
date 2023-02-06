FROM ubuntu:latest

USER root
RUN mkdir /app
WORKDIR /app

# Install basic development environment
RUN apt update -y \
 && apt install -y build-essential gdb valgrind cmake wget unzip vim libasan6 less exa bat git zlib1g-dev

RUN apt install -y libclang-dev
CMD /bin/bash




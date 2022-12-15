
# Parallel Hardware viA Surrogate Models

Clone the repo and go to the directory for it
```bash
git clone https://github.com/nathanwbrei/phasm
cd phasm
```

### Build and run PHASM inside a container

#### CUDA containers (libtorch_cuda)
CUDA containers did not pass the macOS-hostOS test. Therefore, we limit
ourselves to the Linux machines and farm GPUs.

Follow [this guide](docs/farm_guide.md) to compile PHASM inside the singularity container on farm.
All the examples, including the CUDA and non-CUDA ones should be built successfully.

#### CPU containers (libtorch_cpu)
Build the container with the Dockerfile. Run the container and mount PHASM directory to `/app`.

```bash
docker run -it --volume ${PWD}:/app <container_id_or_tag>
```

Inside the container, build and install PHASM.

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="/deps/libtorch;/deps/JANA2/install" \
-DLIBDWARF_DIR="/deps/libdwarf/installdir" -DPIN_ROOT="/deps/pin" ..
```

### Bare-metal build and run
#### Complete all the install steps at once
```bash
#scl enable devtoolset-11   # Make sure you are using a recent compiler
./install.sh

# To Run a PHASM example: 
export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH

# Run the PDE solver example and dump captured data to CSV
PHASM_CALL_MODE=CaptureAndDump install/bin/phasm-example-pdesolver
```

OR if you would prefer to do it manually, follow the steps below

```bash

# Install PyTorch, Intel Pin, and JANA2 dependencies
./download_deps.sh
export DEPS=`pwd`/deps

# Build 
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/pin" ..
make install

# To run one of the examples:
export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH

# Run the PDE solver example and dump captured data to CSV
PHASM_CALL_MODE=CaptureAndDump install/bin/phasm-example-pdesolver

# Run vacuum tool against the example target program
install/bin/phasm-memtrace-pin install/bin/phasm-example-memtrace
```

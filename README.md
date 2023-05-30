
# Parallel Hardware viA Surrogate Models

## Resources
- Test results tracking: https://docs.google.com/spreadsheets/d/19iVKLKfVFlASZSgHDrYQx6XqakzqsAp0i52GIF5nEWs
- Docker images (libtorchx.x.x_cpu and libtorchx.x.x_cuda): https://hub.docker.com/repository/docker/xxmei/phasm/general
- Singularity containers on ifarm: `/work/epsci/shared_pkg/phasm-gpu_xxx-xx.sif`

## Build Instructions
Refer to the top section for docker/singularity image urls.

### Docker container

```bash
# start from PHASM root directory
docker pull <docker_img>  # optional
docker run -it -v ${PWD}:/app <docker_img>  # mount current directory

mkdir build && cd build   # build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" \
 -DLIBDWARF_DIR="$DEPS/libdwarf/installdir" -DPIN_ROOT="$DEPS/pin" ..
make -j32 install  # install
```

### Singularity container (with GPU/CUDA support)

Follow [this guide](docs/farm_guide_singularity.md) to compile PHASM with the singularity container on ifarm nodes.
All the examples, including the CUDA and non-CUDA ones should be built successfully.


### Bare-metal build and run
This method is not recommended. However, the process is recorder [here](/docs/bare_metal_build.md).

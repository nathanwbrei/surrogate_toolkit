
# Parallel Hardware viA Surrogate Models

## Resources
- Test results tracking: https://docs.google.com/spreadsheets/d/19iVKLKfVFlASZSgHDrYQx6XqakzqsAp0i52GIF5nEWs
- Docker images:
    - https://hub.docker.com/r/nbrei/phasm_dev_env
    - https://hub.docker.com/r/xxmei/phasm
- Singularity containers on ifarm: `/work/epsci/shared_pkg/phasm-gpu_xxx-xx.sif`

## Build Instructions
Refer to the top section for docker/singularity image urls.

### Docker container

```bash
# If using Spack-based docker, activate env by
# . /etc/profile.d/phasm_spack_environment.sh

# Start from PHASM root directory
docker pull <docker_img>  # optional
docker run -it -v ${PWD}:/app <docker_img>  # mount current directory

mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" ..
make -j32 install
```
### Singularity container (with GPU/CUDA support)

Follow [this guide](docs/farm_guide_singularity.md) to compile PHASM with the singularity container on ifarm nodes. All the examples, including the CUDA and non-CUDA ones should be built successfully.

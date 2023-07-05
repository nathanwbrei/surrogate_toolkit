
This Dockerfile produces an Ubuntu Jammy container with just Singularity installed.
This is useful for building Singularity containers on macOS.
Run as follows:

```bash
docker build -t phasm_singularity .
docker run -it -v $PATH_TO_PHASM_SOURCE/containers:/containers phasm_singularity 
singularity build /containers/phasm_minus_cuda.sif /containers/phasm_minus_cuda/phasm_minus_cuda.def
```


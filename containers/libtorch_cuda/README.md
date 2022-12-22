# Using self-built singularity containers on the farm

By default, the singularity cache is at `${HOME}/.singularity/cache`. This must be
reset on ifarm due to limited `${HOME}` space. Update your `SINGULARITY_CACHEDIR`
environment variable to another location.

### Build from singularity definition file

Two ways to build the singularity container from [the provided def file](cu-dev.def).

1. On your OWN device, which has singularity installed, build with `sudo` (this is not working in JLab network).
    ```bash
    sudo singularity build <target_img_name>.sif <your_def_name>.def
    ```
   Then `scp` the built `<target_img_name>.sif` to ifarm.
    
   On the farm, run the container with `--nv` option for GPU support. E.g.
    ```bash
    module load singularity
    srun --gres gpu:A100:1 -p gpu --cpus-per-task=4 --mem-per-cpu=8000 \
      --pty bash  # we need more memory here
    singularity run --nv <target_img_name>.sif  # "--nv" is necessary for GPUs
    ```
   The container is about 6.5 GB now which might need a cleanup in the future.

2. Using the singularity remote build. You can paste the content in the `def` file to
the [web GUI](https://cloud.sylabs.io/builder) and follow the GUI instructions.
Once the building is completed, you can pull it to the farm.
   ```bash
   singularity pull library://<your_syslab_img_path_and_tag>
   ```

   You need to register to use the service. The remote build process takes 1 hour.



### Build from Dockerfile

As ifarm does not support docker, to use GPU, eventually a singularity container is needed.
[A libtorch_cuda dockerfile](Dockerfile) is provided. First build the docker container via
`docker build`. Push it to the docker hub and then pull it via singularity.

```bash
singularity pull <singularity_container_name>.sif docker://<docker_hub_container_repo_and_tag>
```

When pulling on ifarm, there might be `PROTOCOL_ERROR` due to slow network. Retry and it will succeed.

### References
- Official guide on building a singularity container:
https://docs.sylabs.io/guides/3.5/user-guide/build_a_container.html
- How to write a `def` file: https://docs.sylabs.io/guides/3.5/user-guide/definition_files.html
- https://cloud.sylabs.io

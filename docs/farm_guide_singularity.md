# Running PHASM on `ifarm` GPU Nodes

## Within a singularity container
[A singularity definition file](../containers/libtorch_cuda/cu-dev.def) is provided to build
a container running PHASM codebase on the farm GPUs.

Clone the PHASM repo to the farm **OTHER THAN** your home directory.

### 1. Load the singularity module

Place the singularity container under your PHASM repo. If you pull the singularity container from
[sylabs](https://cloud.sylabs.io/)
library, remember to reset the singularity cache environment variable `SINGULARITY_CACHEDIR`.

```bash
# module use /apps/modulefiles
module load singularity
```

### 2. Build PHASM without GPU

```bash
singularity run <singularity_container_name>.sif
```

On the login node, launch the container with the above command. Inside the container, each line begins with
`Singularity> `.

Build PHASM as below. Though there is no GPU on the login node, the building process
should be successful.

```bash
Singularity> mkdir build && cd build
Singularity> cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf/installdir" \
 -DPIN_ROOT="$DEPS/pin" -DUSE_CUDA=ON ..
Singularity> make -j32 install
```

Run the pinn-pde-example, it will show that the CPU is doing
the training work.

```bash
Singularity> ./install/bin/phasm-example-pinn-pdesolver 
####### A cpp torch example with PINN heat equation. #######

No CUDA device. Training on CPU.

y_train sizes: [260, 1]
y_train.device().type(): cpu
...
```

### 3. Run PHASM with a GPU

Ask for a GPU node and run the container with option `--nv`.

```bash
# on the login node
srun --gres gpu:A100:1 -p gpu --cpus-per-task=4 --mem-per-cpu=8000 --pty bash  # ask for more memory here

# on the GPU node
singularity run --nv <singularity_container_name>.sif
# run the loading-pt example with GPU
Singularity> ./install/bin/phasm-example-loading-pt /work/epsci/shared_pkg/lstm_model.pt 
Loading gluex-tracking-lstm pt model.................... succeed

Run model on CUDA device 0.
  CUDA device name: NVIDIA A100 80GB PCIe
  CUDA compute capacity: 8.0
  LibTorch version: 1.13.0
```

### 4. Run Geant4 within singularity container

The images copied to `ifarm` `/scigroup/cvmfs/epsci/singularity/images` subdirectory are automatically uploaded to
`/cvmfs/oasis.opensciencegrid.org/jlab/epsci/singularity/images` (within a 4-hr delay).

David has [an instruction](https://wiki.jlab.org/epsciwiki/index.php/HOWTO_build_and_run_PHASM_on_Geant4_examples)
on how to pull `root` and `geant4` into existing containers. Here we walk through the process again.

A CUDA singularity container is available here.

```bash
/cvmfs/oasis.opensciencegrid.org/jlab/epsci/singularity/images/phasm-gpu_Mar-17.sif
```

Note that as X11 forwarding (via `srun/salloc --x11`) is not supported with `ifarm` Slurm, we are not available to see the simulation runtime as we usually do on login nodes.

```bash
# On login node
ifarm1801.jlab.org> module use /apps/modulefiles
ifarm1801.jlab.org> module load singularity
# Ask for a GPU node in interactive mode
ifarm1801.jlab.org> srun --gres gpu:A100:1 -p gpu --cpus-per-task=4 --mem-per-cpu=8000 --pty bash

# On GPU node
bash-4.2$ singularity run --nv -B /cvmfs:/cvmfs /cvmfs/oasis.opensciencegrid.org/jlab/epsci/singularity/images/phasm-gpu_Mar-17.sif

# Inside the container
Singularity> nvidia-smi  # verify CUDA
Singularity> source /cvmfs/oasis.opensciencegrid.org/jlab/epsci/ubuntu/22.04/share/spack/setup-env.sh
Singularity> spack env activate phasm # activate g4&root env
# You will see below errors but donot be afraid! It's working!
# ==> Warning: couldn't get environment settings for geant4@11.1.0 /px46psz
#  Error writing to config file: '[Errno 30] Read-only file system: '/cvmfs/oasis.opensciencegrid.org/jlab/# epsci/ubuntu/22.04/var/spack/environments/phasm/.spack.yaml.tmp''

# Set the g4 dependencies
Singularity> export G4EX=$SPACK_ROOT/opt/spack/linux-ubuntu22.04-x86_64/gcc-11.3.0/geant4-11.1.0-px46pszk3frzg74fdbsqktipkohbyq3u/share/Geant4/examples/basic/B4
# Build and install g4
Singularity> cmake -S $G4EX -B g4/build -DCMAKE_INSTALL_PREFIX=g4/install -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=1
Singularity> cd g4
Singularity> cmake --build build --target install -- -j8

# Run g4 example
Singularity> cp -rp $G4EX/macros  .
Singularity> cd macros  # necessary macro files are loacted here
Singularity> ../install/bin/exampleB4d
# If you are on the login node, a graphic window should come out.
# But if you are on GPU node where X11 is forbidden, there is no GUI.
# Without GUI, the example will complain as below.
# ERROR: G4VisManager::IsValidView(): Current view is not valid.
# ERROR: G4VisManager::PrintInvalidPointers:
#   Graphics system is OpenGLStoredX but:
#   Null scene pointer. Use "/vis/drawVolume" or "/vis/scene/create".
#   Null viewer pointer. Use "/vis/viewer/create".

# G4 exampleB4d is running.
Idle> run/beamOn 1
# Quit the simulation
Idle> exit
Singularity>  # still in the singularity container
```

#### References
- [Ifarm spack admin how-to](https://wiki.jlab.org/epsciwiki/index.php/SPACK_Mirror_on_JLab_CUE#Setting_up_a_new_platform)
- [Ifarm phasm-g4 guide](https://wiki.jlab.org/epsciwiki/index.php/HOWTO_build_and_run_PHASM_on_Geant4_examples)

## Bare-Metal
[This](ifarm_guide_bare_metal.md) is strongly not advised but could serve as a reference on how to build containers.

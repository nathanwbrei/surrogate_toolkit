# Running PHASM on `ifarm` GPU Nodes

## Within a singularity container
[A singularity definition file](../containers/libtorch_cuda/cu-dev.def) is provided to build
a container running PHASM codebase on the farm GPUs.

Clone the PHASM repo to the farm other than your `/home`.

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
`/cvmfs/oasis.opensciencegrid.org/jlab/epsci/singularity/images` (within 4-hr delay).
Currently, a singularity container based on cuda-11.8+libtorch-1.13+cudnn-8.0+gcc-1.13 is available
at the above position.

David has [an instruction](https://wiki.jlab.org/epsciwiki/index.php/HOWTO_build_and_run_PHASM_on_Geant4_examples)
on how to pull `root` and `geant4` into existing containers. Here we walk through the process again.

On **login node**:

```bash
# At phasm root directory
ifarm1802.jlab.org> mkdir g4
ifarm1802.jlab.org> singularity shell -B /cvmfs:/cvmfs \
    /cvmfs/oasis.opensciencegrid.org/jlab/epsci/singularity/images/libtorch_cuda_feb_23.sif

# Inside the container
Singularity> source /cvmfs/oasis.opensciencegrid.org/jlab/epsci/ubuntu/22.04/share/spack/setup-env.sh # takes a while
Singularity> spack env activate phasm # activate g4&root
Singularity> export G4EX=$SPACK_ROOT/opt/spack/linux-ubuntu22.04-x86_64/gcc-11.3.0/geant4-11.1.0-px46pszk3frzg74fdbsqktipkohbyq3u/share/Geant4/examples/basic/B4

# Build and install g4
Singularity> cmake -S $G4EX -B g4/build -DCMAKE_INSTALL_PREFIX=g4/install -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=1
Singularity> cd g4
Singularity> cmake --build build --target install -- -j8

# Run g4 example
Singularity> cp -rp $G4EX/macros  .
Singularity> cd macros
Singularity> ../install/bin/exampleB4d  # a graphic window should come out eventually

# g4 simulation example
Idle> run/beamOn 1
# Quit the simulation
Idle> exit
```



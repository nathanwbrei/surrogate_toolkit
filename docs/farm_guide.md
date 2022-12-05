# Running PHASM on Farm GPU Nodes

Running PHASM on farm nodes is a bit tricky. We create this guide to walk through the building process.

We assume you are already on the farm login node (`ifarmxxxx.jlab.org`).
 Clone the `phasm` folder repo to the farm group repo `/work/<group_name>/<user_account>/`.
 Do not use your `/home` space since it is only of 5GB and will be insufficient for installing the dependencies.

## 1. Load `cmake`, `gcc` and `cuda`
PHASM requires `C++14` and `cmake 3.9+`, and the farm default setting does not satisfy. Reload the modules as below.

```bash
# reload gcc
ifarm1801.jlab.org> gcc --version
gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)
ifarm1801.jlab.org> module unload gcc
ifarm1801.jlab.org> module load gcc/10.2.0
ifarm1801.jlab.org> which gcc
/apps/gcc/10.2.0/bin/gcc

# reload cmake
ifarm1801.jlab.org> module unload cmake
ifarm1801.jlab.org> module load cmake/3.21.1
ifarm1801.jlab.org> which cmake
/apps/cmake/3.21.1/bin/cmake
```

You also need to load the CUDA module manually.

```bash
ifarm1802.jlab.org> module load cuda
ifarm1802.jlab.org> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Aug_15_21:14:11_PDT_2021
Cuda compilation tools, release 11.4, V11.4.120
Build cuda_11.4.r11.4/compiler.30300941_0
```

You need to load these modules every time when you log in to the farm.
 To save time, you can create a bash file including the `module` commands and source it then.

Note that you can either load these modules before or after logging to the compute node.
 But on the compute node, you need to enable the `module` command first.

```
bash-4.2$ source /etc/profile.d/modules.sh
```

## 2. Ask for a GPU node

Use the below Slurm command to ask for one GPU node under `gpu` partition.
 Here I am asking for a Tesla T4 GPU.
 The TitanRTX GPU is available by replacing the `--gres` option with `gpu:TitanRTX:1`.

```bash
ifarm1801.jlab.org> srun --gres gpu:T4:1 -p gpu -n 1 --pty bash
```

Once you are on the GPU node, you should see each line begins with `bash-4.2`.

Make sure your `cmake` and `gcc` are of the newer versions.
```bash
bash-4.2$ which gcc
/apps/gcc/10.2.0/bin/gcc
bash-4.2$ which cmake
/apps/cmake/3.21.1/bin/cmake
```

Check you CUDA version with `nvidia-smi`. 
As of Aug 2022, the version is `11.4`, which is compatible with `libtorch CUDA 11.3` 
and `cuDNN CUDA 11.x`.

```bash
bash-4.2$ nvidia-smi
Tue Aug  9 15:36:37 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
```

Tell `cmake` the newer gcc path.
 Otherwise, it will still use the old `gcc 4.8.5`.
 This need to be done every time you log onto the compute node.
 You can add them to `.bashrc` instead.

```bash
bash-4.2$ export CC=`which gcc`
bash-4.2$ export CXX=`which g++`
```


## 3. Get the PHASM dependencies

Create a folder `deps` under the phasm parent folder and go into it. 

```bash
bash-4.2$ mkdir deps
bash-4.2$ cd deps
```

Export the current path and take down it as `<DEPS_PATH>`.
 For example, my `<DEPS_PATH>` is `/work/epsci/xmei/phasm/deps`.
```bash
bash-4.2$ export DEPS=`pwd`
```

We need to install the dependencies of `libtorch`, `cuDNN` (`libtorch`'s CUDA dependencies), `libdwarf`, `pin` and `JANA2`. The below sub-steps are of arbitrary sequence.

### Install `cuDNN`
If you want to install `libtorch` with CUDA, `cuDNN` is required. 

`cuDNN` is available at NVIDIA official site [here](https://developer.nvidia.com/rdp/cudnn-download), but it will ask you to sign in first.
 To make things easier, `/work/epsci/shared_pkg` has a backup copy of the related packages.
 You can use them as an alternative.

```bash
bash-4.2$ ls /work/epsci/shared_pkg/
cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip  # cxx11 ABI
libtorch-shared-with-deps-1.12.0+cu113.zip  # Pre-cxx11 ABI
```

```bash
# install cuDNN
bash-4.2$ tar -xf /work/epsci/shared_pkg/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
bash-4.2$ mv cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/ cudnn
```

### Install `libtorch`

Farm does not allow you to access the online `libtorch` download path.
 You have to download it to **YOUR OWN DEVICE** and `scp` it.
 An easier way is to use the backup copy.

```bash
bash-4.2$ unzip /work/epsci/shared_pkg/libtorch-shared-with-deps-1.12.0+cu113.zip
```

Note this step requires `cmake`. You must configure the newer `cmake` and `gcc` module/path beforehand. 


### Install `pin`

```bash
bash-4.2$ wget https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
bash-4.2$ tar -xf pin-3.22-98547-g7a303a835-gcc-linux.tar.gz
bash-4.2$ mv pin-3.22-98547-g7a303a835-gcc-linux.tar.gz pin
```

### Build and install `JANA2`

```bash
bash-4.2$ git clone http://github.com/JeffersonLab/JANA2
bash-4.2$ mkdir JANA2/install
bash-4.2$ mkdir JANA2/build
bash-4.2$ cd JANA2/build
bash-4.2$ cmake .. -DCMAKE_INSTALL_PREFIX=../install
bash-4.2$ make -j8 install
bash-4.2$ cd ../..
```

### Build and install `libdwarf`
```bash
bash-4.2$ wget https://github.com/davea42/libdwarf-code/releases/download/v0.3.4/libdwarf-0.3.4.tar.xz
bash-4.2$ tar -xf libdwarf-0.3.4.tar.xz
bash-4.2$ mkdir libdwarf-0.3.4/build
bash-4.2$ mkdir libdwarf-0.3.4/installdir
bash-4.2$ cd libdwarf-0.3.4/build
bash-4.2$ cmake .. -DCMAKE_INSTALL_PREFIX=../installdir
bash-4.2$ make install
```

## 4. Build and run PHASM
Add `libtorch` to the path.

```bash
bash-4.2$ export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH
```

Go to the parent folder and build PHASM following the below steps.
 You need to tell `cmake` all the dependency paths.

```bash
bash-4.2$ mkdir build
bash-4.2$ cd build/
bash-4.2$ cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/cudnn" ..
# bash-4.2$ cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/cudnn;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/pin" ..
bash-4.2$ make
```

### Run the examples
- The pinn-pde-solver is a stand-alone example only utilizes `libtorch`. 
You can run this example to verify whether `libtorch` is configured appropriately.

```bash
bash-4.2$ ./examples/pinn_pde_solver/phasm-example-pinn-pdesolver
####### A cpp torch example with PINN heat equation. #######

CUDA available. Training on GPU.
...
```

## Notes
### The libtorch version

According to libtorch [installation guide](https://pytorch.org/get-started/locally/), the `cxx11 ABI` version should fit the farm node better.
 But when I compiled the code with `cxx11 ABI` `libtorch`, a `glibc` link error occured as below, indicating an old `glibc` (`glibc 2.27` is required but `glibc 2.17` is provided).
 Reinstalling the `glibc 2.27` also failed but `glibc 2.23` succeeded.
 Though including the `glibc 2.23` to the path, the probolem is unsolved.
 Thus I switched to the old `Pre-cxx11 ABI` `libtorch`.

```bash
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `powf@GLIBC_2.27'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `log2f@GLIBC_2.27'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `lgammaf@GLIBC_2.23'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `expf@GLIBC_2.27'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `exp2f@GLIBC_2.27'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `lgamma@GLIBC_2.23'
/work/epsci/xmei/phasm/deps/libtorch/lib/libtorch_cpu.so: undefined reference to `logf@GLIBC_2.27'

# check the libc version
bash-4.2$ ldd --version
ldd (GNU libc) 2.17

# build glibx 2.27
../configure --prefix=/work/epsci/xmei/phasm/deps/glibc-2.27/install --disable-werror
```

### Select GPU or CPU to run PINN
Now the PINN example can run either on (one) GPU or CPU without changes
 in the code. But we do need to switch the `libtorch` libraries.

```bash
[xmei@sciml2103 phasm]$ ls $DEPS/
cudnn	    glibc-2.27	libdwarf-0.3.4	libtorch-cpu	   libtorch-cu-precxx11
glibc-2.23  JANA2	libtorch	libtorch-cu-cxx11  pin

# check the libtorch version. Below is the CUDA version
[xmei@sciml2103 phasm]$ cat $DEPS/libtorch/build-version
1.12.0+cu113
```

This is done with symbol links. As shown above,
 I have different versions of `libtorch` libraries
 under `$DEPS`. Each time before building the project,
 I update the symbol link to let it point to the correct
 version. For example, if I want the code to run on CPU
 on a GPU node, I link `libtorch` to `libtorch-cpu`,
 and it will successfully switch to CPU.

```bash
# Run on GPU
-pinn-pdesolver phasm]$ ./build/examples/pinn_pde_solver/phasm-example-
####### A cpp torch example with PINN heat equation. #######

CUDA available. Training on GPU.

Training started...
  iter=100, loss=11.03283, loss.device().type()=cuda
...
```

```bash
# Run on CPU
...
Training on CPU.

Training started...
  iter=100, loss=23.37409, loss.device().type()=cpu
```

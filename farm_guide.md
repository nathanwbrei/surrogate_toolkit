# Running PHASM on Farm Nodes

Running PHASM on farm nodes is a bit tricky. We create this guide to walk through the installation process.

We assume you are already on the farm login node (`ifarmxxxx.jlab.org`) and you already clone the `phasm` folder from github. 

## 1. Load the newer `cmake` and `gcc`
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
You need to load these modules every time when you login to the farm. To save time, you can create a bash file including the `module` commands and source it then.

Note that you can either load these modules before logging to the compute node, or on the compute node. But on the compute node, you need to find the `module` command first.

```
bash-4.2$ source /etc/profile.d/modules.sh
```

## 2. Ask for a compute node

Use the below Slurm command to find an idle partition. You should see the available compute resources and the partition `<par_name>`.

```bash
ifarm1801.jlab.org> sinfo | grep idle
jupyter          up 1-00:00:00      1   idle farm180302
```

Log onto the compute node.
```bash
ifarm1802.jlab.org> srun -p <par_name> --pty bash
srun: job 61240696 queued and waiting for resources
srun: job 61240696 has been allocated resources
bash-4.2$ 
```

Once you are on the compute node, you should see each line begins with `bash-4.2`.

Make sure your `cmake` and `gcc` are of the newer versions.
```bash
bash-4.2$ which gcc
/apps/gcc/10.2.0/bin/gcc
bash-4.2$ which cmake
/apps/cmake/3.21.1/bin/cmake
```

Tell `cmake` to use the correct gcc path. Otherwise it will still use the old `gcc 4.8.5`. This need to be done every time you log onto the compute node, too. You can add them to `.bashrc` instead.

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

```bash
bash-4.2$ export DEPS=`pwd`
bash-4.2$ echo $DEPS
/home/xmei/projects/phasm/deps
```

We need to install the dependencies of `libtorch`, `libdwarf`, `pin` and `JANA2`. The below four sub-steps are of arbitrary sequence.

### Install `libtorch`
Farm does not allow you to access the online `libtorch` download path, so you have to download it to **YOUR OWN DEVICE** and copy it to the farm. Use `scp` to do the task. Give your farm account `<user_account>` and `<DEPS_PATH>`. You will be asked for the password.

```bash
# ON YOUR OWN DEVICE
wget --no-check-certificate https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip
scp libtorch-*.zip <user_account>@ifarm.jlab.org:<DEPS_PATH>/
```

Once `libtorch` is fully copied to the farm, unzip it. Note this step requires `cmake`. So you must configure the newer `cmake` and `gcc` module/path beforehead.

```bash
# ON THE FARM COMPUTE NODE
bash-4.2$ unzip libtorch-*.zip
```

Add `libtorch` to the path.

```bash
bash-4.2$ export LD_LIBRARY_PATH=$DEPS/libtorch/lib:$LD_LIBRARY_PATH
```

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
Go to the parent folder and build PHASM.
```bash
bash-4.2$ pwd
/home/xmei/projects/phasm
bash-4.2$ mkdir build
bash-4.2$ cd build/
bash-4.2$ cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/pin" ..
```

### Run the examples
- The pinn-pde-solver is a stand-alone example only utilizes `libtorch`. You can run this example to verify whether `libtorch` is configured appropriately.

```bash
bash-4.2$ ./examples/pinn_pde_solver/phasm-example-pinn-pdesolver
####### A cpp torch example with PINN heat equation. #######

Training on CPU.
...
```


## Problems unsolved
My build process stops at `phasm-example-magfieldmap`. There seems to be some problem with linking to JANA2 objects.

```bash
[ 33%] Linking CXX executable phasm-example-magfieldmap
CMakeFiles/phasm-example-magfieldmap.dir/magnetic_field_map.cpp.o: In function `JParameter* JParameterManager::SetDefaultParameter<std::string>(std::string, std::string&, std::string)':
magnetic_field_map.cpp:(.text._ZN17JParameterManager19SetDefaultParameterISsEEP10JParameterSsRT_Ss[_ZN17JParameterManager19SetDefaultParameterISsEEP10JParameterSsRT_Ss]+0x7c): undefined reference to `JParameterManager::to_lower(std::string const&)'
magnetic_field_map.cpp:(.text._ZN17JParameterManager19SetDefaultParameterISsEEP10JParameterSsRT_Ss[_ZN17JParameterManager19SetDefaultParameterISsEEP10JParameterSsRT_Ss]+0x3b7): undefined reference to `JParameterManager::to_lower(std::string const&)'
...
```

When install JANA2, not sure whether the below is expected.
```bash
bash-4.2$ cmake .. -DCMAKE_INSTALL_PREFIX=../install
-- -----------------------
-- Build type is 
-- Installation directory is /home/xmei/projects/phasm/deps/JANA2/install
-- USE_ROOT    Off
-- USE_ZEROMQ  Off
-- USE_XERCES  Off
-- USE_PYTHON  Off
-- USE_ASAN    Off
-- -----------------------
-- Skipping support for libJANA's JGeometryXML because USE_XERCES=Off
-- Skipping examples/StreamingExample because USE_ZEROMQ=Off
-- Skipping plugins/janacontrol because USE_ZEROMQ=Off
-- Skipping plugins/janarate because USE_ROOT=Off
-- Skipping plugins/janaview because USE_ROOT=Off
-- Skipping plugins/JTestRoot because USE_ROOT=Off
-- Skipping plugins/streamDet because USE_ROOT=Off or USE_ZEROMQ=Off
-- Skipping python/* because USE_PYTHON=Off
-- Could NOT find XercesC (missing: XercesC_DIR XercesC_INCLUDE_DIR XercesC_LIBRARY) (found version "3.1.4")
-- Configuring done
-- Generating done
-- Build files have been written to: /home/xmei/projects/phasm/deps/JANA2/build
```

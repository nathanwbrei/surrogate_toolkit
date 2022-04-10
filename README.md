
# Readme

## How to build
```bash
git clone https://github.com/nathanwbrei/surrogate_toolkit
cd surrogate_toolkit

# Install PyTorch, Intel Pin, and JANA2 dependencies
./download_deps.sh
export DEPS=`pwd`/deps

# Build everything except vacuum tool
mkdir build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/install" ..
make install

# Vacuum tool is tricky because it uses a complicated Makefile to be compatible with PIN

```

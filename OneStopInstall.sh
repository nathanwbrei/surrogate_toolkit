#!/bin/bash
# exit when any command fails in the script
set -e

# Record the last command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo the error message given before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT



./download_deps.sh
export DEPS=`pwd`/deps
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/pin" ..
make install


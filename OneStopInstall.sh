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

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Assuming your system is macOS"
  cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/pin" ..
else
  echo "Assuming your system is Linux"
  cmake -DCMAKE_PREFIX_PATH="$DEPS/libtorch;$DEPS/JANA2/install" -DLIBDWARF_DIR="$DEPS/libdwarf-0.3.4/installdir" -DPIN_ROOT="$DEPS/$DEPS/pin-3.22-98547-g7a303a835-gcc-linux" ..
fi


make install


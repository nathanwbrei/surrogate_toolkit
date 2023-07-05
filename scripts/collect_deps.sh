#!/bin/bash
# This script brings into existence the set of environment variables $USE_TORCH, $USE_JULIA, etc, 
# and furnishes them with sensible defaults which the user can override. This lets us download
# the dependencies and set up our environments to include exactly what we need

declare_use_dep() {
    local usevar=$1
    local default=$2

    if [[ -v $usevar ]]; then
        echo "$usevar=${!usevar} (user override)"
    else
        export $usevar=$default
        echo "$usevar=${!usevar} (default)"
    fi
}

declare_use_dep USE_TORCH ON
declare_use_dep USE_JULIA ON
declare_use_dep USE_JANA ON
declare_use_dep USE_LLVM OFF
declare_use_dep USE_GEANT4 OFF
declare_use_dep USE_PIN OFF
declare_use_dep USE_DWARF OFF
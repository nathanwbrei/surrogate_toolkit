#!/bin/bash
# This script brings into existence the set of environment variables $PHASM_USE_TORCH, $PHASM_USE_JULIA, etc, 
# and furnishes them with sensible defaults which the user can override. This lets us download
# the dependencies and set up our environments to include exactly what we need

declare_use_dep() {
    local usevar=$1
    local default=$2

    # Used to be `-v $usevar`, but had to change it because macOS is stuck on bash v3.2 (from 2007!)
    if [ -n "${!usevar}" ]; then
        echo "$usevar = ${!usevar} (explicitly set)"
    else
        export $usevar=$default
        echo "$usevar = ${!usevar} (defaulted)"
    fi
}

declare_use_dep PHASM_USE_CUDA 0
declare_use_dep PHASM_USE_TORCH 1
declare_use_dep PHASM_USE_JULIA 1
declare_use_dep PHASM_USE_JANA 1
declare_use_dep PHASM_USE_LLVM 1
declare_use_dep PHASM_USE_GEANT4 1
declare_use_dep PHASM_USE_PIN 1
declare_use_dep PHASM_USE_DWARF 1

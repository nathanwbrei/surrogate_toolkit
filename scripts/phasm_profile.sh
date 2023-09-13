#!/bin/bash

# We need to call other scripts in our source tree (not our install tree, at least for now)
export PHASM_SOURCE=$(realpath $(dirname $0)/..)
echo "PHASM_SOURCE=$PHASM_SOURCE"

# Give ourselves perf permissions if we don't have them already
#
# export PERF_EVT_PARANOID=$(</proc/sys/kernel/perf_event_paranoid)
# sudo sh -c 'echo "-1" > /proc/sys/kernel/perf_event_paranoid'
#
# export KPTR_RESTRICT=$(</proc/sys/kernel/kptr_restrict)
# sudo sh -c 'echo "0" > /proc/sys/kernel/kptr_restrict'

echo "COMMAND=$1"
perf record -g -F 99 -- $1

perf script | $PHASM_SOURCE/external/stackcollapse-perf.pl >perf.folded

$PHASM_SOURCE/external/flamegraph.pl perf.folded >perf.svg


// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "gpu_perf_tester.hpp"
#include "flamegraph.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    hello_from_gpu_perf_tester();
    auto fg = Flamegraph(argv[1]);
    fg.print(std::cout);
}


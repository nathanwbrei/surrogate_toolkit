
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "dwarf_utils.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Vacuum tool" << std::endl;
    if (argc < 2) {
        std::cout << "Arguments: <path_to_executable> <addr_of_global>" << std::endl;
        return 1;
    }
    std::string path = argv[1]; // Super safe
    DwarfContext dwarfContext {path};
    dwarfContext.identify_global_primitive();
    dwarfContext.print_locals();
    return 0;
}


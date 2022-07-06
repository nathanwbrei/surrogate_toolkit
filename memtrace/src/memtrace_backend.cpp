
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "dwarf_utils.hpp"
#include <iostream>
#include <interpreter.hpp>

int main(int argc, char* argv[]) {
    std::cout << "Vacuum tool" << std::endl;
    if (argc < 3) {
        std::cout << "Arguments: <path_to_executable> <target_function>" << std::endl;
        return 1;
    }
    std::string path = argv[1]; // Super safe
    DwarfContext dwarfContext {path};
    dwarfContext.identify_global_primitive();
    dwarfContext.print_locals();

    std::string target_function_name;
    std::getline(std::cin, target_function_name);
    // Obtain ip from target_function_name via Dwarf
    phasm::memtrace::Interpreter i(0);

    // Idea: Pass routine name instead of int
    return 0;
}


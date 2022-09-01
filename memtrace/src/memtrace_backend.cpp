
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <interpreter.hpp>
#include "dwarf_utils.hpp"
#include <iostream>
#include <sstream>


/// The purpose of memtrace_backend is to define a generic interpreter that can be used in conjunction with
/// any dynamic instrumentation framework. This includes the memory access bookkeeping, and also includes some
/// higher-level tools for working with DWARF data.

int main(int argc, char* argv[]) {

    std::cout << "Model variable discovery tool" << std::endl;
    if (argc < 2) {
        std::cout << "Arguments: <target function ip> [<path_to_executable>]" << std::endl;
        return 1;
    }
    std::string target_fn_ip = argv[1]; // Super safe
    std::string path = argc == 2 ? argv[2] : ""; // Super safe
    /*
    DwarfContext dwarfContext {path};
    dwarfContext.identify_global_primitive();
    dwarfContext.print_locals();
    */

    std::string target_function_name;
    std::getline(std::cin, target_function_name);
    // Obtain ip from target_function_name via Dwarf

    phasm::memtrace::Interpreter interpreter(0);
    bool done = false;
    while (!done) {
        std::string line;
        std::getline(std::cin, line);
        std::stringstream ss(line);

        std::string op;
        ss >> op;

        uintptr_t ip, bp, sp, addr;
        size_t size;

        if (op == "r") {
            // read_mem
            ss >> ip;
            ss >> addr;
            ss >> size;
            ss >> bp;
            ss >> sp;
            interpreter.read_mem(ip, addr, size, bp, sp);
        }
        else if (op == "w") {
            // write_mem
            ss >> ip;
            ss >> addr;
            ss >> size;
            ss >> bp;
            ss >> sp;
            interpreter.write_mem(ip, addr, size, bp, sp);

        }
        else if (op == "et") {
            // Enter target routine
            ss >> ip;
            ss >> bp;
            interpreter.enter_fun(ip, bp);
        }
        else if (op == "xt") {
            // exit_fun
            ss >> ip;
            interpreter.exit_fun(ip);

        }
        else if (op == "er") {
            // Enter routine
            ss >> ip;
            ss >> bp;
            interpreter.enter_fun(ip, bp);
        }
        else if (op == "xr") {
            // exit_fun
            ss >> ip;
            interpreter.exit_fun(ip);

        }
        else if (op == "cm") {
            // request_malloc
            ss >> ip;
            ss >> size;
            interpreter.request_malloc(ip, size);
        }
        else if (op == "rm") {
            // receive_malloc
            ss >> ip;
            ss >> addr;
            interpreter.receive_malloc(ip, addr);
        }
        else if (op == "cf") {
            // free
            ss >> ip;
            ss >> addr;
            interpreter.free(ip, addr);
        }
        else if (op == "q") {
            done = true;
        }
        else {
            std::cout << "Error: Invalid command '" << op << "'" << std::endl;
        }
    }
    return 0;
}


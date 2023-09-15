
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "flamegraph.hpp"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: phasm-flamegraph expects at least one argument." << std::endl;
        exit(1);
    }
    std::string filename;
    bool use_tree_view = false;

    for (int i=1; i<argc; ++i) {
        if (strcmp(argv[i], "-t") == 0) {
            use_tree_view = true;
        }
        else {
            filename = argv[i];
        }
    }
    auto fg = Flamegraph(filename);
    if (use_tree_view) {
        fg.printTree();
    }
    else {
        fg.printFolded(false);
    }
}


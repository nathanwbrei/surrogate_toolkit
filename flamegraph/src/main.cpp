
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include "flamegraph.hpp"
#include <fstream>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: phasm-flamegraph expects at least one argument." << std::endl;
        exit(1);
    }

    std::ofstream filtered("perf.filtered");
    std::ofstream palettemap("palette.map");
    std::string folded_input;
    std::string eventloop_symbol;
    bool verbose = false;

    int position_idx = 0;
    for (int i=1; i<argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
        else {
            if (position_idx == 0) {
                folded_input = argv[i];
                position_idx += 1;
            }
            else if (position_idx == 1) {
                eventloop_symbol = argv[i];
                position_idx += 1;
            }
            else {
                std::cerr << "Too many arguments!" << std::endl;
                exit(1);
            }
        }
    }
    auto fg = Flamegraph(folded_input);
    fg.filter(eventloop_symbol, 0.02, 0.95);
    if (verbose) {
        fg.printTree();
    }
    else {
        fg.printCandidates();
    }
    fg.printFolded(false, filtered);
    fg.printColorPalette(palettemap);
}



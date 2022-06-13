
// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_VACUUM_H
#define SURROGATE_TOOLKIT_VACUUM_H

#include <string>
#include <optional>
#include "dwarf_model.hpp"

#include <dwarf.h>
#include <libdwarf.h>

void hello_from_vacuum_tool();

enum class VariableLocation { Unknown, Global, Stack, Heap, Argument };

struct VariableInfo {
    VariableLocation location;
    std::string name;
    std::string filename;
    int linenumber;
};

class DwarfContext final {

    Dwarf_Debug context = nullptr;
    Dwarf_Error error = nullptr;
    int error_code = 0;

    std::vector<dwarf::Die*> die_stack;

public:
    dwarf::Program program;

    explicit DwarfContext(std::string path_to_executable);
    ~DwarfContext();

    std::optional<VariableInfo> identify_global_primitive();
    void traverse_die(Dwarf_Die die, int level);
    void visit_die(Dwarf_Die die, int level);

    void visit_subprogram_die(Dwarf_Die die, std::string name);
    void visit_variable_die(Dwarf_Die die, std::string name);
    void visit_formal_parameter_die(Dwarf_Die die, std::string name);

    void print_locals();

};


dwarf::Program parseDwarfData(std::string binary);


#endif //SURROGATE_TOOLKIT_VACUUM_H

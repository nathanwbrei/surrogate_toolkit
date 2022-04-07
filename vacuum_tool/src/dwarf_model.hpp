
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_DWARF_MODEL_HPP
#define SURROGATE_TOOLKIT_DWARF_MODEL_HPP

#include <string>
#include <vector>

namespace dwarf {

struct Die {
    std::string tag;
    std::string name;
    std::vector<Die*> children;
    Die(std::string tag, std::string name) : tag(std::move(tag)), name(std::move(name)) {}
    virtual ~Die() = default;
};

struct Variable : public Die {
    std::string type;
    bool has_global_addr = false;
    size_t offset = 0;
    void* addr = nullptr;
    void* get_addr_from_offset(void* base_addr) const {
        return (char*) base_addr + offset;
    }
    Variable(std::string name) : Die("DW_TAG_variable", std::move(name)) {};
};

struct Function : public Die {

    void* instruction_ptr = nullptr;
    std::vector<Variable*> parameters;
    std::vector<Variable*> locals;
    Function(std::string name) : Die("DW_TAG_subprogram", std::move(name)) {};
};

struct Program {
    std::vector<Variable*> globals;
    std::vector<Function*> functions;
};

}

#endif //SURROGATE_TOOLKIT_DWARF_MODEL_HPP

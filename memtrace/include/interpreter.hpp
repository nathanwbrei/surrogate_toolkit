
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_INTERPRETER_HPP
#define SURROGATE_TOOLKIT_INTERPRETER_HPP

#include <vector>
#include <string>
#include <map>

namespace phasm::memtrace {

struct CodeLocation {
    uintptr_t instruction = 0;
    std::string routine_name;
    std::string file;
    int line = -1;
};

struct Variable {
    uintptr_t addr = 0;
    std::string symbol;
    bool is_input = false;
    bool is_output = false;
    std::vector<size_t> sizes; // For now there should be exactly one.
    // Would be different sizes if variable was a union. Also, how do we handle arrays of structs in this case?
    std::vector<CodeLocation> callers; // Eventually replace with CodeLocations
};

struct MemoryAllocation {
    uintptr_t addr = 0;
    size_t size = 0;
    std::vector<Variable> variables;
    CodeLocation location;
};

struct ProgramAddressRanges {

    uintptr_t stack_base = 0;
    uintptr_t heap_start = 0;
    uintptr_t heap_end = 0;
    uintptr_t globals_start = 0;
    uintptr_t globals_end = 0;
    uintptr_t constants_start = 0;
    uintptr_t constants_end = 0;

    bool is_local_var(uintptr_t addr, uintptr_t current_bp, uintptr_t current_sp);

    bool is_stack_var_below_target(uintptr_t addr, uintptr_t current_sp);

    bool is_global(uintptr_t addr);

    bool is_heap(uintptr_t addr);
};


class Interpreter {

public:
    Interpreter(uintptr_t target_fun_ip) : m_target_ip(target_fun_ip) {};
    ~Interpreter() = default;

    void enter_fun(uintptr_t ip, uintptr_t bp);

    void exit_fun(uintptr_t ip);

    void request_malloc(uintptr_t ip, size_t size);

    void receive_malloc(uintptr_t ip, uintptr_t buf);

    void free(uintptr_t ip, uintptr_t buf);

    void read_mem(uintptr_t ip, uintptr_t buf, size_t size, uintptr_t bp, uintptr_t rsp);

    void write_mem(uintptr_t ip, uintptr_t buf, size_t size, uintptr_t bp, uintptr_t rsp);

    // Below are only public so that they are easy to test. Probably move to a helper class eventually.
    // Helper class could read the stack address ranges from /proc


    MemoryAllocation* find_allocation_containing(uintptr_t addr);

    std::vector<Variable> get_variables();

    void print_variables(std::ostream& os);

private:
    std::map<uintptr_t, std::string> m_routine_names;
    std::vector<uintptr_t> m_call_stack;
    std::map<uintptr_t, Variable> m_stack_or_global_variables;
    std::map<uintptr_t, MemoryAllocation> m_open_allocations;
    uintptr_t m_target_ip = -1;
    int m_inside_target_function = 0;  // This is an int instead of a bool so that we can handle recursive functions
    size_t m_last_malloc_request = 0;
    uintptr_t m_target_bp = 0; // Used to decide if a stack address outlives the target function or not
};

} // namespace phasm::vacuumtool


#endif //SURROGATE_TOOLKIT_INTERPRETER_HPP

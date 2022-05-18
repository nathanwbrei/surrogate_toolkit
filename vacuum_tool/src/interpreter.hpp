
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_INTERPRETER_HPP
#define SURROGATE_TOOLKIT_INTERPRETER_HPP

#include <vector>
#include <string>
#include <map>

namespace phasm::memtrace {

struct CodeLocation {
    void* instruction = nullptr;
    int routine_id = -1;
    std::string routine_name;
    std::string file;
    int line = -1;
};

struct Variable {
    void* addr = nullptr;
    std::string symbol;
    bool is_input = false;
    bool is_output = false;
    std::vector<size_t> sizes; // For now there should be exactly one.
    // Would be different sizes if variable was a union. Also, how do we handle arrays of structs in this case?
    std::vector<CodeLocation> callers; // Eventually replace with CodeLocations
};

struct MemoryAllocation {
    void *addr = nullptr;
    size_t size = 0;
    std::vector<Variable> variables;
    CodeLocation location;
};

struct ProgramAddressRanges {

    void* stack_base = nullptr;
    void* heap_start = nullptr;
    void* heap_end = nullptr;
    void* globals_start = nullptr;
    void* globals_end = nullptr;
    void* constants_start = nullptr;
    void* constants_end = nullptr;

    bool is_local_var(void *addr, void *current_rbp, void *current_rsp);

    bool is_stack_var_below_target(void *addr, void *current_rsp);

    bool is_global(void *addr);

    bool is_heap(void *addr);
};


class Interpreter {

public:
    Interpreter(int target_id, std::vector<std::string> routine_names);

    void enter_fun(void *rip, int fun_id, void *rbp);

    void exit_fun(void *rip);

    void request_malloc(void *rip, size_t size);

    void receive_malloc(void *rip, void *addr);

    void free(void *rip, void *buf);

    void read_mem(void *rip, void *addr, size_t size, void *rbp, void *rsp);

    void write_mem(void *rip, void *addr, size_t size, void *rbp, void *rsp);

    // Below are only public so that they are easy to test. Probably move to a helper class eventually.
    // Helper class could read the stack address ranges from /proc


    MemoryAllocation* find_allocation_containing(void *addr);

    std::vector<Variable> get_variables();

    void print_variables(std::ostream& os);

private:
    std::vector<std::string> m_routine_names;
    std::vector<int> m_call_stack;
    std::map<void *, Variable> m_stack_or_global_variables;
    std::map<size_t, MemoryAllocation> m_open_allocations;
    int m_target_id = -1;
    int m_inside_target_function = 0;  // This is an int instead of a bool so that we can handle recursive functions
    size_t m_last_malloc_request = 0;
    void* m_target_rbp = nullptr; // Used to decide if a stack address outlives the target function or not
};

} // namespace phasm::vacuumtool


#endif //SURROGATE_TOOLKIT_INTERPRETER_HPP


// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "interpreter.hpp"

using namespace phasm::memtrace;

Interpreter::Interpreter(int target_id, std::vector<std::string> routine_names)
    : m_routine_names(std::move(routine_names))
    , m_target_id(target_id)
    {}

void Interpreter::enter_fun(void *rip, int fun_id, void *rbp) {
    m_call_stack.push_back(fun_id);
    if (fun_id == m_target_id) m_inside_target_function += 1;
}

void Interpreter::exit_fun(void *rip) {
    m_call_stack.pop_back();
    // This trusts that our frontend is able to properly detect exiting the function.
    // PIN docs suggest this isn't always true. Maybe because of high optimization levels, -fomit-frame-pointer,
    // or maybe because of things like coroutines.
    // We may want to store frame pointers so that we can validate our call stack's integrity.
    // However, I'm holding off on implementing this until I find a case where PIN fails at detecting a function exit.
}

void Interpreter::request_malloc(void *rip, size_t size) {
    assert(m_last_malloc_request == 0);
    m_last_malloc_request = size;
    // This assumes that each thread has its own isolated VacuumInterpreter. This means that there won't be another
    // `request_malloc` event before its corresponding `receive_malloc`. However, in the real world, there is a single
    // malloc shared by all threads, so our representation could run into some weird threading problems still, such as
    // one thread allocating and a different thread freeing. I won't attempt to implement a workaround to these problems
    // before I encounter them in practice.
}

void Interpreter::receive_malloc(void *rip, void *buf) {

    assert(m_last_malloc_request != 0);
    MemoryAllocation alloc;
    alloc.size = m_last_malloc_request;
    alloc.addr = buf;
    alloc.location.routine_id = m_call_stack.back();
    m_last_malloc_request = 0;

    if (m_inside_target_function > 0) {
        size_t highest_addr = (size_t) alloc.addr + alloc.size - 1;
        m_open_allocations[highest_addr] = alloc;
    }
}

void Interpreter::free(void *rip, void *buf) {

    // The goal of all of this is to figure out which memory the target function allocated and didn't deallocate.
    // Anything it allocated+deallocated can be safely ignored, but anything not deallocated is either a memory
    // leak or a large write which we definitely want to know about.
    if (m_inside_target_function > 0) {
        auto it = m_open_allocations.lower_bound((size_t) buf);
        if (it->second.addr == buf) {
            m_open_allocations.erase(it);
        }
    }
}

void Interpreter::read_mem(void *rip, void *addr, size_t size, void *rbp, void *rsp) {
    if (m_inside_target_function > 0) {
        if (find_allocation_containing(addr) == nullptr) {
            // This is not something we allocated. Which means we are reading from the outside.

            // Check if we already have a Variable at this address
            auto it = m_stack_or_global_variables.find(addr);
            if (it != m_stack_or_global_variables.end()) {
                // We have seen this address before. If the first op was a write, we know any subsequent read
                // isn't a movement. If the first op was a read, we haven't gained any new information. So all
                // we have to do is record the caller.
                CodeLocation loc;
                loc.routine_id = m_call_stack.back();
                loc.instruction = rip;
                it->second.callers.push_back(loc);

                // Also check if we have a size contradiction, just because I am curious if we find one
                assert(it->second.sizes[0] == size);
            }
            else {
                // We haven't seen this address before. The first op is a read, so this definitely counts as an input.
                // Remains to be seen whether this is also an output.
                CodeLocation loc;
                loc.routine_id = m_call_stack.back();
                loc.instruction = rip;

                Variable var;
                var.callers.push_back(loc);
                var.is_input = true;
                var.sizes.push_back(size);

                m_stack_or_global_variables[addr] = var;
            }
        }
        else {
            // This IS something we allocated, so even if we read, no memory is being moved
            // Do nothing.
        }
    }
}

void Interpreter::write_mem(void *rip, void *addr, size_t size, void *rbp, void *rsp) {
    if (m_inside_target_function > 0) {
        auto allocation = find_allocation_containing(addr);
        if (allocation != nullptr) {
            // This is something target_function allocated.
            // Do we already have a variable for this?
            // Need to search existing variables
            auto it = m_stack_or_global_variables.find(addr);
            if (it != m_stack_or_global_variables.end()) {
                // We aleady have a variable for this
            }
            else {
                // We don't already have a variable for this
            }
        }
        else {
            // This is NOT something target_function allocated. Could be anything.
            auto it = m_stack_or_global_variables.find(addr);
            if (it != m_stack_or_global_variables.end()) {
                // We aleady have a variable for this
            }
            else {
                // We don't already have a variable for this
            }
        }


    }
}
MemoryAllocation* Interpreter::find_allocation_containing(void *addr) {
    // m_open_allocations takes advantage of the fact that allocations cannot overlap. This means we only have to search
    // for the key closest to the address and then check that we are actually inside that allocation.
    // Wrinkles:
    // - Thanks to how std::map works, we have to key off of the _upper_bound_ of the allocation address range
    //   (i.e. addr+size-1) rather than the lower one (i.e. addr).
    // - I'm not convinced the ugly pointer arithmetic we are doing here will hold up when we have data types with
    //   nontrivial alignments.

    auto it = m_open_allocations.lower_bound((size_t) addr);

    if (   it == m_open_allocations.end()                               // No allocations found
        || addr < it->second.addr                                       // Our addr is below this allocation's address range
        || (size_t) addr > (size_t) it->second.addr + it->second.size   // Our addr is above this allocation's address range
    ) {
        return nullptr;
    }
    else {
        return &it->second;
    }
}

std::vector<Variable> Interpreter::get_variables() {
    std::vector<Variable> vars;
    for (auto pair : m_stack_or_global_variables) {
        vars.push_back(pair.second);
    }
    for (auto pair : m_open_allocations) {
        for (auto v : pair.second.variables)
        vars.push_back(v);
    }
    return vars;
}


bool ProgramAddressRanges::is_local_var(void *addr, void *current_rbp, void *current_rsp) {
    return false;
}

bool ProgramAddressRanges::is_stack_var_below_target(void *addr, void *current_rsp) {
    return false;
}

bool ProgramAddressRanges::is_global(void *addr) {
    return false;
}

bool ProgramAddressRanges::is_heap(void *addr) {
    return false;
}


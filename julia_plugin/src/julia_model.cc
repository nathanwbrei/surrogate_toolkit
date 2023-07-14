
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <julia.h>
#include <julia_model.h>
#include <iostream>

namespace phasm {
void JuliaModel::initialize() {
    std::string inc_str = "include(\"" + m_filepath + "\")";
    jl_eval_string(inc_str.c_str());
    if (jl_exception_occurred()) {
        jl_static_show(jl_stdout_stream(), jl_exception_occurred());
        std::cout << std::endl;
    }

    jl_eval_string("include(\"Phasm.jl\"); using .Phasm");
    if (jl_exception_occurred()) {
        jl_static_show(jl_stdout_stream(), jl_exception_occurred());
        std::cout << std::endl;
    }

    jl_value_t* boxed_model = jl_box_voidpointer(this);
    jl_set_global(jl_main_module, jl_symbol("model"), boxed_model);
    if (jl_exception_occurred()) {
        std::cout << "set_global: exception: " << jl_typeof_str(jl_exception_occurred()) << std::endl;
    }
}

void JuliaModel::train_from_captures() {
    std::cout << "PHASM: Calling JuliaModel::train_from_captures (currently a no-op)" << std::endl;
}

bool JuliaModel::infer() {
    // TODO: Don't want to do the jl_eval_string on every call to infer()
    auto ret = jl_eval_string("Phasm.phasm_infer(reinterpret(Phasm.Model,model), infer)");
    if (jl_exception_occurred()) {
        std::cout << "Julia exception in JuliaModel::infer(): " << jl_typeof_str(jl_exception_occurred()) << std::endl;
        jl_static_show(jl_stdout_stream(), jl_exception_occurred());
        std::cout << std::endl;
        throw std::runtime_error("Exception inside Julia model!");
    }
    if (jl_typeis(ret, jl_bool_type)) {
        return jl_unbox_bool(ret);
    }
    else {
        throw std::runtime_error("JuliaModel::infer() returned a type other than bool. m_filepath=" + m_filepath);
    }
}

} // namespace phasm



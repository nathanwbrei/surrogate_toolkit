
// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <plugin.h>
#include <iostream>
#include <julia.h>
#include <julia_model.h>

// JULIA_DEFINE_FAST_TLS
// TODO: Apparently this needs to be defined in an executable, not a shared library. How do we manage this?

namespace phasm {
struct JuliaPlugin : public phasm::Plugin {

    JuliaPlugin() {
        std::cout << "PHASM: Initializing the Julia interpreter" << std::endl;
        jl_init();
    }

    virtual ~JuliaPlugin() {
        std::cout << "PHASM: Cleanly shutting down the Julia interpreter" << std::endl;
        jl_atexit_hook(0);
    }

    std::string get_name() override {
        return "phasm-julia-plugin";
    }

    std::shared_ptr<phasm::Model> make_model(std::string file_name) override {
        return std::make_shared<phasm::JuliaModel>(file_name);
    }
};

JuliaPlugin g_julia_plugin;
} // namespace phasm

extern "C" {
phasm::Plugin* get_plugin() {
    return &phasm::g_julia_plugin;
};
}





#include "plugin.h"
#include <iostream>
#include <memory>

struct TorchPlugin : public phasm::Plugin {

    void print_hello() override {
        std::cout << "Grüß Gott" << std::endl;
    }

    std::string get_name() {
        return "phasm-torch-plugin";
    }
};

TorchPlugin g_torch_plugin;

extern "C" {
    phasm::Plugin* get_plugin() {
        return &g_torch_plugin;
    };
}


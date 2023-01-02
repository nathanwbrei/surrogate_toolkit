
#include "plugin.h"
#include <iostream>
#include <memory>
#include "feedforward_model.h"

struct TorchPlugin : public phasm::Plugin {

    std::string get_name() {
        return "phasm-torch-plugin";
    }

    std::shared_ptr<phasm::Model> make_model(std::string /*model_name*/) override {
        return std::make_shared<phasm::FeedForwardModel>();
    }
};

TorchPlugin g_torch_plugin;

extern "C" {
    phasm::Plugin* get_plugin() {
        return &g_torch_plugin;
    };
}


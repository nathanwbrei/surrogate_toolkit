
#include "plugin.h"
#include <memory>
#include "feedforward_model.h"
#include "torchscript_model.h"

struct TorchPlugin : public phasm::Plugin {

    std::string get_name() override {
        return "phasm-torch-plugin";
    }

    std::shared_ptr<phasm::Model> make_model(std::string file_name) override {
        if (file_name.empty()) {
            return std::make_shared<phasm::FeedForwardModel>();
        }
        else {
            return std::make_shared<phasm::TorchscriptModel>(file_name);
        }
    }
};

TorchPlugin g_torch_plugin;

extern "C" {
    phasm::Plugin* get_plugin() {
        return &g_torch_plugin;
    };
}


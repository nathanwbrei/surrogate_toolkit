
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef TORCH_PLUGIN_TORCHSCRIPT_MODEL_H
#define TORCH_PLUGIN_TORCHSCRIPT_MODEL_H

#include "surrogate.h"
#include "model.h"
#include <torch/torch.h>
#include <torch/script.h>

namespace phasm {

struct TorchscriptModel : public Model {
private:
    std::string m_filename;
    torch::jit::script::Module m_module;
    torch::Device m_device = torch::kCPU;
    std::vector<std::vector<int64_t>> m_output_shapes;
    std::vector<int64_t> m_output_lengths;

    /// @brief The kernel part of loading *.pt module. Load to m_device manually.
    void LoadModule();

    /// @brief Print the infomation of every layer.
    void PrintModuleLayers();

public:
    TorchscriptModel(std::string filename, bool print_module_layers=false);

    ~TorchscriptModel();

    void ActivateGPU(); // Call before constructor to load pt module to GPU

    void initialize() override;

    void train_from_captures() override;

    bool infer() override;

    torch::jit::script::Module& get_module();
};

} // namespace phasm
#endif //TORCH_PLUGIN_TORCHSCRIPT_MODEL_H


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
    std::vector<std::vector<int64_t>> m_output_shapes;
    std::vector<int64_t> m_output_lengths;

    // This should be included in Model.h but let's leave it here now.
    torch::Device m_device = torch::kCPU;

    /// @brief The kernel part of loading *.pt module. Load to @param m_device manually.
    void LoadModule();

    /// @brief Print the infomation of every layer.
    void PrintModuleLayers();

public:
    TorchscriptModel(std::string filename, bool print_module_layers=false, torch::Device device=torch::kCPU);

    ~TorchscriptModel();

    void initialize() override;

    void train_from_captures() override;

    bool infer() override;

    torch::jit::script::Module& get_module();

    /// @brief @return the shape of the input layer.
    std::vector<int64_t> GetFirstLayerShape();
};

} // namespace phasm
#endif //TORCH_PLUGIN_TORCHSCRIPT_MODEL_H

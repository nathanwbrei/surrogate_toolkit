
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_TORCHSCRIPT_MODEL_H
#define SURROGATE_TOOLKIT_TORCHSCRIPT_MODEL_H

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

public:
    TorchscriptModel(std::string filename);

    ~TorchscriptModel();

    void initialize() override;

    void train_from_captures() override;

    bool infer() override;

    at::Tensor forward(std::vector<torch::jit::IValue> inputs);

};

} // namespace phasm
#endif //SURROGATE_TOOLKIT_FEEDFORWARD_MODEL_H


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

    void infer(std::vector<std::shared_ptr<CallSiteVariable>>& vars) override;

    torch::Tensor flatten_and_join(std::vector<torch::Tensor> inputs);

    std::vector<torch::Tensor> split_and_unflatten_outputs(torch::Tensor output) const;

};

} // namespace phasm
#endif //SURROGATE_TOOLKIT_FEEDFORWARD_MODEL_H

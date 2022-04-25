
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_FEEDFORWARD_MODEL_H
#define SURROGATE_TOOLKIT_FEEDFORWARD_MODEL_H

#include "surrogate.h"
#include "model.h"
#include <torch/torch.h>

struct FeedForwardModel : public Model {
private:

    struct FeedForwardNetwork : public torch::nn::Module {
        int64_t m_dim0 = 0;
        int64_t m_dim1 = 0;
        int64_t m_dim2 = 0;
        int64_t m_dim3 = 0;
        torch::nn::Linear m_input_layer{nullptr};
        torch::nn::Linear m_middle_layer{nullptr};
        torch::nn::Linear m_output_layer{nullptr};

        FeedForwardNetwork(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
                : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2), m_dim3(dim3) {
            m_input_layer = register_module("input_layer", torch::nn::Linear(dim0, dim1));
            m_middle_layer = register_module("middle_layer", torch::nn::Linear(dim1, dim2));
            m_output_layer = register_module("output_layer", torch::nn::Linear(dim2, dim3));
        }

        torch::Tensor forward(torch::Tensor x);

    };

    std::shared_ptr<FeedForwardNetwork> m_network = nullptr;
    std::vector<std::vector<int64_t>> m_output_shapes;
    std::vector<int64_t> m_output_lengths;

public:
    FeedForwardModel() = default;
    ~FeedForwardModel();

    void initialize() override;

    void train_from_captures() override;

    void infer(Surrogate &s) override;

    torch::Tensor flatten_and_join(std::vector<torch::Tensor> inputs);

    std::vector<torch::Tensor> split_and_unflatten_outputs(torch::Tensor output) const;

};
#endif //SURROGATE_TOOLKIT_FEEDFORWARD_MODEL_H

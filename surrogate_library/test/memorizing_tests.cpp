
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate.h"
#include "model.h"

using namespace phasm;
namespace phasm::test::memorizing_tests {

class MemorizingModel : public Model {

    std::unordered_map<int, tensor> memorized_data;
    // TODO: Figure out how to generalize this completely by correctly hashing torch::Tensor

public:

    MemorizingModel() {
        add_var<double>("x", Direction::IN);
        add_var<double>("y", Direction::OUT);
    }

    void train_from_captures() override {
        auto& xs = get_model_var("x")->training_inputs;
        auto& ys = get_model_var("y")->training_outputs;
        for (size_t i = 0; i<get_capture_count(); ++i) {
            tensor x = xs[i];
            tensor y = ys[i];
            double xx = *x.get<double>();
            memorized_data[xx] = y;
        }
    }

    bool infer() override {
        tensor x_val = m_model_vars[0]->inference_input;
        double xx = *x_val.get<double>();

        auto pair = memorized_data.find(xx);
        if (pair != memorized_data.end()) {
            auto y = m_model_vars[1]->inference_output = pair->second;
            return true;
        }
        return false;
    }
};

TEST_CASE("Memorizing model memorizes!") {

    auto m = std::make_shared<MemorizingModel>();
    double x, y;
    auto s = Surrogate();
    s.set_model(m);
    s.bind_locals_to_original_function([&](){y=x*x;});
    s.bind<double>("x", &x);
    s.bind<double>("y", &y);

    x = 2.0;
    y = 7.0;
    s.call_model();
    REQUIRE(y == 7.0);  // MemorizingModel leaves original output value alone if it doesn't have one cached

    s.call_original_and_capture();
    REQUIRE(y == 4.0);  // Correct value comes from original function.

    m->train_from_captures();  // Load the captures into the cache

    y = 7.0;  // Reset to garbage value
    s.call_model();
    REQUIRE(y == 4.0);  // Correct value comes from cache
}

} // namespace phasm::test::memorizing_tests






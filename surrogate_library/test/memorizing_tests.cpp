
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate.h"

using namespace phasm;
namespace phasm::test::memorizing_tests {

class MemorizingModel : public Model {

    std::unordered_map<int, torch::Tensor> memorized_data;
    // TODO: Figure out how to generalize this completely by correctly hashing torch::Tensor

public:

    MemorizingModel() {
        add_var<double>("x", Direction::Input);
        add_var<double>("y", Direction::Output);
    }

    void train_from_captures() override {
        auto& xs = get_model_var("x")->training_inputs;
        auto& ys = get_model_var("y")->training_outputs;
        for (size_t i = 0; i<get_capture_count(); ++i) {
            torch::Tensor x = xs[i];
            torch::Tensor y = ys[i];
            double xx = *x.data_ptr<double>();
            memorized_data[xx] = y;
        }
    }

    void infer(Surrogate& surrogate) override {
        auto x = surrogate.get_binding("x");
        x->captureAllInferenceInputs();

        torch::Tensor x_val = x->model_vars[0]->accessor->unsafe_to(x->binding);
        double xx = *x_val.data_ptr<double>();

        auto pair = memorized_data.find(xx);
        if (pair != memorized_data.end()) {
            auto y = surrogate.get_binding("y");
            y->model_vars[0]->accessor->unsafe_from(pair->second, y->binding);
        }
    }
};

TEST_CASE("Memorizing model memorizes!") {

    auto m = std::make_shared<MemorizingModel>();
    double x, y;
    auto s = Surrogate([&](){y=x*x;}, m);
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






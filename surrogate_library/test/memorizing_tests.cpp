
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate.h"

class MemorizingModel : public Model {

    std::unordered_map<int, torch::Tensor> memorized_data;
    // TODO: Figure out how to generalize this completely by correctly hashing torch::Tensor

public:

    MemorizingModel() {
        input<double>("x");
        output<double>("y");
    }

    void train(torch::Tensor) override {
        auto& xs = get_input<double>("x")->captures;
        auto& ys = get_output<double>("y")->captures;
        for (size_t i = 0; i<get_capture_count(); ++i) {
            torch::Tensor x = xs[i];
            torch::Tensor y = ys[i];
            double xx = *x.data_ptr<float>();
            memorized_data[xx] = y;
        }
    }

    void infer(Surrogate& surrogate) override {
        auto x = surrogate.get_input_binding<double>("x");
        torch::Tensor x_val = x->parameter->accessor->to(x->binding_root);
        double xx = *x_val.data_ptr<float>();

        auto pair = memorized_data.find(xx);
        if (pair != memorized_data.end()) {
            auto y = surrogate.get_output_binding<double>("y");
            y->parameter->accessor->from(pair->second, y->binding_root);
        }
    }
};

TEST_CASE("Memorizing model memorizes!") {

    auto m = std::make_shared<MemorizingModel>();
    double x, y;
    auto s = Surrogate([&](){y=x*x;}, m);
    s.bind_input<double>("x", &x);
    s.bind_output<double>("y", &y);

    x = 2.0;
    y = 7.0;
    s.call_model();
    REQUIRE(y == 7.0);  // MemorizingModel leaves original output value alone if it doesn't have one cached

    s.call_original_and_capture();
    REQUIRE(y == 4.0);  // Correct value comes from original function.

    m->train({});  // Load the captures into the cache

    y = 7.0;  // Reset to garbage value
    s.call_model();
    REQUIRE(y == 4.0);  // Correct value comes from cache
}







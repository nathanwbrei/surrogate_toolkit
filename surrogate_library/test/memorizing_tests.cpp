
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <catch.hpp>
#include "surrogate.h"

class MemorizingModel : public Model {

    std::map<double, double> memorized_data;

public:

    MemorizingModel() {
        input<double>("x");
        output<double>("y");
    }

    void train() override {
        auto& xs = get_input<double>("x")->captures;
        auto& ys = get_output<double>("y")->captures;
        for (size_t i = 0; i<get_capture_count(); ++i) {
            double x = xs[i];
            double y = ys[i];
            memorized_data[x] = y;
        }
    }

    void infer(Surrogate& surrogate) override {
        double x_val = surrogate.get_input_binding<double>("x")->sample;
        auto pair = memorized_data.find(x_val);
        if (pair != memorized_data.end()) {
            auto y = surrogate.get_output_binding<double>("y");
            *(y->slot) = pair->second;
        }
    }
};

TEST_CASE("Memorizing model memorizes!") {

    REQUIRE(true == false);
}







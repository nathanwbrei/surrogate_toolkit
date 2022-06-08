
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "feedforward_model.h"
#include "surrogate.h"
#include "fluent.h"

std::shared_ptr<phasm::FeedForwardModel> make_model() {
    phasm::OpticBuilder builder;
    builder.local_primitive<double>("x", phasm::Direction::Input)
           .local_primitive<double>("y", phasm::Direction::Input)
           .local_primitive<double>("z", phasm::Direction::Input)
           .local_primitive<double>("Bx", phasm::Direction::Output)
           .local_primitive<double>("By", phasm::Direction::Output)
           .local_primitive<double>("Bz", phasm::Direction::Output);

    auto model = std::make_shared<phasm::FeedForwardModel>();
    model->add_vars(builder);
    model->initialize(); // TODO: If we forget this, everything crashes when we try to train
    return model;
}

static auto s_model = make_model();

struct ToyMagFieldMap {
    void getField(double x, double y, double z, double& Bx, double& By, double& Bz) {
        auto surrogate = phasm::Surrogate([&](){ return this->getFieldOriginal(x,y,z,Bx,By,Bz);}, s_model);
        // TODO: If we only have one call site, do we have to recreate the lambda every time?
        // TODO: Fluent interface for surrogate!?
        surrogate.bind_all_locals(&x, &y, &z, &Bx, &By, &Bz);
        // surrogate.bind("x", &x);
        // surrogate.bind("y", &y);
        // surrogate.bind("z", &z);
        // surrogate.bind("Bx", &Bx); // TODO: Verify that &Bz changes depending on caller, so we have to rebind every time
        // surrogate.bind("By", &By);
        // surrogate.bind("Bz", &Bz);
        // surrogate.bind("result", &result);
        surrogate.call();
        // return result;
    }
    void getFieldOriginal(double x, double y, double z, double& Bx, double& By, double& Bz) {
        Bx = 2;
        By = x + 3;
        Bz = x * x;
    }
};

TEST_CASE("Toy magnetic field map") {

    // TODO: Can we make this any less awkward?
    phasm::Surrogate::set_call_mode(phasm::Surrogate::CallMode::CaptureAndDump);

    ToyMagFieldMap tmfm;
    double Bx, By, Bz;

    tmfm.getField(1, 2, 3, Bx, By, Bz);
    REQUIRE(Bx == 2);
    REQUIRE(By == 4);
    REQUIRE(Bz == 1);

    tmfm.getField(2, 2, 2, Bx, By, Bz);
    REQUIRE(Bx == 2);
    REQUIRE(By == 5);
    REQUIRE(Bz == 4);


    s_model->dump_captures_to_csv(std::cout);

    REQUIRE(s_model->get_capture_count() == 2);
    REQUIRE(s_model->get_model_var("x")->training_inputs[0].get<double>()[0] == 1);
    REQUIRE(s_model->get_model_var("Bz")->training_outputs[1].get<double>()[0] == 4);
}
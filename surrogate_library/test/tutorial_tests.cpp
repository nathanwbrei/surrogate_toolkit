
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "feedforward_model.h"
#include "surrogate.h"
#include "fluent.h"

std::shared_ptr<phasm::FeedForwardModel> make_model() {
    phasm::OpticBuilder builder;
    // TODO: builder.local_primitive<double>("x", In);
    builder.local<double>("x").primitive("x", phasm::Direction::Input);
    builder.local<double>("y").primitive("y", phasm::Direction::Input);
    builder.local<double>("z").primitive("z", phasm::Direction::Input);
    builder.local<double>("Bx").primitive("Bx", phasm::Direction::Output);
    builder.local<double>("By").primitive("By", phasm::Direction::Output);
    builder.local<double>("Bz").primitive("Bz", phasm::Direction::Output);
    auto model = std::make_shared<phasm::FeedForwardModel>();
    model->add_vars(builder);
    model->initialize(); // TODO: Does everything crash if we forget this?
    return model;
}

static auto s_model = make_model();

struct ToyMagFieldMap {
    void getField(double x, double y, double z, double& Bx, double& By, double& Bz) {
        auto surrogate = phasm::Surrogate([&](){ return this->getFieldOriginal(x,y,z,Bx,By,Bz);}, s_model);
        // TODO: If we only have one call site, do we have to recreate the lambda every time?
        // TODO: Create an impl of bind that doesn't need to do any string lookups, e.g. bind_all_locals
        // TODO: Fluent interface for surrogate!?
        surrogate.bind("x", &x);
        surrogate.bind("y", &y);
        surrogate.bind("z", &z);
        surrogate.bind("Bx", &Bx); // TODO: Verify that &Bz changes depending on caller, so we have to rebind every time
        surrogate.bind("By", &By);
        surrogate.bind("Bz", &Bz);
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

    REQUIRE(s_model->get_capture_count() == 2);
}
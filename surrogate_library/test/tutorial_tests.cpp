
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <catch.hpp>
#include "feedforward_model.h"
#include "surrogate.h"
#include "surrogate_builder.h"

// For now this is a static global, so that we can inspect it and write assertions
// against it. However, in general we probably want to declare it static inside the
// wrapper function itself. That way, the entirety of PHASM machinery lives in one place.

using namespace phasm;
static Surrogate s_surrogate = SurrogateBuilder()
        .set_model(std::make_shared<FeedForwardModel>())
        .set_callmode(phasm::CallMode::CaptureAndDump)
        .local_primitive<double>("x", IN)
        .local_primitive<double>("y", IN)
        .local_primitive<double>("z", IN)
        .local_primitive<double>("Bx", OUT)
        .local_primitive<double>("By", OUT)
        .local_primitive<double>("Bz", OUT)
        .finish();

struct ToyMagFieldMap {

    void getField(double x, double y, double z, double& Bx, double& By, double& Bz) {
        std::cout << "Binding &Bz=" << &Bz << std::endl;
        // Because Bz is a reference, &Bz changes depending on the caller, so we have to rebind on every call!
        // This includes both the lambda and the CallSiteVariables.
        // In theory we don't have to re-copy the CallSiteVariables over from the model every time, but
        // that is an optimization that can wait until the next time we rejigger the whole domain model.

        s_surrogate.bind_locals_to_original_function([&](){ return this->getFieldOriginal(x,y,z,Bx,By,Bz);});
        s_surrogate.bind_locals_to_model(&x, &y, &z, &Bx, &By, &Bz);
        s_surrogate.call();
    }
    void getFieldOriginal(double x, double y, double z, double& Bx, double& By, double& Bz) {
        Bx = 2;
        By = x + 3;
        Bz = x * x;
    }
};

TEST_CASE("Toy magnetic field map") {

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

    double Bz2;

    tmfm.getField(7, 2, 2, Bx, By, Bz2);
    REQUIRE(Bz == 4);
    REQUIRE(Bz2 == 49);

    auto model = s_surrogate.get_model();
    model->dump_captures_to_csv(std::cout);

    REQUIRE(model->get_capture_count() == 3);
    REQUIRE(model->get_model_var("x")->training_inputs[0].get<double>()[0] == 1);
    REQUIRE(model->get_model_var("Bz")->training_outputs[1].get<double>()[0] == 4);
}
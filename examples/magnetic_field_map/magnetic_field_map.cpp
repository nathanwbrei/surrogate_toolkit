

// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>

#include <JANA/JApplication.h>
#include <JANA/Calibrations/JCalibrationManager.h>
#include "DMagneticFieldMapFineMesh.h"
#include "surrogate_builder.h"
#include "JCalibrationGenerator_TestFixture.hpp"


int main(int argc, char* argv[]) {

    using phasm::SurrogateBuilder, phasm::Direction, phasm::CallMode;


    phasm::CallMode call_mode;
    std::string modelname;
    if (argc == 1) {
        std::cout << "PHASM: No model specified, so we are dumping training data (from the original function) to CSV" << std::endl;
        call_mode = CallMode::DumpTrainingData;
    }
    else {
        std::cout << "PHASM: Provided model '" << argv[1] << "', so we are dumping validation data (from the model) to CSV" << std::endl;
        call_mode = CallMode::DumpValidationData;
        modelname = argv[1];
    }

    phasm::Surrogate surrogate = SurrogateBuilder()
            .set_model("phasm-torch-plugin", modelname, true)
            .set_callmode(call_mode)
            .local_primitive<double>("x", Direction::IN)
            .local_primitive<double>("y", Direction::IN)
            .local_primitive<double>("z", Direction::IN)
            .local_primitive<double>("Bx", Direction::OUT)
            .local_primitive<double>("By", Direction::OUT)
            .local_primitive<double>("Bz", Direction::OUT)
            .finish();

    std::map<std::string, std::string> magnet_calib_data;
    magnet_calib_data["URL_base"] = "https://halldweb.jlab.org/resources";
    magnet_calib_data["path"] = "Magnets/Solenoid/solenoid_1350A_poisson_20160222";
    magnet_calib_data["md5"] = "a96263c5e2f3936241b3eadb85e6559f";
    JCalibration_TestFixture calib;
    calib.SetCalib("Magnets/Solenoid/solenoid_1350A_poisson_20160222", magnet_calib_data);
    auto calib_man = std::make_shared<JCalibrationManager>();
    calib_man->AddCalibrationGenerator(new JCalibrationGenerator_TestFixture(calib));
    japp = new JApplication;
    japp->ProvideService(calib_man);

    DMagneticFieldMapFineMesh mfmfm(japp, 1, "Magnets/Solenoid/solenoid_1350A_poisson_20160222");

    double x, y, z, bx, by, bz;

    surrogate.bind_original_function([&]() { mfmfm.GetField(x, y, z, bx, by, bz); });
    surrogate.bind_all_callsite_vars(&x, &y, &z, &bx, &by, &bz);

    size_t sample_count = 0;
    for (x = -20.0; x < 20.0; x+=4.0) {
        for (y = -20.0; y < 20.0; y+=4.0) {
            for (z = 0.0; z < 100.0; z+=2.0) {
                // This is the function being surrogated:
                // mfmfm.GetField(x,y,z,bx,by,bz);

                surrogate.call();
                // This will either call the original function and dump training data, or
                // call the model and dump validation data, depending on the CallMode set at the top
                sample_count++;
                // std::cout << "x=" << x << ", y=" << y << ", z=" << z << ", Bx=" << bx << ", By=" << by << ", Bz=" << bz << std::endl;
            }
        }
    }
    std::cout << "Obtained " << sample_count << " rows of data" << std::endl;
}


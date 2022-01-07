

// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <iostream>

#include <JANA/JApplication.h>
#include <JANA/Calibrations/JCalibrationManager.h>
#include <JANA/Calibrations/JCalibrationGeneratorCCDB.h>
#include "DMagneticFieldMapFineMesh.h"

int main() {
    std::cout << "Hello from magnetic_field_map_before" << std::endl;
    japp = new JApplication;
    auto calib_man = std::make_shared<JCalibrationManager>();
    // calib_man->AddCalibrationGenerator(new JCalibrationGeneratorCCDB);
    japp->ProvideService(calib_man);
    DMagneticFieldMapFineMesh mfmfm(japp);

    std::cout << mfmfm.GetBz(0.0, 0.0, 0.0) << std::endl;
}


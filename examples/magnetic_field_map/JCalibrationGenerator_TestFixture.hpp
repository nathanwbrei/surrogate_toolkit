
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_J_CALIBRATION_GENERATOR_TEST_FIXTURE_HPP
#define SURROGATE_TOOLKIT_J_CALIBRATION_GENERATOR_TEST_FIXTURE_HPP

#include <JANA/Calibrations/JCalibration.h>
#include <JANA/Calibrations/JCalibrationGenerator.h>

#include "JCalibration_TestFixture.hpp"


class JCalibrationGenerator_TestFixture : public JCalibrationGenerator {

    JCalibration_TestFixture m_prototype;

public:

    JCalibrationGenerator_TestFixture(JCalibration_TestFixture prototype);
    JCalibration_TestFixture& GetPrototype();

    const char* Description() override;
    double CheckOpenable(std::string, int32_t, std::string) override;
    JCalibration* MakeJCalibration(std::string, int32_t, std::string) override;

};

#endif //SURROGATE_TOOLKIT_J_CALIBRATION_GENERATOR_TEST_FIXTURE_HPP

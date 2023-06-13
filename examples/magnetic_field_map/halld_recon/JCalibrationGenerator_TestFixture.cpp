
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include "JCalibrationGenerator_TestFixture.hpp"

JCalibrationGenerator_TestFixture::JCalibrationGenerator_TestFixture(JCalibration_TestFixture prototype)
: m_prototype(prototype) {
}

JCalibration_TestFixture& JCalibrationGenerator_TestFixture::GetPrototype() {
    return m_prototype;
}

const char* JCalibrationGenerator_TestFixture::Description() {
    return "";
}

double JCalibrationGenerator_TestFixture::CheckOpenable(std::string, int32_t, std::string) {
    return 1.0;
}

JCalibration* JCalibrationGenerator_TestFixture::MakeJCalibration(std::string, int32_t, std::string) {
    return new JCalibration_TestFixture(m_prototype);
}

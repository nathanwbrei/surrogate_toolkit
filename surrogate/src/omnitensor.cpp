#include <omnitensor.hpp>
#include <iostream>

namespace phasm {


omnitensor::~omnitensor() {
    switch (m_dtype) {
        case DType::UI8: delete[] static_cast<uint8_t*>(m_data); break;
        case DType::I16: delete[] static_cast<int16_t*>(m_data); break;
        case DType::I32: delete[] static_cast<int32_t*>(m_data); break;
        case DType::I64: delete[] static_cast<int64_t*>(m_data); break;
        case DType::F32: delete[] static_cast<float*>(m_data); break;
        case DType::F64: delete[] static_cast<double*>(m_data); break;
        default:
            if (m_length > 0) {
                std::cout << "PHASM: Memory leak due to invalid (corrupt?) tensor dtype" << std::endl;
                std::terminate();
            }
            break;
    }
};



} // namespace phasm
    

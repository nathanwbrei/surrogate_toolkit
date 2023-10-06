
#pragma once
#include "tensor.hpp"
#include <stddef.h>
#include <vector>
#include <iostream>


namespace phasm {

class omnitensor_index {
    friend class omnitensor;
    std::vector<int> m_indices;

public:

    omnitensor_index(std::vector<int> indices) : m_indices(indices) {}

};

static omnitensor_index index(std::vector<int> indices) {
    return omnitensor_index(indices);
}


class omnitensor {

public:
    enum class DimType { Array, Vector, List };

    void* m_data;
    size_t* m_sizes;
    size_t m_capacity;
    size_t m_length = 0;
    size_t m_dim_count;
    std::vector<int64_t> m_shape; // 0 for variably-sized dimensions
    std::vector<DimType> m_dimtypes;
    std::vector<size_t> m_dim_offsets;
    DType m_dtype;

public:
    omnitensor(DType dtype, std::vector<int64_t> shapes, size_t capacity=1024) {

        assert(capacity > 0);
        m_capacity = capacity;
        m_shape = shapes;
        m_dim_count = shapes.size();
        m_dtype = dtype;
        m_sizes = new size_t[capacity];

        switch(dtype) {
            case DType::UI8: m_data = new uint8_t[capacity]; break;
            case DType::I16: m_data = new int16_t[capacity]; break;
            case DType::I32: m_data = new int32_t[capacity]; break;
            case DType::I64: m_data = new int64_t[capacity]; break;
            case DType::F32: m_data = new float[capacity];   break;
            case DType::F64: m_data = new double[capacity];  break;
            default: throw std::runtime_error("Bad dtype");
        }
        
        m_length = 1;
        for (size_t i = 0; i < m_dim_count; ++i) {
            m_length *= m_shape[i]; // If we have any zero-dimension dims, length will be 0
                                    
            if (i == m_dim_count - 1) {
                if (m_shape[i] == 0) {
                    m_dimtypes.push_back(DimType::Vector);
                }
                else {
                    m_dimtypes.push_back(DimType::Array);
                }
            }
            else {
                if (m_shape[i] == 0 && m_shape[i+1] == 0) {
                    m_dimtypes.push_back(DimType::List);
                }
                else if (m_shape[i] == 0 && m_shape[i+1] != 0) {
                    m_dimtypes.push_back(DimType::Vector);
                }
                else if (m_shape[i] != 0 && m_shape[i+1] == 0) {
                    assert(false);
                }
                else if (m_shape[i] != 0 && m_shape[i+1] != 0) {
                    m_dimtypes.push_back(DimType::Array);
                }
            }
        }

        size_t offset = 1;
        m_dim_offsets.resize(m_dim_count);
        for (size_t dim = 0; dim < m_dim_count; ++dim) {
            size_t rdim = m_dim_count - 1 - dim;
            m_dim_offsets[rdim] = offset;
            offset *= m_shape[rdim];
        }

    };
    size_t length() {
        return m_length;
    }

    ~omnitensor();

    size_t offset(const omnitensor_index& index) {
        size_t offset = 0;
        for (int dim=0; dim<index.m_indices.size(); ++dim) {
            if (m_dimtypes[dim] == DimType::Array) {
                offset += m_dim_offsets[dim]*index.m_indices[dim];
            }
        }
        return offset;
    }

    size_t length(const omnitensor_index& current) {
        if (current.m_indices.size() == 0) {
            return m_length;
        }
        omnitensor_index next = current;
        size_t last_dim = next.m_indices.size() - 1;
        next.m_indices[last_dim]++;
        return offset(next) - offset(current);
    }


    template <typename T>
    std::pair<T*, size_t> data(const omnitensor_index& index) {
        return { static_cast<T*>(m_data)+offset(index), length(index)};
    }
};

}; // namespace phasm

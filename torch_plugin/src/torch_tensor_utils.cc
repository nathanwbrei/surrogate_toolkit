
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.
#include "torch_tensor_utils.h"

namespace phasm {

template <typename T>
torch::Tensor to_torch_tensor_typed(const phasm::tensor& t) {
    const T* data = t.get_data<T>();
    int64_t length = t.get_length();
    auto result = torch::tensor(at::ArrayRef<T>(data,length));
    result = result.reshape(t.get_shape());
    return result;
}

torch::Tensor to_torch_tensor(const phasm::tensor& t) {
    switch(t.get_dtype()) {
        case DType::UI8: return to_torch_tensor_typed<uint8_t>(t);
        case DType::I16: return to_torch_tensor_typed<int16_t>(t);
        case DType::I32: return to_torch_tensor_typed<int32_t>(t);
        case DType::I64: return to_torch_tensor_typed<int64_t>(t);
        case DType::F32: return to_torch_tensor_typed<float>(t);
        case DType::F64: return to_torch_tensor_typed<double>(t);
        default: throw std::runtime_error("Undefined tensor");
    }
}

template <typename T>
phasm::tensor to_phasm_tensor_typed(const torch::Tensor& t) {

    T* torch_data = t.data_ptr<T>();
    int64_t length = t.numel();
    T* phasm_data = new T[length];
    for (int64_t i=0; i<length; ++i) {
        phasm_data[i] = torch_data[i];
    }
    size_t dims = t.dim();
    std::vector<int64_t> phasm_dims;
    for (size_t dim=0; dim<dims; ++dim) {
        phasm_dims.push_back(t.size(dim));
    }
    return phasm::tensor(std::unique_ptr<T[]>(phasm_data), phasm_dims);
}

phasm::tensor to_phasm_tensor(const torch::Tensor& t) {
    torch::Dtype dtype = t.dtype().toScalarType();
    if (dtype == torch::kUInt8) return to_phasm_tensor_typed<uint8_t>(t);
    if (dtype == torch::kInt16) return to_phasm_tensor_typed<int16_t>(t);
    if (dtype == torch::kInt32) return to_phasm_tensor_typed<int32_t>(t);
    if (dtype == torch::kInt64) return to_phasm_tensor_typed<int64_t>(t);
    if (dtype == torch::kFloat32) return to_phasm_tensor_typed<float>(t);
    if (dtype == torch::kFloat64) return to_phasm_tensor_typed<double>(t);
    throw std::runtime_error("Torch tensor has invalid or incompatible dtype!");
}

torch::Tensor flatten_and_join(std::vector<torch::Tensor> inputs) {
    for (auto& input : inputs) {
        input = input.flatten(0, -1).toType(c10::ScalarType::Float);
    }
    auto result = torch::cat(inputs);
    return result;
}

std::vector<torch::Tensor> split_and_unflatten_outputs(torch::Tensor output,
                                                       const std::vector<int64_t>& output_lengths,
                                                       const std::vector<std::vector<int64_t>>& output_shapes) {
    std::vector<torch::Tensor> outputs;
    int64_t start = 0;
    for (size_t i=0; i<output_lengths.size(); ++i) {
        torch::Tensor o = output.slice(0, start, start+output_lengths[i]).reshape(output_shapes[i]);
        outputs.push_back(o);
        start += output_lengths[i];
    }
    return outputs;
}

phasm::DType to_phasm_dtype(torch::Dtype t) {
    if (t == torch::kUInt8) return phasm::DType::UI8;
    if (t == torch::kInt16) return phasm::DType::I16;
    if (t == torch::kInt32) return phasm::DType::I32;
    if (t == torch::kInt64) return phasm::DType::I64;
    if (t == torch::kFloat32) return phasm::DType::F32;
    if (t == torch::kFloat64) return phasm::DType::F64;
    return phasm::DType::Undefined;
}


} // namespace phasm

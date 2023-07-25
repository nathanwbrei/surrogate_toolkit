#pragma once
#include <vector>
#include <variant>
#include <memory>
#include <string>

namespace phasm {
namespace mvdt {


enum class DType { U8, I16, I32, I64, F32, F64 };
struct PrimitiveType;
struct StructType;
struct PointerType;
struct ArrayType;

using Type = std::variant<std::unique_ptr<PrimitiveType>, 
                          std::unique_ptr<StructType>,
                          std::unique_ptr<ArrayType>,
                          std::unique_ptr<PointerType>>;

struct Variable {
    std::string symbol;
    Type type;
};

struct PrimitiveType {
    DType dtype;
    bool isRead = false;
    bool isWritten = false;
};

struct StructType {
    std::string symbol;
    std::vector<Variable> fields;
};

struct ArrayType {
    Type innerType;
    size_t length;
};

struct PointerType {
    Type innerType;
};

struct FunctionCall {
    std::string symbol;
};


struct AnalysisResult {
    FunctionCall functionBeingAnalyzed;
    std::vector<FunctionCall> callees;
    std::vector<Variable> formalParameters;
    std::vector<Variable> globals;
    std::vector<std::string> mysteries;
};


void printAnalysisResults(const std::vector<AnalysisResult>&);



} // namespace mvdt
} // namespace phasm


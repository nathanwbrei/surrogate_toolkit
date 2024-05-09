
#include "AnalysisResult.h"
#include <iostream>

namespace phasm {
namespace mvdt {

void printAnalysisResults(const std::vector<AnalysisResult>& results) {
    std::cout << "============" << std::endl;
    std::cout << "MVDT results" << std::endl;
    std::cout << "============" << std::endl;
    for (const AnalysisResult& result : results) {
        std::cout << "Function: " << result.functionBeingAnalyzed.symbol << std::endl;
        std::cout << "-----------------------" << std::endl;
        std::cout << "Callees" << std::endl;
        std::cout << "-----------------------" << std::endl;
        for (const FunctionCall& callee : result.callees) {
            std::cout << callee.symbol << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
        std::cout << "Formal parameters" << std::endl;
        std::cout << "-----------------------" << std::endl;
        for (const Variable& param : result.formalParameters) {
            std::cout << param.symbol << std::endl;
        }
        std::cout << "-----------------------" << std::endl;
        std::cout << "Globals " << std::endl;
        std::cout << "-----------------------" << std::endl;
        for (const Variable& global : result.globals) {
            std::cout << global.symbol << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }
}

} // namespace mvdt
} // namespace phasm
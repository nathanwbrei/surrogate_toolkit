

#include <iostream>
#include "AnalysisResult.h"

using namespace phasm::mvdt;

int main() {
    std::cout << "PHASM Model Variable Discovery Tool" << std::endl;
    std::vector<AnalysisResult> results;
    printAnalysisResults(results);

}

// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

#include <cstdint>
#pragma once

/// These are meant to be called from Phasm.jl. They expose a subset of phasm's ModelVariable to Julia
extern "C" {

int64_t phasm_modelvars_count(void* model);
const char* phasm_modelvars_getname(void* model, int64_t index);
bool phasm_modelvars_isinput(void* model, int64_t index);
bool phasm_modelvars_isoutput(void* model, int64_t index);
double* phasm_modelvars_getinputdata(void* model, int64_t index);
void phasm_modelvars_setoutputdata(void* model, int64_t index, double* data, size_t length);
void phasm_modelvars_setoutputdata2(void* model, int64_t index, double* data, int64_t* shape, size_t dims);

} // extern "C"

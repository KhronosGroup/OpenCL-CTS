//
// Copyright (c) 2021-2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef COMMON_H
#define COMMON_H

#include "harness/typeWrappers.h"
#include "utility.h"

#include <array>
#include <string>
#include <vector>

// Array of thread-specific kernels for each vector size.
using KernelMatrix =
    std::array<std::vector<clKernelWrapper>, VECTOR_SIZE_COUNT>;

// Array of programs for each vector size.
using Programs = std::array<clProgramWrapper, VECTOR_SIZE_COUNT>;

// Array of buffers for each vector size.
using Buffers = std::array<clMemWrapper, VECTOR_SIZE_COUNT>;

// Types supported for kernel code generation.
enum class ParameterType
{
    Half,
    Float,
    Double,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
};

// Return kernel name suffixed with vector size.
std::string GetKernelName(int vector_size_index);

// Generate kernel code for the given builtin function/operator.
std::string GetUnaryKernel(const std::string &kernel_name, const char *builtin,
                           ParameterType retType, ParameterType type1,
                           int vector_size_index);
std::string GetUnaryKernel(const std::string &kernel_name, const char *builtin,
                           ParameterType retType1, ParameterType retType2,
                           ParameterType type1, int vector_size_index);
std::string GetBinaryKernel(const std::string &kernel_name, const char *builtin,
                            ParameterType retType, ParameterType type1,
                            ParameterType type2, int vector_size_index);
std::string GetBinaryKernel(const std::string &kernel_name, const char *builtin,
                            ParameterType retType1, ParameterType retType2,
                            ParameterType type1, ParameterType type2,
                            int vector_size_index);
std::string GetTernaryKernel(const std::string &kernel_name,
                             const char *builtin, ParameterType retType,
                             ParameterType type1, ParameterType type2,
                             ParameterType type3, int vector_size_index);

// Information to generate OpenCL kernels.
struct BuildKernelInfo
{
    // Number of kernels to build, one for each thread to avoid data races.
    cl_uint threadCount;

    KernelMatrix &kernels;

    Programs &programs;

    // Function, macro or symbol tested by the kernel.
    const char *nameInCode;

    // Whether to build with -cl-fast-relaxed-math.
    bool relaxedMode;

    // Whether to build with -cl-fp32-correctly-rounded-divide-sqrt.
    bool correctlyRounded;
};

// Thread specific data for a unary function worker thread.
struct ThreadInfoUnary
{
    // Input and output buffer for the thread.
    clMemWrapper inBuf;
    Buffers outBuf;

    // Max error value found in this thread.
    float maxError = 0.0f;

    // Position of the max error value (param 1).
    double maxErrorValue = 0.0;

    // Per thread pseudorandom number generator to ensure determinism.
    MTdataHolder d;

    // Per thread command queue to improve performance.
    clCommandQueueWrapper tQueue;
};

// Thread specific data for a binary function worker thread.
template <typename Param2Ty>
struct ThreadInfoBinaryBase : public ThreadInfoUnary
{
    // Input buffer for parameter 2.
    clMemWrapper inBuf2;

    // Position of the max error value (param 2).
    Param2Ty maxErrorValue2 = {};
};

// Thread specific data for an (fp, fp) binary function worker thread.
using ThreadInfoBinary = ThreadInfoBinaryBase<double>;

// Thread specific data for an (fp, int) binary function worker thread.
using ThreadInfoBinaryFPInt = ThreadInfoBinaryBase<cl_int>;

// Data common to all math tests.
struct TestInfoBase
{
    TestInfoBase() = default;
    ~TestInfoBase() = default;

    // Prevent accidental copy/move.
    TestInfoBase(const TestInfoBase &) = delete;
    TestInfoBase &operator=(const TestInfoBase &) = delete;
    TestInfoBase(TestInfoBase &&h) = delete;
    TestInfoBase &operator=(TestInfoBase &&h) = delete;

    // Size of the sub-buffer in elements.
    size_t subBufferSize = 0;
    // Function info.
    const Func *f = nullptr;

    // Number of worker threads.
    cl_uint threadCount = 0;
    // Number of jobs.
    cl_uint jobCount = 0;
    // max_allowed ulps.
    float ulps = -1.f;
    // non-zero if running in flush to zero mode.
    int ftz = 0;

    // 1 if running the fdim test.
    int isFDim = 0;
    // 1 if input/output NaNs and INFs are skipped.
    int skipNanInf = 0;
    // 1 if running the nextafter test.
    int isNextafter = 0;

    // 1 if the function is only to be evaluated over a range.
    int isRangeLimited = 0;

    // Result limit for half_sin/half_cos/half_tan.
    float half_sin_cos_tan_limit = -1.f;

    // Whether the test is being run in relaxed mode.
    bool relaxedMode = false;
};

using SourceGenerator = std::string (*)(const std::string &kernel_name,
                                        const char *builtin,
                                        cl_uint vector_size_index);

/// Build kernels for all threads in "info" for the given job_id.
cl_int BuildKernels(BuildKernelInfo &info, cl_uint job_id,
                    SourceGenerator generator);

const size_t getInputCount();
void initInputCount(int wimpyReductionFactor);

void fillHalfUnaryInput(cl_half *data, size_t num_elems, size_t base_elem,
                        MTdata d, bool testAll = false);
void fillFloatUnaryInput(float *data, size_t num_elems, size_t base_elem,
                         MTdata d, bool testAll = false);
void fillDoubleUnaryInput(double *data, size_t num_elems, size_t base_elem,
                          MTdata d);

void fillHalfBinaryInput(cl_half *data1, cl_half *data2, size_t num_elems,
                         size_t base_elem, MTdata d);
void fillFloatBinaryInput(float *data1, float *data2, size_t num_elems,
                          size_t base_elem, MTdata d);
void fillDoubleBinaryInput(double *data1, double *data2, size_t num_elems,
                           size_t base_elem, MTdata d);

void fillIntHalfBinaryInput(int *data1, cl_half *data2, size_t num_elems,
                            size_t base_elem, MTdata d);
void fillIntFloatBinaryInput(int *data1, float *data2, size_t num_elems,
                             size_t base_elem, MTdata d);
void fillIntDoubleBinaryInput(int *data1, double *data2, size_t num_elems,
                              size_t base_elem, MTdata d);

void fillHalfTernaryInput(cl_half *data1, cl_half *data2, cl_half *data3,
                          size_t num_elems, size_t base_elem, MTdata d);
void fillFloatTernaryInput(float *data1, float *data2, float *data3,
                           size_t num_elems, size_t base_elem, MTdata d);
void fillDoubleTernaryInput(double *data1, double *data2, double *data3,
                            size_t num_elems, size_t base_elem, MTdata d);

#endif /* COMMON_H */

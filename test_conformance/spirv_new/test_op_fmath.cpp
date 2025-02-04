//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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

#include "testBase.h"
#include "types.hpp"

#include <sstream>
#include <string>

template<typename T>
int test_fmath(cl_device_id deviceID,
               cl_context context,
               cl_command_queue queue,
               const char *spvName,
               const char *funcName,
               const char *Tname,
               bool fast_math,
               std::vector<T> &h_lhs,
               std::vector<T> &h_rhs)
{

    if(std::string(Tname).find("double") != std::string::npos) {
        if(!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return 0;
        }
    }
    cl_int err = CL_SUCCESS;
    int num = (int)h_lhs.size();
    size_t bytes = num * sizeof(T);

    clMemWrapper lhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create lhs buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, bytes, &h_lhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper rhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create rhs buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, bytes, &h_rhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to rhs buffer");

    std::string kernelStr;

    {
        std::stringstream kernelStream;

        if (is_double<T>::value) {
            kernelStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        } else if (sizeof(T) == sizeof(cl_half)) {
            kernelStream << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }

        kernelStream << "#define spirv_fadd(a, b) (a) + (b)               \n";
        kernelStream << "#define spirv_fsub(a, b) (a) - (b)               \n";
        kernelStream << "#define spirv_fmul(a, b) (a) * (b)               \n";
        kernelStream << "#define spirv_fdiv(a, b) (a) / (b)               \n";
        kernelStream << "#define spirv_frem(a, b) fmod(a, b)              \n";
        kernelStream << "#define spirv_fmod(a, b) copysign(fmod(a,b),b)   \n";
        kernelStream << "#define T " << Tname                         << "\n";
        kernelStream << "#define FUNC spirv_" << funcName             << "\n";
        kernelStream << "__kernel void fmath_cl(__global T *out,          \n";
        kernelStream << "const __global T *lhs, const __global T *rhs)    \n";
        kernelStream << "{                                                \n";
        kernelStream << "    int id = get_global_id(0);                   \n";
        kernelStream << "    out[id] = FUNC(lhs[id], rhs[id]);            \n";
        kernelStream << "}                                                \n";
        kernelStr = kernelStream.str();
    }

    const char *kernelBuf = kernelStr.c_str();

    std::vector<T> h_ref(num);

    {
        // Run the cl kernel for reference results
        clProgramWrapper prog;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &prog, &kernel, 1,
                                          &kernelBuf, "fmath_cl");
        SPIRV_CHECK_ERROR(err, "Failed to create cl kernel");

        clMemWrapper ref = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
        SPIRV_CHECK_ERROR(err, "Failed to create ref buffer");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ref);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

        size_t global = num;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

        err = clEnqueueReadBuffer(queue, ref, CL_TRUE, 0, bytes, &h_ref[0], 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to read from ref");
    }

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, spvName);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, "fmath_spv", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create res buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<T> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_res[i] != h_ref[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

#define TEST_FMATH_FUNC(TYPE, FUNC, MODE)                                      \
    REGISTER_TEST(op_##FUNC##_##TYPE##_##MODE)                                 \
    {                                                                          \
        if (sizeof(cl_##TYPE) == 2)                                            \
        {                                                                      \
            PASSIVE_REQUIRE_FP16_SUPPORT(device);                              \
        }                                                                      \
        const int num = 1 << 20;                                               \
        std::vector<cl_##TYPE> lhs(num);                                       \
        std::vector<cl_##TYPE> rhs(num);                                       \
                                                                               \
        RandomSeed seed(gRandomSeed);                                          \
                                                                               \
        for (int i = 0; i < num; i++)                                          \
        {                                                                      \
            lhs[i] = genrandReal<cl_##TYPE>(seed);                             \
            rhs[i] = genrandReal<cl_##TYPE>(seed);                             \
        }                                                                      \
                                                                               \
        const char *mode = #MODE;                                              \
        return test_fmath(device, context, queue, #FUNC "_" #TYPE, #FUNC,      \
                          #TYPE, mode[0] == 'f', lhs, rhs);                    \
    }

#define TEST_FMATH_MODE(TYPE, MODE)                                            \
    TEST_FMATH_FUNC(TYPE, fadd, MODE)                                          \
    TEST_FMATH_FUNC(TYPE, fsub, MODE)                                          \
    TEST_FMATH_FUNC(TYPE, fmul, MODE)                                          \
    TEST_FMATH_FUNC(TYPE, fdiv, MODE)                                          \
    // disable those tests until we figure out what the precision requirements
    // are
    //    TEST_FMATH_FUNC(TYPE, frem, MODE)
    //    TEST_FMATH_FUNC(TYPE, fmod, MODE)

#define TEST_FMATH_TYPE(TYPE)                   \
    TEST_FMATH_MODE(TYPE, regular)              \
    TEST_FMATH_MODE(TYPE, fast)                 \

TEST_FMATH_TYPE(float)
TEST_FMATH_TYPE(double)

TEST_FMATH_TYPE(float4)
TEST_FMATH_TYPE(double2)

TEST_FMATH_TYPE(half)

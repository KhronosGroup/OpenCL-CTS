//
// Copyright (c) 2018-2023 The Khronos Group Inc.
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
#include "spirvInfo.hpp"
#include "types.hpp"

#include <sstream>
#include <string>
#include <type_traits>


template <typename T>
int test_no_integer_wrap_decoration(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, const char *spvName,
                                    const char *funcName, const char *Tname)
{

    cl_int err = CL_SUCCESS;
    const int num = 10;
    std::vector<T> h_lhs(num);
    std::vector<T> h_rhs(num);
    std::vector<T> expected_results(num);
    std::vector<T> h_ref(num);

    /*Test with some values that do not cause overflow*/
    if (std::is_signed<T>::value == true)
    {
        h_lhs.push_back((T)-25000);
        h_lhs.push_back((T)-3333);
        h_lhs.push_back((T)-7);
        h_lhs.push_back((T)-1);
        h_lhs.push_back(0);
        h_lhs.push_back(1);
        h_lhs.push_back(1024);
        h_lhs.push_back(2048);
        h_lhs.push_back(4094);
        h_lhs.push_back(10000);
    }
    else
    {
        h_lhs.push_back(0);
        h_lhs.push_back(1);
        h_lhs.push_back(3);
        h_lhs.push_back(5);
        h_lhs.push_back(10);
        h_lhs.push_back(100);
        h_lhs.push_back(1024);
        h_lhs.push_back(2048);
        h_lhs.push_back(4094);
        h_lhs.push_back(52888);
    }

    h_rhs.push_back(0);
    h_rhs.push_back(1);
    h_rhs.push_back(2);
    h_rhs.push_back(3);
    h_rhs.push_back(4);
    h_rhs.push_back(5);
    h_rhs.push_back(6);
    h_rhs.push_back(7);
    h_rhs.push_back(8);
    h_rhs.push_back(9);
    size_t bytes = num * sizeof(T);

    clMemWrapper lhs =
        clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create lhs buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, bytes, &h_lhs[0], 0,
                               NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper rhs =
        clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create rhs buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, bytes, &h_rhs[0], 0,
                               NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to rhs buffer");

    std::string kernelStr;

    {
        std::stringstream kernelStream;
        kernelStream << "#define spirv_fadd(a, b) (a) + (b)               \n";
        kernelStream << "#define spirv_fsub(a, b) (a) - (b)               \n";
        kernelStream << "#define spirv_fmul(a, b) (a) * (b)               \n";
        kernelStream << "#define spirv_fshiftleft(a, b) (a) << (b)        \n";
        kernelStream << "#define spirv_fnegate(a, b)  (-a)                \n";

        kernelStream << "#define T " << Tname << "\n";
        kernelStream << "#define FUNC spirv_" << funcName << "\n";
        kernelStream << "__kernel void fmath_cl(__global T *out,          \n";
        kernelStream << "const __global T *lhs, const __global T *rhs)    \n";
        kernelStream << "{                                                \n";
        kernelStream << "    int id = get_global_id(0);                   \n";
        kernelStream << "    out[id] = FUNC(lhs[id], rhs[id]);            \n";
        kernelStream << "}                                                \n";
        kernelStr = kernelStream.str();
    }

    const char *kernelBuf = kernelStr.c_str();

    for (int i = 0; i < num; i++)
    {
        if (std::string(funcName) == std::string("fadd"))
        {
            expected_results[i] = h_lhs[i] + h_rhs[i];
        }
        else if (std::string(funcName) == std::string("fsub"))
        {
            expected_results[i] = h_lhs[i] - h_rhs[i];
        }
        else if (std::string(funcName) == std::string("fmul"))
        {
            expected_results[i] = h_lhs[i] * h_rhs[i];
        }
        else if (std::string(funcName) == std::string("fshiftleft"))
        {
            expected_results[i] = h_lhs[i] << h_rhs[i];
        }
        else if (std::string(funcName) == std::string("fnegate"))
        {
            expected_results[i] = 0 - h_lhs[i];
        }
    }

    {
        // Run the cl kernel for reference results
        clProgramWrapper prog;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &prog, &kernel, 1,
                                          &kernelBuf, "fmath_cl");
        SPIRV_CHECK_ERROR(err, "Failed to create cl kernel");

        clMemWrapper ref =
            clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
        SPIRV_CHECK_ERROR(err, "Failed to create ref buffer");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ref);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

        size_t global = num;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                     NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

        err = clEnqueueReadBuffer(queue, ref, CL_TRUE, 0, bytes, &h_ref[0], 0,
                                  NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to read from ref");
    }

    for (int i = 0; i < num; i++)
    {
        if (expected_results[i] != h_ref[i])
        {
            log_error(
                "Values do not match at index %d expected = %d got = %d\n", i,
                expected_results[i], h_ref[i]);
            return -1;
        }
    }

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, spvName);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, "fmath_cl", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    clMemWrapper res =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create res buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL,
                                 NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<T> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, bytes, &h_res[0], 0, NULL,
                              NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++)
    {
        if (expected_results[i] != h_res[i])
        {
            log_error(
                "Values do not match at location %d expected = %d got = %d\n",
                i, expected_results[i], h_res[i]);
            return -1;
        }
    }

    return TEST_PASS;
}

#define TEST_FMATH_FUNC_KHR(TYPE, FUNC)                                        \
    REGISTER_TEST(ext_cl_khr_spirv_no_integer_wrap_decoration_##FUNC##_##TYPE) \
    {                                                                          \
        if (!is_extension_available(                                           \
                device, "cl_khr_spirv_no_integer_wrap_decoration"))            \
        {                                                                      \
            log_info("Extension cl_khr_spirv_no_integer_wrap_decoration not "  \
                     "supported; skipping tests.\n");                          \
            return TEST_SKIPPED_ITSELF;                                        \
        }                                                                      \
        return test_no_integer_wrap_decoration<cl_##TYPE>(                     \
            device, context, queue,                                            \
            "ext_cl_khr_spirv_no_integer_wrap_decoration_" #FUNC "_" #TYPE,    \
            #FUNC, #TYPE);                                                     \
    }

TEST_FMATH_FUNC_KHR(int, fadd)
TEST_FMATH_FUNC_KHR(int, fsub)
TEST_FMATH_FUNC_KHR(int, fmul)
TEST_FMATH_FUNC_KHR(int, fshiftleft)
TEST_FMATH_FUNC_KHR(int, fnegate)
TEST_FMATH_FUNC_KHR(uint, fadd)
TEST_FMATH_FUNC_KHR(uint, fsub)
TEST_FMATH_FUNC_KHR(uint, fmul)
TEST_FMATH_FUNC_KHR(uint, fshiftleft)

#define TEST_FMATH_FUNC_14(TYPE, FUNC)                                         \
    REGISTER_TEST(spirv14_no_integer_wrap_decoration_##FUNC##_##TYPE)          \
    {                                                                          \
        if (!is_spirv_version_supported(device, "SPIR-V_1.4"))                 \
        {                                                                      \
            log_info("SPIR-V 1.4 not supported; skipping tests.\n");           \
            return TEST_SKIPPED_ITSELF;                                        \
        }                                                                      \
        return test_no_integer_wrap_decoration<cl_##TYPE>(                     \
            device, context, queue,                                            \
            "spv1.4/no_integer_wrap_decoration_" #FUNC "_" #TYPE, #FUNC,       \
            #TYPE);                                                            \
    }

TEST_FMATH_FUNC_14(int, fadd)
TEST_FMATH_FUNC_14(int, fsub)
TEST_FMATH_FUNC_14(int, fmul)
TEST_FMATH_FUNC_14(int, fshiftleft)
TEST_FMATH_FUNC_14(int, fnegate)
TEST_FMATH_FUNC_14(uint, fadd)
TEST_FMATH_FUNC_14(uint, fsub)
TEST_FMATH_FUNC_14(uint, fmul)
TEST_FMATH_FUNC_14(uint, fshiftleft)

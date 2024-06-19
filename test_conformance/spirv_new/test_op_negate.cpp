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

template<typename Tv>
int test_negation(cl_device_id deviceID,
                  cl_context context,
                  cl_command_queue queue,
                  const char *Tname,
                  const char *funcName,
                  const std::vector<Tv> &h_in,
                  Tv (*negate)(Tv) = negOp<Tv>)
{
    if(std::string(Tname).find("double") != std::string::npos) {
        if(!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return 0;
        }
    }
    if (std::string(Tname).find("half") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp16"))
        {
            log_info(
                "Extension cl_khr_fp16 not supported; skipping half tests.\n");
            return 0;
        }
    }

    cl_int err = CL_SUCCESS;
    int num = (int)h_in.size();
    size_t bytes = sizeof(Tv) * num;

    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    std::string spvStr = std::string(funcName) + "_" + std::string(Tname);
    const char *spvName = spvStr.c_str();

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, spvName);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, spvName, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<Tv> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != negate(h_in[i])) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

#define TEST_NEGATION(TYPE, Tv, OP, FUNC)                                      \
    TEST_SPIRV_FUNC(OP##_##TYPE)                                               \
    {                                                                          \
        int num = 1 << 20;                                                     \
        std::vector<Tv> in(num);                                               \
        RandomSeed seed(gRandomSeed);                                          \
        for (int i = 0; i < num; i++)                                          \
        {                                                                      \
            in[i] = genrand<Tv>(seed);                                         \
        }                                                                      \
        return test_negation<Tv>(deviceID, context, queue, #TYPE, #OP, in,     \
                                 FUNC);                                        \
    }


#define TEST_NEG_HALF TEST_NEGATION(half, cl_half, op_neg, negOpHalf)
#define TEST_NEG(TYPE)        TEST_NEGATION(TYPE, cl_##TYPE, op_neg, negOp<cl_##TYPE>)
#define TEST_NOT(TYPE)        TEST_NEGATION(TYPE, cl_##TYPE, op_not, notOp<cl_##TYPE>)
#define TEST_NEG_VEC(TYPE, N) TEST_NEGATION(TYPE##N, cl_##TYPE##N, op_neg, (negOpVec<cl_##TYPE##N, N>))
#define TEST_NOT_VEC(TYPE, N) TEST_NEGATION(TYPE##N, cl_##TYPE##N, op_not, (notOpVec<cl_##TYPE##N, N>))

TEST_NEG_HALF
TEST_NEG(float)
TEST_NEG(double)
TEST_NEG(int)
TEST_NEG(long)
TEST_NOT(int)
TEST_NOT(long)

#ifdef __GNUC__
// std::vector<cl_short> is causing compilation errors on GCC 5.3 (works on gcc 4.8)
// Needs further investigation
TEST_NEGATION(short, short, op_neg, negOp<cl_short>)
TEST_NEGATION(short, short, op_not, notOp<cl_short>)
#else
TEST_NEG(short)
TEST_NOT(short)
#endif

TEST_NEG_VEC(float  , 4)
TEST_NEG_VEC(int    , 4)
TEST_NOT_VEC(int    , 4)

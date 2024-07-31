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

int test_function(cl_device_id deviceID,
                  cl_context context,
                  cl_command_queue queue,
                  const char *funcType,
                  const std::vector<float> &h_in)
{
    cl_int err = CL_SUCCESS;
    int num = (int)h_in.size();
    size_t bytes = sizeof(float) * num;

    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    std::string spvStr = std::string("op_function") + "_" + std::string(funcType);
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

    std::vector<cl_float> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != -h_in[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}


#define TEST_FUNCTION(TYPE)                     \
    TEST_SPIRV_FUNC(function_##TYPE)            \
    {                                           \
        int num = 1 << 20;                      \
        std::vector<cl_float> in(num);          \
        RandomSeed seed(gRandomSeed);           \
        for (int i = 0; i < num; i++) {         \
            in[i] = genrand<cl_float>(seed);    \
        }                                       \
        return test_function(deviceID,          \
                             context,           \
                             queue,             \
                             #TYPE,             \
                             in);               \
    }                                           \

TEST_FUNCTION(none)
TEST_FUNCTION(inline)
TEST_FUNCTION(noinline)
TEST_FUNCTION(pure)
TEST_FUNCTION(const)
TEST_FUNCTION(pure_ptr)

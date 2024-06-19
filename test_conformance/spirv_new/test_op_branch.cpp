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

template<typename T>
int test_branch_simple(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, const char *name,
                       std::vector<T> &results,
                       bool (*notEqual)(const T&, const T&) = isNotEqual<T>)
{
    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    int num = (int)results.size();

    size_t bytes = num * sizeof(T);
    clMemWrapper in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clEnqueueWriteBuffer(queue, in_mem, CL_TRUE, 0, bytes, &results[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    std::vector<T> host(num);
    err = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, bytes, &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    for (int i = 0; i < num; i++) {
        if (notEqual(host[i], results[i])) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

#define TEST_BRANCH_SIMPLE(NAME)                            \
    TEST_SPIRV_FUNC(op_##NAME##_simple)                     \
    {                                                       \
        RandomSeed seed(gRandomSeed);                       \
        int num = 1 << 10;                                  \
        std::vector<cl_int> results(num);                   \
        for (int i = 0; i < num; i++) {                     \
            results[i] = genrand<cl_int>(seed);             \
        }                                                   \
        return test_branch_simple(deviceID, context, queue, \
                                  #NAME "_simple",          \
                                  results);                 \
    }                                                       \


TEST_BRANCH_SIMPLE(label)
TEST_BRANCH_SIMPLE(branch)
TEST_BRANCH_SIMPLE(unreachable)

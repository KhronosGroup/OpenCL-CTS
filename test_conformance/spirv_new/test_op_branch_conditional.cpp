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
int test_branch_conditional(cl_device_id deviceID,
                         cl_context context,
                         cl_command_queue queue,
                         const char *name,
                         const std::vector<T> &h_lhs,
                         const std::vector<T> &h_rhs,
                         const std::vector<T> &h_ref)
{

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

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build spv program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
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

#define TEST_BRANCH_CONDITIONAL(name)                                   \
    TEST_SPIRV_FUNC(op_##name)                                          \
    {                                                                   \
        const int num = 1 << 10;                                        \
        RandomSeed seed(gRandomSeed);                                   \
                                                                        \
        std::vector<cl_int> lhs(num);                                   \
        std::vector<cl_int> rhs(num);                                   \
        std::vector<cl_int> out(num);                                   \
                                                                        \
        for (int i = 0; i < num; i++) {                                 \
            lhs[i] = genrand<cl_int>(seed);                             \
            rhs[i] = genrand<cl_int>(seed);                             \
            out[i] = lhs[i] < rhs[i] ?                                  \
                              (rhs[i] - lhs[i]) : (lhs[i] - rhs[i]);    \
        }                                                               \
                                                                        \
        return test_branch_conditional(deviceID, context, queue,        \
                                       #name,                           \
                                       lhs, rhs, out);                  \
    }                                                                   \

TEST_BRANCH_CONDITIONAL(branch_conditional)
TEST_BRANCH_CONDITIONAL(branch_conditional_weighted)

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
int test_selection_merge(cl_device_id deviceID,
                         cl_context context,
                         cl_command_queue queue,
                         const char *name,
                         const std::vector<T> &h_in,
                         const std::vector<T> &h_ref,
                         const int rep)
{

    cl_int err = CL_SUCCESS;
    int num = (int)h_ref.size();
    size_t bytes = num * sizeof(T);
    size_t in_bytes = rep * bytes;

    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_ONLY, in_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, in_bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create out buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(int), &rep);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    err = clSetKernelArg(kernel, 3, sizeof(int), &num);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 3");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<T> h_out(num);
    err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != h_ref[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

#define TEST_LOOP_BRANCH(control)                                              \
    REGISTER_TEST(op_loop_merge_branch_##control)                              \
    {                                                                          \
        const int num = 1 << 10;                                               \
        RandomSeed seed(gRandomSeed);                                          \
                                                                               \
        int rep = 4;                                                           \
        std::vector<cl_int> in(rep *num);                                      \
        std::vector<cl_int> out(num);                                          \
                                                                               \
        for (int i = 0; i < num; i++)                                          \
        {                                                                      \
            int res = 0;                                                       \
            for (int j = 0; j < rep; j++)                                      \
            {                                                                  \
                cl_int val = genrand<cl_int>(seed) % 1024;                     \
                res += val;                                                    \
                in[j * num + i] = val;                                         \
            }                                                                  \
            out[i] = res;                                                      \
        }                                                                      \
                                                                               \
        return test_selection_merge(device, context, queue,                    \
                                    "loop_merge_branch_" #control, in, out,    \
                                    rep);                                      \
    }

TEST_LOOP_BRANCH(none)
TEST_LOOP_BRANCH(unroll)
TEST_LOOP_BRANCH(dont_unroll)


TEST_LOOP_BRANCH(conditional_none)
TEST_LOOP_BRANCH(conditional_unroll)
TEST_LOOP_BRANCH(conditional_dont_unroll)

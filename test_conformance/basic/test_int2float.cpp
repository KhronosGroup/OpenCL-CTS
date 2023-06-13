//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>
#include <vector>

#include "procs.h"

namespace {
const char *int2float_kernel_code = R"(
__kernel void test_X2Y(__global TYPE_X *src, __global TYPE_Y *dst)
{
    int  tid = get_global_id(0);

    dst[tid] = (TYPE_Y)src[tid];

})";

template <typename T> const char *Type2str() { return ""; }
template <> const char *Type2str<cl_int>() { return "int"; }
template <> const char *Type2str<cl_float>() { return "float"; }

template <typename T> void generate_random_inputs(std::vector<T> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31),
                                MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), seed);
    };

    std::generate(v.begin(), v.end(), random_generator);
}

template <typename Tx, typename Ty> bool equal_value(Tx a, Ty b)
{
    return a == (Tx)b;
}

template <typename Tx, typename Ty>
int verify_X2Y(std::vector<Tx> input, std::vector<Ty> output,
               const char *test_name)
{

    if (!std::equal(output.begin(), output.end(), input.begin(),
                    equal_value<Tx, Ty>))
    {
        log_error("%s test failed\n", test_name);
        return -1;
    }

    log_info("%s test passed\n", test_name);
    return 0;
}
template <typename Tx, typename Ty>
int test_X2Y(cl_device_id device, cl_context context, cl_command_queue queue,
             int num_elements, const char *test_name)
{
    clMemWrapper streams[2];
    clProgramWrapper program;
    clKernelWrapper kernel;
    int err;


    std::vector<Tx> input(num_elements);
    std::vector<Ty> output(num_elements);

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(Tx) * num_elements, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(Ty) * num_elements, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");

    generate_random_inputs(input);

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                               sizeof(Tx) * num_elements, input.data(), 0,
                               nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");

    std::string build_options;
    build_options.append("-DTYPE_X=").append(Type2str<Tx>());
    build_options.append(" -DTYPE_Y=").append(Type2str<Ty>());
    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &int2float_kernel_code, "test_X2Y",
                                      build_options.c_str());
    test_error(err, "create_single_kernel_helper failed.");

    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    test_error(err, "clSetKernelArg failed.");

    size_t threads[] = { (size_t)num_elements };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, threads, nullptr, 0,
                                 nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                              sizeof(Ty) * num_elements, output.data(), 0,
                              nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed.");

    err = verify_X2Y(input, output, test_name);

    return err;
}
}
int test_int2float(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    return test_X2Y<cl_int, cl_float>(device, context, queue, num_elements,
                                      "INT2FLOAT");
}
int test_float2int(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    return test_X2Y<cl_float, cl_int>(device, context, queue, num_elements,
                                      "FLOAT2INT");
}

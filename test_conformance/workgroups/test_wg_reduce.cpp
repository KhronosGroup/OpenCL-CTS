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

//#include <stdio.h>
//#include <string.h>
//#include <sys/types.h>
//#include <sys/stat.h>

#include <vector>

#include "procs.h"

static constexpr const char *kernel_source_reduce = R"CLC(
__kernel void test_wg_reduce_add(global TYPE *input, global TYPE *output)
{
    int  tid = get_global_id(0);

    output[tid] = work_group_reduce_add(input[tid]);
}
)CLC";

template <typename T> struct ReduceTestInfo
{
};

template <> struct ReduceTestInfo<cl_int>
{
    static constexpr const char *deviceTypeName = "int";
};

template <> struct ReduceTestInfo<cl_uint>
{
    static constexpr const char *deviceTypeName = "uint";
};

template <> struct ReduceTestInfo<cl_long>
{
    static constexpr const char *deviceTypeName = "long";
};

template <> struct ReduceTestInfo<cl_ulong>
{
    static constexpr const char *deviceTypeName = "ulong";
};

template <typename T>
static int verify_wg_reduce_add(T *inptr, T *outptr, size_t n, size_t wg_size)
{
    size_t i, j;

    for (i = 0; i < n; i += wg_size)
    {
        T sum = 0;
        for (j = 0; j < ((n - i) > wg_size ? wg_size : (n - i)); j++)
            sum += inptr[i + j];

        for (j = 0; j < ((n - i) > wg_size ? wg_size : (n - i)); j++)
        {
            if (sum != outptr[i + j])
            {
                log_info("work_group_reduce_add: Error at %u\n", i + j);
                return -1;
            }
        }
    }

    return 0;
}

template <typename T>
static int test_reduce_add_type(cl_device_id device, cl_context context,
                                cl_command_queue queue, int n_elems)
{
    cl_int err = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t wg_size[1];
    int i;

    std::string buildOptions;
    buildOptions += " -DTYPE=";
    buildOptions += ReduceTestInfo<T>::deviceTypeName;

    const char *kernel_source = kernel_source_reduce;
    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &kernel_source, "test_wg_reduce_add",
                                      buildOptions.c_str());
    test_error(err, "Unable to create test kernel");

    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    clMemWrapper src = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(T) * n_elems, NULL, &err);
    test_error(err, "Unable to create source buffer");

    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(T) * n_elems, NULL, &err);
    test_error(err, "Unable to create destination buffer");

    std::vector<T> input_ptr(n_elems);

    MTdataHolder d(gRandomSeed);
    for (int i = 0; i < n_elems; i++)
    {
        input_ptr[i] = genrand_int64(d);
    }

    err = clEnqueueWriteBuffer(queue, src, true, 0, sizeof(T) * n_elems,
                               input_ptr.data(), 0, NULL, NULL);
    test_error(err, "clWriteBuffer to initialize src buffer failed");

    err = clSetKernelArg(kernel, 0, sizeof(src), &src);
    test_error(err, "Unable to set src buffer kernel arg");
    err |= clSetKernelArg(kernel, 1, sizeof(dst), &dst);
    test_error(err, "Unable to set dst buffer kernel arg");

    size_t global_work_size[] = { n_elems };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 wg_size, 0, NULL, NULL);
    test_error(err, "Unable to enqueue test kernel");

    std::vector<T> output_ptr(n_elems);

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr.data(), &dead, sizeof(T) * n_elems);
    err = clEnqueueReadBuffer(queue, dst, true, 0, sizeof(T) * n_elems,
                              output_ptr.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueReadBuffer to read read dst buffer failed");

    if (verify_wg_reduce_add(input_ptr.data(), output_ptr.data(), n_elems,
                             wg_size[0]))
    {
        log_error("work_group_reduce_add %s failed\n",
                  ReduceTestInfo<T>::deviceTypeName);
        return TEST_FAIL;
    }

    log_info("work_group_reduce_add %s passed\n",
             ReduceTestInfo<T>::deviceTypeName);
    return TEST_PASS;
}

int test_work_group_reduce_add(cl_device_id device, cl_context context,
                               cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |= test_reduce_add_type<cl_int>(device, context, queue, n_elems);
    result |= test_reduce_add_type<cl_uint>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |=
            test_reduce_add_type<cl_long>(device, context, queue, n_elems);
        result |=
            test_reduce_add_type<cl_ulong>(device, context, queue, n_elems);
    }

    return result;
}

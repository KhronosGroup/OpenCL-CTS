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
#include "testBase.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include <sstream>
#include <string>

const char *queue_hint_test_kernel[] = {
"__kernel void vec_cpy(__global int *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src[tid];\n"
"\n"
"}\n" };

int test_enqueue(cl_context context, clCommandQueueWrapper& queue, clKernelWrapper& kernel, size_t num_elements)
{
    clMemWrapper            streams[2];
    int error;

    int* buf = new int[num_elements];

    for (int i = 0; i < static_cast<int>(num_elements); ++i)
    {
        buf[i] = i;
    }


    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, num_elements * sizeof(int), buf, &error);
    test_error( error, "clCreateBuffer failed." );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, num_elements * sizeof(int), NULL, &error);
    test_error( error, "clCreateBuffer failed." );

    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error( error, "clSetKernelArg failed." );

    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error( error, "clSetKernelArg failed." );

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &num_elements, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, num_elements * sizeof(int), buf, 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed." );

    for (int i = 0; i < static_cast<int>(num_elements); ++i)
    {
        if (buf[i] != i)
        {
            log_error("ERROR: Incorrect vector copy result.");
            return -1;
        }
    }

    delete [] buf;

    return 0;
}


REGISTER_TEST(queue_hint)
{
    if (num_elements <= 0)
    {
        num_elements = 128;
    }

    int err = 0;

    // Query extension
    clProgramWrapper program;
    clKernelWrapper kernel;

    err = create_single_kernel_helper_with_build_options(context, &program, &kernel, 1, queue_hint_test_kernel, "vec_cpy", NULL);
    if (err != 0)
    {
        return err;
    }

    if (is_extension_available(device, "cl_khr_priority_hints"))
    {
        log_info("Testing cl_khr_priority_hints...\n");

        cl_queue_properties queue_prop[][3] =
        {
            {
                CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR,
                0
            },
            {
                CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_MED_KHR,
                0
            },
            {
                CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR,
                0
            }
        };

        for (int i = 0; i < 3; ++i)
        {
            clCommandQueueWrapper q = clCreateCommandQueueWithProperties(
                context, device, queue_prop[i], &err);
            test_error(err, "clCreateCommandQueueWithProperties failed");

            err = test_enqueue(context, q, kernel, (size_t)num_elements);
            if (err != 0)
            {
                return err;
            }
        }
    }
    else
    {
        log_info("cl_khr_priority_hints is not supported.\n");
    }

    if (is_extension_available(device, "cl_khr_throttle_hints"))
    {
        log_info("Testing cl_khr_throttle_hints...\n");
        cl_queue_properties queue_prop[][3] =
        {
            {
                CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR,
                0
            },
            {
                CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_MED_KHR,
                0
            },
            {
                CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR,
                0
            }
        };

        for (int i = 0; i < 3; ++i)
        {
            clCommandQueueWrapper q = clCreateCommandQueueWithProperties(
                context, device, queue_prop[i], &err);
            test_error(err, "clCreateCommandQueueWithProperties failed");

            err = test_enqueue(context, q, kernel, (size_t)num_elements);
            if (err != 0)
            {
                return err;
            }
        }

    }
    else
    {
        log_info("cl_khr_throttle_hints is not supported.\n");
    }

    return 0;
}

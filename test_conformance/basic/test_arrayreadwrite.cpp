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
#include <vector>

#include "testBase.h"

static int test_arrayreadwrite_impl(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements,
                                    cl_mem_flags flags)
{
    clMemWrapper buffer;
    int                 num_tries = 400;
    num_elements = 1024 * 1024 * 4;
    MTdataHolder d(gRandomSeed);

    std::vector<cl_uint> reference_vals(num_elements);
    std::vector<cl_uint> inptr(num_elements);
    std::vector<cl_uint> outptr(num_elements);

    // randomize data
    for (int i = 0; i < num_elements; i++)
    {
        inptr[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);
        reference_vals[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);
    }

    void* host_ptr = nullptr;
    if ((flags & CL_MEM_USE_HOST_PTR) || (flags & CL_MEM_COPY_HOST_PTR))
    {
        host_ptr = inptr.data();
    }

    cl_int err = CL_SUCCESS;
    buffer = clCreateBuffer(context, flags, sizeof(cl_uint) * num_elements,
                            host_ptr, &err);
    test_error(err, "clCreateBuffer failed");

    for (int i = 0; i < num_tries; i++)
    {
        int        offset;
        int        cb;

        do {
            offset = (int)(genrand_int32(d) & 0x7FFFFFFF);
            if (offset > 0 && offset < num_elements)
                break;
        } while (1);
        cb = (int)(genrand_int32(d) & 0x7FFFFFFF);
        if (cb > (num_elements - offset))
            cb = num_elements - offset;

        err = clEnqueueWriteBuffer(
            queue, buffer, CL_TRUE, offset * sizeof(cl_uint),
            sizeof(cl_uint) * cb, &reference_vals[offset], 0, nullptr, nullptr);
        if (flags & CL_MEM_IMMUTABLE_EXT)
        {
            test_failure_error_ret(err, CL_INVALID_OPERATION,
                                   "clEnqueueWriteBuffer is expected to fail "
                                   "with CL_INVALID_OPERATION when the buffer "
                                   "is created with CL_MEM_IMMUTABLE_EXT",
                                   TEST_FAIL);
        }
        else
        {
            test_error(err, "clEnqueueWriteBuffer failed");
        }

        err = clEnqueueReadBuffer(
            queue, buffer, CL_TRUE, offset * sizeof(cl_uint),
            cb * sizeof(cl_uint), &outptr[offset], 0, nullptr, nullptr);
        test_error(err, "clEnqueueReadBuffer failed");

        const cl_uint* expected_buffer_values = nullptr;
        if (flags & CL_MEM_IMMUTABLE_EXT)
        {
            expected_buffer_values = inptr.data();
        }
        else
        {
            expected_buffer_values = reference_vals.data();
        }
        for (int j = offset; j < offset + cb; j++)
        {
            if (expected_buffer_values[j] != outptr[j])
            {
                log_error("ARRAY read, write test failed\n");
                err = -1;
                break;
            }
        }

        if (err)
            break;
    }

    if (!err)
        log_info("ARRAY read, write test passed\n");

    return err;
}


REGISTER_TEST(arrayreadwrite)
{
    return test_arrayreadwrite_impl(device, context, queue, num_elements,
                                    CL_MEM_READ_WRITE);
}

REGISTER_TEST(immutable_arrayreadwrite)
{
    REQUIRE_EXTENSION("cl_ext_immutable_memory_objects");

    return test_arrayreadwrite_impl(device, context, queue, num_elements,
                                    CL_MEM_IMMUTABLE_EXT | CL_MEM_USE_HOST_PTR);
}

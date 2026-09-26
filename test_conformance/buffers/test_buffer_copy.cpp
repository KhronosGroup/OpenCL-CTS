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

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "testBase.h"
#include "harness/errorHelpers.h"

static int verify_copy_buffer(int *inptr, int *outptr, int n)
{
    int         i;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}

using alignedOwningPtr = std::unique_ptr<cl_int[], decltype(&align_free)>;

static int test_copy(cl_device_id device, cl_command_queue queue,
                     cl_context context, int num_elements, MTdata d)
{
    clMemWrapper buffers[2];
    cl_int err = CL_SUCCESS;

    size_t  min_alignment = get_min_alignment(context);

    alignedOwningPtr invalid_ptr{
        (cl_int *)align_malloc(sizeof(cl_int) * num_elements, min_alignment),
        align_free
    };
    if (!invalid_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }
    alignedOwningPtr out_ptr{ (cl_int *)align_malloc(
                                  sizeof(cl_int) * num_elements, min_alignment),
                              align_free };
    if (!out_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }
    alignedOwningPtr reference_ptr{
        (cl_int *)align_malloc(sizeof(cl_int) * num_elements, min_alignment),
        align_free
    };
    if (!reference_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }

    const bool has_immutable_memory_extension =
        is_extension_available(device, "cl_ext_immutable_memory_objects");

    for (int src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
    {
        for (int dst_flag_id = 0; dst_flag_id < NUM_FLAGS; dst_flag_id++)
        {
            if (((flag_set[src_flag_id] & CL_MEM_IMMUTABLE_EXT)
                 || (flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT))
                && !has_immutable_memory_extension)
            {
                continue;
            }
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            for (int i = 0; i < num_elements; i++)
            {
                invalid_ptr[i] = static_cast<int>(0xdeaddead);
                out_ptr[i] = static_cast<int>(0xdeadbeef);
                reference_ptr[i] = (int)genrand_int32(d);
            }

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            reference_ptr.get(), &err);
            else
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            nullptr, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, "clCreateBuffer failed\n");
                return TEST_FAIL;
            }

            if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            invalid_ptr.get(), &err);
            else
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            nullptr, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, "clCreateBuffer failed\n");
                return TEST_FAIL;
            }

            if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR)
                && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
            {
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0,
                                           sizeof(cl_int) * num_elements,
                                           reference_ptr.get(), 0, nullptr,
                                           nullptr);
                if ( err != CL_SUCCESS ){
                    print_error(err, "clEnqueueWriteBuffer failed\n");
                    return TEST_FAIL;
                }
            }

            err = clEnqueueCopyBuffer(queue, buffers[0], buffers[1], 0, 0,
                                      sizeof(cl_int) * num_elements, 0, nullptr,
                                      nullptr);
            if ((flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT))
            {
                if (err != CL_INVALID_OPERATION)
                {
                    test_failure_error_ret(err, CL_INVALID_OPERATION,
                                           "clEnqueueCopyBuffer should return "
                                           "CL_INVALID_OPERATION when: "
                                           "\"dst_buffer is created with "
                                           "CL_MEM_IMMUTABLE_EXT flag\"",
                                           TEST_FAIL);
                    return TEST_FAIL;
                }
            }
            else if (err != CL_SUCCESS)
            {
                print_error(err, "clCopyArray failed\n");
                return TEST_FAIL;
            }

            err = clEnqueueReadBuffer(queue, buffers[0], true, 0,
                                      sizeof(int) * num_elements, out_ptr.get(),
                                      0, nullptr, nullptr);
            if (verify_copy_buffer(reference_ptr.get(), out_ptr.get(),
                                   num_elements))
            {
                log_error("test failed\n");
                return TEST_FAIL;
            }
            else
            {
                log_info("test passed\n");
            }

            // Reset out_ptr
            for (int i = 0; i < num_elements; i++)
            {
                out_ptr[i] = (int)0xdeadbeef; // seed with incorrect data
            }
            err = clEnqueueReadBuffer(queue, buffers[1], true, 0,
                                      sizeof(int) * num_elements, out_ptr.get(),
                                      0, nullptr, nullptr);
            if ( err != CL_SUCCESS ){
                print_error(err, "clEnqueueReadBuffer failed\n");
                return TEST_FAIL;
            }

            int *target_buffer = reference_ptr.get();
            if (flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT)
            {
                target_buffer = invalid_ptr.get();
            }

            if (verify_copy_buffer(target_buffer, out_ptr.get(), num_elements))
            {
                log_error("test failed\n");
                return TEST_FAIL;
            }
            else
            {
                log_info("test passed\n");
            }
        } // dst flags
    } // src flags

    return TEST_PASS;

}   // end test_copy()


static int testPartialCopy(cl_device_id device, cl_command_queue queue,
                           cl_context context, int num_elements,
                           cl_uint srcStart, cl_uint dstStart, int size,
                           MTdata d)
{
    clMemWrapper buffers[2];
    cl_int err = CL_SUCCESS;

    size_t  min_alignment = get_min_alignment(context);

    alignedOwningPtr invalid_ptr{
        (cl_int *)align_malloc(sizeof(cl_int) * num_elements, min_alignment),
        align_free
    };
    if (!invalid_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }
    alignedOwningPtr out_ptr{ (cl_int *)align_malloc(
                                  sizeof(cl_int) * num_elements, min_alignment),
                              align_free };
    if (!out_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }
    alignedOwningPtr reference_ptr{
        (cl_int *)align_malloc(sizeof(cl_int) * num_elements, min_alignment),
        align_free
    };
    if (!reference_ptr)
    {
        log_error(" unable to allocate %zu bytes of memory\n",
                  sizeof(cl_int) * num_elements);
        return TEST_FAIL;
    }

    const bool has_immutable_memory_extension =
        is_extension_available(device, "cl_ext_immutable_memory_objects");

    for (int src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
    {
        for (int dst_flag_id = 0; dst_flag_id < NUM_FLAGS; dst_flag_id++)
        {
            if (((flag_set[src_flag_id] & CL_MEM_IMMUTABLE_EXT)
                 || (flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT))
                && !has_immutable_memory_extension)
            {
                continue;
            }
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            for (int i = 0; i < num_elements; i++)
            {
                invalid_ptr[i] = static_cast<int>(0xdeaddead);
                out_ptr[i] = static_cast<int>(0xdeadbeef);
                reference_ptr[i] = (int)genrand_int32(d);
            }

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            reference_ptr.get(), &err);
            else
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            nullptr, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, "clCreateBuffer failed\n");
                return TEST_FAIL;
            }

            if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            invalid_ptr.get(), &err);
            else
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],
                                            sizeof(cl_int) * num_elements,
                                            nullptr, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, "clCreateBuffer failed\n");
                return TEST_FAIL;
            }

            if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR)
                && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
            {
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0,
                                           sizeof(cl_int) * num_elements,
                                           reference_ptr.get(), 0, nullptr,
                                           nullptr);
                if ( err != CL_SUCCESS ){
                    print_error(err, "clEnqueueWriteBuffer failed\n");
                    return TEST_FAIL;
                }
            }

            err = clEnqueueCopyBuffer(
                queue, buffers[0], buffers[1], srcStart * sizeof(cl_int),
                dstStart * sizeof(cl_int), sizeof(cl_int) * size, 0, nullptr,
                nullptr);
            if ((flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT))
            {
                if (err != CL_INVALID_OPERATION)
                {
                    test_failure_error_ret(err, CL_INVALID_OPERATION,
                                           "clEnqueueCopyBuffer should return "
                                           "CL_INVALID_OPERATION when: "
                                           "\"dst_buffer is created with "
                                           "CL_MEM_IMMUTABLE_EXT flag\"",
                                           TEST_FAIL);
                }
            }
            else if (err != CL_SUCCESS)
            {
                print_error(err, "clCopyArray failed\n");
                return TEST_FAIL;
            }

            err = clEnqueueReadBuffer(queue, buffers[0], true, 0,
                                      sizeof(int) * num_elements, out_ptr.get(),
                                      0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                print_error(err, "clEnqueueReadBuffer failed\n");
                return TEST_FAIL;
            }
            if (verify_copy_buffer(reference_ptr.get(), out_ptr.get(),
                                   num_elements))
            {
                log_error("test failed\n");
                return TEST_FAIL;
            }
            else
            {
                log_info("test passed\n");
            }

            // Reset out_ptr
            for (int i = 0; i < num_elements; i++)
            {
                out_ptr[i] = (int)0xdeadbeef; // seed with incorrect data
            }
            err = clEnqueueReadBuffer(queue, buffers[1], true, 0,
                                      sizeof(int) * num_elements, out_ptr.get(),
                                      0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                print_error(err, "clEnqueueReadBuffer failed\n");
                return TEST_FAIL;
            }

            cl_int *target_buffer = reference_ptr.get() + srcStart;
            if (flag_set[dst_flag_id] & CL_MEM_IMMUTABLE_EXT)
            {
                target_buffer = invalid_ptr.get();
            }

            if (verify_copy_buffer(target_buffer, out_ptr.get() + dstStart,
                                   size))
            {
                log_error("test failed\n");
                return TEST_FAIL;
            }
            else
            {
                log_info("test passed\n");
            }
        } // dst mem flags
    } // src mem flags

    return TEST_PASS;

}   // end testPartialCopy()


REGISTER_TEST(buffer_copy)
{
    int     i, err = 0;
    int     size;
    MTdata  d = init_genrand( gRandomSeed );

    // test the preset size
    log_info( "set size: %d: ", num_elements );
    if (test_copy(device, queue, context, num_elements, d) != TEST_PASS)
    {
        err++;
    }

    // now test random sizes
    for ( i = 0; i < 8; i++ ){
        size = (int)get_random_float(2.f,131072.f, d);
        log_info( "random size: %d: ", size );
        if (test_copy(device, queue, context, size, d) != TEST_PASS)
        {
            err++;
        }
    }

    free_mtdata(d);

    return err;

}   // end test_buffer_copy()


REGISTER_TEST(buffer_partial_copy)
{
    int     i, err = 0;
    int     size;
    cl_uint srcStart, dstStart;
    MTdata  d = init_genrand( gRandomSeed );

    // now test copy of partial sizes
    for ( i = 0; i < 8; i++ ){
        srcStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - 8), d );
        size = (int)get_random_float( 8.f, (float)(num_elements - srcStart), d );
        dstStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - size), d );
        log_info( "random partial copy from %d to %d, size: %d: ", (int)srcStart, (int)dstStart, size );
        if (testPartialCopy(device, queue, context, num_elements, srcStart,
                            dstStart, size, d)
            != TEST_PASS)
        {
            err++;
        }
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_partial_copy()


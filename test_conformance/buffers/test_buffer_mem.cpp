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

#include "testBase.h"

#ifndef uchar
typedef unsigned char uchar;
#endif


const char *mem_read_write_kernel_code =
"__kernel void test_mem_read_write(__global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = dst[tid]+1;\n"
"}\n";

const char *mem_read_kernel_code =
    "__kernel void test_mem_read(__global int *dst, __global int *src)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid]+1;\n"
    "}\n";

const char *mem_write_kernel_code =
"__kernel void test_mem_write(__global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = dst[tid]+1;\n"
"}\n";


static int verify_mem( int *outptr, int n )
{
    int i;

    for ( i = 0; i < n; i++ ){
        if ( outptr[i] != ( i + 1 ) )
            return -1;
    }

    return 0;
}


static int test_mem_flags(cl_context context, cl_command_queue queue,
                          int num_elements, cl_mem_flags flags,
                          const char **kernel_program, const char *kernel_name)
{
    clMemWrapper buffers[2];
    cl_int      *inptr, *outptr;
    clProgramWrapper program;
    clKernelWrapper kernel;
    size_t      global_work_size[3];
    cl_int      err;
    int         i;

    size_t      min_alignment = get_min_alignment(context);
    bool test_read_only = (flags & CL_MEM_READ_ONLY) != 0;
    bool test_write_only = (flags & CL_MEM_WRITE_ONLY) != 0;
    bool copy_host_ptr = (flags & CL_MEM_COPY_HOST_PTR) != 0;

    global_work_size[0] = (cl_uint)num_elements;

    inptr = (cl_int*)align_malloc(sizeof(cl_int)  * num_elements, min_alignment);
    if (!inptr)
    {
        log_error(" unable to allocate %d bytes of memory\n",
                  (int)sizeof(cl_int) * num_elements);
        return -1;
    }
    outptr = (cl_int*)align_malloc(sizeof(cl_int) * num_elements, min_alignment);
    if (!outptr)
    {
        log_error(" unable to allocate %d bytes of memory\n",
                  (int)sizeof(cl_int) * num_elements);
        align_free((void *)inptr);
        return -1;
    }

    for (i = 0; i < num_elements; i++) inptr[i] = i;

    buffers[0] = clCreateBuffer(context, flags, sizeof(cl_int) * num_elements,
                                copy_host_ptr ? inptr : NULL, &err);
    if (err != CL_SUCCESS)
    {
        print_error(err, "clCreateBuffer failed");
        align_free((void *)outptr);
        align_free((void *)inptr);
        return -1;
    }
    if (!copy_host_ptr)
    {
        err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0,
                                   sizeof(cl_int) * num_elements, (void *)inptr,
                                   0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clEnqueueWriteBuffer failed");
            align_free((void *)outptr);
            align_free((void *)inptr);
            return -1;
        }
    }

    if (test_read_only)
    {
        /* The read only buffer for mem_read_only_flags should be created above
        with the correct flags as in other tests. However to make later test
        code simpler, the additional read_write buffer required is stored as
        the first buffer */
        buffers[1] = buffers[0];
        buffers[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(cl_int) * num_elements, NULL, &err);
        if (err != CL_SUCCESS)
        {
            print_error(err, " clCreateBuffer failed \n");
            align_free((void *)inptr);
            align_free((void *)outptr);
            return -1;
        }
    }

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      kernel_program, kernel_name);
    if (err){
        print_error(err, "creating kernel failed");
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[0]);
    if (test_read_only && (err == CL_SUCCESS))
    {
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffers[1]);
    }
    if ( err != CL_SUCCESS ){
        print_error( err, "clSetKernelArg failed" );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL,
                                 0, NULL, NULL);
    if (err != CL_SUCCESS){
        log_error("clEnqueueNDRangeKernel failed\n");
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[0], true, 0,
                              sizeof(cl_int) * num_elements, (void *)outptr, 0,
                              NULL, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueReadBuffer failed" );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    if (!test_write_only)
    {
        if (verify_mem(outptr, num_elements))
        {
            log_error("test failed\n");
            err = -1;
        }
        else
        {
            log_info("test passed\n");
            err = 0;
        }
    }

    // cleanup
    align_free( (void *)outptr );
    align_free( (void *)inptr );

    return err;
} // end test_mem_flags()

REGISTER_TEST(mem_read_write_flags)
{
    return test_mem_flags(context, queue, num_elements, CL_MEM_READ_WRITE,
                          &mem_read_write_kernel_code, "test_mem_read_write");
}


REGISTER_TEST(mem_write_only_flags)
{
    return test_mem_flags(context, queue, num_elements, CL_MEM_WRITE_ONLY,
                          &mem_write_kernel_code, "test_mem_write");
}


REGISTER_TEST(mem_read_only_flags)
{
    return test_mem_flags(context, queue, num_elements, CL_MEM_READ_ONLY,
                          &mem_read_kernel_code, "test_mem_read");
}


REGISTER_TEST(mem_copy_host_flags)
{
    return test_mem_flags(context, queue, num_elements,
                          CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                          &mem_read_write_kernel_code, "test_mem_read_write");
}

REGISTER_TEST(mem_alloc_ref_flags)
{
    return test_mem_flags(context, queue, num_elements,
                          CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                          &mem_read_write_kernel_code, "test_mem_read_write");
}

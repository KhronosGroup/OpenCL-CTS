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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"


const char *wg_reduce_min_kernel_code_int =
"__kernel void test_wg_reduce_min_int(global int *input, global int *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    int result = work_group_reduce_min(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


const char *wg_reduce_min_kernel_code_uint =
"__kernel void test_wg_reduce_min_uint(global uint *input, global uint *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    uint result = work_group_reduce_min(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";

const char *wg_reduce_min_kernel_code_long =
"__kernel void test_wg_reduce_min_long(global long *input, global long *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    long result = work_group_reduce_min(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


const char *wg_reduce_min_kernel_code_ulong =
"__kernel void test_wg_reduce_min_ulong(global ulong *input, global ulong *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    ulong result = work_group_reduce_min(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


static int
verify_wg_reduce_min_int(int *inptr, int *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;

    for (i=0; i<n; i+=wg_size)
    {
        int min = CL_INT_MAX;
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
            min = (min < inptr[i+j]) ? min : inptr[i+j];

        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if ( min != outptr[i+j] )
            {
                log_info("work_group_reduce_min int: Error at %u: expected = %d, got = %d\n", i+j, min, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}

static int
verify_wg_reduce_min_uint(unsigned int *inptr, unsigned int *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;

    for (i=0; i<n; i+=wg_size)
    {
        unsigned int min = CL_UINT_MAX;
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
            min = (min < inptr[i+j]) ? min : inptr[i+j];

        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if ( min != outptr[i+j] )
            {
                log_info("work_group_reduce_min uint: Error at %u: expected = %d, got = %d\n", i+j, min, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}

static int
verify_wg_reduce_min_long(cl_long *inptr, cl_long *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;

    for (i=0; i<n; i+=wg_size)
    {
        cl_long min = CL_ULONG_MAX;
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
            min = (min < inptr[i+j]) ? min : inptr[i+j];

        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if ( min != outptr[i+j] )
            {
                log_info("work_group_reduce_min long: Error at %u: expected = %lld, got = %lld\n", i+j, min, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}

static int
verify_wg_reduce_min_ulong(cl_ulong *inptr, cl_ulong *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;

    for (i=0; i<n; i+=wg_size)
    {
        cl_ulong min = CL_ULONG_MAX;
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
            min = (min < inptr[i+j]) ? min : inptr[i+j];

        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if ( min != outptr[i+j] )
            {
                log_info("work_group_reduce_min ulong: Error at %u: expected = %llu, got = %llu\n", i+j, min, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}



int
test_work_group_reduce_min_int(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_int       *input_ptr[1], *p;
    cl_int       *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       wg_sizes_per_dimension[3];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &wg_reduce_min_kernel_code_int, "test_wg_reduce_min_int", "-cl-std=CL2.0" );
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, wg_sizes_per_dimension, NULL);
    if (err)
        return -1;
    if(wg_sizes_per_dimension[0] < wg_size[0])
    {
        wg_size[0] = wg_sizes_per_dimension[0];
    }

    num_elements = n_elems;

    input_ptr[0] = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    output_ptr = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_int) * num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    values[0] = streams[0];
    values[1] = streams[1];
    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    // Line below is troublesome...
    threads[0] = (size_t)num_elements;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_int)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_reduce_min_int(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_reduce_min int failed\n");
        return -1;
    }
    log_info("work_group_reduce_min int passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_reduce_min_uint(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_uint      *input_ptr[1], *p;
    cl_uint      *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       wg_sizes_per_dimension[3];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &wg_reduce_min_kernel_code_uint, "test_wg_reduce_min_uint", "-cl-std=CL2.0" );
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, wg_sizes_per_dimension, NULL);
    if (err)
        return -1;
    if(wg_sizes_per_dimension[0] < wg_size[0])
    {
        wg_size[0] = wg_sizes_per_dimension[0];
    }

    num_elements = n_elems;

    input_ptr[0] = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);
    output_ptr = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_uint) * num_elements, NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_uint) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_uint)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    values[0] = streams[0];
    values[1] = streams[1];
    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    // Line below is troublesome...
    threads[0] = (size_t)n_elems;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_uint)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_uint)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_reduce_min_uint(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_reduce_min uint failed\n");
        return -1;
    }
    log_info("work_group_reduce_min uint passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}

int
test_work_group_reduce_min_long(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_long      *input_ptr[1], *p;
    cl_long      *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       wg_sizes_per_dimension[3];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &wg_reduce_min_kernel_code_long, "test_wg_reduce_min_long", "-cl-std=CL2.0" );
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, wg_sizes_per_dimension, NULL);
    if (err)
        return -1;
    if(wg_sizes_per_dimension[0] < wg_size[0])
    {
        wg_size[0] = wg_sizes_per_dimension[0];
    }

    num_elements = n_elems;

    input_ptr[0] = (cl_long*)malloc(sizeof(cl_long) * num_elements);
    output_ptr = (cl_long*)malloc(sizeof(cl_long) * num_elements);
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_long) * num_elements, NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_long) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        p[i] = genrand_int64(d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_long)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    values[0] = streams[0];
    values[1] = streams[1];
    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    // Line below is troublesome...
    threads[0] = (size_t)n_elems;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_long)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_long)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_reduce_min_long(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_reduce_min long failed\n");
        return -1;
    }
    log_info("work_group_reduce_min long passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_reduce_min_ulong(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_ulong     *input_ptr[1], *p;
    cl_ulong     *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       wg_sizes_per_dimension[3];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &wg_reduce_min_kernel_code_ulong, "test_wg_reduce_min_ulong", "-cl-std=CL2.0" );
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, wg_sizes_per_dimension, NULL);
    if (err)
        return -1;
    if(wg_sizes_per_dimension[0] < wg_size[0])
    {
        wg_size[0] = wg_sizes_per_dimension[0];
    }

    num_elements = n_elems;

    input_ptr[0] = (cl_ulong*)malloc(sizeof(cl_ulong) * num_elements);
    output_ptr = (cl_ulong*)malloc(sizeof(cl_ulong) * num_elements);
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_ulong) * num_elements, NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_ulong) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        p[i] = genrand_int64(d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_ulong)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    values[0] = streams[0];
    values[1] = streams[1];
    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    // Line below is troublesome...
    threads[0] = (size_t)n_elems;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_ulong)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_ulong)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_reduce_min_ulong(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_reduce_min ulong failed\n");
        return -1;
    }
    log_info("work_group_reduce_min ulong passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_reduce_min(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int err;

    err = test_work_group_reduce_min_int(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_reduce_min_uint(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_reduce_min_long(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_reduce_min_ulong(device, context, queue, n_elems);
    return err;
}


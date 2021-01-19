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


const char *wg_scan_inclusive_add_kernel_code_int =
"__kernel void test_wg_scan_inclusive_add_int(global int *input, global int *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    int result = work_group_scan_inclusive_add(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


const char *wg_scan_inclusive_add_kernel_code_uint =
"__kernel void test_wg_scan_inclusive_add_uint(global uint *input, global uint *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    uint result = work_group_scan_inclusive_add(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";

const char *wg_scan_inclusive_add_kernel_code_long =
"__kernel void test_wg_scan_inclusive_add_long(global long *input, global long *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    long result = work_group_scan_inclusive_add(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


const char *wg_scan_inclusive_add_kernel_code_ulong =
"__kernel void test_wg_scan_inclusive_add_ulong(global ulong *input, global ulong *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    ulong result = work_group_scan_inclusive_add(input[tid]);\n"
"    output[tid] = result;\n"
"}\n";


static int
verify_wg_scan_inclusive_add_int(int *inptr, int *outptr, size_t n, size_t wg_size)
{
    size_t i, j, m;
    int s;

    for (j=0; j<n; j+=wg_size) {
    m = n - j;
    if (m > wg_size)
        m = wg_size;

    s = 0;
    for (i=0; i<m; ++i) {
        s += inptr[j+i];
        if (outptr[j+i] != s) {
                log_info("work_group_scan_inclusive_add int: Error at %u: expected = %d, got = %d\n", (unsigned int)(j+i), s, outptr[j+i]);
                return -1;
            }
        }
    }
    return 0;
}

static int
verify_wg_scan_inclusive_add_uint(unsigned int *inptr, unsigned int *outptr, size_t n, size_t wg_size)
{
    size_t i, j, m;
    unsigned int s;

    for (j=0; j<n; j+=wg_size) {
    m = n - j;
    if (m > wg_size)
        m = wg_size;

    s = 0;
    for (i=0; i<m; ++i) {
        s += inptr[j+i];
        if (outptr[j+i] != s) {
                log_info("work_group_scan_inclusive_add uint: Error at %u: expected = %u, got = %u\n", (unsigned int)(j+i), s, outptr[j+i]);
                return -1;
            }
        }
    }
    return 0;
}

static int
verify_wg_scan_inclusive_add_long(cl_long *inptr, cl_long *outptr, size_t n, size_t wg_size)
{
    size_t i, j, m;
    cl_long s;

    for (j=0; j<n; j+=wg_size) {
    m = n - j;
    if (m > wg_size)
        m = wg_size;

    s = 0;
    for (i=0; i<m; ++i) {
        s += inptr[j+i];
        if (outptr[j+i] != s) {
                log_info("work_group_scan_inclusive_add long: Error at %u: expected = %lld, got = %lld\n",
            (unsigned int)(j+i), (long long)s, (long long)outptr[j+i]);
                return -1;
            }
        }
    }
    return 0;
}

static int
verify_wg_scan_inclusive_add_ulong(cl_ulong *inptr, cl_ulong *outptr, size_t n, size_t wg_size)
{
    size_t i, j, m;
    cl_ulong s;

    for (j=0; j<n; j+=wg_size) {
    m = n - j;
    if (m > wg_size)
        m = wg_size;

    s = 0;
    for (i=0; i<m; ++i) {
        s += inptr[j+i];
        if (outptr[j+i] != s) {
                log_info("work_group_scan_inclusive_add int: Error at %u: expected = %llu, got = %llu\n",
            (unsigned int)(j+i), (unsigned long long)s, (unsigned long long)outptr[j+i]);
                return -1;
            }
        }
    }
    return 0;
}


int
test_work_group_scan_inclusive_add_int(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_int       *input_ptr[1], *p;
    cl_int       *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_scan_inclusive_add_kernel_code_int,
                                      "test_wg_scan_inclusive_add_int");
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    output_ptr = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, NULL);
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

    if (verify_wg_scan_inclusive_add_int(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_scan_inclusive_add int failed\n");
        return -1;
    }
    log_info("work_group_scan_inclusive_add int passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_scan_inclusive_add_uint(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_uint      *input_ptr[1], *p;
    cl_uint      *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_scan_inclusive_add_kernel_code_uint,
                                      "test_wg_scan_inclusive_add_uint");
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);
    output_ptr = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_uint) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_uint) * num_elements, NULL, NULL);
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

    if (verify_wg_scan_inclusive_add_uint(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_scan_inclusive_add uint failed\n");
        return -1;
    }
    log_info("work_group_scan_inclusive_add uint passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}

int
test_work_group_scan_inclusive_add_long(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_long      *input_ptr[1], *p;
    cl_long      *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_scan_inclusive_add_kernel_code_long,
                                      "test_wg_scan_inclusive_add_long");
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_long*)malloc(sizeof(cl_long) * num_elements);
    output_ptr = (cl_long*)malloc(sizeof(cl_long) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_long) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_long) * num_elements, NULL, NULL);
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

    if (verify_wg_scan_inclusive_add_long(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_scan_inclusive_add long failed\n");
        return -1;
    }
    log_info("work_group_scan_inclusive_add long passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_scan_inclusive_add_ulong(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_ulong     *input_ptr[1], *p;
    cl_ulong     *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_scan_inclusive_add_kernel_code_ulong,
                                      "test_wg_scan_inclusive_add_ulong");
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_ulong*)malloc(sizeof(cl_ulong) * num_elements);
    output_ptr = (cl_ulong*)malloc(sizeof(cl_ulong) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_ulong) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_ulong) * num_elements, NULL, NULL);
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

    if (verify_wg_scan_inclusive_add_ulong(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_scan_inclusiveadd ulong failed\n");
        return -1;
    }
    log_info("work_group_scan_inclusive_add ulong passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_scan_inclusive_add(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int err;

    err = test_work_group_scan_inclusive_add_int(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_scan_inclusive_add_uint(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_scan_inclusive_add_long(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_scan_inclusive_add_ulong(device, context, queue, n_elems);
    return err;
}


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

#include <algorithm>

#include "procs.h"


const char *wg_broadcast_1D_kernel_code =
"__kernel void test_wg_broadcast_1D(global float *input, global float *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    float result = work_group_broadcast(input[tid], get_group_id(0) % get_local_size(0));\n"
"    output[tid] = result;\n"
"}\n";

const char *wg_broadcast_2D_kernel_code =
"__kernel void test_wg_broadcast_2D(global float *input, global float *output)\n"
"{\n"
"    size_t tid_x = get_global_id(0);\n"
"    size_t tid_y = get_global_id(1);\n"
"    size_t x = get_group_id(0) % get_local_size(0);\n"
"    size_t y = get_group_id(1) % get_local_size(1);\n"
"\n"
"    size_t indx = (tid_y * get_global_size(0)) + tid_x;\n"
"    float result = work_group_broadcast(input[indx], x, y);\n"
"    output[indx] = result;\n"
"}\n";

const char *wg_broadcast_3D_kernel_code =
"__kernel void test_wg_broadcast_3D(global float *input, global float *output)\n"
"{\n"
"    size_t tid_x = get_global_id(0);\n"
"    size_t tid_y = get_global_id(1);\n"
"    size_t tid_z = get_global_id(2);\n"
"    size_t x = get_group_id(0) % get_local_size(0);\n"
"    size_t y = get_group_id(1) % get_local_size(1);\n"
"    size_t z = get_group_id(2) % get_local_size(2);\n"
"\n"
"    size_t indx = (tid_z * get_global_size(1) * get_global_size(0)) + (tid_y * get_global_size(0)) + tid_x;\n"
"    float result = work_group_broadcast(input[indx], x, y, z);\n"
"    output[indx] = result;\n"
"}\n";

static int
verify_wg_broadcast_1D(float *inptr, float *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;
    size_t     group_id;

    for (i=0,group_id=0; i<n; i+=wg_size,group_id++)
    {
        int local_size = (n-i) > wg_size ? wg_size : (n-i);
        float broadcast_result = inptr[i + (group_id % local_size)];
        for (j=0; j<local_size; j++)
        {
            if ( broadcast_result != outptr[i+j] )
            {
                log_info("work_group_broadcast: Error at %u: expected = %f, got = %f\n", i+j, broadcast_result, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}

static int
verify_wg_broadcast_2D(float *inptr, float *outptr, size_t nx, size_t ny, size_t wg_size_x, size_t wg_size_y)
{
    size_t i, j, _i, _j;
    size_t group_id_x, group_id_y;

    for (i=0,group_id_y=0; i<ny; i+=wg_size_y,group_id_y++)
    {
        size_t y = group_id_y % wg_size_y;
        size_t local_size_y = (ny-i) > wg_size_y ? wg_size_y : (ny-i);
        for (_i=0; _i < local_size_y; _i++)
        {
            for (j=0,group_id_x=0; j<nx; j+=wg_size_x,group_id_x++)
            {
                size_t x = group_id_x % wg_size_x;
                size_t local_size_x = (nx-j) > wg_size_x ? wg_size_x : (nx-j);
                float  broadcast_result = inptr[(i + y) * nx + (j + x)];
                for (_j=0; _j < local_size_x; _j++)
                {
                    size_t indx = (i + _i) * nx + (j + _j);
                    if ( broadcast_result != outptr[indx] )
                    {
                        log_info("work_group_broadcast: Error at (%u, %u): expected = %f, got = %f\n", j+_j, i+_i, broadcast_result, outptr[indx]);
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}

static int
verify_wg_broadcast_3D(float *inptr, float *outptr, size_t nx, size_t ny, size_t nz, size_t wg_size_x, size_t wg_size_y, size_t wg_size_z)
{
    size_t i, j, k, _i, _j, _k;
    size_t group_id_x, group_id_y, group_id_z;

    for (i=0,group_id_z=0; i<nz; i+=wg_size_z,group_id_z++)
    {
        size_t z = group_id_z % wg_size_z;
        size_t local_size_z = (nz-i) > wg_size_z ? wg_size_z : (nz-i);
        for (_i=0; _i < local_size_z; _i++)
        {
            for (j=0,group_id_y=0; j<ny; j+=wg_size_y,group_id_y++)
            {
                size_t y = group_id_y % wg_size_y;
                size_t local_size_y = (ny-j) > wg_size_y ? wg_size_y : (ny-j);
                for (_j=0; _j < local_size_y; _j++)
                {
                    for (k=0,group_id_x=0; k<nx; k+=wg_size_x,group_id_x++)
                    {
                        size_t x = group_id_x % wg_size_x;
                        size_t local_size_x = (nx-k) > wg_size_x ? wg_size_x : (nx-k);
                        float  broadcast_result = inptr[(i + z) * ny * nz + (j + y) * nx + (k + x)];
                        for (_k=0; _k < local_size_x; _k++)
                        {
                            size_t indx = (i + _i) * ny * nx + (j + _j) * nx + (k + _k);
                            if ( broadcast_result != outptr[indx] )
                            {
                                log_info("work_group_broadcast: Error at (%u, %u, %u): expected = %f, got = %f\n", k+_k, j+_j, i+_i, broadcast_result, outptr[indx]);
                                return -1;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}


int
test_work_group_broadcast_1D(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_float     *input_ptr[1], *p;
    cl_float     *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       globalsize[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_broadcast_1D_kernel_code,
                                      "test_wg_broadcast_1D");
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float((float)(-100000.f * M_PI), (float)(100000.f * M_PI) ,d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
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
    globalsize[0] = (size_t)n_elems;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, globalsize, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_float)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_broadcast_1D(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_broadcast_1D test failed\n");
        return -1;
    }
    log_info("work_group_broadcast_1D test passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_broadcast_2D(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_float     *input_ptr[1], *p;
    cl_float     *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       globalsize[2];
    size_t       localsize[2];
    size_t       wg_size[1];
    size_t       num_workgroups;
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_broadcast_2D_kernel_code,
                                      "test_wg_broadcast_2D");
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    if (wg_size[0] >= 256)
    {
        localsize[0] = localsize[1] = 16;
    }
    else if (wg_size[0] >=64)
    {
        localsize[0] = localsize[1] = 8;
    }
    else if (wg_size[0] >= 16)
    {
        localsize[0] = localsize[1] = 4;
    }
    else
    {
        localsize[0] = localsize[1] = 1;
    }

    num_workgroups = std::max(n_elems / wg_size[0], (size_t)16);
    globalsize[0] = num_workgroups * localsize[0];
    globalsize[1] = num_workgroups * localsize[1];
    num_elements = globalsize[0] * globalsize[1];

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float((float)(-100000.f * M_PI), (float)(100000.f * M_PI) ,d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
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

    err = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, globalsize, localsize, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_float)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_broadcast_2D(input_ptr[0], output_ptr, globalsize[0], globalsize[1], localsize[0], localsize[1]))
    {
        log_error("work_group_broadcast_2D test failed\n");
        return -1;
    }
    log_info("work_group_broadcast_2D test passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_broadcast_3D(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_float     *input_ptr[1], *p;
    cl_float     *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       globalsize[3];
    size_t       localsize[3];
    size_t       wg_size[1];
    size_t       num_workgroups;
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &wg_broadcast_3D_kernel_code,
                                      "test_wg_broadcast_3D");
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), wg_size, NULL);
    if (err)
        return -1;

    if (wg_size[0] >=512)
    {
        localsize[0] = localsize[1] = localsize[2] = 8;
    }
    else if (wg_size[0] >= 64)
    {
        localsize[0] = localsize[1] = localsize[2] = 4;
    }
    else if (wg_size[0] >= 8)
    {
        localsize[0] = localsize[1] = localsize[2] = 2;
    }
    else
    {
        localsize[0] = localsize[1] = localsize[2] = 1;
    }

    num_workgroups = std::max(n_elems / wg_size[0], (size_t)8);
    globalsize[0] = num_workgroups * localsize[0];
    globalsize[1] = num_workgroups * localsize[1];
    globalsize[2] = num_workgroups * localsize[2];
    num_elements = globalsize[0] * globalsize[1] * globalsize[2];

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float((float)(-100000.f * M_PI), (float)(100000.f * M_PI) ,d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
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

    err = clEnqueueNDRangeKernel( queue, kernel, 3, NULL, globalsize, localsize, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_float)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_broadcast_3D(input_ptr[0], output_ptr, globalsize[0], globalsize[1], globalsize[2], localsize[0], localsize[1], localsize[2]))
    {
        log_error("work_group_broadcast_3D test failed\n");
        return -1;
    }
    log_info("work_group_broadcast_3D test passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


int
test_work_group_broadcast(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int err;

    err = test_work_group_broadcast_1D(device, context, queue, n_elems);
    if (err) return err;
    err = test_work_group_broadcast_2D(device, context, queue, n_elems);
    if (err) return err;
    return err;
}



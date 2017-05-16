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
#ifndef TEST_CONFORMANCE_CLCPP_WG_TEST_WG_BROADCAST_HPP
#define TEST_CONFORMANCE_CLCPP_WG_TEST_WG_BROADCAST_HPP

#include <vector>
#include <limits>
#include <algorithm>

// Common for all OpenCL C++ tests
#include "../common.hpp"
// Common for tests of work-group functions
#include "common.hpp"

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_wg_broadcast_1D_kernel_code()
{
    return
        "__kernel void test_wg_broadcast(global uint *input, global uint *output)\n"
        "{\n"
        "    ulong tid = get_global_id(0);\n"
        "    uint result = work_group_broadcast(input[tid], get_group_id(0) % get_local_size(0));\n"
        "    output[tid] = result;\n"
        "}\n";
}
std::string generate_wg_broadcast_2D_kernel_code()
{
    return
        "__kernel void test_wg_broadcast(global uint *input, global uint *output)\n"
        "{\n"
        "    ulong tid_x = get_global_id(0);\n"
        "    ulong tid_y = get_global_id(1);\n"
        "    size_t x = get_group_id(0) % get_local_size(0);\n"
        "    size_t y = get_group_id(1) % get_local_size(1);\n"
        "    size_t idx = (tid_y * get_global_size(0)) + tid_x;\n"
        "    uint result = work_group_broadcast(input[idx], x, y);\n"
        "    output[idx] = result;\n"
        "}\n";
}
std::string generate_wg_broadcast_3D_kernel_code()
{
    return
        "__kernel void test_wg_broadcast(global uint *input, global uint *output)\n"
        "{\n"
        "    ulong tid_x = get_global_id(0);\n"
        "    ulong tid_y = get_global_id(1);\n"
        "    ulong tid_z = get_global_id(2);\n"
        "    size_t x = get_group_id(0) % get_local_size(0);\n"
        "    size_t y = get_group_id(1) % get_local_size(1);\n"
        "    size_t z = get_group_id(2) % get_local_size(2);\n"
        "    ulong idx = (tid_z * get_global_size(1) * get_global_size(0)) + (tid_y * get_global_size(0)) + tid_x;\n"
        "    uint result = work_group_broadcast(input[idx], x, y, z);\n"
        "    output[idx] = result;\n"
        "}\n";
}
#else
std::string generate_wg_broadcast_1D_kernel_code()
{
    return "#include <opencl_memory>\n"
           "#include <opencl_work_item>\n"
           "#include <opencl_work_group>\n"
           "using namespace cl;\n"
           "__kernel void test_wg_broadcast(global_ptr<uint[]> input, global_ptr<uint[]> output)\n"
           "{\n"
           "    ulong tid = get_global_id(0);\n"
           "    uint result = work_group_broadcast(input[tid], get_group_id(0) % get_local_size(0));\n"
           "    output[tid] = result;\n"
           "}\n";
}
std::string generate_wg_broadcast_2D_kernel_code()
{
    return "#include <opencl_memory>\n"
           "#include <opencl_work_item>\n"
           "#include <opencl_work_group>\n"
           "using namespace cl;\n"
           "__kernel void test_wg_broadcast(global_ptr<uint[]> input, global_ptr<uint[]> output)\n"
           "{\n"
           "    ulong tid_x = get_global_id(0);\n"
           "    ulong tid_y = get_global_id(1);\n"
           "    size_t x = get_group_id(0) % get_local_size(0);\n"
           "    size_t y = get_group_id(1) % get_local_size(1);\n"
           "    size_t idx = (tid_y * get_global_size(0)) + tid_x;\n"
           "    uint result = work_group_broadcast(input[idx], x, y);\n"
           "    output[idx] = result;\n"
           "}\n";
}
std::string generate_wg_broadcast_3D_kernel_code()
{
    return "#include <opencl_memory>\n"
           "#include <opencl_work_item>\n"
           "#include <opencl_work_group>\n"
           "using namespace cl;\n"
           "__kernel void test_wg_broadcast(global_ptr<uint[]> input, global_ptr<uint[]> output)\n"
           "{\n"
           "    ulong tid_x = get_global_id(0);\n"
           "    ulong tid_y = get_global_id(1);\n"
           "    ulong tid_z = get_global_id(2);\n"
           "    size_t x = get_group_id(0) % get_local_size(0);\n"
           "    size_t y = get_group_id(1) % get_local_size(1);\n"
           "    size_t z = get_group_id(2) % get_local_size(2);\n"
           "    ulong idx = (tid_z * get_global_size(1) * get_global_size(0)) + (tid_y * get_global_size(0)) + tid_x;\n"
           "    uint result = work_group_broadcast(input[idx], x, y, z);\n"
           "    output[idx] = result;\n"
           "}\n";
}
#endif

int
verify_wg_broadcast_1D(const std::vector<cl_uint> &in, const std::vector<cl_uint> &out, size_t n, size_t wg_size)
{
    size_t i, j;
    size_t group_id;

    for (i=0,group_id=0; i<n; i+=wg_size,group_id++)
    {
        int local_size = (n-i) > wg_size ? wg_size : (n-i);
        cl_uint broadcast_result = in[i + (group_id % local_size)];
        for (j=0; j<local_size; j++)
        {
            if ( broadcast_result != out[i+j] )
            {
                log_info("work_group_broadcast: Error at %lu: expected = %u, got = %u\n", i+j, broadcast_result, out[i+j]);
                return -1;
            }
        }
    }

    return CL_SUCCESS;
}

int
verify_wg_broadcast_2D(const std::vector<cl_uint> &in, const std::vector<cl_uint> &out,
                       size_t nx, size_t ny,
                       size_t wg_size_x, size_t wg_size_y)
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
                cl_uint broadcast_result = in[(i + y) * nx + (j + x)];
                for (_j=0; _j < local_size_x; _j++)
                {
                    size_t indx = (i + _i) * nx + (j + _j);
                    if ( broadcast_result != out[indx] )
                    {
                        log_info("%lu\n", indx);
                        log_info("%lu\n", ((i + y) * nx + (j + x)));
                         log_info("%lu\n", out.size());
                        log_info("work_group_broadcast: Error at (%lu, %lu): expected = %u, got = %u\n", j+_j, i+_i, broadcast_result, out[indx]);
                        return -1;
                    }
                }
            }
        }
    }

    return CL_SUCCESS;
}

int
verify_wg_broadcast_3D(const std::vector<cl_uint> &in, const std::vector<cl_uint> &out,
                       size_t nx, size_t ny, size_t nz,
                       size_t wg_size_x, size_t wg_size_y, size_t wg_size_z)
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
                        cl_uint broadcast_result = in[(i + z) * ny * nz + (j + y) * nx + (k + x)];
                        for (_k=0; _k < local_size_x; _k++)
                        {
                            size_t indx = (i + _i) * ny * nx + (j + _j) * nx + (k + _k);
                            if ( broadcast_result != out[indx] )
                            {
                                log_info(
                                    "work_group_broadcast: Error at (%lu, %lu, %lu): expected = %u, got = %u\n",
                                    k+_k, j+_j, i+_i,
                                    broadcast_result, out[indx]);
                                return -1;
                            }
                        }
                    }
                }
            }
        }
    }
    return CL_SUCCESS;
}

std::vector<cl_uint> generate_input_wg_broadcast(size_t count, size_t wg_size)
{
    std::vector<cl_uint> input(count, cl_uint(0));
    size_t j = wg_size;
    for(size_t i = 0; i < count; i++)
    {
        input[i] = static_cast<cl_uint>(j);
        j--;
        if(j == 0)
        {
            j = wg_size;
        }
    }
    return input;
}

std::vector<cl_uint> generate_output_wg_broadcast(size_t count, size_t wg_size)
{
    (void) wg_size;
    return std::vector<cl_uint>(count, cl_uint(1));
}

int work_group_broadcast(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, size_t dim)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t flat_wg_size;
    size_t wg_size[] = { 1, 1, 1};
    size_t work_size[] = { 1, 1, 1};
    int err;

    // Get kernel source code
    std::string code_str;
    if(dim > 2) code_str = generate_wg_broadcast_3D_kernel_code();
    else if(dim > 1) code_str = generate_wg_broadcast_2D_kernel_code();
    else code_str = generate_wg_broadcast_1D_kernel_code();

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_wg_broadcast");
    RETURN_ON_ERROR(err)
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_wg_broadcast", "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)
#else
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_wg_broadcast");
    RETURN_ON_ERROR(err)
#endif

    // Get max flat workgroup size
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &flat_wg_size, NULL);
    RETURN_ON_CL_ERROR(err, "clGetKernelWorkGroupInfo")

    // Set local work size
    wg_size[0] = flat_wg_size;
    if(dim > 2)
    {
        if (flat_wg_size >=512)
        {
            wg_size[0] = wg_size[1] = wg_size[2] = 8;
        }
        else if (flat_wg_size >= 64)
        {
            wg_size[0] = wg_size[1] = wg_size[2] = 4;
        }
        else if (flat_wg_size >= 8)
        {
            wg_size[0] = wg_size[1] = wg_size[2] = 2;
        }
        else
        {
            wg_size[0] = wg_size[1] = wg_size[2] = 1;
        }
    }
    else if(dim > 1)
    {
        if (flat_wg_size >= 256)
        {
            wg_size[0] = wg_size[1] = 16;
        }
        else if (flat_wg_size >=64)
        {
            wg_size[0] = wg_size[1] = 8;
        }
        else if (flat_wg_size >= 16)
        {
            wg_size[0] = wg_size[1] = 4;
        }
        else
        {
            wg_size[0] = wg_size[1] = 1;
        }
    }

    // Calculate flat local work size
    flat_wg_size = wg_size[0];
    if(dim > 1) flat_wg_size *= wg_size[1];
    if(dim > 2) flat_wg_size *= wg_size[2];

    // Calculate global work size
    size_t flat_work_size = count;
    // 3D
    if(dim > 2)
    {
        size_t wg_number = static_cast<size_t>(
            std::ceil(static_cast<double>(count / 3) / (wg_size[0] * wg_size[1] * wg_size[2]))
        );
        work_size[0] = wg_number * wg_size[0];
        work_size[1] = wg_number * wg_size[1];
        work_size[2] = wg_number * wg_size[2];
        flat_work_size = work_size[0] * work_size[1] * work_size[2];
    }
    // 2D
    else if(dim > 1)
    {
        size_t wg_number = static_cast<size_t>(
            std::ceil(static_cast<double>(count / 2) / (wg_size[0] * wg_size[1]))
        );
        work_size[0] = wg_number * wg_size[0];
        work_size[1] = wg_number * wg_size[1];
        flat_work_size = work_size[0] * work_size[1];
    }
    // 1D
    else
    {
        size_t wg_number = static_cast<size_t>(
            std::ceil(static_cast<double>(count) / wg_size[0])
        );
        flat_work_size = wg_number * wg_size[0];
        work_size[0] = flat_work_size;
    }

    std::vector<cl_uint> input = generate_input_wg_broadcast(flat_work_size, flat_wg_size);
    std::vector<cl_uint> output = generate_output_wg_broadcast(flat_work_size, flat_wg_size);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * input.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(cl_uint) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, wg_size, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    int result = CL_SUCCESS;
    // 3D
    if(dim > 2)
    {
        result = verify_wg_broadcast_3D(
            input, output,
            work_size[0], work_size[1], work_size[2],
            wg_size[0], wg_size[1], wg_size[2]
        );
    }
    // 2D
    else if(dim > 1)
    {
        result = verify_wg_broadcast_2D(
            input, output,
            work_size[0], work_size[1],
            wg_size[0], wg_size[1]
        );
    }
    // 1D
    else
    {
        result = verify_wg_broadcast_1D(
            input, output,
            work_size[0],
            wg_size[0]
        );
    }

    RETURN_ON_ERROR_MSG(result, "work_group_broadcast_%luD failed", dim);
    log_info("work_group_broadcast_%luD passed\n", dim);

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

AUTO_TEST_CASE(test_work_group_broadcast)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int local_error = CL_SUCCESS;

    local_error = work_group_broadcast(device, context, queue, n_elems, 1);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = work_group_broadcast(device, context, queue, n_elems, 2);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = work_group_broadcast(device, context, queue, n_elems, 3);
    CHECK_ERROR(local_error)
    error |= local_error;

    if(error != CL_SUCCESS)
        return -1;
    return CL_SUCCESS;
}

#endif // TEST_CONFORMANCE_CLCPP_WG_TEST_WG_BROADCAST_HPP

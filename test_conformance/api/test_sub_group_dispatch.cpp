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

const char *subgroup_dispatch_kernel[] = {
"#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
"__kernel void subgroup_dispatch_kernel(__global int *output)\n"
"{\n"
"    size_t size = get_num_sub_groups ();\n"
"\n"
"    output[0] = size;\n"
"\n"
"}\n" };

size_t flatten_ndrange(size_t* ndrange, size_t dim)
{
    switch(dim)
    {
    case 1:
        return *ndrange;
    case 2:
        return ndrange[0] * ndrange[1];
    case 3:
        return ndrange[0] * ndrange[1] * ndrange[2];
    default:
        log_error("ERROR: bad ndrange value");
        return 0;
    }
}

cl_int get_sub_group_num(cl_command_queue queue, cl_kernel kernel, clMemWrapper& out, size_t& size, size_t local_size, size_t dim)
{
    size_t ndrange[3] = {local_size, 1, 1};
    cl_int error = CL_SUCCESS;
    size = 0;
    error = clSetKernelArg(kernel, 0, sizeof(out), &out);
    error += clEnqueueNDRangeKernel(queue, kernel, dim, NULL, ndrange, ndrange, 0, NULL, NULL);
    error += clEnqueueReadBuffer(queue, out, CL_TRUE, 0, 4, &size, 0, NULL, NULL);
    return error;
}

int test_sub_group_dispatch(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t realSize;
    size_t kernel_max_subgroup_size, kernel_subgroup_count;
    size_t max_local;

    cl_platform_id platform;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper out;

    size_t ret_ndrange1d;
    size_t ret_ndrange2d[2];
    size_t ret_ndrange3d[3];

    size_t ret_ndrange2d_flattened;
    size_t ret_ndrange3d_flattened;

    if (get_device_cl_version(deviceID) >= Version(3, 0))
    {
        int error;
        cl_uint max_num_sub_groups;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                sizeof(max_num_sub_groups), &max_num_sub_groups,
                                NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get max num subgroups");
            return error;
        }

        if (max_num_sub_groups == 0)
        {
            return TEST_SKIPPED_ITSELF;
        }
    }

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        subgroup_dispatch_kernel,
                                        "subgroup_dispatch_kernel");
    if (error != 0)
        return error;

    out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(size_t), NULL, &error);
    test_error(error, "clCreateBuffer failed");

    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_local, NULL);
    test_error(error, "clGetDeviceInfo failed");


    error = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform), (void *)&platform, NULL);
    test_error(error, "clDeviceInfo failed for CL_DEVICE_PLATFORM");

    // Get the max subgroup size
    error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
            sizeof(max_local), &max_local, sizeof(kernel_max_subgroup_size), (void *)&kernel_max_subgroup_size, &realSize);
    test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE");
    log_info("The CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE for the kernel is %d.\n", (int)kernel_max_subgroup_size);

    if (realSize != sizeof(kernel_max_subgroup_size)) {
        log_error( "ERROR: Returned size of max sub group size not valid! (Expected %d, got %d)\n", (int)sizeof(kernel_max_subgroup_size), (int)realSize );
        return -1;
    }

    // Get the number of subgroup for max local size
    error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
            sizeof(max_local), &max_local, sizeof(kernel_subgroup_count), (void *)&kernel_subgroup_count, &realSize);
    test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE");
    log_info("The CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE for the kernel is %d.\n", (int)kernel_subgroup_count);

    if (realSize != sizeof(kernel_subgroup_count)) {
        log_error( "ERROR: Returned size of sub group count not valid! (Expected %d, got %d)\n", (int)sizeof(kernel_subgroup_count), (int)realSize );
        return -1;
    }

    // test CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT
    for (size_t i = kernel_subgroup_count; i > 0; --i)
    {
        // test all 3 different dimention of requested local size
        size_t expect_size = kernel_max_subgroup_size * i;
        size_t kernel_ret_size = 0;
        error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(i), &i, sizeof(ret_ndrange1d), &ret_ndrange1d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
        if (realSize != sizeof(ret_ndrange1d)) {
            log_error( "ERROR: Returned size of sub group count not valid! (Expected %d, got %d)\n", (int)sizeof(kernel_subgroup_count), (int)realSize );
            return -1;
        }

        if (ret_ndrange1d != expect_size)
        {
            log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT! (Expected %d, got %d)\n", (int)expect_size, (int)ret_ndrange1d );
            return -1;
        }

        error = get_sub_group_num(queue, kernel, out, kernel_ret_size, ret_ndrange1d, 1);
        test_error(error, "Failed to query number of subgroups from kernel");
        if (i != kernel_ret_size)
        {
            log_error( "ERROR: Mismatch between requested number of subgroups and what get_num_sub_groups() in kernel returned! (Expected %d, got %d)\n", (int)i, (int)kernel_ret_size );
            return -1;
        }

        error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(i), &i, sizeof(ret_ndrange2d), ret_ndrange2d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
        if (realSize != sizeof(ret_ndrange2d)) {
            log_error( "ERROR: Returned size of sub group count not valid! (Expected %d, got %d)\n", (int)sizeof(kernel_subgroup_count), (int)realSize );
            return -1;
        }

        ret_ndrange2d_flattened = flatten_ndrange(ret_ndrange2d, 2);
        if (ret_ndrange2d_flattened != expect_size ||
            ret_ndrange2d[1] != 1)
        {
            log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT! (Expected %d, got %d)\n", (int)expect_size, (int)ret_ndrange2d_flattened );
            return -1;
        }

        error = get_sub_group_num(queue, kernel, out, kernel_ret_size, ret_ndrange2d_flattened, 2);
        test_error(error, "Failed to query number of subgroups from kernel");
        if (i != kernel_ret_size)
        {
            log_error( "ERROR: Mismatch between requested number of subgroups and what get_num_sub_groups() in kernel returned! (Expected %d, got %d)\n", (int)i, (int)kernel_ret_size );
            return -1;
        }

        error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(i), &i, sizeof(ret_ndrange3d), ret_ndrange3d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
        if (realSize != sizeof(ret_ndrange3d)) {
            log_error( "ERROR: Returned size of sub group count not valid! (Expected %d, got %d)\n", (int)sizeof(kernel_subgroup_count), (int)realSize );
            return -1;
        }

        ret_ndrange3d_flattened = flatten_ndrange(ret_ndrange3d, 3);
        if (ret_ndrange3d_flattened != expect_size ||
            ret_ndrange3d[1] != 1 ||
            ret_ndrange3d[2] != 1)
        {
            log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT! (Expected %d, got %d)\n", (int)expect_size, (int)ret_ndrange3d_flattened );
            return -1;
        }

        error = get_sub_group_num(queue, kernel, out, kernel_ret_size, ret_ndrange3d_flattened, 3);
        test_error(error, "Failed to query number of subgroups from kernel");
        if (i != kernel_ret_size)
        {
            log_error( "ERROR: Mismatch between requested number of subgroups and what get_num_sub_groups() in kernel returned! (Expected %d, got %d)\n", (int)i, (int)kernel_ret_size );
            return -1;
        }
    }

    // test when input subgroup count exceeds max wg size
    size_t large_sg_size = kernel_subgroup_count + 1;
    error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(size_t), &large_sg_size, sizeof(ret_ndrange1d), &ret_ndrange1d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
    if (ret_ndrange1d != 0)
    {
        log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT! (Expected %d, got %d)\n", 0, (int)ret_ndrange1d );
            return -1;
    }

    error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(size_t), &large_sg_size, sizeof(ret_ndrange2d), ret_ndrange2d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
    if (ret_ndrange2d[0] != 0 ||
        ret_ndrange2d[1] != 0)
    {
        log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT!" );
            return -1;
    }

    error = clGetKernelSubGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, sizeof(size_t), &large_sg_size, sizeof(ret_ndrange3d), ret_ndrange3d, &realSize);
        test_error(error, "clGetKernelSubGroupInfo failed for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT");
    if (ret_ndrange3d[0] != 0 ||
        ret_ndrange3d[1] != 0 ||
        ret_ndrange3d[2] != 0)
    {
        log_error( "ERROR: Incorrect value returned for CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT!" );
            return -1;
    }

    return 0;
}

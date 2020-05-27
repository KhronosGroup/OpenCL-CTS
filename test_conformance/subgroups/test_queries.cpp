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
#include "procs.h"
#include "subhelpers.h"

typedef struct
{
    cl_uint maxSubGroupSize;
    cl_uint numSubGroups;
} result_data;


int test_sub_group_info(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements,
                        bool useCoreSubgroups)
{
    static const size_t gsize0 = 80;
    int i, error;
    size_t realSize;
    size_t kernel_max_subgroup_size, kernel_subgroup_count;
    size_t global[] = { gsize0, 14, 10 };
    size_t local[] = { 0, 0, 0 };
    result_data result[gsize0];

    cl_uint max_dimensions;

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(max_dimensions), &max_dimensions, NULL);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");

    cl_platform_id platform;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper out;
    std::stringstream kernel_sstr;
    if (useCoreSubgroups)
    {
        kernel_sstr << "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
    }
    kernel_sstr
        << "\n"
           "typedef struct {\n"
           "    uint maxSubGroupSize;\n"
           "    uint numSubGroups;\n"
           "} result_data;\n"
           "\n"
           "__kernel void query_kernel( __global result_data *outData )\n"
           "{\n"
           "    int gid = get_global_id( 0 );\n"
           "    outData[gid].maxSubGroupSize = get_max_sub_group_size();\n"
           "    outData[gid].numSubGroups = get_num_sub_groups();\n"
           "}";

    const std::string &kernel_str = kernel_sstr.str();
    const char *kernel_src = kernel_str.c_str();
    error = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &kernel_src, "query_kernel",
        "-cl-std=CL2.0");
    if (error != 0) return error;

    // Determine some local dimensions to use for the test.
    if (max_dimensions == 1)
    {
        error = get_max_common_work_group_size(context, kernel, global[0],
                                               &local[0]);
        test_error(error, "get_max_common_work_group_size failed");
    }
    else if (max_dimensions == 2)
    {
        error =
            get_max_common_2D_work_group_size(context, kernel, global, local);
        test_error(error, "get_max_common_2D_work_group_size failed");
    }
    else
    {
        error =
            get_max_common_3D_work_group_size(context, kernel, global, local);
        test_error(error, "get_max_common_3D_work_group_size failed");
    }

    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                            (void *)&platform, NULL);
    test_error(error, "clDeviceInfo failed for CL_DEVICE_PLATFORM");

    subgroupsAPI subgroupsApiSet(platform, useCoreSubgroups);
    clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfo_ptr =
        subgroupsApiSet.clGetKernelSubGroupInfo_ptr();
    if (clGetKernelSubGroupInfo_ptr == NULL)
    {
        log_error("ERROR: %s function not available",
                  subgroupsApiSet.clGetKernelSubGroupInfo_name);
        return TEST_FAIL;
    }

    error = clGetKernelSubGroupInfo_ptr(
        kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof(local),
        (void *)&local, sizeof(kernel_max_subgroup_size),
        (void *)&kernel_max_subgroup_size, &realSize);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: %s function error for "
                  "CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE",
                  subgroupsApiSet.clGetKernelSubGroupInfo_name);
        return TEST_FAIL;
    }
    log_info(
        "The CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE for the kernel is %d.\n",
        (int)kernel_max_subgroup_size);
    if (realSize != sizeof(kernel_max_subgroup_size))
    {
        log_error("ERROR: Returned size of max sub group size not valid! "
                  "(Expected %d, got %d)\n",
                  (int)sizeof(kernel_max_subgroup_size), (int)realSize);
        return TEST_FAIL;
    }
    error = clGetKernelSubGroupInfo_ptr(
        kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, sizeof(local),
        (void *)&local, sizeof(kernel_subgroup_count),
        (void *)&kernel_subgroup_count, &realSize);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: %s function error "
                  "for CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE",
                  subgroupsApiSet.clGetKernelSubGroupInfo_name);
        return TEST_FAIL;
    }
    log_info(
        "The CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE for the kernel is %d.\n",
        (int)kernel_subgroup_count);

    if (realSize != sizeof(kernel_subgroup_count))
    {
        log_error("ERROR: Returned size of sub group count not valid! "
                  "(Expected %d, got %d)\n",
                  (int)sizeof(kernel_subgroup_count), (int)realSize);
        return TEST_FAIL;
    }

    // Verify that the kernel gets the same max_subgroup_size and subgroup_count
    out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(result), NULL,
                         &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(out), &out);
    test_error(error, "clSetKernelArg failed");

    error = clEnqueueNDRangeKernel(queue, kernel, max_dimensions, NULL, global,
                                   local, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, out, CL_FALSE, 0, sizeof(result),
                                &result, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    for (i = 0; i < (int)gsize0; ++i)
    {
        if (result[i].maxSubGroupSize != (cl_uint)kernel_max_subgroup_size)
        {
            log_error("ERROR: get_max_subgroup_size() doesn't match result "
                      "from clGetKernelSubGroupInfoKHR, %u vs %u\n",
                      result[i].maxSubGroupSize,
                      (cl_uint)kernel_max_subgroup_size);
            return -1;
        }
        if (result[i].numSubGroups != (cl_uint)kernel_subgroup_count)
        {
            log_error("ERROR: get_num_sub_groups() doesn't match result from "
                      "clGetKernelSubGroupInfoKHR, %u vs %u\n",
                      result[i].numSubGroups, (cl_uint)kernel_subgroup_count);
            return -1;
        }
    }

    return 0;
}

int test_sub_group_info_core(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    return test_sub_group_info(device, context, queue, num_elements, true);
}

int test_sub_group_info_ext(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_sub_group_info(device, context, queue, num_elements, false);
}
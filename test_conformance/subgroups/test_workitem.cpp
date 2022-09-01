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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include <CL/cl.h>

struct get_test_data
{
    cl_uint subGroupSize;
    cl_uint maxSubGroupSize;
    cl_uint numSubGroups;
    cl_uint enqNumSubGroups;
    cl_uint subGroupId;
    cl_uint subGroupLocalId;
    bool operator==(get_test_data x)
    {
        return subGroupSize == x.subGroupSize
            && maxSubGroupSize == x.maxSubGroupSize
            && numSubGroups == x.numSubGroups && subGroupId == x.subGroupId
            && subGroupLocalId == x.subGroupLocalId;
    }
};

static int check_group(const get_test_data *result, int nw, cl_uint ensg,
                       int maxwgs)
{
    int first = -1;
    int last = -1;
    int i, j;
    cl_uint hit[32];

    for (i = 0; i < nw; ++i)
    {
        if (result[i].subGroupId == 0 && result[i].subGroupLocalId == 0)
            first = i;
        if (result[i].subGroupId == result[0].numSubGroups - 1
            && result[i].subGroupLocalId == 0)
            last = i;
        if (first != -1 && last != -1) break;
    }

    if (first == -1 || last == -1)
    {
        log_error("ERROR: expected sub group id's are missing\n");
        return -1;
    }

    // Check them
    if (result[first].subGroupSize == 0)
    {
        log_error("ERROR: get_sub_group_size() returned 0\n");
        return -1;
    }
    if (result[first].maxSubGroupSize == 0
        || result[first].maxSubGroupSize > maxwgs)
    {
        log_error(
            "ERROR: get_max_subgroup_size() returned incorrect result: %u\n",
            result[first].maxSubGroupSize);
        return -1;
    }
    if (result[first].subGroupSize > result[first].maxSubGroupSize)
    {
        log_error("ERROR: get_sub_group_size() > get_max_sub_group_size()\n");
        return -1;
    }
    if (result[last].subGroupSize > result[first].subGroupSize)
    {
        log_error("ERROR: last sub group larger than first sub group\n");
        return -1;
    }
    if (result[first].numSubGroups == 0 || result[first].numSubGroups > ensg)
    {
        log_error(
            "ERROR: get_num_sub_groups() returned incorrect result:  %u \n",
            result[first].numSubGroups);
        return -1;
    }

    memset(hit, 0, sizeof(hit));
    for (i = 0; i < nw; ++i)
    {
        if (result[i].maxSubGroupSize != result[first].maxSubGroupSize
            || result[i].numSubGroups != result[first].numSubGroups)
        {
            log_error("ERROR: unexpected variation in get_*_sub_group_*()\n");
            return -1;
        }
        if (result[i].subGroupId >= result[first].numSubGroups)
        {
            log_error(
                "ERROR: get_sub_group_id() returned out of range value: %u\n",
                result[i].subGroupId);
            return -1;
        }
        if (result[i].enqNumSubGroups != ensg)
        {
            log_error("ERROR: get_enqueued_num_sub_groups() returned incorrect "
                      "value: %u\n",
                      result[i].enqNumSubGroups);
            return -1;
        }
        if (result[first].numSubGroups > 1)
        {
            if (result[i].subGroupId < result[first].numSubGroups - 1)
            {
                if (result[i].subGroupSize != result[first].subGroupSize)
                {
                    log_error(
                        "ERROR: unexpected variation in get_*_sub_group_*()\n");
                    return -1;
                }
                if (result[i].subGroupLocalId >= result[first].subGroupSize)
                {
                    log_error("ERROR: get_sub_group_local_id() returned out of "
                              "bounds value: %u \n",
                              result[i].subGroupLocalId);
                    return -1;
                }
            }
            else
            {
                if (result[i].subGroupSize != result[last].subGroupSize)
                {
                    log_error(
                        "ERROR: unexpected variation in get_*_sub_group_*()\n");
                    return -1;
                }
                if (result[i].subGroupLocalId >= result[last].subGroupSize)
                {
                    log_error("ERROR: get_sub_group_local_id() returned out of "
                              "bounds value: %u \n",
                              result[i].subGroupLocalId);
                    return -1;
                }
            }
        }
        else
        {
            if (result[i].subGroupSize != result[first].subGroupSize)
            {
                log_error(
                    "ERROR: unexpected variation in get_*_sub_group_*()\n");
                return -1;
            }
            if (result[i].subGroupLocalId >= result[first].subGroupSize)
            {
                log_error("ERROR: get_sub_group_local_id() returned out of "
                          "bounds value: %u \n",
                          result[i].subGroupLocalId);
                return -1;
            }
        }

        j = (result[first].subGroupSize + 31) / 32 * result[i].subGroupId
            + (result[i].subGroupLocalId >> 5);
        if (j < sizeof(hit) / 4)
        {
            cl_uint b = 1U << (result[i].subGroupLocalId & 0x1fU);
            if ((hit[j] & b) != 0)
            {
                log_error("ERROR: get_sub_group_local_id() repeated a result "
                          "in the same sub group\n");
                return -1;
            }
            hit[j] |= b;
        }
    }

    return 0;
}

int test_work_item_functions(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements,
                             bool useCoreSubgroups)
{
    static const size_t lsize = 200;
    int error;
    int i, j, k, q, r, nw;
    int maxwgs;
    cl_uint ensg;
    size_t global;
    size_t local;
    get_test_data result[lsize * 6];
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
           "\n"
           "typedef struct {\n"
           "    uint subGroupSize;\n"
           "    uint maxSubGroupSize;\n"
           "    uint numSubGroups;\n"
           "    uint enqNumSubGroups;\n"
           "    uint subGroupId;\n"
           "    uint subGroupLocalId;\n"
           "} get_test_data;\n"
           "\n"
           "__kernel void get_test( __global get_test_data *outData )\n"
           "{\n"
           "    int gid = get_global_id( 0 );\n"
           "    outData[gid].subGroupSize = get_sub_group_size();\n"
           "    outData[gid].maxSubGroupSize = get_max_sub_group_size();\n"
           "    outData[gid].numSubGroups = get_num_sub_groups();\n"
           "    outData[gid].enqNumSubGroups = get_enqueued_num_sub_groups();\n"
           "    outData[gid].subGroupId = get_sub_group_id();\n"
           "    outData[gid].subGroupLocalId = get_sub_group_local_id();\n"
           "}";
    const std::string &kernel_str = kernel_sstr.str();
    const char *kernel_src = kernel_str.c_str();
    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_src, "get_test");
    if (error != 0) return error;

    error = get_max_allowed_work_group_size(context, kernel, &local, NULL);
    if (error != 0) return error;

    maxwgs = (int)local;

    // Limit it a bit so we have muliple work groups
    // Ideally this will still be large enough to give us multiple subgroups
    if (local > lsize) local = lsize;

    // Create our buffer
    out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(result), NULL,
                         &error);
    test_error(error, "clCreateBuffer failed");

    // Set argument
    error = clSetKernelArg(kernel, 0, sizeof(out), &out);
    test_error(error, "clSetKernelArg failed");

    global = local * 5;

    // Non-uniform work-groups are an optional feature from 3.0 onward.
    cl_bool device_supports_non_uniform_wg = CL_TRUE;
    if (get_device_cl_version(device) >= Version(3, 0))
    {
        error = clGetDeviceInfo(
            device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, sizeof(cl_bool),
            &device_supports_non_uniform_wg, nullptr);
        test_error(error, "clGetDeviceInfo failed");
    }

    if (device_supports_non_uniform_wg)
    {
        // Make sure we have a flexible range
        global += 3 * local / 4;
    }

    // Collect the data
    memset((void *)&result, 0xf0, sizeof(result));

    error = clEnqueueWriteBuffer(queue, out, CL_FALSE, 0, sizeof(result),
                                 (void *)&result, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                   NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, out, CL_FALSE, 0, sizeof(result),
                                (void *)&result, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    nw = (int)local;
    ensg = result[0].enqNumSubGroups;

    // Check the first group
    error = check_group(result, nw, ensg, maxwgs);
    if (error) return error;

    q = (int)global / nw;
    r = (int)global % nw;

    // Check the remaining work groups including the last if it is the same size
    for (k = 1; k < q; ++k)
    {
        for (j = 0; j < nw; ++j)
        {
            i = k * nw + j;
            if (!(result[i] == result[i - nw]))
            {
                log_error("ERROR: sub group mapping is not identical for all "
                          "work groups\n");
                return -1;
            }
        }
    }

    // Check the last group if it wasn't the same size
    if (r != 0)
    {
        error = check_group(result + q * nw, r, ensg, maxwgs);
        if (error) return error;
    }

    return 0;
}

int test_work_item_functions_core(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    return test_work_item_functions(device, context, queue, num_elements, true);
}

int test_work_item_functions_ext(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_work_item_functions(device, context, queue, num_elements,
                                    false);
}

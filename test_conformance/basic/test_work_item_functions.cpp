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

#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

namespace {

struct work_item_data
{
    cl_uint workDim;
    cl_uint globalSize[ 3 ];
    cl_uint globalID[ 3 ];
    cl_uint localSize[ 3 ];
    cl_uint localID[ 3 ];
    cl_uint numGroups[ 3 ];
    cl_uint groupID[ 3 ];
    cl_uint globalOffset[3];
    cl_uint enqueuedLocalSize[3];
};

const char *workItemKernelCode =
    R"(typedef struct {
    uint workDim;
    uint globalSize[ 3 ];
    uint globalID[ 3 ];
    uint localSize[ 3 ];
    uint localID[ 3 ];
    uint numGroups[ 3 ];
    uint groupID[ 3 ];
    uint globalOffset[ 3 ];
    uint enqueuedLocalSize[ 3 ];
 } work_item_data;

__kernel void sample_kernel( __global work_item_data *outData )
{
    int id = get_global_id(0);
   outData[ id ].workDim = (uint)get_work_dim();
    for( uint i = 0; i < get_work_dim(); i++ )
   {
       outData[ id ].globalSize[ i ] = (uint)get_global_size( i );
       outData[ id ].globalID[ i ] = (uint)get_global_id( i );
       outData[ id ].localSize[ i ] = (uint)get_local_size( i );
       outData[ id ].localID[ i ] = (uint)get_local_id( i );
       outData[ id ].numGroups[ i ] = (uint)get_num_groups( i );
       outData[ id ].groupID[ i ] = (uint)get_group_id( i );
   }
})";

struct work_item_data_out_of_range
{
    cl_uint workDim;
    cl_uint globalSize;
    cl_uint globalID;
    cl_uint localSize;
    cl_uint localID;
    cl_uint numGroups;
    cl_uint groupID;
    cl_uint globalOffset;
    cl_uint enqueuedLocalSize;
};

const char *outOfRangeWorkItemKernelCode =
    R"(typedef struct {
    uint workDim;
    uint globalSize;
    uint globalID;
    uint localSize;
    uint localID;
    uint numGroups;
    uint groupID;
    uint globalOffset;
    uint enqueuedLocalSize;
 } work_item_data;

__kernel void sample_kernel( __global work_item_data *outData, int dim_param )
{
    int ind_mul=1;
    int ind=0;
    for( uint i = 0; i < get_work_dim(); i++ )
    {
        ind += (uint)get_global_id(i) * ind_mul;
        ind_mul *= get_global_size(i);
    }
    outData[ind].workDim = (uint)get_work_dim();

    uint dimindx=dim_param;
    outData[ind].globalSize = (uint)get_global_size(dimindx);
    outData[ind].globalID = (uint)get_global_id(dimindx);
    outData[ind].localSize = (uint)get_local_size(dimindx);
    outData[ind].localID = (uint)get_local_id(dimindx);
    outData[ind].numGroups = (uint)get_num_groups(dimindx);
    outData[ind].groupID = (uint)get_group_id(dimindx);
#if __OPENCL_VERSION__ >= CL_VERSION_2_0
    outData[ind].enqueuedLocalSize = (uint)get_enqueued_local_size(dimindx);
    outData[ind].globalOffset = (uint)get_global_offset(dimindx);
#elif __OPENCL_VERSION__ >= CL_VERSION_1_1
    outData[ind].globalOffset = (uint)get_global_offset(dimindx);
#endif
})";

const char *outOfRangeWorkItemHardcodedKernelCode =
    R"(typedef struct {
    uint workDim;
    uint globalSize;
    uint globalID;
    uint localSize;
    uint localID;
    uint numGroups;
    uint groupID;
    uint globalOffset;
    uint enqueuedLocalSize;
 } work_item_data;

__kernel void sample_kernel( __global work_item_data *outData, int dim_param )
{
    int ind_mul=1;
    int ind=0;
    for( uint i = 0; i < get_work_dim(); i++ )
    {
        ind += (uint)get_global_id(i) * ind_mul;
        ind_mul *= get_global_size(i);
    }
    outData[ind].workDim = (uint)get_work_dim();
    outData[ind].globalSize = (uint)get_global_size(4);
    outData[ind].globalID = (uint)get_global_id(4);
    outData[ind].localSize = (uint)get_local_size(4);
    outData[ind].localID = (uint)get_local_id(4);
    outData[ind].numGroups = (uint)get_num_groups(4);
    outData[ind].groupID = (uint)get_group_id(4);
#if __OPENCL_VERSION__ >= CL_VERSION_2_0
    outData[ind].enqueuedLocalSize = (uint)get_enqueued_local_size(4);
    outData[ind].globalOffset = (uint)get_global_offset(4);
#elif __OPENCL_VERSION__ >= CL_VERSION_1_1
    outData[ind].globalOffset = (uint)get_global_offset(4);
#endif
})";

#define NUM_TESTS 1

struct TestWorkItemFns
{
    TestWorkItemFns(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue)
        : device(deviceID), context(context), queue(queue), program(nullptr),
          kernel(nullptr), outData(nullptr), d_holder(gRandomSeed),
          testData(10240)
    {}

    cl_int SetUp(const char *src)
    {
        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &src, "sample_kernel");
        test_error(error, "Unable to create testing kernel");

        outData = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(work_item_data) * testData.size(), NULL,
                                 &error);
        test_error(error, "Unable to create output buffer");

        error = clSetKernelArg(kernel, 0, sizeof(outData), &outData);
        test_error(error, "Unable to set kernel arg");

        return CL_SUCCESS;
    }

    cl_int Run()
    {
        cl_int error = SetUp(workItemKernelCode);
        test_error(error, "SetUp failed");

        size_t threads[3] = { 0, 0, 0 };
        size_t localThreads[3] = { 0, 0, 0 };
        for (size_t dim = 1; dim <= 3; dim++)
        {
            for (int i = 0; i < NUM_TESTS; i++)
            {
                for (size_t j = 0; j < dim; j++)
                {
                    // All of our thread sizes should be within the max local
                    // sizes, since they're all <= 20
                    threads[j] = (size_t)random_in_range(1, 20, d_holder);
                    localThreads[j] = threads[j]
                        / (size_t)random_in_range(1, (int)threads[j], d_holder);
                    while (localThreads[j] > 1
                           && (threads[j] % localThreads[j] != 0))
                        localThreads[j]--;

                    // Hack for now: localThreads > 1 are iffy
                    localThreads[j] = 1;
                }
                error = clEnqueueNDRangeKernel(queue, kernel, (cl_uint)dim,
                                               NULL, threads, localThreads, 0,
                                               NULL, NULL);
                test_error(error, "Unable to run kernel");

                error = clEnqueueReadBuffer(queue, outData, CL_TRUE, 0,
                                            sizeof(work_item_data)
                                                * testData.size(),
                                            testData.data(), 0, NULL, NULL);
                test_error(error, "Unable to read results");

                // Validate
                for (size_t q = 0; q < threads[0]; q++)
                {
                    // We can't really validate the actual value of each one,
                    // but we can validate that they're within a sane range
                    if (testData[q].workDim != (cl_uint)dim)
                    {
                        log_error(
                            "ERROR: get_work_dim() did not return proper value "
                            "for %d dimensions (expected %d, got %d)\n",
                            (int)dim, (int)dim, (int)testData[q].workDim);
                        return -1;
                    }
                    for (size_t j = 0; j < dim; j++)
                    {
                        if (testData[q].globalSize[j] != (cl_uint)threads[j])
                        {
                            log_error("ERROR: get_global_size(%d) did not "
                                      "return proper value for %d dimensions "
                                      "(expected %d, got %d)\n",
                                      (int)j, (int)dim, (int)threads[j],
                                      (int)testData[q].globalSize[j]);
                            return -1;
                        }
                        if (testData[q].globalID[j] >= (cl_uint)threads[j])
                        {
                            log_error("ERROR: get_global_id(%d) did not return "
                                      "proper value for %d dimensions (max %d, "
                                      "got %d)\n",
                                      (int)j, (int)dim, (int)threads[j],
                                      (int)testData[q].globalID[j]);
                            return -1;
                        }
                        if (testData[q].localSize[j]
                            != (cl_uint)localThreads[j])
                        {
                            log_error("ERROR: get_local_size(%d) did not "
                                      "return proper value for %d dimensions "
                                      "(expected %d, got %d)\n",
                                      (int)j, (int)dim, (int)localThreads[j],
                                      (int)testData[q].localSize[j]);
                            return -1;
                        }
                        if (testData[q].localID[j] >= (cl_uint)localThreads[j])
                        {
                            log_error(
                                "ERROR: get_local_id(%d) did not return proper "
                                "value for %d dimensions (max %d, got %d)\n",
                                (int)j, (int)dim, (int)localThreads[j],
                                (int)testData[q].localID[j]);
                            return -1;
                        }
                        size_t groupCount = (threads[j] + localThreads[j] - 1)
                            / localThreads[j];
                        if (testData[q].numGroups[j] != (cl_uint)groupCount)
                        {
                            log_error("ERROR: get_num_groups(%d) did not "
                                      "return proper value for %d dimensions "
                                      "(expected %d with global dim %d and "
                                      "local dim %d, got %d)\n",
                                      (int)j, (int)dim, (int)groupCount,
                                      (int)threads[j], (int)localThreads[j],
                                      (int)testData[q].numGroups[j]);
                            return -1;
                        }
                        if (testData[q].groupID[j] >= (cl_uint)groupCount)
                        {
                            log_error(
                                "ERROR: get_group_id(%d) did not return proper "
                                "value for %d dimensions (max %d, got %d)\n",
                                (int)j, (int)dim, (int)groupCount,
                                (int)testData[q].groupID[j]);
                            return -1;
                        }
                    }
                }
            }
        }
        return 0;
    }

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outData;
    MTdataHolder d_holder;

    std::vector<work_item_data> testData;
};

struct TestWorkItemFnsOutOfRange
{
    size_t threads[3] = { 0, 0, 0 };

    TestWorkItemFnsOutOfRange(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, const char *ksrc)
        : device(deviceID), context(context), queue(queue), program(nullptr),
          kernel(nullptr), outData(nullptr), d_holder(gRandomSeed),
          testData(10240), max_workgroup_size(0), kernel_src(ksrc)
    {}

    virtual cl_int SetUp(const char *src)
    {
        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &src, "sample_kernel");
        test_error(error, "Unable to create testing kernel");

        outData = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(work_item_data_out_of_range)
                                     * testData.size(),
                                 NULL, &error);
        test_error(error, "Unable to create output buffer");

        error = clSetKernelArg(kernel, 0, sizeof(outData), &outData);
        test_error(error, "Unable to set kernel arg");

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                sizeof(size_t) * maxWorkItemSizes.size(),
                                maxWorkItemSizes.data(), NULL);
        test_error(error,
                   "clDeviceInfo for CL_DEVICE_MAX_WORK_ITEM_SIZES failed");

        error = clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(max_workgroup_size), &max_workgroup_size, NULL);
        test_error(error, "clGetKernelWorkgroupInfo failed.");

        return CL_SUCCESS;
    }

    bool Validate(const cl_uint dim)
    {
        cl_uint threads_to_verify = 1;
        for (size_t j = 0; j < dim; j++) threads_to_verify *= threads[j];

        for (size_t q = 0; q < threads_to_verify; q++)
        {
            if (testData[q].workDim != (cl_uint)dim)
            {
                log_error("ERROR: get_work_dim() did not return proper value "
                          "for %d dimensions (expected %d, got %d)\n",
                          (int)dim, (int)dim, (int)testData[q].workDim);
                return false;
            }
            if (testData[q].globalSize != 1)
            {
                log_error("ERROR: get_global_size(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim, (int)testData[q].globalSize);
                return false;
            }
            if (testData[q].globalID != 0)
            {
                log_error("ERROR: get_global_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim, (int)testData[q].globalID);
                return false;
            }
            if (testData[q].localSize != 1)
            {
                log_error("ERROR: get_local_size(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim, (int)testData[q].localSize);
                return false;
            }
            if (testData[q].localID != 0)
            {
                log_error("ERROR: get_local_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim, (int)testData[q].localID);
                return false;
            }
            if (testData[q].numGroups != 1)
            {
                log_error("ERROR: get_num_groups(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim, (int)testData[q].numGroups);
                return false;
            }
            if (testData[q].groupID != 0)
            {
                log_error("ERROR: get_group_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim, (int)testData[q].groupID);
                return false;
            }
        }

        const Version version = get_device_cl_version(device);
        if (version >= Version(2, 0))
        {
            for (size_t q = 0; q < threads_to_verify; q++)
            {
                if (testData[q].globalOffset != 0)
                {
                    log_error(
                        "ERROR: get_global_offset(%d) did not return "
                        "proper value "
                        "for the argument out of range  (expected 0, got %d)\n",
                        (int)dim, (int)testData[q].globalOffset);
                    return false;
                }
                if (testData[q].enqueuedLocalSize != 1)
                {
                    log_error(
                        "ERROR: get_enqueued_local_size(%d) did not return "
                        "proper value for the argument out of range "
                        "(expected 1, got %d)\n",
                        (int)dim, (int)testData[q].globalSize);
                    return false;
                }
            }
        }
        else if (version >= Version(1, 1))
        {
            for (size_t q = 0; q < threads_to_verify; q++)
            {
                if (testData[q].globalOffset != 0)
                {
                    log_error(
                        "ERROR: get_global_offset(%d) did not return "
                        "proper value "
                        "for the argument out of range  (expected 0, got %d)\n",
                        (int)dim, (int)testData[q].globalOffset);
                    return false;
                }
            }
        }

        return true;
    }

    cl_int Run()
    {
        cl_int error = SetUp(kernel_src);
        test_error(error, "SetUp failed");

        size_t localThreads[3] = { 0, 0, 0 };

        for (size_t dim = 1; dim <= 3; dim++)
        {
            size_t local_workgroup_size[3] = { maxWorkItemSizes[0],
                                               maxWorkItemSizes[1],
                                               maxWorkItemSizes[2] };
            // check if maximum work group size for current dimention is not
            // exceeded
            cl_uint work_group_size = max_workgroup_size + 1;
            while (max_workgroup_size < work_group_size && work_group_size != 1)
            {
                work_group_size = 1;
                for (size_t j = 0; j < dim; j++)
                    work_group_size *= local_workgroup_size[j];
                if (max_workgroup_size < work_group_size)
                {
                    for (size_t j = 0; j < dim; j++)
                        local_workgroup_size[j] =
                            std::max(1, (int)local_workgroup_size[j] / 2);
                }
            };

            // compute max number of work groups based on buffer size and max
            // group size
            cl_uint max_work_groups = testData.size() / work_group_size;
            // take into account number of dimentions
            cl_uint work_groups_per_dim =
                std::max(1, (int)pow(max_work_groups, 1.f / dim));

            for (size_t j = 0; j < dim; j++)
            {
                // generate ranges for uniform work group size
                localThreads[j] =
                    random_in_range(1, (int)local_workgroup_size[j], d_holder);
                size_t num_groups =
                    (size_t)random_in_range(1, work_groups_per_dim, d_holder);
                threads[j] = num_groups * localThreads[j];
            }

            cl_int dim_param = dim + 1;
            error = clSetKernelArg(kernel, 1, sizeof(cl_int), &dim_param);
            test_error(error, "Unable to set kernel arg");

            error =
                clEnqueueNDRangeKernel(queue, kernel, (cl_uint)dim, NULL,
                                       threads, localThreads, 0, NULL, NULL);
            test_error(error, "Unable to run kernel");

            error = clEnqueueReadBuffer(queue, outData, CL_TRUE, 0,
                                        sizeof(work_item_data_out_of_range)
                                            * testData.size(),
                                        testData.data(), 0, NULL, NULL);
            test_error(error, "Unable to read results");

            // Validate
            if (!Validate(dim))
            {
                log_error("Validation failed");
                return TEST_FAIL;
            }
        }
        return TEST_PASS;
    }

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outData;
    MTdataHolder d_holder;

    std::vector<work_item_data_out_of_range> testData;

    std::array<size_t, 3> maxWorkItemSizes;
    size_t max_workgroup_size;

    const char *kernel_src;
};

} // anonymous namespace

int test_work_item_functions(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    TestWorkItemFns fnct(deviceID, context, queue);
    return fnct.Run();
}

int test_work_item_functions_out_of_range(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    TestWorkItemFnsOutOfRange fnct(deviceID, context, queue,
                                   outOfRangeWorkItemKernelCode);
    return fnct.Run();
}

int test_work_item_functions_out_of_range_hardcoded(cl_device_id deviceID,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    TestWorkItemFnsOutOfRange fnct(deviceID, context, queue,
                                   outOfRangeWorkItemHardcodedKernelCode);
    return fnct.Run();
}

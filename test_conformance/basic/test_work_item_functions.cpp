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
#include <vector>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/stringHelpers.h"

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

const char *outOfRangeWorkItemKernelCode =
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
    int ind_mul=1;
    int ind=0;
    for( uint i = 0; i < get_work_dim(); i++ )
    {
        ind += (uint)get_global_id(i) * ind_mul;
        ind_mul *= get_global_size(i);
    }
    outData[ind].workDim = (uint)get_work_dim();
%s
})";

const char *outOfRangeWorkItemKernelCodeExt =
    R"(
    uint dimindx=get_work_dim()+1;
    outData[ind].globalSize[0] = (uint)get_global_size(dimindx);
    outData[ind].globalID[0] = (uint)get_global_id(dimindx);
    outData[ind].localSize[0] = (uint)get_local_size(dimindx);
    outData[ind].localID[0] = (uint)get_local_id(dimindx);
    outData[ind].numGroups[0] = (uint)get_num_groups(dimindx);
    outData[ind].groupID[0] = (uint)get_group_id(dimindx);
)";

const char *outOfRangeWorkItemKernelCodeExt11 =
    R"(
    uint dimindx=get_work_dim()+1;
    outData[ind].globalSize[0] = (uint)get_global_size(dimindx);
    outData[ind].globalID[0] = (uint)get_global_id(dimindx);
    outData[ind].localSize[0] = (uint)get_local_size(dimindx);
    outData[ind].localID[0] = (uint)get_local_id(dimindx);
    outData[ind].numGroups[0] = (uint)get_num_groups(dimindx);
    outData[ind].groupID[0] = (uint)get_group_id(dimindx);
    outData[ind].globalOffset[0] = (uint)get_global_offset(dimindx);
)";

const char *outOfRangeWorkItemKernelCodeExt20 =
    R"(
    uint dimindx=get_work_dim()+1;
    outData[ind].globalSize[0] = (uint)get_global_size(dimindx);
    outData[ind].globalID[0] = (uint)get_global_id(dimindx);
    outData[ind].localSize[0] = (uint)get_local_size(dimindx);
    outData[ind].localID[0] = (uint)get_local_id(dimindx);
    outData[ind].numGroups[0] = (uint)get_num_groups(dimindx);
    outData[ind].groupID[0] = (uint)get_group_id(dimindx);
    outData[ind].globalOffset[0] = (uint)get_global_offset(dimindx);
    outData[ind].enqueuedLocalSize[0] = (uint)get_enqueued_local_size(dimindx);
)";


const char *outOfRangeWorkItemHardcodedKernelCodeExt =
    R"(
    outData[ind].globalSize[0] = (uint)get_global_size(4);
    outData[ind].globalID[0] = (uint)get_global_id(4);
    outData[ind].localSize[0] = (uint)get_local_size(4);
    outData[ind].localID[0] = (uint)get_local_id(4);
    outData[ind].numGroups[0] = (uint)get_num_groups(4);
    outData[ind].groupID[0] = (uint)get_group_id(4);
)";

const char *outOfRangeWorkItemHardcodedKernelCodeExt11 =
    R"(
    outData[ind].globalSize[0] = (uint)get_global_size(4);
    outData[ind].globalID[0] = (uint)get_global_id(4);
    outData[ind].localSize[0] = (uint)get_local_size(4);
    outData[ind].localID[0] = (uint)get_local_id(4);
    outData[ind].numGroups[0] = (uint)get_num_groups(4);
    outData[ind].groupID[0] = (uint)get_group_id(4);
    outData[ind].globalOffset[0] = (uint)get_global_offset(4);
)";

const char *outOfRangeWorkItemHardcodedKernelCodeExt20 =
    R"(
    outData[ind].globalSize[0] = (uint)get_global_size(4);
    outData[ind].globalID[0] = (uint)get_global_id(4);
    outData[ind].localSize[0] = (uint)get_local_size(4);
    outData[ind].localID[0] = (uint)get_local_id(4);
    outData[ind].numGroups[0] = (uint)get_num_groups(4);
    outData[ind].groupID[0] = (uint)get_group_id(4);
    outData[ind].globalOffset[0] = (uint)get_global_offset(4);
    outData[ind].enqueuedLocalSize[0] = (uint)get_enqueued_local_size(4);
)";


struct TestWorkItemBase
{
    TestWorkItemBase(cl_device_id did, cl_context c, cl_command_queue q,
                     cl_uint data_size, const char *src)
        : device(did), context(c), queue(q), program(nullptr), kernel(nullptr),
          outData(nullptr), d_holder(gRandomSeed), testData(data_size),
          kernel_source(src), max_workgroup_size(0)
    {}

    virtual cl_int SetUp(const char *kernel_source)
    {
        cl_int error = create_single_kernel_helper(
            context, &program, &kernel, 1, &kernel_source, "sample_kernel");
        test_error(error, "Unable to create testing kernel");

        outData = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(work_item_data) * testData.size(), NULL,
                                 &error);
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
    virtual cl_int Run() = 0;

protected:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outData;
    MTdataHolder d_holder;
    std::vector<work_item_data> testData;
    const char *kernel_source;

    std::array<size_t, 3> maxWorkItemSizes;
    size_t max_workgroup_size;
};

#define NUM_TESTS 1

struct TestWorkItemFns : public TestWorkItemBase
{

    TestWorkItemFns(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue)
        : TestWorkItemBase(deviceID, context, queue, 10240, workItemKernelCode)
    {}

    cl_int Run() override
    {
        cl_int error = SetUp(kernel_source);
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
};

template <bool hardcoded>
struct TestWorkItemFnsOutOfRange : public TestWorkItemBase
{

    size_t threads[3] = { 0, 0, 0 };

    TestWorkItemFnsOutOfRange(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue)
        : TestWorkItemBase(deviceID, context, queue, 10240,
                           outOfRangeWorkItemKernelCode)
    {}

    virtual cl_int SetUp(const char *kernel_source) override
    {
        std::ostringstream sstr;
        std::string program_source;
        const Version version = get_device_cl_version(device);
        if (version >= Version(2, 0))
        {
            program_source = str_sprintf(
                std::string(kernel_source),
                hardcoded ? outOfRangeWorkItemHardcodedKernelCodeExt20
                          : outOfRangeWorkItemKernelCodeExt20);
        }
        else if (version >= Version(1, 1))
        {
            program_source = str_sprintf(
                std::string(kernel_source),
                hardcoded ? outOfRangeWorkItemHardcodedKernelCodeExt11
                          : outOfRangeWorkItemKernelCodeExt11);
        }
        else
        {
            program_source =
                str_sprintf(std::string(kernel_source),
                            hardcoded ? outOfRangeWorkItemHardcodedKernelCodeExt
                                      : outOfRangeWorkItemKernelCodeExt);
        }

        return TestWorkItemBase::SetUp(program_source.c_str());
    }


    virtual cl_int Validate(const cl_uint dim)
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
                return -1;
            }
            if (testData[q].globalSize[0] != 1)
            {
                log_error("ERROR: get_global_size(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim + 1, (int)testData[q].globalSize[0]);
                return -1;
            }
            if (testData[q].globalID[0] != 0)
            {
                log_error("ERROR: get_global_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim + 1, (int)testData[q].globalID[0]);
                return -1;
            }
            if (testData[q].localSize[0] != 1)
            {
                log_error("ERROR: get_local_size(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim + 1, (int)testData[q].localSize[0]);
                return -1;
            }
            if (testData[q].localID[0] != 0)
            {
                log_error("ERROR: get_local_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim + 1, (int)testData[q].localID[0]);
                return -1;
            }
            if (testData[q].numGroups[0] != 1)
            {
                log_error("ERROR: get_num_groups(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 1, got %d)\n",
                          (int)dim + 1, (int)testData[q].numGroups[0]);
                return -1;
            }
            if (testData[q].groupID[0] != 0)
            {
                log_error("ERROR: get_group_id(%d) did not return "
                          "proper value for the argument out of range "
                          "(expected 0, got %d)\n",
                          (int)dim + 1, (int)testData[q].groupID[0]);
                return -1;
            }
        }

        const Version version = get_device_cl_version(device);
        if (version >= Version(2, 0))
        {
            for (size_t q = 0; q < threads_to_verify; q++)
            {
                if (testData[q].globalOffset[0] != 0)
                {
                    log_error("ERROR: get_global_offset() did not return "
                              "proper value "
                              "for %d dimensions (expected %d, got %d)\n",
                              (int)dim, (int)dim, (int)testData[q].workDim);
                    return -1;
                }
                if (testData[q].enqueuedLocalSize[0] != 1)
                {
                    log_error(
                        "ERROR: get_enqueued_local_size(%d) did not return "
                        "proper value for the argument out of range "
                        "(expected 1, got %d)\n",
                        (int)dim + 1, (int)testData[q].globalSize[0]);
                    return -1;
                }
            }
        }
        else if (version >= Version(1, 1))
        {
            for (size_t q = 0; q < threads_to_verify; q++)
            {
                if (testData[q].globalOffset[0] != 0)
                {
                    log_error("ERROR: get_global_offset() did not return "
                              "proper value "
                              "for %d dimensions (expected %d, got %d)\n",
                              (int)dim, (int)dim, (int)testData[q].workDim);
                    return -1;
                }
            }
        }

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = SetUp(kernel_source);
        test_error(error, "SetUp failed");

        size_t localThreads[3] = { 0, 0, 0 };

        for (size_t dim = 1; dim <= 3; dim++)
        {
            cl_uint local_workgroup_size[3] = { maxWorkItemSizes[0],
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

            error =
                clEnqueueNDRangeKernel(queue, kernel, (cl_uint)dim, NULL,
                                       threads, localThreads, 0, NULL, NULL);
            test_error(error, "Unable to run kernel");

            error =
                clEnqueueReadBuffer(queue, outData, CL_TRUE, 0,
                                    sizeof(work_item_data) * testData.size(),
                                    testData.data(), 0, NULL, NULL);
            test_error(error, "Unable to read results");

            // Validate
            error = Validate(dim);
            test_error(error, "Validation failed");
        }
        return 0;
    }
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
    TestWorkItemFnsOutOfRange<false> fnct(deviceID, context, queue);
    return fnct.Run();
}

int test_work_item_functions_out_of_range_hardcoded(cl_device_id deviceID,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    TestWorkItemFnsOutOfRange<true> fnct(deviceID, context, queue);
    return fnct.Run();
}

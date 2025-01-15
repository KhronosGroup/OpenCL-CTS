//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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
#include "types.hpp"

template<typename T>
int test_composite_construct(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, const char *name,
                std::vector<T> &results,
                bool (*notEqual)(const T&, const T&) = isNotEqual<T>)
{
    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    int num = (int)results.size();

    size_t bytes = num * sizeof(T);
    clMemWrapper mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    std::vector<T> host(num);
    err = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes, &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    for (int i = 0; i < num; i++) {
        if (notEqual(host[i], results[i])) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

REGISTER_TEST(op_composite_construct_int4)
{
    cl_int4 value = { { 123, 122, 121, 119 } };
    std::vector<cl_int4> results(256, value);
    return test_composite_construct(device, context, queue,
                                    "composite_construct_int4", results);
}

REGISTER_TEST(op_composite_construct_struct)
{
    typedef AbstractStruct2<int, char> CustomType1;
    typedef AbstractStruct2<cl_int2, CustomType1> CustomType2;

    CustomType1 value1 = { 2100483600, (char)128 };
    cl_int2 intvals = { { 2100480000, 2100480000 } };
    CustomType2 value2 = {intvals, value1};

    std::vector<CustomType2> results(256, value2);
    return test_composite_construct(device, context, queue,
                                    "composite_construct_struct", results);
}

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
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/mt19937.h"
#include "base.h"

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

class CStressTest : public CTest {
public:
    CStressTest(const std::vector<std::string>& kernel) : CTest(), _kernels(kernel) {

    }

    CStressTest(const std::string& kernel) : CTest(), _kernels(1, kernel) {

    }

    int ExecuteSubcase(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, const std::string& src) {
        cl_int error;

        clProgramWrapper program;
        clKernelWrapper kernel;

        const char *srcPtr = src.c_str();

        if (create_single_kernel_helper_with_build_options(context, &program, &kernel, 1, &srcPtr, "testKernel", "-cl-std=CL2.0")) {
            log_error("create_single_kernel_helper failed");
            return -1;
        }

        size_t bufferSize = num_elements * sizeof(cl_uint);
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        test_error(error, "clSetKernelArg failed");

        size_t globalWorkGroupSize = num_elements;
        size_t localWorkGroupSize = 0;
        error = get_max_common_work_group_size(context, kernel, globalWorkGroupSize, &localWorkGroupSize);
        test_error(error, "Unable to get common work group size");

        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkGroupSize, &localWorkGroupSize, 0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed");

        // verify results
        std::vector<cl_uint> results(num_elements);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize, &results[0], 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        size_t passCount = std::count(results.begin(), results.end(), 1);
        if (passCount != results.size()) {
            std::vector<cl_uint>::iterator iter = std::find(results.begin(), results.end(), 0);
            log_error("Verification on device failed at index %ld\n", std::distance(results.begin(), iter));
            log_error("%ld out of %ld failed\n", (results.size()-passCount), results.size());
            return -1;
        }

        return CL_SUCCESS;
    }

    int Execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
        cl_int result = CL_SUCCESS;

        for (std::vector<std::string>::const_iterator it = _kernels.begin(); it != _kernels.end(); ++it) {
            log_info("Executing subcase #%ld out of %ld\n", (it - _kernels.begin() + 1), _kernels.size());

            result |= ExecuteSubcase(deviceID, context, queue, num_elements, *it);
        }

        return result;
    }

private:
    const std::vector<std::string> _kernels;
};

int test_max_number_of_params(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    cl_int error;

    size_t deviceMaxParameterSize;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(deviceMaxParameterSize), &deviceMaxParameterSize, NULL);
    test_error(error, "clGetDeviceInfo failed");

    size_t deviceAddressBits;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(deviceAddressBits), &deviceAddressBits, NULL);
    test_error(error, "clGetDeviceInfo failed");

    size_t maxParams = deviceMaxParameterSize / (deviceAddressBits / 8);

    const std::string KERNEL_FUNCTION_TEMPLATE[] = {
        common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(int *ptr0 ",
            // the rest of arguments goes here
           ") {"
        NL "    // check first pointer only"
        NL "    if (!isFenceValid(get_fence(ptr0)))"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __global int * gptr;"
        NL "    __local int * lptr;"
        NL "    __private int * pptr;"
        NL
        NL "    size_t failures = 0;"
        NL
        NL,
            // the function body goes here
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    };

    std::ostringstream type_params;
    std::ostringstream function_calls;

    for (size_t i = 0; i < maxParams; i++) {
        type_params << ", int *ptr" << i+1;
    }

    // use pseudo random generator to shuffle params
    MTdata d = init_genrand(gRandomSeed);
    if (!d)
        return -1;

    std::string pointers[] = { "gptr", "lptr", "pptr" };

    size_t totalCalls = maxParams / 2;
    for (size_t i = 0; i < totalCalls; i++) {
        function_calls << "\tif (!helperFunction(gptr";

        for (size_t j = 0; j < maxParams; j++) {
            function_calls << ", " << pointers[genrand_int32(d)%3];
        }

        function_calls << ")) failures++;" << NL;
    }

    free_mtdata(d);
    d = NULL;

    const std::string KERNEL_FUNCTION = KERNEL_FUNCTION_TEMPLATE[0] + type_params.str() + KERNEL_FUNCTION_TEMPLATE[1] + function_calls.str() + KERNEL_FUNCTION_TEMPLATE[2];

    CStressTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

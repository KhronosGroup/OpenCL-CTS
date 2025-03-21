//
// Copyright (c) 2017-2020 The Khronos Group Inc.
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

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <atomic>
#include <string>

namespace {

const char *sample_async_kernel[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    size_t tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n"
};

const char *sample_async_kernel_error[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    size_t tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = badcodehere;\n"
    "\n"
    "}\n"
};

// Data passed to a program completion callback
struct TestData
{
    cl_device_id device;
    cl_build_status expectedStatus;
};

std::atomic<int> callbackResult;

}

void CL_CALLBACK test_notify_build_complete(cl_program program, void *userData)
{
    TestData *data = reinterpret_cast<TestData *>(userData);

    // Check user data is valid
    if (data == nullptr)
    {
        log_error("ERROR: User data passed to callback was not valid!\n");
        callbackResult = -1;
        return;
    }

    // Get program build status
    cl_build_status status;
    cl_int err =
        clGetProgramBuildInfo(program, data->device, CL_PROGRAM_BUILD_STATUS,
                              sizeof(cl_build_status), &status, NULL);
    if (err != CL_SUCCESS)
    {
        log_info("ERROR: failed to get build status from callback\n");
        callbackResult = -1;
        return;
    }

    log_info("Program completion callback received build status %d\n", status);

    // Check program build status matches expectation
    if (status != data->expectedStatus)
    {
        log_info("ERROR: build status %d != expected status %d\n", status,
                 data->expectedStatus);
        callbackResult = -1;
    }
    else
    {
        callbackResult = 1;
    }
}

REGISTER_TEST(async_build)
{
    cl_int error;

    struct TestDef
    {
        const char **source;
        cl_build_status expectedStatus;
    };

    TestDef testDefs[] = { { sample_async_kernel, CL_BUILD_SUCCESS },
                           { sample_async_kernel_error, CL_BUILD_ERROR } };
    for (TestDef &testDef : testDefs)
    {
        log_info("\nTesting program that should produce status %d\n",
                 testDef.expectedStatus);

        // Create the program
        clProgramWrapper program;
        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           testDef.source);
        test_error(error, "Unable to create program from source");

        // Start an asynchronous build, registering the completion callback
        TestData testData = { device, testDef.expectedStatus };
        callbackResult = 0;
        error = clBuildProgram(program, 1, &device, NULL,
                               test_notify_build_complete, (void *)&testData);
        // Allow implementations to return synchronous build failures.
        // They still need to call the callback.
        if (!(error == CL_BUILD_PROGRAM_FAILURE
              && testDef.expectedStatus == CL_BUILD_ERROR))
            test_error(error, "Unable to start build");

        // Wait for callback to fire
        int timeout = 20;
        while (callbackResult == 0)
        {
            if (timeout < 0)
            {
                log_error("Timeout while waiting for callback to fire.\n\n");
                return -1;
            }

            log_info(" -- still waiting for callback...\n");
            sleep(1);
            timeout--;
        }

        // Check the callback result
        if (callbackResult == 1)
        {
            log_error("Test passed.\n\n");
        }
        else
        {
            log_error("Async build callback indicated test failure.\n\n");
            return -1;
        }
    }

    return 0;
}

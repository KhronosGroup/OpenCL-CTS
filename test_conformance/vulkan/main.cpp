//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>

#if !defined(__APPLE__)
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif

#include "harness/testHarness.h"

unsigned int numCQ;
bool multiImport;
bool multiCtx;
bool debug_trace = false;
bool useSingleImageKernel = false;
bool useDeviceLocal = false;
bool disableNTHandleType = false;
bool enableOffset = false;

static void printUsage(const char *execName)
{
    const char *p = strrchr(execName, '/');
    if (p != NULL) execName = p + 1;

    log_info("Usage: %s [test_names] [options]\n", execName);
    log_info("Test names:\n");
    for (int i = 0; i < test_registry::getInstance().num_tests(); i++)
    {
        log_info("\t%s\n", test_registry::getInstance().definitions()[i].name);
    }
    log_info("\n");
    log_info("Options:\n");
    log_info("\t--debug_trace - Enables additional debug info logging\n");
    log_info("\t--non_dedicated - Choose dedicated Vs. non_dedicated \n");
}

bool isDeviceSelection(const char *arg)
{

    return strcmp(arg, "gpu") == 0 || strcmp(arg, "CL_DEVICE_TYPE_GPU") == 0
        || strcmp(arg, "cpu") == 0 || strcmp(arg, "CL_DEVICE_TYPE_CPU") == 0
        || strcmp(arg, "accelerator") == 0
        || strcmp(arg, "CL_DEVICE_TYPE_ACCELERATOR") == 0
        || strcmp(arg, "custom") == 0
        || strcmp(arg, "CL_DEVICE_TYPE_CUSTOM") == 0
        || strcmp(arg, "CL_DEVICE_TYPE_DEFAULT") == 0;
}

size_t parseParams(int argc, const char *argv[], const char **argList)
{
    size_t argCount = 1;
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == NULL) break;
        if (argv[i][0] == '-')
        {
            if (!strcmp(argv[i], "--debug_trace"))
            {
                debug_trace = true;
            }
            if (!strcmp(argv[i], "--useSingleImageKernel"))
            {
                useSingleImageKernel = true;
            }
            if (!strcmp(argv[i], "--useDeviceLocal"))
            {
                useDeviceLocal = true;
            }
            if (!strcmp(argv[i], "--disableNTHandleType"))
            {
                disableNTHandleType = true;
            }
            if (strcmp(argv[i], "-h") == 0)
            {
                printUsage(argv[0]);
                argCount = 0; // Returning argCount=0 to assert error in main()
                break;
            }
        }
        else if (isDeviceSelection(argv[i]))
        {
            if (strcmp(argv[i], "gpu") != 0
                && strcmp(argv[i], "CL_DEVICE_TYPE_GPU") != 0
                && strcmp(argv[i], "CL_DEVICE_TYPE_DEFAULT") != 0)
            {
                log_info("Vulkan tests can only run on a GPU device.\n");
                return 0;
            }
        }
        else
        {
            argList[argCount] = argv[i];
            argCount++;
        }
    }
    return argCount;
}

int main(int argc, const char *argv[])
{
    test_start();

    cl_device_type requestedDeviceType = CL_DEVICE_TYPE_GPU;
    char *force_cpu = getenv("CL_DEVICE_TYPE");
    if (force_cpu != NULL)
    {
        if (strcmp(force_cpu, "gpu") == 0
            || strcmp(force_cpu, "CL_DEVICE_TYPE_GPU") == 0)
            requestedDeviceType = CL_DEVICE_TYPE_GPU;
        else if (strcmp(force_cpu, "cpu") == 0
                 || strcmp(force_cpu, "CL_DEVICE_TYPE_CPU") == 0)
            requestedDeviceType = CL_DEVICE_TYPE_CPU;
        else if (strcmp(force_cpu, "accelerator") == 0
                 || strcmp(force_cpu, "CL_DEVICE_TYPE_ACCELERATOR") == 0)
            requestedDeviceType = CL_DEVICE_TYPE_ACCELERATOR;
        else if (strcmp(force_cpu, "CL_DEVICE_TYPE_DEFAULT") == 0)
            requestedDeviceType = CL_DEVICE_TYPE_DEFAULT;
    }

    if (requestedDeviceType != CL_DEVICE_TYPE_GPU)
    {
        log_info("Vulkan tests can only run on a GPU device.\n");
        return 0;
    }

    const char **argList = (const char **)calloc(argc, sizeof(char *));
    size_t argCount = parseParams(argc, argv, argList);
    if (argCount == 0) return 0;

    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}

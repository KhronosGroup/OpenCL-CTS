//
// Copyright (c) 2022 The Khronos Group Inc.
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


#include "procs.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"
#include "harness/deviceInfo.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <vulkan_interop_common.hpp>
#include <vulkan_wrapper.hpp>

#define BUFFERSIZE 3000

static void params_reset()
{
    numCQ = 1;
    multiImport = false;
    multiCtx = false;
}

extern int test_buffer_common(cl_device_id device_, cl_context context_,
                              cl_command_queue queue_, int numElements_);
extern int test_image_common(cl_device_id device_, cl_context context_,
                             cl_command_queue queue_, int numElements_);

int test_buffer_single_queue(cl_device_id device_, cl_context context_,
                             cl_command_queue queue_, int numElements_)
{
    params_reset();
    log_info("RUNNING TEST WITH ONE QUEUE...... \n\n");
    return test_buffer_common(device_, context_, queue_, numElements_);
}
int test_buffer_multiple_queue(cl_device_id device_, cl_context context_,
                               cl_command_queue queue_, int numElements_)
{
    params_reset();
    numCQ = 2;
    log_info("RUNNING TEST WITH TWO QUEUE...... \n\n");
    return test_buffer_common(device_, context_, queue_, numElements_);
}
int test_buffer_multiImport_sameCtx(cl_device_id device_, cl_context context_,
                                    cl_command_queue queue_, int numElements_)
{
    params_reset();
    multiImport = true;
    log_info("RUNNING TEST WITH MULTIPLE DEVICE MEMORY IMPORT "
             "IN SAME CONTEXT...... \n\n");
    return test_buffer_common(device_, context_, queue_, numElements_);
}
int test_buffer_multiImport_diffCtx(cl_device_id device_, cl_context context_,
                                    cl_command_queue queue_, int numElements_)
{
    params_reset();
    multiImport = true;
    multiCtx = true;
    log_info("RUNNING TEST WITH MULTIPLE DEVICE MEMORY IMPORT "
             "IN DIFFERENT CONTEXT...... \n\n");
    return test_buffer_common(device_, context_, queue_, numElements_);
}
int test_image_single_queue(cl_device_id device_, cl_context context_,
                            cl_command_queue queue_, int numElements_)
{
    params_reset();
    log_info("RUNNING TEST WITH ONE QUEUE...... \n\n");
    return test_image_common(device_, context_, queue_, numElements_);
}
int test_image_multiple_queue(cl_device_id device_, cl_context context_,
                              cl_command_queue queue_, int numElements_)
{
    params_reset();
    numCQ = 2;
    log_info("RUNNING TEST WITH TWO QUEUE...... \n\n");
    return test_image_common(device_, context_, queue_, numElements_);
}

test_definition test_list[] = { ADD_TEST(buffer_single_queue),
                                ADD_TEST(buffer_multiple_queue),
                                ADD_TEST(buffer_multiImport_sameCtx),
                                ADD_TEST(buffer_multiImport_diffCtx),
                                ADD_TEST(image_single_queue),
                                ADD_TEST(image_multiple_queue),
                                ADD_TEST(consistency_external_buffer),
                                ADD_TEST(consistency_external_image),
                                ADD_TEST(consistency_external_semaphore),
                                ADD_TEST(platform_info),
                                ADD_TEST(device_info) };

const int test_num = ARRAY_SIZE(test_list);

cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
char *choosen_platform_name = NULL;
cl_platform_id platform = NULL;
cl_int choosen_platform_index = -1;
char platform_name[1024] = "";
cl_platform_id select_platform = NULL;
char *extensions = NULL;
size_t extensionSize = 0;
cl_uint num_devices = 0;
cl_uint device_no = 0;
cl_device_id *devices;
const size_t bufsize = BUFFERSIZE;
char buf[BUFFERSIZE];
cl_uchar uuid[CL_UUID_SIZE_KHR];
unsigned int numCQ;
bool multiImport;
bool multiCtx;
bool debug_trace = false;
bool useSingleImageKernel = false;
bool useDeviceLocal = false;
bool disableNTHandleType = false;
bool enableOffset = false;
bool non_dedicated = false;

static void printUsage(const char *execName)
{
    const char *p = strrchr(execName, '/');
    if (p != NULL) execName = p + 1;

    log_info("Usage: %s [test_names] [options]\n", execName);
    log_info("Test names:\n");
    for (int i = 0; i < test_num; i++)
    {
        log_info("\t%s\n", test_list[i].name);
    }
    log_info("\n");
    log_info("Options:\n");
    log_info("\t--debug_trace - Enables additional debug info logging\n");
    log_info("\t--non_dedicated - Choose dedicated Vs. non_dedicated \n");
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
            if (!strcmp(argv[i], "--enableOffset"))
            {
                enableOffset = true;
            }
            if (!strcmp(argv[i], "--non_dedicated"))
            {
                non_dedicated = true;
            }
            if (strcmp(argv[i], "-h") == 0)
            {
                printUsage(argv[0]);
                argCount = 0; // Returning argCount=0 to assert error in main()
                break;
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
    int errNum = 0;

    test_start();
    params_reset();

    if (!checkVkSupport())
    {
        log_info("Vulkan supported GPU not found \n");
        log_info("TEST SKIPPED \n");
        return 0;
    }

    VulkanDevice vkDevice;

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
    gDeviceType = CL_DEVICE_TYPE_GPU;

    const char **argList = (const char **)calloc(argc, sizeof(char *));
    size_t argCount = parseParams(argc, argv, argList);
    if (argCount == 0) return 0;
    // get the platform ID
    errNum = clGetPlatformIDs(1, &platform, NULL);
    if (errNum != CL_SUCCESS)
    {
        print_error(errNum, "Error: Failed to get platform\n");
        return errNum;
    }

    errNum =
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (CL_SUCCESS != errNum)
    {
        print_error(errNum, "clGetDeviceIDs failed in returning of devices\n");
        return errNum;
    }
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    if (NULL == devices)
    {
        print_error(errNum, "Unable to allocate memory for devices\n");
        return CL_OUT_OF_HOST_MEMORY;
    }
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices,
                            NULL);
    if (CL_SUCCESS != errNum)
    {
        print_error(errNum, "Failed to get deviceID.\n");
        return errNum;
    }
    for (device_no = 0; device_no < num_devices; device_no++)
    {
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS, 0,
                                 NULL, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            log_error("Error in clGetDeviceInfo for getting "
                      "device_extension size....\n");
            return errNum;
        }
        extensions = (char *)malloc(extensionSize);
        if (NULL == extensions)
        {
            log_error("Unable to allocate memory for extensions\n");
            return CL_OUT_OF_HOST_MEMORY;
        }
        errNum =
            clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS,
                            extensionSize, extensions, NULL /*&extensionSize*/);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum,
                        "Error in clGetDeviceInfo for getting "
                        "device_extension\n");
            return errNum;
        }
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_UUID_KHR,
                                 CL_UUID_SIZE_KHR, uuid, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum, "clGetDeviceInfo failed with error\n ");
            return errNum;
        }
        errNum =
            memcmp(uuid, vkDevice.getPhysicalDevice().getUUID(), VK_UUID_SIZE);
        if (errNum == 0)
        {
            break;
        }
    }
    if (device_no >= num_devices)
    {
        fprintf(stderr,
                "OpenCL error: "
                "No Vulkan-OpenCL Interop capable GPU found.\n");
    }
    if (!(is_extension_available(devices[device_no], "cl_khr_external_memory")
          && is_extension_available(devices[device_no],
                                    "cl_khr_external_semaphore")))
    {
        log_info("Device does not support cl_khr_external_memory "
                 "or cl_khr_external_semaphore\n");
        log_info(" TEST SKIPPED\n");
        return CL_SUCCESS;
    }
    init_cl_vk_ext(platform);

    // Execute tests.
    // Note: don't use the entire harness, because we have a different way of
    // obtaining the device (via the context)
    errNum = parseAndCallCommandLineTests(argCount, argList, devices[device_no],
                                          test_num, test_list, true, 0, 1024);
    return errNum;
}

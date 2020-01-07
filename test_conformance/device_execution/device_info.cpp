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
#include <stdio.h>
#include <string.h>
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include "procs.h"
#include "utils.h"

static const cl_uint MIN_DEVICE_PREFFERED_QUEUE_SIZE =  16 * 1024;
static const cl_uint MAX_DEVICE_QUEUE_SIZE           = 256 * 1024;
static const cl_uint MAX_DEVICE_EMBEDDED_QUEUE_SIZE  =  64 * 1024;

#ifdef CL_VERSION_2_0

int test_device_info(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_int err_ret;
    int embedded = 0;
    size_t ret_len;
    char profile[32] = {0};
    cl_command_queue_properties properties;
    cl_uint size;

    err_ret = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), profile, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_PROFILE) failed");
    if(ret_len < sizeof(profile) && strcmp(profile, "FULL_PROFILE") == 0) embedded = 0;
    else if(ret_len < sizeof(profile) && strcmp(profile, "EMBEDDED_PROFILE") == 0) embedded = 1;
    else
    {
        log_error("Unknown device profile: %s\n", profile);
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, sizeof(properties), &properties, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES) failed");
    if(!(properties&CL_QUEUE_PROFILING_ENABLE))
    {
        log_error("Host command-queue does not support mandated minimum capability: CL_QUEUE_PROFILING_ENABLE\n");
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, sizeof(properties), &properties, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES) failed");
    if(!(properties&CL_QUEUE_PROFILING_ENABLE))
    {
        log_error("Device command-queue does not support mandated minimum capability: CL_QUEUE_PROFILING_ENABLE\n");
        return -1;
    }
    if(!(properties&CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
    {
        log_error("Device command-queue does not support mandated minimum capability: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, sizeof(size), &size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE) failed");
    if(size < MIN_DEVICE_PREFFERED_QUEUE_SIZE)
    {
        log_error("Device command-queue preferred size is less than minimum %dK: %dK\n", MIN_DEVICE_PREFFERED_QUEUE_SIZE/1024, size/1024);
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, sizeof(size), &size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");
    if(size < (embedded ? MAX_DEVICE_EMBEDDED_QUEUE_SIZE : MAX_DEVICE_QUEUE_SIZE))
    {
        log_error("Device command-queue maximum size is less than minimum %dK: %dK\n", (embedded ? MAX_DEVICE_EMBEDDED_QUEUE_SIZE : MAX_DEVICE_QUEUE_SIZE)/1024, size/1024);
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof(size), &size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_ON_DEVICE_QUEUES) failed");
    if(size < 1)
    {
        log_error("Maximum number of device queues is less than minimum 1: %d\n", size);
        return -1;
    }

    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_EVENTS, sizeof(size), &size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_ON_DEVICE_EVENTS) failed");
    if(size < 1024)
    {
        log_error("Maximum number of events in use by a device queue is less than minimum 1024: %d\n", size);
        return -1;
    }

    return 0;
}

#endif


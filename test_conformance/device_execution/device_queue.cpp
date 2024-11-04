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

#include <vector>

#include "procs.h"
#include "utils.h"

static int check_device_queue(cl_device_id device, cl_context context, cl_command_queue queue, cl_uint size)
{
    cl_int err_ret;
    cl_context q_context;
    cl_device_id q_device;
    cl_command_queue_properties q_properties;
    cl_uint q_size;
    size_t size_ret;

    err_ret = clRetainCommandQueue(queue);
    test_error(err_ret, "clRetainCommandQueue() failed");

    err_ret = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(q_context), &q_context, &size_ret);
    test_error(err_ret, "clGetCommandQueueInfo(CL_QUEUE_CONTEXT) failed");
    if(size_ret != sizeof(q_context) || q_context != context)
    {
        log_error("clGetCommandQueueInfo(CL_QUEUE_CONTEXT) returned invalid context\n");
        return -1;
    }

    err_ret = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(q_device), &q_device, &size_ret);
    test_error(err_ret, "clGetCommandQueueInfo(CL_QUEUE_DEVICE) failed");
    if(size_ret != sizeof(q_device) || q_device != device)
    {
        log_error("clGetCommandQueueInfo(CL_QUEUE_DEVICE) returned invalid device\n");
        return -1;
    }

    err_ret = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(q_properties), &q_properties, &size_ret);
    test_error(err_ret, "clGetCommandQueueInfo(CL_QUEUE_PROPERTIES) failed");
    if(size_ret != sizeof(q_properties) || !(q_properties & (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE)))
    {
        log_error("clGetCommandQueueInfo(CL_QUEUE_PROPERTIES) returned invalid properties\n");
        return -1;
    }

    err_ret = clGetCommandQueueInfo(queue, CL_QUEUE_SIZE, sizeof(q_size), &q_size, &size_ret);
    test_error(err_ret, "clGetCommandQueueInfo(CL_QUEUE_SIZE) failed");
    if(size_ret != sizeof(q_size) || q_size < 1)
    {
        log_error("clGetCommandQueueInfo(CL_QUEUE_SIZE) returned invalid queue size\n");
        return -1;
    }

    err_ret = clReleaseCommandQueue(queue);
    test_error(err_ret, "clReleaseCommandQueue() failed");


    return 0;
}

static int check_device_queues(cl_device_id device, cl_context context, cl_uint num_queues, cl_queue_properties *properties, cl_uint size)
{
    cl_int err_ret, res = 0;
    cl_uint i;
    std::vector<clCommandQueueWrapper> queue(num_queues);

    // Create all queues
    for(i = 0; i < num_queues; ++i)
    {
        queue[i] = clCreateCommandQueueWithProperties(context, device, properties, &err_ret);
        test_error(err_ret, "clCreateCommandQueueWithProperties failed");
    }

    // Validate all queues
    for(i = 0; i < num_queues; ++i)
    {
        err_ret = check_device_queue(device, context, queue[i], size);
        if(check_error(err_ret, "Device queue[%d] validation failed", i)) res = -1;

    }
    return res;
}

int test_device_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_int err_ret, res = 0;
    size_t ret_len;
    clCommandQueueWrapper dev_queue;
    cl_uint preffered_size, max_size, max_queues;

    cl_queue_properties queue_prop_def[] =
    {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE|CL_QUEUE_ON_DEVICE_DEFAULT,
        0
    };

    cl_queue_properties queue_prop[] =
    {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE,
        0
    };

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, sizeof(preffered_size), &preffered_size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE) failed");

    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, sizeof(max_size), &max_size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");

    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof(max_queues), &max_queues, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_ON_DEVICE_QUEUES) failed");

    if(max_queues > MAX_QUEUES) max_queues = MAX_QUEUES;

    dev_queue = clCreateCommandQueueWithProperties(context, device, queue_prop_def, &err_ret);
    test_error(err_ret,
               "clCreateCommandQueueWithProperties(CL_QUEUE_ON_DEVICE | "
               "CL_QUEUE_ON_DEVICE_DEFAULT) failed");

    err_ret = check_device_queue(device, context, dev_queue, preffered_size);
    if(check_error(err_ret, "Default device queue validation failed")) res = -1;

    log_info("Default device queue is OK.\n");

    if(max_queues > 1) // Check more queues if supported.
    {
        cl_uint q_size = preffered_size-1024;
        cl_queue_properties queue_prop_size[] =
        {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE,
            CL_QUEUE_SIZE, q_size,
            0
        };

        cl_queue_properties queue_prop_max[] =
        {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE,
            CL_QUEUE_SIZE, max_size,
            0
        };
        {
            err_ret = check_device_queues(device, context, 1, queue_prop, preffered_size);
            if(check_error(err_ret, "Second device queue validation failed")) res = -1;
            else log_info("Second device queue is OK.\n");
        }
        {
            err_ret = check_device_queues(device, context, 1, queue_prop_size, q_size);
            if(check_error(err_ret, "Device queue with size validation failed")) res = -1;
            else log_info("Device queue with size is OK.\n");
        }
        {
            err_ret = check_device_queues(device, context, 1, queue_prop_max, max_size);
            if(check_error(err_ret, "Device queue max size validation failed")) res = -1;
            else log_info("Device queue max size is OK.\n");
        }
        {
            err_ret = check_device_queues(device, context, max_queues, queue_prop, preffered_size);
            if(check_error(err_ret, "Max number device queue validation failed")) res = -1;
            else log_info("Max number device queue is OK.\n");
        }
        {
            err_ret = check_device_queues(device, context, max_queues, queue_prop_size, q_size);
            if(check_error(err_ret, "Max number device queue with size validation failed")) res = -1;
            else log_info("Max number device queue with size is OK.\n");
        }
        {
            err_ret = check_device_queues(device, context, max_queues, queue_prop_max, max_size);
            if(check_error(err_ret, "Max number device queue with max size validation failed")) res = -1;
            else log_info("Max number device queue with max size is OK.\n");
        }
    }

    return res;
}


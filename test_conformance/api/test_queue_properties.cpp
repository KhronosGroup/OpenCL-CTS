//
// Copyright (c) 2018 The Khronos Group Inc.
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
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include <sstream>
#include <string>
#include <vector>

/*
The test against cl_khr_create_command_queue extension. It validates if devices with Opencl 1.X can use clCreateCommandQueueWithPropertiesKHR function.
Based on device capabilities test will create queue with NULL properties, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property and
CL_QUEUE_PROFILING_ENABLE property. Finally simple kernel will be executed on such queue.
*/

const char *queue_test_kernel[] = {
"__kernel void vec_cpy(__global int *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src[tid];\n"
"\n"
"}\n" };

int enqueue_kernel(cl_context context,
                   const cl_queue_properties_khr *queue_prop_def,
                   cl_device_id device, clKernelWrapper &kernel,
                   size_t num_elements)
{
    clMemWrapper streams[2];
    int error;
    std::vector<int> buf(num_elements);
    clCreateCommandQueueWithPropertiesKHR_fn clCreateCommandQueueWithPropertiesKHR = NULL;
    cl_platform_id platform;
    clEventWrapper event;

    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

    clCreateCommandQueueWithPropertiesKHR = (clCreateCommandQueueWithPropertiesKHR_fn) clGetExtensionFunctionAddressForPlatform(platform, "clCreateCommandQueueWithPropertiesKHR");
    if (clCreateCommandQueueWithPropertiesKHR == NULL)
    {
        log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed\n");
        return -1;
    }

    clCommandQueueWrapper queue = clCreateCommandQueueWithPropertiesKHR(
        context, device, queue_prop_def, &error);
    test_error(error, "clCreateCommandQueueWithPropertiesKHR failed");

    for (size_t i = 0; i < num_elements; ++i)
    {
        buf[i] = i;
    }

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, num_elements * sizeof(int), buf.data(), &error);
    test_error( error, "clCreateBuffer failed." );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, num_elements * sizeof(int), NULL, &error);
    test_error( error, "clCreateBuffer failed." );

    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error( error, "clSetKernelArg failed." );

    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error( error, "clSetKernelArg failed." );

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &num_elements, NULL, 0, NULL, &event);
    test_error( error, "clEnqueueNDRangeKernel failed." );
    
    error = clWaitForEvents(1, &event);
    test_error(error, "clWaitForEvents failed.");
    
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, num_elements, buf.data(), 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed." );

    for (size_t i = 0; i < num_elements; ++i)
    {
        if (static_cast<size_t>(buf[i]) != i)
        {
            log_error("ERROR: Incorrect vector copy result.");
            return -1;
        }
    }

    return 0;
}

REGISTER_TEST(queue_properties)
{
    if (num_elements <= 0)
    {
        num_elements = 128;
    }
    int error = 0;

    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_command_queue_properties device_props = 0;
    cl_command_queue_properties queue_prop_def[] = { CL_QUEUE_PROPERTIES, 0,
                                                     0 };

    // Query extension
    if (!is_extension_available(device, "cl_khr_create_command_queue"))
    {
        log_info("extension cl_khr_create_command_queue is not supported.\n");
        return 0;
    }

    error = create_single_kernel_helper(context, &program, &kernel, 1, queue_test_kernel, "vec_cpy");
    test_error(error, "create_single_kernel_helper failed");

    log_info("Queue property NULL. Testing ... \n");
    error = enqueue_kernel(context, NULL, device, kernel, (size_t)num_elements);
    test_error(error, "enqueue_kernel failed");

    error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(device_props), &device_props, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

    if (device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
        log_info("Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE supported. Testing ... \n");
        queue_prop_def[1] = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        error = enqueue_kernel(context, queue_prop_def, device, kernel,
                               (size_t)num_elements);
        test_error(error, "enqueue_kernel failed");
    } else
    {
        log_info("Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE not supported \n");
    }

    if (device_props & CL_QUEUE_PROFILING_ENABLE)
    {
        log_info("Queue property CL_QUEUE_PROFILING_ENABLE supported. Testing ... \n");
        queue_prop_def[1] = CL_QUEUE_PROFILING_ENABLE;
        error = enqueue_kernel(context, queue_prop_def, device, kernel,
                               (size_t)num_elements);
        test_error(error, "enqueue_kernel failed");
    } else
    {
        log_info("Queue property CL_QUEUE_PROFILING_ENABLE not supported \n");
    }

    if (device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE && device_props & CL_QUEUE_PROFILING_ENABLE)
    {
        log_info("Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE & CL_QUEUE_PROFILING_ENABLE supported. Testing ... \n");
        queue_prop_def[1] = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE;
        error = enqueue_kernel(context, queue_prop_def, device, kernel,
                               (size_t)num_elements);
        test_error(error, "enqueue_kernel failed");
    }
    else
    {
        log_info("Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE or CL_QUEUE_PROFILING_ENABLE not supported \n");
    }

    return 0;
}

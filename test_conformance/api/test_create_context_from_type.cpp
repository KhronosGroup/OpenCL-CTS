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
#include "testBase.h"
#include "harness/testHarness.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#include "harness/conversions.h"
#include <bitset>

REGISTER_TEST(create_context_from_type)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper    streams[2];
    clContextWrapper context_to_test;
    clCommandQueueWrapper queue_to_test;
    size_t    threads[1], localThreads[1];
    cl_float inputData[10];
    cl_int outputData[10];
    int i;
    RandomSeed seed( gRandomSeed );

    const char *sample_single_test_kernel[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n" };

    cl_device_type type;
    error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed\n");

    cl_platform_id platform;
    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                            &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed\n");

    cl_context_properties properties[3] = {
      (cl_context_properties)CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform,
      0
    };

    context_to_test = clCreateContextFromType(properties, type, notify_callback, NULL, &error);
    test_error(error, "clCreateContextFromType failed");
    if (context_to_test == NULL) {
        log_error("clCreateContextFromType returned NULL, but error was CL_SUCCESS.");
        return -1;
    }

    queue_to_test = clCreateCommandQueue(context_to_test, device, 0, &error);
    test_error(error, "clCreateCommandQueue failed");
    if (queue_to_test == NULL) {
        log_error("clCreateCommandQueue returned NULL, but error was CL_SUCCESS.");
        return -1;
    }

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context_to_test, &program, &kernel, 1, sample_single_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context_to_test, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context_to_test, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Write some test data */
    memset( outputData, 0, sizeof( outputData ) );

    for (i=0; i<10; i++)
        inputData[i] = get_random_float(-(float) 0x7fffffff, (float) 0x7fffffff, seed);

    error = clEnqueueWriteBuffer(queue_to_test, streams[0], CL_TRUE, 0, sizeof(cl_float)*10, (void *)inputData, 0, NULL, NULL);
    test_error( error, "Unable to set testing kernel data" );

    /* Test setting the arguments by index manually */
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = (size_t)10;

    error = get_max_common_work_group_size( context_to_test, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue_to_test, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer( queue_to_test, streams[1], CL_TRUE, 0, sizeof(cl_int)*10, (void *)outputData, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );

    for (i=0; i<10; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

  return 0;
}

REGISTER_TEST(create_context_from_type_device_type_all)
{
    cl_device_type type;
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed\n");

    std::bitset<sizeof(cl_device_type)> type_bits(type);

    if (type_bits.count() > 1 || (type & CL_DEVICE_TYPE_DEFAULT))
    {
        log_error("clGetDeviceInfo(CL_DEVICE_TYPE) must report a single device "
                  "type, which must not be CL_DEVICE_TYPE_DEFAULT or "
                  "CL_DEVICE_TYPE_ALL.\n");
        return -1;
    }
    cl_platform_id platform;
    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                            &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed\n");

    cl_context_properties properties[3] = {
        (cl_context_properties)CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform, 0
    };

    clContextWrapper context_to_test = clCreateContextFromType(
        properties, CL_DEVICE_TYPE_ALL, notify_callback, NULL, &error);
    test_error(error, "clCreateContextFromType failed");

    cl_uint num_devices = 0;
    error = clGetContextInfo(context_to_test, CL_CONTEXT_NUM_DEVICES,
                             sizeof(cl_uint), &num_devices, nullptr);
    test_error(error, "clGetContextInfo CL_CONTEXT_NUM_DEVICES failed\n");

    test_assert_error(num_devices >= 1,
                      "Context must contain at least one device\n");

    return 0;
}

REGISTER_TEST(create_context_from_type_device_type_default)
{
    cl_device_type type;
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed\n");

    std::bitset<sizeof(cl_device_type)> type_bits(type);

    if (type_bits.count() > 1 || (type & CL_DEVICE_TYPE_DEFAULT))
    {
        log_error("clGetDeviceInfo(CL_DEVICE_TYPE) must report a single device "
                  "type, which must not be CL_DEVICE_TYPE_DEFAULT or "
                  "CL_DEVICE_TYPE_ALL.\n");
        return -1;
    }
    cl_platform_id platform;
    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                            &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed\n");

    cl_context_properties properties[3] = {
        (cl_context_properties)CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform, 0
    };

    clContextWrapper context_to_test = clCreateContextFromType(
        properties, CL_DEVICE_TYPE_DEFAULT, notify_callback, NULL, &error);
    test_error(error, "clCreateContextFromType failed");

    cl_uint num_devices = 0;
    error = clGetContextInfo(context_to_test, CL_CONTEXT_NUM_DEVICES,
                             sizeof(cl_uint), &num_devices, nullptr);
    test_error(error, "clGetContextInfo CL_CONTEXT_NUM_DEVICES failed\n");

    std::vector<cl_device_id> devices(num_devices);
    error = clGetContextInfo(context_to_test, CL_CONTEXT_DEVICES,
                             num_devices * sizeof(cl_device_id), devices.data(),
                             nullptr);
    test_error(error, "clGetContextInfo CL_CONTEXT_DEVICES failed\n");

    test_assert_error(devices.size() == 1,
                      "Context must contain exactly one device\n");

    cl_uint num_platform_devices;

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL,
                           &num_platform_devices);
    test_error(error, "clGetDeviceIDs failed.\n");
    test_assert_error(num_platform_devices == 1,
                      "clGetDeviceIDs must return exactly one device\n");

    std::vector<cl_device_id> platform_devices(num_platform_devices);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT,
                           num_platform_devices, platform_devices.data(), NULL);
    test_error(error, "clGetDeviceIDs failed.\n");

    test_assert_error(platform_devices[0] == devices[0],
                      "device in the context must be equivalent to device "
                      "returned by clGetDeviceIDs\n");

    return 0;
}

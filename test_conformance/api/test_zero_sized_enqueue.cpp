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
#include "harness/typeWrappers.h"
#include "harness/conversions.h"

const char* zero_sized_enqueue_test_kernel[] = {
    "__kernel void foo_kernel(__global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = 1;\n"
    "\n"
    "}\n"
};

const int bufSize = 128;

cl_int test_zero_sized_enqueue_and_test_output_buffer(cl_command_queue queue, clKernelWrapper& kernel, clMemWrapper& buf, size_t dim, size_t ndrange[])
{
    cl_int error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, ndrange, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    clFinish(queue);

    // check output buffer has not changed.
    int* output = reinterpret_cast<int*>(clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * bufSize, 0, NULL, NULL, &error));
    if (error != CL_SUCCESS)
    {
        return error;
    }

    for (int i = 0; i < bufSize; ++i)
    {
        if (output[i] != 0)
        {
            log_error( "ERROR: output buffer value has changed.\n" );
            return CL_INVALID_OPERATION;
        }
    }

    return clEnqueueUnmapMemObject(queue, buf, output, 0, NULL, NULL);
}

int test_zero_sized_enqueue_helper(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper output_stream;
    size_t    ndrange1 = 0;
    size_t    ndrange20[2] = {0, 0};
    size_t    ndrange21[2] = {1, 0};
    size_t    ndrange22[2] = {0, 1};

    size_t    ndrange30[3] = {0, 0, 0};
    size_t    ndrange31[3] = {1, 0, 0};
    size_t    ndrange32[3] = {0, 1, 0};
    size_t    ndrange33[3] = {0, 0, 1};
    size_t    ndrange34[3] = {0, 1, 1};
    size_t    ndrange35[3] = {1, 0, 1};
    size_t    ndrange36[3] = {1, 1, 0};

    output_stream =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       bufSize * sizeof(int), NULL, &error);

    // Initialise output buffer.
    int output_buffer_data = 0;
    error = clEnqueueFillBuffer(queue, output_stream, &output_buffer_data,
                                sizeof(int), 0, sizeof(int) * bufSize, 0, NULL,
                                NULL);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, zero_sized_enqueue_test_kernel, "foo_kernel" ) != 0 )
    {
        return -1;
    }

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_stream);
    test_error( error, "clSetKernelArg failed." );

    // Simple API return code tests for 1D, 2D and 3D zero sized ND range.
    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 1, &ndrange1);
    test_error( error, "1D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 2, ndrange20);
    test_error( error, "2D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 2, ndrange21);
    test_error( error, "2D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 2, ndrange22);
    test_error( error, "2D zero sized kernel enqueue failed." );


    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange30);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange31);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange32);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange33);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange34);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange35);
    test_error( error, "3D zero sized kernel enqueue failed." );

    error = test_zero_sized_enqueue_and_test_output_buffer(
        queue, kernel, output_stream, 3, ndrange36);
    test_error( error, "3D zero sized kernel enqueue failed." );

    // Verify zero-sized ND range kernel still satisfy event wait list and correct event object
    // is returned
    clEventWrapper ev = NULL;
    clEventWrapper user_ev = clCreateUserEvent(context, &error);
    test_error( error, "user event creation failed." );
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, ndrange30, NULL, 1, &user_ev, &ev);
    test_error( error, "3D zero sized kernel enqueue failed." );
    if (ev == NULL)
    {
        log_error( "ERROR: failed to create an event object\n" );
        return -1;
    }

    cl_int sta;
    error = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &sta, NULL);
    test_error( error, "Failed to get event status.");

    if (sta != CL_QUEUED && sta != CL_SUBMITTED)
    {
        log_error( "ERROR: incorrect zero sized kernel enqueue event status.\n" );
        return -1;
    }

    // now unblock zero-sized enqueue
    error = clSetUserEventStatus(user_ev, CL_COMPLETE);
    test_error( error, "Failed to set user event status.");

    clFinish(queue);

    // now check zero sized enqueue event status
    error = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &sta, NULL);
    test_error( error, "Failed to get event status.");

    if (sta != CL_COMPLETE)
    {
        log_error( "ERROR: incorrect zero sized kernel enqueue event status.\n" );
        return -1;
    }

    return 0;
}


REGISTER_TEST_VERSION(zero_sized_enqueue, Version(2, 1))
{
    int res =
        test_zero_sized_enqueue_helper(device, context, queue, num_elements);
    if (res != 0)
    {
        return res;
    }

    // now test out of order queue
    cl_command_queue_properties props;
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                        sizeof(cl_command_queue_properties), &props, NULL);
    test_error( error, "clGetDeviceInfo failed.");

    if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
        // test out of order queue
        cl_queue_properties queue_prop_def[] =
        {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            0
        };

        clCommandQueueWrapper ooqueue = clCreateCommandQueueWithProperties(
            context, device, queue_prop_def, &error);
        test_error( error, "clCreateCommandQueueWithProperties failed.");

        res = test_zero_sized_enqueue_helper(device, context, ooqueue,
                                             num_elements);
    }

    return res;
}

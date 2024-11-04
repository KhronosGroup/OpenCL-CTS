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

const char *write_kernels[] = {
    "__kernel void write_up(__global int *dst, int length)\n"
    "{\n"
    "\n"
    " dst[get_global_id(0)] *= 2;\n"
    "\n"
    "}\n"
    "__kernel void write_down(__global int *dst, int length)\n"
    "{\n"
    "\n"
    " dst[get_global_id(0)]--;\n"
    "\n"
    "}\n"
};

#define TEST_SIZE 10000
#define TEST_COUNT 10
#define RANDOMIZE 1
#define DEBUG_OUT 0

/*
 Tests event dependencies by running two kernels that use the same buffer.
 If two_queues is set they are run in separate queues.
 If test_enqueue_wait_for_events is set then clEnqueueWaitForEvent is called
 between them. If test_barrier is set then clEnqueueBarrier is called between
 them (only for single queue). If neither are set, nothing is done to prevent
 them from executing in the wrong order. This can be used for verification.
 */
int test_event_enqueue_wait_for_events_run_test(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements, int two_queues, int two_devices,
    int test_enqueue_wait_for_events, int test_barrier, int use_waitlist,
    int use_marker)
{
    cl_int error = CL_SUCCESS;
    size_t threads[3] = { TEST_SIZE, 0, 0 };
    int i, loop_count, expected_value, failed;
    int expected_if_only_queue[2];
    int max_count = TEST_SIZE;

    cl_platform_id platform;
    cl_command_queue
        queues[2]; // Not a wrapper so we don't autorelease if they are the same
    clCommandQueueWrapper queueWrappers[2]; // If they are different, we use the
                                            // wrapper so it will auto release
    clContextWrapper context_to_use;
    clMemWrapper data;
    clProgramWrapper program;
    clKernelWrapper kernel1[TEST_COUNT], kernel2[TEST_COUNT];

    if (test_enqueue_wait_for_events)
        log_info("\tTesting with clEnqueueBarrierWithWaitList as barrier "
                 "function.\n");
    if (test_barrier)
        log_info("\tTesting with clEnqueueBarrierWithWaitList as barrier "
                 "function.\n");
    if (use_waitlist)
        log_info(
            "\tTesting with waitlist-based depenednecies between kernels.\n");
    if (use_marker)
        log_info("\tTesting with clEnqueueMarker as a barrier function.\n");
    if (test_barrier && (two_queues || two_devices))
    {
        log_error("\tTest requested with clEnqueueBarrier across two queues. "
                  "This is not a valid combination.\n");
        return -1;
    }

    error = clGetPlatformIDs(1, &platform, NULL);
    test_error(error, "clGetPlatformIDs failed.");

    // If we are to use two devices, then get them and create a context with
    // both.
    cl_device_id *two_device_ids;
    if (two_devices)
    {
        two_device_ids = (cl_device_id *)malloc(sizeof(cl_device_id) * 2);
        cl_uint number_returned;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 2, two_device_ids,
                               &number_returned);
        test_error(error, "clGetDeviceIDs for CL_DEVICE_TYPE_ALL failed.");
        if (number_returned < 2)
        {
            log_info("Failed to obtain two devices. Test can not run.\n");
            free(two_device_ids);
            return 0;
        }

        for (i = 0; i < 2; i++)
        {
            cl_device_type type;
            error = clGetDeviceInfo(two_device_ids[i], CL_DEVICE_TYPE,
                                    sizeof(cl_device_type), &type, NULL);
            test_error(error, "clGetDeviceInfo failed.");
            if (type & CL_DEVICE_TYPE_CPU)
                log_info("\tDevice %d is CL_DEVICE_TYPE_CPU.\n", i);
            if (type & CL_DEVICE_TYPE_GPU)
                log_info("\tDevice %d is CL_DEVICE_TYPE_GPU.\n", i);
            if (type & CL_DEVICE_TYPE_ACCELERATOR)
                log_info("\tDevice %d is CL_DEVICE_TYPE_ACCELERATOR.\n", i);
            if (type & CL_DEVICE_TYPE_DEFAULT)
                log_info("\tDevice %d is CL_DEVICE_TYPE_DEFAULT.\n", i);
        }

        context_to_use = clCreateContext(NULL, 2, two_device_ids,
                                         notify_callback, NULL, &error);
        test_error(error, "clCreateContext failed for two devices.");

        log_info("\tTesting with two devices.\n");
    }
    else
    {
        context_to_use =
            clCreateContext(NULL, 1, &deviceID, NULL, NULL, &error);
        test_error(error, "clCreateContext failed for one device.");

        log_info("\tTesting with one device.\n");
    }

    // If we are using two queues then create them
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (two_queues)
    {
        // Get a second queue
        if (two_devices)
        {
            if (!checkDeviceForQueueSupport(two_device_ids[0], props)
                || !checkDeviceForQueueSupport(two_device_ids[1], props))
            {
                log_info(
                    "WARNING: One or more device for multi-device test does "
                    "not support out-of-order exec mode; skipping test.\n");
                return -1942;
            }

            queueWrappers[0] = clCreateCommandQueue(
                context_to_use, two_device_ids[0], props, &error);
            test_error(
                error,
                "clCreateCommandQueue for first queue on first device failed.");
            queueWrappers[1] = clCreateCommandQueue(
                context_to_use, two_device_ids[1], props, &error);
            test_error(error,
                       "clCreateCommandQueue for second queue on second device "
                       "failed.");
        }
        else
        {
            // Single device has already been checked for out-of-order exec
            // support
            queueWrappers[0] =
                clCreateCommandQueue(context_to_use, deviceID, props, &error);
            test_error(error, "clCreateCommandQueue for first queue failed.");
            queueWrappers[1] =
                clCreateCommandQueue(context_to_use, deviceID, props, &error);
            test_error(error, "clCreateCommandQueue for second queue failed.");
        }
        // Ugly hack to make sure we only have the wrapper auto-release if they
        // are different queues
        queues[0] = queueWrappers[0];
        queues[1] = queueWrappers[1];
        log_info("\tTesting with two queues.\n");
    }
    else
    {
        // (Note: single device has already been checked for out-of-order exec
        // support) Otherwise create one queue and have the second one be the
        // same
        queueWrappers[0] =
            clCreateCommandQueue(context_to_use, deviceID, props, &error);
        test_error(error, "clCreateCommandQueue for first queue failed.");
        queues[0] = queueWrappers[0];
        queues[1] = (cl_command_queue)queues[0];
        log_info("\tTesting with one queue.\n");
    }


    // Setup - create a buffer and the two kernels
    data = clCreateBuffer(context_to_use, CL_MEM_READ_WRITE,
                          TEST_SIZE * sizeof(cl_int), NULL, &error);
    test_error(error, "clCreateBuffer failed");


    // Initialize the values to zero
    cl_int *values = (cl_int *)malloc(TEST_SIZE * sizeof(cl_int));
    for (i = 0; i < (int)TEST_SIZE; i++) values[i] = 0;
    error =
        clEnqueueWriteBuffer(queues[0], data, CL_TRUE, 0,
                             TEST_SIZE * sizeof(cl_int), values, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");
    expected_value = 0;

    // Build the kernels
    if (create_single_kernel_helper(context_to_use, &program, &kernel1[0], 1,
                                    write_kernels, "write_up"))
        return -1;

    error = clSetKernelArg(kernel1[0], 0, sizeof(data), &data);
    error |= clSetKernelArg(kernel1[0], 1, sizeof(max_count), &max_count);
    test_error(error, "clSetKernelArg 1 failed");

    for (i = 1; i < TEST_COUNT; i++)
    {
        kernel1[i] = clCreateKernel(program, "write_up", &error);
        test_error(error, "clCreateKernel 1 failed");

        error = clSetKernelArg(kernel1[i], 0, sizeof(data), &data);
        error |= clSetKernelArg(kernel1[i], 1, sizeof(max_count), &max_count);
        test_error(error, "clSetKernelArg 1 failed");
    }

    for (i = 0; i < TEST_COUNT; i++)
    {
        kernel2[i] = clCreateKernel(program, "write_down", &error);
        test_error(error, "clCreateKernel 2 failed");

        error = clSetKernelArg(kernel2[i], 0, sizeof(data), &data);
        error |= clSetKernelArg(kernel2[i], 1, sizeof(max_count), &max_count);
        test_error(error, "clSetKernelArg 2 failed");
    }

    // Execution - run the first kernel, then enqueue the wait on the events,
    // then the second kernel If clEnqueueBarrierWithWaitList works, the buffer
    // will be filled with 1s, then multiplied by 4s, then incremented to 5s,
    // repeatedly. Otherwise the values may be 2s (if the first one doesn't
    // work) or 8s (if the second one doesn't work).
    if (RANDOMIZE)
        log_info("Queues chosen randomly for each kernel execution.\n");
    else
        log_info("Queues chosen alternatily for each kernel execution.\n");

    clEventWrapper pre_loop_event;
    clEventWrapper last_loop_event;

    for (i = 0; i < (int)TEST_SIZE; i++) values[i] = 1;
    error = clEnqueueWriteBuffer(queues[0], data, CL_FALSE, 0,
                                 TEST_SIZE * sizeof(cl_int), values, 0, NULL,
                                 &pre_loop_event);
    test_error(error, "clEnqueueWriteBuffer 2 failed");
    expected_value = 1;
    expected_if_only_queue[0] = 1;
    expected_if_only_queue[1] = 1;

    int queue_to_use = 1;
    if (test_enqueue_wait_for_events)
    {
        error = clEnqueueBarrierWithWaitList(queues[queue_to_use], 1,
                                             &pre_loop_event, NULL);
        test_error(error, "Unable to queue wait for events");
    }
    else if (test_barrier)
    {
        error =
            clEnqueueBarrierWithWaitList(queues[queue_to_use], 0, NULL, NULL);
        test_error(error, "Unable to queue barrier");
    }

    for (loop_count = 0; loop_count < TEST_COUNT; loop_count++)
    {
        int event_count = 0;
        clEventWrapper first_dependency =
            (loop_count == 0) ? pre_loop_event : last_loop_event;
        clEventWrapper
            event[5]; // A maximum of 5 events are created in the loop
        event[event_count] = first_dependency;

        // Execute kernel 1
        event_count++;
        if (use_waitlist | use_marker)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueNDRangeKernel(queues[%d], kernel1[%d], 1, "
                         "NULL, threads, NULL, 1, &event[%d], &event[%d])\n",
                         queue_to_use, loop_count, event_count - 1,
                         event_count);
            error = clEnqueueNDRangeKernel(
                queues[queue_to_use], kernel1[loop_count], 1, NULL, threads,
                NULL, 1, &event[event_count - 1], &event[event_count]);
        }
        else
        {
            if (DEBUG_OUT)
                log_info("clEnqueueNDRangeKernel(queues[%d], kernel1[%d], 1, "
                         "NULL, threads, NULL, 0, NULL, &event[%d])\n",
                         queue_to_use, loop_count, event_count);
            error = clEnqueueNDRangeKernel(
                queues[queue_to_use], kernel1[loop_count], 1, NULL, threads,
                NULL, 0, NULL, &event[event_count]);
        }
        if (error)
        {
            log_info("\tLoop count %d\n", loop_count);
            print_error(error, "clEnqueueNDRangeKernel for kernel 1 failed");
            return error;
        }
        expected_value *= 2;
        expected_if_only_queue[queue_to_use] *= 2;

        // If we are using a marker, it needs to go in the same queue
        if (use_marker)
        {
            event_count++;
            if (DEBUG_OUT)
                log_info("clEnqueueMarker(queues[%d], event[%d])\n",
                         queue_to_use, event_count);

#ifdef CL_VERSION_1_2
            error = clEnqueueMarkerWithWaitList(queues[queue_to_use], 0, NULL,
                                                &event[event_count]);
#else
            error = clEnqueueMarker(queues[queue_to_use], &event[event_count]);
#endif
        }

        // Pick the next queue to run
        if (RANDOMIZE)
            queue_to_use = rand() % 2;
        else
            queue_to_use = (queue_to_use + 1) % 2;

        // Put in a barrier if requested
        if (test_enqueue_wait_for_events)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueBarrierWithWaitList(queues[%d], 1, "
                         "&event[%d], NULL)\n",
                         queue_to_use, event_count);
            error = clEnqueueBarrierWithWaitList(queues[queue_to_use], 1,
                                                 &event[event_count], NULL);
            test_error(error, "Unable to queue wait for events");
        }
        else if (test_barrier)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueBarrierWithWaitList(queues[%d])\n",
                         queue_to_use);
            error = clEnqueueBarrierWithWaitList(queues[queue_to_use], 0, NULL,
                                                 NULL);
            test_error(error, "Unable to queue barrier");
        }

        // Execute Kernel 2
        event_count++;
        if (use_waitlist | use_marker)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueNDRangeKernel(queues[%d], kernel2[%d], 1, "
                         "NULL, threads, NULL, 1, &event[%d], &event[%d])\n",
                         queue_to_use, loop_count, event_count - 1,
                         event_count);
            error = clEnqueueNDRangeKernel(
                queues[queue_to_use], kernel2[loop_count], 1, NULL, threads,
                NULL, 1, &event[event_count - 1], &event[event_count]);
        }
        else
        {
            if (DEBUG_OUT)
                log_info("clEnqueueNDRangeKernel(queues[%d], kernel2[%d], 1, "
                         "NULL, threads, NULL, 0, NULL, &event[%d])\n",
                         queue_to_use, loop_count, event_count);
            error = clEnqueueNDRangeKernel(
                queues[queue_to_use], kernel2[loop_count], 1, NULL, threads,
                NULL, 0, NULL, &event[event_count]);
        }
        if (error)
        {
            log_info("\tLoop count %d\n", loop_count);
            print_error(error, "clEnqueueNDRangeKernel for kernel 2 failed");
            return error;
        }
        expected_value--;
        expected_if_only_queue[queue_to_use]--;

        // If we are using a marker, it needs to go in the same queue
        if (use_marker)
        {
            event_count++;
            if (DEBUG_OUT)
                log_info("clEnqueueMarker(queues[%d], event[%d])\n",
                         queue_to_use, event_count);

#ifdef CL_VERSION_1_2
            error = clEnqueueMarkerWithWaitList(queues[queue_to_use], 0, NULL,
                                                &event[event_count]);
#else
            error = clEnqueueMarker(queues[queue_to_use], &event[event_count]);
#endif
        }

        // Pick the next queue to run
        if (RANDOMIZE)
            queue_to_use = rand() % 2;
        else
            queue_to_use = (queue_to_use + 1) % 2;

        // Put in a barrier if requested
        if (test_enqueue_wait_for_events)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueBarrierWithWaitList(queues[%d], 1, "
                         "&event[%d], NULL)\n",
                         queue_to_use, event_count);
            error = clEnqueueBarrierWithWaitList(queues[queue_to_use], 1,
                                                 &event[event_count], NULL);
            test_error(error, "Unable to queue wait for events");
        }
        else if (test_barrier)
        {
            if (DEBUG_OUT)
                log_info("clEnqueueBarrierWithWaitList(queues[%d])\n",
                         queue_to_use);
            error = clEnqueueBarrierWithWaitList(queues[queue_to_use], 0, NULL,
                                                 NULL);
            test_error(error, "Unable to queue barrier");
        }
        last_loop_event = event[event_count];
    }

    // Now finish up everything
    if (two_queues)
    {
        error = clFlush(queues[1]);
        test_error(error, "clFlush[1] failed");
    }

    error = clEnqueueReadBuffer(queues[0], data, CL_TRUE, 0,
                                TEST_SIZE * sizeof(cl_int), values, 1,
                                &last_loop_event, NULL);

    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queues[0]);
    test_error(error, "clFinish(queues[0]) failed");
    if (two_queues)
    {
        error = clFinish(queues[1]);
        test_error(error, "clFinish(queues[1]) failed");
    }

    failed = 0;
    for (i = 0; i < (int)TEST_SIZE; i++)
        if (values[i] != expected_value)
        {
            failed = 1;
            log_info("\tvalues[%d] = %d, expected %d (If only queue 1 accessed "
                     "memory: %d only queue 2 accessed memory: %d)\n",
                     i, values[i], expected_value, expected_if_only_queue[0],
                     expected_if_only_queue[1]);
            break;
        }

    free(values);
    if (two_devices) free(two_device_ids);

    return failed;
}

int test(cl_device_id deviceID, cl_context context, cl_command_queue queue,
         int num_elements, int two_queues, int two_devices,
         int test_enqueue_wait_for_events, int test_barrier, int use_waitlists,
         int use_marker)
{
    if (!checkDeviceForQueueSupport(deviceID,
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
    {
        log_info("WARNING: Device does not support out-of-order exec mode; "
                 "skipping test.\n");
        return 0;
    }

    log_info("Running test for baseline results to determine if out-of-order "
             "execution can be detected...\n");
    int baseline_results = test_event_enqueue_wait_for_events_run_test(
        deviceID, context, queue, num_elements, two_queues, two_devices, 0, 0,
        0, 0);
    if (baseline_results == 0)
    {
        if (test_enqueue_wait_for_events)
            log_info(
                "WARNING: could not detect any out-of-order execution without "
                "using clEnqueueBarrierWithWaitList, so this test is not a "
                "valid test of out-of-order event dependencies.\n");
        if (test_barrier)
            log_info(
                "WARNING: could not detect any out-of-order execution without "
                "using clEnqueueBarrierWithWaitList, so this test is not a "
                "valid test of out-of-order event dependencies.\n");
        if (use_waitlists)
            log_info("WARNING: could not detect any out-of-order execution "
                     "without using waitlists, so this test is not a valid "
                     "test of out-of-order event dependencies.\n");
        if (use_marker)
            log_info("WARNING: could not detect any out-of-order execution "
                     "without using clEnqueueMarker, so this test is not a "
                     "valid test of out-of-order event dependencies.\n");
    }
    else if (baseline_results == 1)
    {
        if (test_enqueue_wait_for_events)
            log_info("Detected incorrect execution (possibly out-of-order) "
                     "without clEnqueueBarrierWithWaitList. Test can be a "
                     "valid test of out-of-order event dependencies.\n");
        if (test_barrier)
            log_info("Detected incorrect execution (possibly out-of-order) "
                     "without clEnqueueBarrierWithWaitList. Test can be a "
                     "valid test of out-of-order event dependencies.\n");
        if (use_waitlists)
            log_info("Detected incorrect execution (possibly out-of-order) "
                     "without waitlists. Test can be a valid test of "
                     "out-of-order event dependencies.\n");
        if (use_marker)
            log_info("Detected incorrect execution (possibly out-of-order) "
                     "without clEnqueueMarker. Test can be a valid test of "
                     "out-of-order event dependencies.\n");
    }
    else if (baseline_results == -1942)
    {
        // Just ignore and return (out-of-order exec mode not supported)
        return 0;
    }
    else
    {
        print_error(baseline_results, "Baseline run failed");
        return baseline_results;
    }
    log_info("Running test for actual results...\n");
    return test_event_enqueue_wait_for_events_run_test(
        deviceID, context, queue, num_elements, two_queues, two_devices,
        test_enqueue_wait_for_events, test_barrier, use_waitlists, use_marker);
}


int test_out_of_order_event_waitlist_single_queue(cl_device_id deviceID,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements)
{
    int two_queues = 0;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 1;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}

int test_out_of_order_event_waitlist_multi_queue(cl_device_id deviceID,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements)
{
    int two_queues = 1;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 1;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}

int test_out_of_order_event_waitlist_multi_queue_multi_device(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int two_queues = 1;
    int two_devices = 1;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 1;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}


int test_out_of_order_event_enqueue_wait_for_events_single_queue(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int two_queues = 0;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 1;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}

int test_out_of_order_event_enqueue_wait_for_events_multi_queue(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int two_queues = 1;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 1;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}


int test_out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int two_queues = 1;
    int two_devices = 1;
    int test_enqueue_wait_for_events = 1;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}


int test_out_of_order_event_enqueue_barrier_single_queue(cl_device_id deviceID,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements)
{
    int two_queues = 0;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 1;
    int use_waitlists = 0;
    int use_marker = 0;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}


int test_out_of_order_event_enqueue_marker_single_queue(cl_device_id deviceID,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements)
{
    int two_queues = 0;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 1;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}

int test_out_of_order_event_enqueue_marker_multi_queue(cl_device_id deviceID,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    int two_queues = 1;
    int two_devices = 0;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 1;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}


int test_out_of_order_event_enqueue_marker_multi_queue_multi_device(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int two_queues = 1;
    int two_devices = 1;
    int test_enqueue_wait_for_events = 0;
    int test_barrier = 0;
    int use_waitlists = 0;
    int use_marker = 1;
    return test(deviceID, context, queue, num_elements, two_queues, two_devices,
                test_enqueue_wait_for_events, test_barrier, use_waitlists,
                use_marker);
}

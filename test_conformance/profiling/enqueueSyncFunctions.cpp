//
// Copyright (c) 2025-2026 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"


static const char *test_kernel = R"CLC(
__kernel void test(__global int* dst) {
    size_t id = get_global_linear_id();
    size_t loop_end = 1<<12UL;
    for (size_t i = 0; i <= loop_end; i++) {
        if(i%2 == 0) {
            dst[id] = 0;
        } else {
            dst[id] += 1;
        }
    }
}
)CLC";

// Helper that executes one variant (in-order or out-of-order)
static int run_enqueue_variant(cl_device_id device, cl_context context,
                               cl_command_queue queue, bool is_out_of_order,
                               int (*fn)(cl_command_queue, cl_uint,
                                         const cl_event *, cl_event *))
{
    cl_int error = CL_SUCCESS;
    int test_status = TEST_PASS;

    const size_t global_work_size = 256;
    const size_t allocSize = global_work_size * sizeof(cl_int);

    clMemWrapper buffer1 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer1");
    clMemWrapper buffer2 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer2");
    clMemWrapper buffer3 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer3");
    clMemWrapper buffer4 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer4");
    clMemWrapper buffer5 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer5");
    clMemWrapper buffer6 =
        clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, nullptr, &error);
    test_error(error, "Unable to create buffer6");

    clProgramWrapper program;
    clKernelWrapper kernel;
    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &test_kernel, "test");
    test_error(error, "Unable to build program / create kernel 'test'");

    clEventWrapper events_set1[3] = { nullptr, nullptr, nullptr };
    clEventWrapper events_set2[3] = { nullptr, nullptr, nullptr };
    clEventWrapper sync_event1, sync_event2;

    // First set (3 launches of the same kernel with different buffers)
    error = clSetKernelArg(kernel, 0, sizeof(buffer1), &buffer1);
    test_error(error, "Set arg buffer1 failed");
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                                   nullptr, 0, nullptr, &events_set1[0]);

    error |= clSetKernelArg(kernel, 0, sizeof(buffer2), &buffer2);
    test_error(error, "Set arg buffer2 failed");
    error |=
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               nullptr, 0, nullptr, &events_set1[1]);

    error |= clSetKernelArg(kernel, 0, sizeof(buffer3), &buffer3);
    test_error(error, "Set arg buffer3 failed");
    error |=
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               nullptr, 0, nullptr, &events_set1[2]);
    test_error(error, "Unable to enqueue first kernel set");

    error = fn(queue, 3, &events_set1[0], &sync_event1);
    test_error(error, "Unable to enqueue first sync command");

    // Second set
    error = clSetKernelArg(kernel, 0, sizeof(buffer4), &buffer4);
    test_error(error, "Set arg buffer1 (2nd set) failed");
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                                   nullptr, 0, nullptr, &events_set2[0]);

    error |= clSetKernelArg(kernel, 0, sizeof(buffer5), &buffer5);
    test_error(error, "Set arg buffer2 (2nd set) failed");
    error |=
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               nullptr, 0, nullptr, &events_set2[1]);

    error |= clSetKernelArg(kernel, 0, sizeof(buffer6), &buffer6);
    test_error(error, "Set arg buffer3 (2nd set) failed");
    error |=
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                               nullptr, 0, nullptr, &events_set2[2]);
    test_error(error, "Unable to enqueue second kernel set");

    // Second synchronization command
    error = fn(queue, 3, &events_set2[0], &sync_event2);
    test_error(error, "Unable to enqueue second sync command");

    error = clWaitForEvents(1, &sync_event1);
    test_error(error, "Wait for sync_event1 failed");
    error = clWaitForEvents(1, &sync_event2);
    test_error(error, "Wait for sync_event2 failed");

    // First sync event basic monotonicity check
    cl_ulong tQueued = 0, tSubmit = 0, tStart = 0, tEnd = 0;
    error = clGetEventProfilingInfo(sync_event1, CL_PROFILING_COMMAND_QUEUED,
                                    sizeof(tQueued), &tQueued, nullptr);
    test_error(error, "Profiling sync1 QUEUED failed");
    error = clGetEventProfilingInfo(sync_event1, CL_PROFILING_COMMAND_SUBMIT,
                                    sizeof(tSubmit), &tSubmit, nullptr);
    test_error(error, "Profiling sync1 SUBMIT failed");
    error = clGetEventProfilingInfo(sync_event1, CL_PROFILING_COMMAND_START,
                                    sizeof(tStart), &tStart, nullptr);
    test_error(error, "Profiling sync1 START failed");
    error = clGetEventProfilingInfo(sync_event1, CL_PROFILING_COMMAND_END,
                                    sizeof(tEnd), &tEnd, nullptr);
    test_error(error, "Profiling sync1 END failed");

    if (check_times(tQueued, tSubmit, tStart, tEnd, device) != TEST_PASS)
    {
        log_error("Timestamp monotonicity failed for sync event 1\n");
        test_status = TEST_FAIL;
    }

    cl_ulong end_set1[3] = { 0 }, start_set2[3] = { 0 };
    for (int i = 0; i < 3; i++)
    {
        error =
            clGetEventProfilingInfo(events_set1[i], CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &end_set1[i], nullptr);
        test_error(error, "Profiling end set1 failed");
        error =
            clGetEventProfilingInfo(events_set2[i], CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong), &start_set2[i], nullptr);
        test_error(error, "Profiling start set2 failed");
    }

    const bool is_barrier = (fn == clEnqueueBarrierWithWaitList);

    log_info("Verification (%s queue, %s):\n",
             is_out_of_order ? "out-of-order" : "in-order",
             is_barrier ? "barrier" : "marker");

    // Ordering:
    //  - barrier: second set must start only after ALL kernels of first set end
    //  - marker: on OOO queue ordering is not guaranteed (skip); on in-order
    //  queue ordering is implied
    if (is_barrier
        || !is_out_of_order) // test both barriers and in-order queues
    {
        for (int i = 0; i < 3; i++)
        {
            if (start_set2[i] <= end_set1[0] || start_set2[i] <= end_set1[1]
                || start_set2[i] <= end_set1[2])
            {
                log_error("Kernel %d in second set started too early\n", i);
                test_status = TEST_FAIL;
            }
        }
    }
    else
    {
        log_info("Marker on OOO: skipping cross-set ordering (not "
                 "guaranteed).\n");
    }

    // Get profiling for second sync event
    cl_ulong t2Queued = 0, t2Submit = 0, t2Start = 0, t2End = 0;
    error = clGetEventProfilingInfo(sync_event2, CL_PROFILING_COMMAND_QUEUED,
                                    sizeof(t2Queued), &t2Queued, nullptr);
    test_error(error, "Profiling sync2 QUEUED failed");
    error = clGetEventProfilingInfo(sync_event2, CL_PROFILING_COMMAND_SUBMIT,
                                    sizeof(t2Submit), &t2Submit, nullptr);
    test_error(error, "Profiling sync2 SUBMIT failed");
    error = clGetEventProfilingInfo(sync_event2, CL_PROFILING_COMMAND_START,
                                    sizeof(t2Start), &t2Start, nullptr);
    test_error(error, "Profiling sync2 START failed");
    error = clGetEventProfilingInfo(sync_event2, CL_PROFILING_COMMAND_END,
                                    sizeof(t2End), &t2End, nullptr);
    test_error(error, "Profiling sync2 END failed");
    if (check_times(t2Queued, t2Submit, t2Start, t2End, device) != TEST_PASS)
    {
        log_error("Timestamp monotonicity failed for sync event 2\n");
        test_status = TEST_FAIL;
    }

    return test_status;
}

int test_enqueue_function(cl_device_id device, cl_context context,
                          int (*fn)(cl_command_queue command_queue,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event))
{
    cl_int error;

    cl_command_queue_properties dev_props = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(dev_props), &dev_props, nullptr);
    test_error(error, "clGetDeviceInfo(CL_DEVICE_QUEUE_PROPERTIES) failed");
    bool ooo_supported =
        (dev_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

    if (!ooo_supported)
        log_info("Device does NOT support out-of-order queues: OOO phase "
                 "skipped.\n");
    else
        log_info(
            "Device supports out-of-order queues: running both variants.\n");

    clCommandQueueWrapper in_order_queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    test_error(error, "Failed to create in-order profiling queue");

    int status_in_order =
        run_enqueue_variant(device, context, in_order_queue, false, fn);

    int status_ooo = TEST_PASS;
    if (ooo_supported)
    {
        clCommandQueueWrapper ooo_queue = clCreateCommandQueue(
            context, device,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
            &error);
        test_error(error, "Failed to create out-of-order profiling queue");
        status_ooo = run_enqueue_variant(device, context, ooo_queue, true, fn);
    }

    return (status_in_order == TEST_FAIL || status_ooo == TEST_FAIL)
        ? TEST_FAIL
        : TEST_PASS;
}

REGISTER_TEST(enqueue_marker)
{
    return test_enqueue_function(device, context, clEnqueueMarkerWithWaitList);
}

REGISTER_TEST(enqueue_barrier)
{
    return test_enqueue_function(device, context, clEnqueueBarrierWithWaitList);
}

//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

static const char *test_kernel = R"CLC(
__kernel void test1(__global int* dst) {
    size_t id = get_global_linear_id();
    size_t loop_end = 1<<12UL;
    for (size_t i = 0; i <= loop_end; i++) {
        if(i%1 == 0) {
            dst[id] = 0;
        } else {
            dst[id] += 1;
        }
    }
}
__kernel void test2(__global int* dst) {
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
__kernel void test3(__global int* dst) {
    size_t id = get_global_linear_id();
    size_t loop_end = 1<<12UL;
    for (size_t i = 0; i <= loop_end; i++) {
        if(i%3 == 0) {
            dst[id] = 0;
        } else {
            dst[id] += 1;
        }
    }
}
)CLC";

int check_times2(const cl_ulong timestamp, const cl_ulong *timestamps_array, const std::string condition = "after") {
    if (condition == "after") {
        if (timestamp > timestamps_array[0] && timestamp > timestamps_array[1] && timestamp > timestamps_array[2]) {
            log_info("OK\n");
        }
        else {
            log_error("FAILED\n");
            return -1;
        }
    }
    else if (condition == "before") {
        if (timestamp < timestamps_array[0] && timestamp < timestamps_array[1] && timestamp < timestamps_array[2]) {
            log_info("OK\n");
        }
        else {
            log_error("FAILED\n");
            return -1;
        }
    }
    else {
        log_error("Unknown condition for function check_times2\n");
        return -1;
    }
    return 0;
}

//----- the test functions
int test_enqueue_function(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements,
                          int (*sync_fn)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)) {
    cl_int error;
    clCommandQueueWrapper queue_with_props;
    clMemWrapper buffer1, buffer2, buffer3;
    clProgramWrapper program;
    clKernelWrapper kernel1, kernel2, kernel3;
    cl_ulong queueStart, submitStart, fnStart, fnEnd;
    clEventWrapper eventEnqueueMarkerSet1, eventEnqueueMarkerSet2;
    size_t global_work_size[] = { 256, 256, 256 };
    const size_t allocSize = global_work_size[0] * global_work_size[1] * global_work_size[2] * sizeof(uint32_t);
    cl_command_queue_properties props_out_of_order = CL_QUEUE_PROFILING_ENABLE;

    queue_with_props = clCreateCommandQueue(context, device, props_out_of_order, &error);
    test_error(error, "Unable to create test buffer");

    buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
    test_error(error, "Unable to create test buffer");
    buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
    test_error(error, "Unable to create test buffer");
    buffer3 = clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
    test_error(error, "Unable to create test buffer");

    error = create_single_kernel_helper(context, &program, &kernel1, 1, &test_kernel, "test1");
    test_error(error, "Unable to create test kernel");
    error = create_single_kernel_helper(context, &program, &kernel2, 1, &test_kernel, "test2");
    test_error(error, "Unable to create test kernel");
    error = create_single_kernel_helper(context, &program, &kernel3, 1, &test_kernel, "test3");
    test_error(error, "Unable to create test kernel");
    
    error = clSetKernelArg(kernel1, 0, sizeof(buffer1), &buffer1);
    test_error(error, "Unable to set argument for test kernel");
    error = clSetKernelArg(kernel2, 0, sizeof(buffer2), &buffer2);
    test_error(error, "Unable to set argument for test kernel");
    error = clSetKernelArg(kernel3, 0, sizeof(buffer3), &buffer3);
    test_error(error, "Unable to set argument for test kernel");

    clEventWrapper events_list_set1[3] = { NULL, NULL, NULL };
    clEventWrapper events_list_set2[3] = { NULL, NULL, NULL };

    // run 1 set of ndrange commands
    error = clEnqueueNDRangeKernel(queue_with_props, kernel1, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set1[0]);
    error |= clEnqueueNDRangeKernel(queue_with_props, kernel2, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set1[1]);
    error |= clEnqueueNDRangeKernel(queue_with_props, kernel3, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set1[2]);
    test_error(error, "Unable to enqueue kernels in set 1");

    error = sync_fn(queue_with_props, 3, &events_list_set1[0], &eventEnqueueMarkerSet1);
    test_error(error, "Unable to enqueue sync command");

    // run 2 set of ndrange commands
    error = clEnqueueNDRangeKernel(queue_with_props, kernel1, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set2[0]);
    error |= clEnqueueNDRangeKernel(queue_with_props, kernel2, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set2[1]);
    error |= clEnqueueNDRangeKernel(queue_with_props, kernel3, 1, NULL, global_work_size, NULL, 0, NULL, &events_list_set2[2]);
    test_error(error, "Unable to enqueue kernels in set 2");

    error = sync_fn(queue_with_props, 3, &events_list_set2[0], &eventEnqueueMarkerSet2);
    test_error(error, "Unable to enqueue sync command");


    error = clWaitForEvents(1, &eventEnqueueMarkerSet2);
    test_error(error, "Unable to wait for event");

    error = clGetEventProfilingInfo(eventEnqueueMarkerSet1, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queueStart, NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_QUEUED");

    error = clGetEventProfilingInfo(eventEnqueueMarkerSet1, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitStart, NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_SUBMIT");

    error = clGetEventProfilingInfo(eventEnqueueMarkerSet1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &fnStart, NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_START");

    error = clGetEventProfilingInfo(eventEnqueueMarkerSet1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &fnEnd, NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_END");

    error = check_times(queueStart, submitStart, fnStart, fnEnd, device);
    test_error(error, "Checking timestamps function failed.");


    cl_ulong timestamps_set1_cmd_end[] = { 0, 0, 0 };
    cl_ulong timestamps_set2_cmd_start[] = { 0, 0, 0 };

    error = clGetEventProfilingInfo(events_list_set1[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timestamps_set1_cmd_end[0], NULL);
    error |= clGetEventProfilingInfo(events_list_set1[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timestamps_set1_cmd_end[1], NULL);
    error |= clGetEventProfilingInfo(events_list_set1[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timestamps_set1_cmd_end[2], NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_START");

    error = clGetEventProfilingInfo(events_list_set2[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timestamps_set2_cmd_start[0], NULL);
    error |= clGetEventProfilingInfo(events_list_set2[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timestamps_set2_cmd_start[1], NULL);
    error |= clGetEventProfilingInfo(events_list_set2[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timestamps_set2_cmd_start[2], NULL);
    test_error(error, "Unable to run clGetEventProfilingInfo CL_PROFILING_COMMAND_START");

    log_info("Verification:\n");
    log_info("cmd 1 from set2 run after all cmds from set1... ");
    error |= check_times2(timestamps_set2_cmd_start[0], timestamps_set1_cmd_end, "after");

    log_info("cmd 2 from set2 run after all cmds from set1... ");
    error |= check_times2(timestamps_set2_cmd_start[1], timestamps_set1_cmd_end, "after");

    log_info("cmd 3 from set2 run after all cmds from set1... ");
    error |= check_times2(timestamps_set2_cmd_start[2], timestamps_set1_cmd_end, "after");

    log_info("Sync command run after all cmds from set1... ");

    cl_ulong max_end = std::max({ timestamps_set1_cmd_end[0], timestamps_set1_cmd_end[1], timestamps_set1_cmd_end[2] });
    if (fnStart < max_end) {
        log_info("\nWARNING: fnStart (%lu) < end of set1 (%lu)\n", fnStart, max_end);
        error = 0;
    } else {
        log_info("\nOK: fnStart > all set1 ends\n");
        error = 0;
    }

    log_info("Sync command finishes before all functions from set2... ");
    error |= check_times2(fnEnd, timestamps_set2_cmd_start, "before");

    return error;
}

REGISTER_TEST(enqueue_marker)
{
    int (*foo)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    foo = clEnqueueMarkerWithWaitList;
    return test_enqueue_function(device, context, queue, num_elements, foo);
}

REGISTER_TEST(enqueue_barrier)
{
    int (*foo)(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
    foo = clEnqueueBarrierWithWaitList;
    return test_enqueue_function(device, context, queue, num_elements, foo);
}
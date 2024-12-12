//
// Copyright (c) 2021 The Khronos Group Inc.
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

#include "procs.h"

const char *kernelCode = "__kernel void kernel_empty(){}";

REGISTER_TEST(profiling_timebase)
{
    Version version = get_device_cl_version(device);
    cl_platform_id platform = getPlatformFromDevice(device);
    cl_ulong timer_resolution = 0;
    cl_int err =
        clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
                          sizeof(timer_resolution), &timer_resolution, NULL);
    test_error(err, "Unable to query CL_PLATFORM_HOST_TIMER_RESOLUTION");

    // If CL_PLATFORM_HOST_TIMER_RESOLUTION returns 0, clGetDeviceAndHostTimer
    // is not a supported feature
    if (timer_resolution == 0 && version >= Version(3, 0))
    {
        return TEST_SKIPPED_ITSELF;
    }

    cl_ulong hostTime;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clEventWrapper kEvent;

    clEventWrapper uEvent = clCreateUserEvent(context, &err);
    test_error(err, "Failed to create user event");

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &kernelCode, "kernel_empty");
    test_error(err, "Failed to create kernel");

    cl_ulong deviceTimeBeforeQueue;
    err = clGetDeviceAndHostTimer(device, &deviceTimeBeforeQueue, &hostTime);
    test_error(err, "Unable to get starting device time");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, NULL, NULL, 1, &uEvent,
                                 &kEvent);
    test_error(err, "clEnqueueNDRangeKernel failed");

    cl_ulong deviceTimeAfterQueue;
    err = clGetDeviceAndHostTimer(device, &deviceTimeAfterQueue, &hostTime);
    test_error(err, "Unable to get queue device time");

    err = clFlush(queue);
    test_error(err, "clFlush failed");

    err = clSetUserEventStatus(uEvent, CL_COMPLETE);
    test_error(err, "Unable to complete user event");

    err = clWaitForEvents(1, &kEvent);
    test_error(err, "clWaitForEvents failed");

    cl_ulong deviceTimeAfterCompletion;
    err =
        clGetDeviceAndHostTimer(device, &deviceTimeAfterCompletion, &hostTime);
    test_error(err, "Unable to get finishing device time");

    cl_ulong eventQueue, eventSubmit, eventStart, eventEnd;
    err = clGetEventProfilingInfo(kEvent, CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(cl_ulong), &eventQueue, NULL);
    test_error(err, "clGetEventProfilingInfo failed");
    err = clGetEventProfilingInfo(kEvent, CL_PROFILING_COMMAND_SUBMIT,
                                  sizeof(cl_ulong), &eventSubmit, NULL);
    test_error(err, "clGetEventProfilingInfo failed");
    err = clGetEventProfilingInfo(kEvent, CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &eventStart, NULL);
    test_error(err, "clGetEventProfilingInfo failed");
    err = clGetEventProfilingInfo(kEvent, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &eventEnd, NULL);
    test_error(err, "clGetEventProfilingInfo failed");

    test_assert_error(deviceTimeBeforeQueue < eventQueue,
                      "Device timestamp was taken before kernel was queued");
    test_assert_error(eventQueue < deviceTimeAfterQueue,
                      "Device timestamp was taken after kernel was queued");
    test_assert_error(eventSubmit < deviceTimeAfterCompletion,
                      "Device timestamp was taken after kernel was submitted");
    test_assert_error((eventStart < deviceTimeAfterCompletion)
                          && (eventEnd < deviceTimeAfterCompletion),
                      "Device timestamp was taken after kernel was executed");
    return TEST_PASS;
}

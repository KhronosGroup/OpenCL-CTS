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
#include <cinttypes>
#include <CL/cl.h>
#include "harness/errorHelpers.h"
#include "harness/compat.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
    #include "unistd.h" // For "sleep"
#endif

#define ALLOWED_ERROR 0.005f

REGISTER_TEST(device_and_host_timers)
{
    int errors = 0;
    cl_int result = CL_SUCCESS;
    cl_ulong deviceStartTime, deviceEndTime, deviceTimeDiff;
    cl_ulong hostStartTime, hostEndTime, hostTimeDiff;
    cl_ulong hostOnlyStartTime, hostOnlyEndTime, hostOnlyTimeDiff;
    cl_ulong observedDiff;
    cl_ulong allowedDiff;

    result = clGetDeviceAndHostTimer(device, &deviceStartTime, &hostStartTime);
    if (result != CL_SUCCESS) {
        log_error("clGetDeviceAndHostTimer failed with error %s\n", IGetErrorString(result));
        errors++;
        goto End;
    }

    result = clGetHostTimer(device, &hostOnlyStartTime);
    if (result != CL_SUCCESS) {
        log_error("clGetHostTimer failed with error %s\n", IGetErrorString(result));
        errors++;
        goto End;
    }

    // Wait for a while to allow the timers to increment substantially.
    sleep(5);

    result = clGetDeviceAndHostTimer(device, &deviceEndTime, &hostEndTime);
    if (result != CL_SUCCESS) {
        log_error("clGetDeviceAndHostTimer failed with error %s\n", IGetErrorString(result));
        errors++;
        goto End;
    }

    result = clGetHostTimer(device, &hostOnlyEndTime);
    if (result != CL_SUCCESS) {
        log_error("clGetHostTimer failed with error %s\n", IGetErrorString(result));
        errors++;
        goto End;
    }

    deviceTimeDiff = deviceEndTime - deviceStartTime ;
    hostTimeDiff = hostEndTime - hostStartTime ;
    hostOnlyTimeDiff = hostOnlyEndTime - hostOnlyStartTime;

    log_info("Checking results from clGetDeviceAndHostTimer ...\n");

    if (deviceEndTime <= deviceStartTime) {
        log_error("Device timer is not monotonically increasing.\n");
        log_error("    deviceStartTime: %" PRIu64 ", deviceEndTime: %" PRIu64
                  "\n",
                  deviceStartTime, deviceEndTime);
        errors++;
    }

    if (hostEndTime <= hostStartTime) {
        log_error("Error: Host timer is not monotonically increasing.\n");
        log_error("    hostStartTime: %" PRIu64 ", hostEndTime: %" PRIu64 "\n",
                  hostStartTime, hostEndTime);
        errors++;
    }

    if (deviceTimeDiff > hostTimeDiff) {
        observedDiff = deviceTimeDiff - hostTimeDiff;
        allowedDiff = (cl_ulong)(hostTimeDiff * ALLOWED_ERROR);
    }
    else {
        observedDiff = hostTimeDiff - deviceTimeDiff;
        allowedDiff = (cl_ulong)(deviceTimeDiff * ALLOWED_ERROR);
    }

    if (observedDiff > allowedDiff) {
        log_error("Error: Device and host timers did not increase by same amount\n");
        log_error("    Observed difference between timers %" PRIu64
                  " (max allowed %" PRIu64 ").\n",
                  observedDiff, allowedDiff);
        errors++;
    }

    log_info("Cross-checking results with clGetHostTimer ...\n");

    if (hostOnlyEndTime <= hostOnlyStartTime) {
        log_error("Error: Host timer is not monotonically increasing.\n");
        log_error("    hostStartTime: %" PRIu64 ", hostEndTime: %" PRIu64 "\n",
                  hostOnlyStartTime, hostOnlyEndTime);
        errors++;
    }

    if (hostOnlyStartTime < hostStartTime) {
        log_error("Error: Host start times do not correlate.\n");
        log_error("clGetDeviceAndHostTimer was called before clGetHostTimer but timers are not in that order.\n");
        log_error("    clGetDeviceAndHostTimer: %" PRIu64
                  ", clGetHostTimer: %" PRIu64 "\n",
                  hostStartTime, hostOnlyStartTime);
        errors++;
    }

    if (hostOnlyEndTime < hostEndTime) {
        log_error("Error: Host end times do not correlate.\n");
        log_error("clGetDeviceAndHostTimer was called before clGetHostTimer but timers are not in that order.\n");
        log_error("    clGetDeviceAndHostTimer: %" PRIu64
                  ", clGetHostTimer: %" PRIu64 "\n",
                  hostEndTime, hostOnlyEndTime);
        errors++;
    }

End:
    return errors;
}

REGISTER_TEST(timer_resolution_queries)
{
    int errors = 0;
    cl_int result = CL_SUCCESS;
    cl_platform_id platform = 0;
    cl_ulong deviceTimerResolution = 0;
    cl_ulong hostTimerResolution = 0;

    result = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                             &platform, NULL);
    if (result != CL_SUCCESS) {
        log_error("clGetDeviceInfo(CL_DEVICE_PLATFORM) failed with error %s.\n", IGetErrorString(result));
        errors++;
    }

    result = clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                             sizeof(deviceTimerResolution),
                             &deviceTimerResolution, NULL);
    if (result != CL_SUCCESS) {
        log_error("clGetDeviceInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION) failed with error %s.\n", IGetErrorString(result));
        errors++;
    }
    else {
        log_info("CL_DEVICE_PROFILING_TIMER_RESOLUTION == %" PRIu64
                 " nanoseconds\n",
                 deviceTimerResolution);
    }
   
    if (platform) {
        result = clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION, sizeof(hostTimerResolution), &hostTimerResolution, NULL);
        if (result != CL_SUCCESS) {
            log_error("clGetPlatformInfo(CL_PLATFORM_HOST_TIMER_RESOLUTION) failed with error %s.\n", IGetErrorString(result));
            errors++;
        }
        else {
            log_info("CL_PLATFORM_HOST_TIMER_RESOLUTION == %" PRIu64
                     " nanoseconds\n",
                     hostTimerResolution);
        }
    }
    else {
        log_error("Could not find platform ID to query CL_PLATFORM_HOST_TIMER_RESOLUTION\n");
    }

    return errors;
}

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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include "harness/testHarness.h"

// FIXME: To use certain functions in harness/imageHelpers.h
// (for example, generate_random_image_data()), the tests are required to declare
// the following variables (<rdar://problem/11111245>):

// FIXME: use timer resolution rather than hardcoding 1µs per tick.

#define QUEUE_SECONDS_LIMIT 30
#define SUBMIT_SECONDS_LIMIT 30
#define COMMAND_SECONDS_LIMIT 30
int check_times(cl_ulong queueStart, cl_ulong commandSubmit, cl_ulong commandStart, cl_ulong commandEnd, cl_device_id device) {
  int err = 0;

  size_t profiling_resolution = 0;
  err = clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(profiling_resolution), &profiling_resolution, NULL);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_PROFILING_TIMER_RESOLUTION failed.\n");

    log_info("CL_PROFILING_COMMAND_QUEUED: %" PRIu64
             " CL_PROFILING_COMMAND_SUBMIT: %" PRIu64
             " CL_PROFILING_COMMAND_START: %" PRIu64
             " CL_PROFILING_COMMAND_END: %" PRIu64
             " CL_DEVICE_PROFILING_TIMER_RESOLUTION: %zu\n",
             queueStart, commandSubmit, commandStart, commandEnd,
             profiling_resolution);

    double queueTosubmitTimeS = (double)(commandSubmit - queueStart) * 1e-9;
    double submitToStartTimeS = (double)(commandStart - commandSubmit) * 1e-9;
    double startToEndTimeS = (double)(commandEnd - commandStart) * 1e-9;

    log_info( "Profiling info:\n" );
    log_info( "Time from queue to submit : %fms\n", (double)(queueTosubmitTimeS) * 1000.f );
    log_info( "Time from submit to start : %fms\n", (double)(submitToStartTimeS) * 1000.f );
    log_info( "Time from start to end: %fms\n", (double)(startToEndTimeS) * 1000.f );

  if(queueStart > commandSubmit) {
    log_error("CL_PROFILING_COMMAND_QUEUED > CL_PROFILING_COMMAND_SUBMIT.\n");
    err = -1;
  }

  if (commandSubmit > commandStart) {
    log_error("CL_PROFILING_COMMAND_SUBMIT > CL_PROFILING_COMMAND_START.\n");
    err = -1;
  }

  if (commandStart > commandEnd) {
    log_error("CL_PROFILING_COMMAND_START > CL_PROFILING_COMMAND_END.\n");
    err = -1;
  }

  if (queueStart == 0 && commandStart == 0 && commandEnd == 0) {
    log_error("All values are 0. This is exceedingly unlikely.\n");
    err = -1;
  }

  if (queueTosubmitTimeS > QUEUE_SECONDS_LIMIT) {
    log_error("Time between queue and submit is too big: %fs, test limit: %fs.\n",
              queueTosubmitTimeS , (double)QUEUE_SECONDS_LIMIT);
    err = -1;
  }

   if (submitToStartTimeS > SUBMIT_SECONDS_LIMIT) {
    log_error("Time between submit and start is too big: %fs, test limit: %fs.\n",
              submitToStartTimeS , (double)QUEUE_SECONDS_LIMIT);
    err = -1;
  }

  if (startToEndTimeS > COMMAND_SECONDS_LIMIT) {
    log_error("Time between queue and start is too big: %fs, test limit: %fs.\n",
             startToEndTimeS , (double)QUEUE_SECONDS_LIMIT);
    err = -1;
  }
  return err;
}

int main( int argc, const char *argv[] )
{
    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false,
                          CL_QUEUE_PROFILING_ENABLE);
}


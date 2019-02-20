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
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

basefn    basefn_list[] = {
    read_int_array,
    read_uint_array,
    read_long_array,
    read_ulong_array,
    read_short_array,
    read_ushort_array,
    read_float_array,
    read_char_array,
    read_uchar_array,
    read_struct_array,
    write_int_array,
    write_uint_array,
    write_long_array,
    write_ulong_array,
    write_short_array,
    write_ushort_array,
    write_float_array,
    write_char_array,
    write_uchar_array,
    write_struct_array,
    read_float_image,
    read_char_image,
    read_uchar_image,
    write_float_image,
    write_char_image,
    write_uchar_image,
    copy_array,
    copy_partial_array,
    copy_image,
    copy_array_to_image,
    execute
};


const char *basefn_names[] = {
"read_array_int",
"read_array_uint",
"read_array_long",
"read_array_ulong",
"read_array_short",
"read_array_ushort",
"read_array_float",
"read_array_char",
"read_array_uchar",
"read_array_struct",
"write_array_int",
"write_array_uint",
"write_array_long",
"write_array_ulong",
"write_array_short",
"write_array_ushort",
"write_array_float",
"write_array_char",
"write_array_uchar",
"write_array_struct",
"read_image_float",
"read_image_int",
"read_image_uint",
"write_image_float",
"write_image_char",
"write_image_uchar",
"copy_array",
"copy_partial_array",
"copy_image",
"copy_array_to_image",
"execute",
"all"
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_streamfns = sizeof(basefn_names) / sizeof(char *);

// FIXME: use timer resolution rather than hardcoding 1µs per tick.

#define QUEUE_SECONDS_LIMIT 30
#define SUBMIT_SECONDS_LIMIT 30
#define COMMAND_SECONDS_LIMIT 30
int check_times(cl_ulong queueStart, cl_ulong commandSubmit, cl_ulong commandStart, cl_ulong commandEnd, cl_device_id device) {
  int err = 0;

  size_t profiling_resolution = 0;
  err = clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(profiling_resolution), &profiling_resolution, NULL);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_PROFILING_TIMER_RESOLUTION failed.\n");

  log_info("CL_PROFILING_COMMAND_QUEUED: %llu CL_PROFILING_COMMAND_SUBMIT: %llu CL_PROFILING_COMMAND_START: %llu CL_PROFILING_COMMAND_END: %llu CL_DEVICE_PROFILING_TIMER_RESOLUTION: %ld\n",
           queueStart, commandSubmit, commandStart, commandEnd, profiling_resolution);

  double queueTosubmitTimeS = (double)(commandSubmit - queueStart)*1e-9;
  double submitToStartTimeS = (double)(commandStart - commandSubmit)*1e-9;
  double startToEndTimeS = (double)(commandEnd - commandStart)*1e-9;

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
    return runTestHarness( argc, argv, num_streamfns, basefn_list, basefn_names,
                           false, false, CL_QUEUE_PROFILING_ENABLE );
}



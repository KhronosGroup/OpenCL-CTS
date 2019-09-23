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
#ifndef __PROCS_H__
#define __PROCS_H__

#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/mt19937.h"
#include "harness/conversions.h"

#ifndef __APPLE__
#include <CL/cl.h>
#endif

extern int      test_pipe_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int        test_pipe_workgroup_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_workgroup_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int        test_pipe_subgroup_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_subgroup_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int        test_pipe_convenience_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_pipe_convenience_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int      test_pipe_info( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_pipe_max_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pipe_max_packet_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pipe_max_active_reservations(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pipe_query_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pipe_readwrite_errors(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pipe_subgroups_divergence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);


#endif    // #ifndef __PROCS_H__


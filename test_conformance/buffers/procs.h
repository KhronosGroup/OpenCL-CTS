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

extern const cl_mem_flags flag_set[];
extern const char* flag_set_names[];
#define NUM_FLAGS 5

extern int      test_buffer_read_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_random_size( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_async_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_read_array_barrier_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_write_async_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_copy( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_partial_copy( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_array_info_size( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_mem_read_write_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_mem_write_only_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_mem_read_only_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_mem_copy_host_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_mem_alloc_ref_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_read_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int      test_buffer_map_write_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_map_write_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

extern int      test_sub_buffers_read_write( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_sub_buffers_read_write_dual_devices( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_sub_buffers_overlapping( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_migrate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_image_migrate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_buffer_fill_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_buffer_fill_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

#endif    // #ifndef __PROCS_H__


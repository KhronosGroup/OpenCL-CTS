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

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/imageHelpers.h"
#include "harness/mt19937.h"


extern int check_times(cl_ulong queueStart, cl_ulong submitStart, cl_ulong commandStart, cl_ulong commandEnd, cl_device_id device);

extern int        test_read_array_int( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_uint( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_long( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_ulong( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_short( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_ushort( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_float( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_half( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_char( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_array_struct( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_int( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_uint( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_long( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_ulong( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_short( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_ushort( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_float( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_half( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_char( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_array_struct( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_image_float( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_image_char( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_read_image_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_image_float( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_image_char( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_write_image_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_copy_array( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_copy_partial_array( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_copy_image( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_copy_array_to_image( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_execute( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_parallel_kernels( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );


#endif    // #ifndef __PROCS_H__



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
//#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

basefn  bufferfn_list[] = {
    test_buffer_read_async_int,
    test_buffer_read_async_uint,
    test_buffer_read_async_long,
    test_buffer_read_async_ulong,
    test_buffer_read_async_short,
    test_buffer_read_async_ushort,
    test_buffer_read_async_char,
    test_buffer_read_async_uchar,
    test_buffer_read_async_float,
    test_buffer_read_array_barrier_int,
    test_buffer_read_array_barrier_uint,
    test_buffer_read_array_barrier_long,
    test_buffer_read_array_barrier_ulong,
    test_buffer_read_array_barrier_short,
    test_buffer_read_array_barrier_ushort,
    test_buffer_read_array_barrier_char,
    test_buffer_read_array_barrier_uchar,
    test_buffer_read_array_barrier_float,
    test_buffer_read_int,
    test_buffer_read_uint,
    test_buffer_read_long,
    test_buffer_read_ulong,
    test_buffer_read_short,
    test_buffer_read_ushort,
    test_buffer_read_float,
    0, //test_buffer_read_half,
    test_buffer_read_char,
    test_buffer_read_uchar,
    test_buffer_read_struct,
    test_buffer_read_random_size,
    test_buffer_map_read_int,
    test_buffer_map_read_uint,
    test_buffer_map_read_long,
    test_buffer_map_read_ulong,
    test_buffer_map_read_short,
    test_buffer_map_read_ushort,
    test_buffer_map_read_char,
    test_buffer_map_read_uchar,
    test_buffer_map_read_float,
    test_buffer_map_read_struct,

    test_buffer_map_write_int,
    test_buffer_map_write_uint,
    test_buffer_map_write_long,
    test_buffer_map_write_ulong,
    test_buffer_map_write_short,
    test_buffer_map_write_ushort,
    test_buffer_map_write_char,
    test_buffer_map_write_uchar,
    test_buffer_map_write_float,
    test_buffer_map_write_struct,

    test_buffer_write_int,
    test_buffer_write_uint,
    test_buffer_write_short,
    test_buffer_write_ushort,
    test_buffer_write_char,
    test_buffer_write_uchar,
    test_buffer_write_float,
    0, //test_buffer_write_half,
    test_buffer_write_long,
    test_buffer_write_ulong,
    test_buffer_write_struct,
    test_buffer_write_async_int,
    test_buffer_write_async_uint,
    test_buffer_write_async_short,
    test_buffer_write_async_ushort,
    test_buffer_write_async_char,
    test_buffer_write_async_uchar,
    test_buffer_write_async_float,
    test_buffer_write_async_long,
    test_buffer_write_async_ulong,
    test_buffer_copy,
    test_buffer_partial_copy,
    test_mem_read_write_flags,
    test_mem_write_flags,
    test_mem_read_flags,
    test_mem_copy_host_flags,
    0, //test_mem_alloc_ref_flags,
    testBufferSize,

    test_sub_buffers_read_write,
    test_sub_buffers_read_write_dual_devices,
    test_sub_buffers_overlapping,

    test_buffer_fill_int,
    test_buffer_fill_uint,
    test_buffer_fill_short,
    test_buffer_fill_ushort,
    test_buffer_fill_char,
    test_buffer_fill_uchar,
    test_buffer_fill_long,
    test_buffer_fill_ulong,
    test_buffer_fill_float,
    test_buffer_fill_struct,
  
    test_buffer_migrate,
    test_image_migrate,
};

const char *bufferfn_names[] = {
    "buffer_read_async_int",
    "buffer_read_async_uint",
    "buffer_read_async_long",
    "buffer_read_async_ulong",
    "buffer_read_async_short",
    "buffer_read_async_ushort",
    "buffer_read_async_char",
    "buffer_read_async_uchar",
    "buffer_read_async_float",
    "buffer_read_array_barrier_int",
    "buffer_read_array_barrier_uint",
    "buffer_read_array_barrier_long",
    "buffer_read_array_barrier_ulong",
    "buffer_read_array_barrier_short",
    "buffer_read_array_barrier_ushort",
    "buffer_read_array_barrier_char",
    "buffer_read_array_barrier_uchar",
    "buffer_read_array_barrier_float",
    "buffer_read_int",
    "buffer_read_uint",
    "buffer_read_long",
    "buffer_read_ulong",
    "buffer_read_short",
    "buffer_read_ushort",
    "buffer_read_float",
    "buffer_read_half",
    "buffer_read_char",
    "buffer_read_uchar",
    "buffer_read_struct",
    "buffer_read_random_size",
    "buffer_map_read_int",
    "buffer_map_read_uint",
    "buffer_map_read_long",
    "buffer_map_read_ulong",
    "buffer_map_read_short",
    "buffer_map_read_ushort",
    "buffer_map_read_char",
    "buffer_map_read_uchar",
    "buffer_map_read_float",
    "buffer_map_read_struct",

    "buffer_map_write_int",
    "buffer_map_write_uint",
    "buffer_map_write_long",
    "buffer_map_write_ulong",
    "buffer_map_write_short",
    "buffer_map_write_ushort",
    "buffer_map_write_char",
    "buffer_map_write_uchar",
    "buffer_map_write_float",
    "buffer_map_write_struct",

    "buffer_write_int",
    "buffer_write_uint",
    "buffer_write_short",
    "buffer_write_ushort",
    "buffer_write_char",
    "buffer_write_uchar",
    "buffer_write_float",
    "buffer_write_half",
    "buffer_write_long",
    "buffer_write_ulong",
    "buffer_write_struct",
    "buffer_write_async_int",
    "buffer_write_async_uint",
    "buffer_write_async_short",
    "buffer_write_async_ushort",
    "buffer_write_async_char",
    "buffer_write_async_uchar",
    "buffer_write_async_float",
    "buffer_write_async_long",
    "buffer_write_async_ulong",
    "buffer_copy",
    "buffer_partial_copy",
    "mem_read_write_flags",
    "mem_write_only_flags",
    "mem_read_only_flags",
    "mem_copy_host_flags",
    "mem_alloc_ref_flags",
    "array_info_size",
    "sub_buffers_read_write",
    "sub_buffers_read_write_dual_devices",
    "sub_buffers_overlapping",
    "buffer_fill_int",
    "buffer_fill_uint",
    "buffer_fill_short",
    "buffer_fill_ushort",
    "buffer_fill_char",
    "buffer_fill_uchar",
    "buffer_fill_long",
    "buffer_fill_ulong",
    "buffer_fill_float",
    "buffer_fill_struct",
    "buffer_migrate",
    "image_migrate",
    "all"
};

ct_assert((sizeof(bufferfn_names) / sizeof(bufferfn_names[0]) - 1) == (sizeof(bufferfn_list) / sizeof(bufferfn_list[0])));

int num_bufferfns = sizeof(bufferfn_names) / sizeof(char *);

const cl_mem_flags flag_set[] = {
    CL_MEM_ALLOC_HOST_PTR, 
    CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
    CL_MEM_USE_HOST_PTR,
    CL_MEM_COPY_HOST_PTR,
    0
};
const char* flag_set_names[] = {
    "CL_MEM_ALLOC_HOST_PTR", 
    "CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR",
    "CL_MEM_USE_HOST_PTR",
    "CL_MEM_COPY_HOST_PTR",
    "0"
};  

int main( int argc, const char *argv[] )
{
    return runTestHarness( argc, argv, num_bufferfns, bufferfn_list, bufferfn_names,
                           false, false, 0 );
}

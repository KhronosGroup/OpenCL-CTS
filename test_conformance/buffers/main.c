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
#include <stdlib.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST( buffer_read_async_int ),
    ADD_TEST( buffer_read_async_uint ),
    ADD_TEST( buffer_read_async_long ),
    ADD_TEST( buffer_read_async_ulong ),
    ADD_TEST( buffer_read_async_short ),
    ADD_TEST( buffer_read_async_ushort ),
    ADD_TEST( buffer_read_async_char ),
    ADD_TEST( buffer_read_async_uchar ),
    ADD_TEST( buffer_read_async_float ),
    ADD_TEST( buffer_read_array_barrier_int ),
    ADD_TEST( buffer_read_array_barrier_uint ),
    ADD_TEST( buffer_read_array_barrier_long ),
    ADD_TEST( buffer_read_array_barrier_ulong ),
    ADD_TEST( buffer_read_array_barrier_short ),
    ADD_TEST( buffer_read_array_barrier_ushort ),
    ADD_TEST( buffer_read_array_barrier_char ),
    ADD_TEST( buffer_read_array_barrier_uchar ),
    ADD_TEST( buffer_read_array_barrier_float ),
    ADD_TEST( buffer_read_int ),
    ADD_TEST( buffer_read_uint ),
    ADD_TEST( buffer_read_long ),
    ADD_TEST( buffer_read_ulong ),
    ADD_TEST( buffer_read_short ),
    ADD_TEST( buffer_read_ushort ),
    ADD_TEST( buffer_read_float ),
    NOT_IMPLEMENTED_TEST( buffer_read_half ),
    ADD_TEST( buffer_read_char ),
    ADD_TEST( buffer_read_uchar ),
    ADD_TEST( buffer_read_struct ),
    ADD_TEST( buffer_read_random_size ),
    ADD_TEST( buffer_map_read_int ),
    ADD_TEST( buffer_map_read_uint ),
    ADD_TEST( buffer_map_read_long ),
    ADD_TEST( buffer_map_read_ulong ),
    ADD_TEST( buffer_map_read_short ),
    ADD_TEST( buffer_map_read_ushort ),
    ADD_TEST( buffer_map_read_char ),
    ADD_TEST( buffer_map_read_uchar ),
    ADD_TEST( buffer_map_read_float ),
    ADD_TEST( buffer_map_read_struct ),

    ADD_TEST( buffer_map_write_int ),
    ADD_TEST( buffer_map_write_uint ),
    ADD_TEST( buffer_map_write_long ),
    ADD_TEST( buffer_map_write_ulong ),
    ADD_TEST( buffer_map_write_short ),
    ADD_TEST( buffer_map_write_ushort ),
    ADD_TEST( buffer_map_write_char ),
    ADD_TEST( buffer_map_write_uchar ),
    ADD_TEST( buffer_map_write_float ),
    ADD_TEST( buffer_map_write_struct ),

    ADD_TEST( buffer_write_int ),
    ADD_TEST( buffer_write_uint ),
    ADD_TEST( buffer_write_short ),
    ADD_TEST( buffer_write_ushort ),
    ADD_TEST( buffer_write_char ),
    ADD_TEST( buffer_write_uchar ),
    ADD_TEST( buffer_write_float ),
    NOT_IMPLEMENTED_TEST( buffer_write_half ),
    ADD_TEST( buffer_write_long ),
    ADD_TEST( buffer_write_ulong ),
    ADD_TEST( buffer_write_struct ),
    ADD_TEST( buffer_write_async_int ),
    ADD_TEST( buffer_write_async_uint ),
    ADD_TEST( buffer_write_async_short ),
    ADD_TEST( buffer_write_async_ushort ),
    ADD_TEST( buffer_write_async_char ),
    ADD_TEST( buffer_write_async_uchar ),
    ADD_TEST( buffer_write_async_float ),
    ADD_TEST( buffer_write_async_long ),
    ADD_TEST( buffer_write_async_ulong ),
    ADD_TEST( buffer_copy ),
    ADD_TEST( buffer_partial_copy ),
    ADD_TEST( mem_read_write_flags ),
    ADD_TEST( mem_write_only_flags ),
    ADD_TEST( mem_read_only_flags ),
    ADD_TEST( mem_copy_host_flags ),
    NOT_IMPLEMENTED_TEST( mem_alloc_ref_flags ),
    ADD_TEST( array_info_size ),

    ADD_TEST( sub_buffers_read_write ),
    ADD_TEST( sub_buffers_read_write_dual_devices ),
    ADD_TEST( sub_buffers_overlapping ),

    ADD_TEST( buffer_fill_int ),
    ADD_TEST( buffer_fill_uint ),
    ADD_TEST( buffer_fill_short ),
    ADD_TEST( buffer_fill_ushort ),
    ADD_TEST( buffer_fill_char ),
    ADD_TEST( buffer_fill_uchar ),
    ADD_TEST( buffer_fill_long ),
    ADD_TEST( buffer_fill_ulong ),
    ADD_TEST( buffer_fill_float ),
    ADD_TEST( buffer_fill_struct ),

    ADD_TEST( buffer_migrate ),
    ADD_TEST( image_migrate ),
};

const int test_num = ARRAY_SIZE( test_list );

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
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

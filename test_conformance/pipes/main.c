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
#include "procs.h"
#include "harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST_VERSION( pipe_readwrite_int, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_uint, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_long, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_ulong, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_short, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_ushort, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_float, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_half, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_char, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_uchar, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_double, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_struct, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_int, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_uint, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_long, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_ulong, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_short, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_ushort, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_float, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_half, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_char, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_uchar, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_double, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_workgroup_readwrite_struct, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_int, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_uint, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_long, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_ulong, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_short, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_ushort, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_float, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_half, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_char, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_uchar, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_double, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroup_readwrite_struct, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_int, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_uint, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_long, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_ulong, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_short, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_ushort, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_float, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_half, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_char, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_uchar, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_double, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_convenience_readwrite_struct, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_info, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_max_args, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_max_packet_size, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_max_active_reservations, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_query_functions, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_readwrite_errors, Version(2, 0) ),
    ADD_TEST_VERSION( pipe_subgroups_divergence, Version(2, 0) ),
};

const int test_num = ARRAY_SIZE( test_list );

int main( int argc, const char *argv[] )
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

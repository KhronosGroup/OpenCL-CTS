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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST( pipe_readwrite_int ),
    ADD_TEST( pipe_readwrite_uint ),
    ADD_TEST( pipe_readwrite_long ),
    ADD_TEST( pipe_readwrite_ulong ),
    ADD_TEST( pipe_readwrite_short ),
    ADD_TEST( pipe_readwrite_ushort ),
    ADD_TEST( pipe_readwrite_float ),
    ADD_TEST( pipe_readwrite_half ),
    ADD_TEST( pipe_readwrite_char ),
    ADD_TEST( pipe_readwrite_uchar ),
    ADD_TEST( pipe_readwrite_double ),
    ADD_TEST( pipe_readwrite_struct ),
    ADD_TEST( pipe_workgroup_readwrite_int ),
    ADD_TEST( pipe_workgroup_readwrite_uint ),
    ADD_TEST( pipe_workgroup_readwrite_long ),
    ADD_TEST( pipe_workgroup_readwrite_ulong ),
    ADD_TEST( pipe_workgroup_readwrite_short ),
    ADD_TEST( pipe_workgroup_readwrite_ushort ),
    ADD_TEST( pipe_workgroup_readwrite_float ),
    ADD_TEST( pipe_workgroup_readwrite_half ),
    ADD_TEST( pipe_workgroup_readwrite_char ),
    ADD_TEST( pipe_workgroup_readwrite_uchar ),
    ADD_TEST( pipe_workgroup_readwrite_double ),
    ADD_TEST( pipe_workgroup_readwrite_struct ),
    ADD_TEST( pipe_subgroup_readwrite_int ),
    ADD_TEST( pipe_subgroup_readwrite_uint ),
    ADD_TEST( pipe_subgroup_readwrite_long ),
    ADD_TEST( pipe_subgroup_readwrite_ulong ),
    ADD_TEST( pipe_subgroup_readwrite_short ),
    ADD_TEST( pipe_subgroup_readwrite_ushort ),
    ADD_TEST( pipe_subgroup_readwrite_float ),
    ADD_TEST( pipe_subgroup_readwrite_half ),
    ADD_TEST( pipe_subgroup_readwrite_char ),
    ADD_TEST( pipe_subgroup_readwrite_uchar ),
    ADD_TEST( pipe_subgroup_readwrite_double ),
    ADD_TEST( pipe_subgroup_readwrite_struct ),
    ADD_TEST( pipe_convenience_readwrite_int ),
    ADD_TEST( pipe_convenience_readwrite_uint ),
    ADD_TEST( pipe_convenience_readwrite_long ),
    ADD_TEST( pipe_convenience_readwrite_ulong ),
    ADD_TEST( pipe_convenience_readwrite_short ),
    ADD_TEST( pipe_convenience_readwrite_ushort ),
    ADD_TEST( pipe_convenience_readwrite_float ),
    ADD_TEST( pipe_convenience_readwrite_half ),
    ADD_TEST( pipe_convenience_readwrite_char ),
    ADD_TEST( pipe_convenience_readwrite_uchar ),
    ADD_TEST( pipe_convenience_readwrite_double ),
    ADD_TEST( pipe_convenience_readwrite_struct ),
    ADD_TEST( pipe_info ),
    ADD_TEST( pipe_max_args ),
    ADD_TEST( pipe_max_packet_size ),
    ADD_TEST( pipe_max_active_reservations ),
    ADD_TEST( pipe_query_functions ),
    ADD_TEST( pipe_readwrite_errors ),
    ADD_TEST( pipe_subgroups_divergence ),
};

const int test_num = ARRAY_SIZE( test_list );

int main( int argc, const char *argv[] )
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

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

basefn  pipefn_list[] = {
    test_pipe_readwrite_int,
    test_pipe_readwrite_uint,
    test_pipe_readwrite_long,
    test_pipe_readwrite_ulong,
    test_pipe_readwrite_short,
    test_pipe_readwrite_ushort,
    test_pipe_readwrite_float,
    test_pipe_readwrite_half,
    test_pipe_readwrite_char,
    test_pipe_readwrite_uchar,
    test_pipe_readwrite_double,
    test_pipe_readwrite_struct,
    test_pipe_workgroup_readwrite_int,
    test_pipe_workgroup_readwrite_uint,
    test_pipe_workgroup_readwrite_long,
    test_pipe_workgroup_readwrite_ulong,
    test_pipe_workgroup_readwrite_short,
    test_pipe_workgroup_readwrite_ushort,
    test_pipe_workgroup_readwrite_float,
    test_pipe_workgroup_readwrite_half,
    test_pipe_workgroup_readwrite_char,
    test_pipe_workgroup_readwrite_uchar,
    test_pipe_workgroup_readwrite_double,
    test_pipe_workgroup_readwrite_struct,
    test_pipe_subgroup_readwrite_int,
    test_pipe_subgroup_readwrite_uint,
    test_pipe_subgroup_readwrite_long,
    test_pipe_subgroup_readwrite_ulong,
    test_pipe_subgroup_readwrite_short,
    test_pipe_subgroup_readwrite_ushort,
    test_pipe_subgroup_readwrite_float,
    test_pipe_subgroup_readwrite_half,
    test_pipe_subgroup_readwrite_char,
    test_pipe_subgroup_readwrite_uchar,
    test_pipe_subgroup_readwrite_double,
    test_pipe_subgroup_readwrite_struct,
    test_pipe_convenience_readwrite_int,
    test_pipe_convenience_readwrite_uint,
    test_pipe_convenience_readwrite_long,
    test_pipe_convenience_readwrite_ulong,
    test_pipe_convenience_readwrite_short,
    test_pipe_convenience_readwrite_ushort,
    test_pipe_convenience_readwrite_float,
    test_pipe_convenience_readwrite_half,
    test_pipe_convenience_readwrite_char,
    test_pipe_convenience_readwrite_uchar,
    test_pipe_convenience_readwrite_double,
    test_pipe_convenience_readwrite_struct,
    test_pipe_info,
    test_pipe_max_args,
    test_pipe_max_packet_size,
    test_pipe_max_active_reservations,
    test_pipe_query_functions,
    test_pipe_readwrite_errors,
    test_pipe_subgroups_divergence
};

const char *pipefn_names[] = {
    "pipe_readwrite_int",
    "pipe_readwrite_uint",
    "pipe_readwrite_long",
    "pipe_readwrite_ulong",
    "pipe_readwrite_short",
    "pipe_readwrite_ushort",
    "pipe_readwrite_float",
    "pipe_readwrite_half",
    "pipe_readwrite_char",
    "pipe_readwrite_uchar",
    "pipe_readwrite_double",
    "pipe_readwrite_struct",
    "pipe_workgroup_readwrite_int",
    "pipe_workgroup_readwrite_uint",
    "pipe_workgroup_readwrite_long",
    "pipe_workgroup_readwrite_ulong",
    "pipe_workgroup_readwrite_short",
    "pipe_workgroup_readwrite_ushort",
    "pipe_workgroup_readwrite_float",
    "pipe_workgroup_readwrite_half",
    "pipe_workgroup_readwrite_char",
    "pipe_workgroup_readwrite_uchar",
    "pipe_workgroup_readwrite_double",
    "pipe_workgroup_readwrite_struct",
    "pipe_subgroup_readwrite_int",
    "pipe_subgroup_readwrite_uint",
    "pipe_subgroup_readwrite_long",
    "pipe_subgroup_readwrite_ulong",
    "pipe_subgroup_readwrite_short",
    "pipe_subgroup_readwrite_ushort",
    "pipe_subgroup_readwrite_float",
    "pipe_subgroup_readwrite_half",
    "pipe_subgroup_readwrite_char",
    "pipe_subgroup_readwrite_uchar",
    "pipe_subgroup_readwrite_double",
    "pipe_subgroup_readwrite_struct",
    "pipe_convenience_readwrite_int",
    "pipe_convenience_readwrite_uint",
    "pipe_convenience_readwrite_long",
    "pipe_convenience_readwrite_ulong",
    "pipe_convenience_readwrite_short",
    "pipe_convenience_readwrite_ushort",
    "pipe_convenience_readwrite_float",
    "pipe_convenience_readwrite_half",
    "pipe_convenience_readwrite_char",
    "pipe_convenience_readwrite_uchar",
    "pipe_convenience_readwrite_double",
    "pipe_convenience_readwrite_struct",
    "pipe_info",
    "pipe_max_args",
    "pipe_max_packet_size",
    "pipe_max_active_reservations",
    "pipe_query_functions",
    "pipe_readwrite_errors",
    "pipe_subgroups_divergence",
};

ct_assert((sizeof(pipefn_names) / sizeof(pipefn_names[0])) == (sizeof(pipefn_list) / sizeof(pipefn_list[0])));

int num_pipefns = sizeof(pipefn_names) / sizeof(char *);

int main( int argc, const char *argv[] )
{
    return runTestHarness( argc, argv, num_pipefns, pipefn_list, pipefn_names,
                           false, false, 0 );
}

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
#include "../../test_common/harness/mt19937.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
            test_partition_equally,
            test_partition_by_counts,
            test_partition_by_affinity_domain_numa,
            test_partition_by_affinity_domain_l4_cache,
            test_partition_by_affinity_domain_l3_cache,
            test_partition_by_affinity_domain_l2_cache,
            test_partition_by_affinity_domain_l1_cache,
            test_partition_by_affinity_domain_next_partitionable,
            test_partition
};


const char    *basefn_names[] = {
            "device_partition_equally",
            "device_partition_by_counts",
            "device_partition_by_affinity_domain_numa",
            "device_partition_by_affinity_domain_l4_cache",
            "device_partition_by_affinity_domain_l3_cache",
            "device_partition_by_affinity_domain_l2_cache",
            "device_partition_by_affinity_domain_l1_cache",
            "device_partition_by_affinity_domain_next_partitionable",
            "device_partition_all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, true, 0 );
}

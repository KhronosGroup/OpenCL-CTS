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
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/mt19937.h"

extern int      test_partition_all(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_equally(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_counts(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_numa(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_l4_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_l3_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_l2_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_l1_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_partition_by_affinity_domain_next_partitionable(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

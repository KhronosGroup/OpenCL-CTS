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
#ifndef _procs_h
#define _procs_h

#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/mt19937.h"

extern MTdata gMTdata;

extern int test_sub_group_info_ext(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements);
extern int test_sub_group_info_core(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_work_item_functions_ext(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_work_item_functions_core(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements);
extern int test_subgroup_functions_ext(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_subgroup_functions_core(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_barrier_functions_ext(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_barrier_functions_core(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_pipe_functions(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_ifp_ext(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements);
extern int test_ifp_core(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements);
extern int test_subgroup_functions_extended_types(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements);
extern int test_subgroup_functions_non_uniform_vote(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_subgroup_functions_non_uniform_arithmetic(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_subgroup_functions_ballot(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_subgroup_functions_clustered_reduce(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_subgroup_functions_shuffle(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_subgroup_functions_shuffle_relative(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_subgroup_functions_rotate(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
#endif /*_procs_h*/

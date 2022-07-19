//
// Copyright (c) 2022 The Khronos Group Inc.
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
#ifndef _CL_KHR_COMMAND_BUFFER_PROCS_H
#define _CL_KHR_COMMAND_BUFFER_PROCS_H

#include <CL/cl.h>

// Basic command-buffer tests
extern int test_single_ndrange(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_interleaved_enqueue(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_mixed_commands(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_explicit_flush(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_user_events(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements);
extern int test_out_of_order(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements);

#endif /*_CL_KHR_COMMAND_BUFFER_PROCS_H*/

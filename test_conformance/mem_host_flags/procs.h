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

#include "testBase.h"

#define NUM_FLAGS 4

extern int test_mem_host_read_only_buffer(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_mem_host_read_only_subbuffer(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);

extern int test_mem_host_write_only_buffer(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_mem_host_write_only_subbuffer(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);

extern int test_mem_host_no_access_buffer(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_mem_host_no_access_subbuffer(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);

extern int test_mem_host_read_only_image(cl_device_id deviceID,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements);
extern int test_mem_host_write_only_image(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_mem_host_no_access_image(cl_device_id deviceID,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements);

#endif // #ifndef __PROCS_H__

//
// Copyright (c) 2023 The Khronos Group Inc.
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
#ifndef CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_PROCS_H
#define CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_PROCS_H

#include <CL/cl.h>


// Basic mutable dispatch tests
extern int test_mutable_command_info_device_query(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements);
extern int test_mutable_command_info_buffer(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_mutable_command_info_type(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_mutable_command_info_queue(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_mutable_command_properties_array(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements);
extern int test_mutable_command_kernel(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_mutable_command_dimensions(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_mutable_command_info_global_work_offset(cl_device_id device,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements);
extern int test_mutable_command_info_local_work_size(cl_device_id device,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements);
extern int test_mutable_command_info_global_work_size(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements);
extern int test_mutable_dispatch_image_1d_arguments(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_mutable_dispatch_image_2d_arguments(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_mutable_dispatch_global_arguments(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements);
extern int test_mutable_dispatch_local_arguments(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements);
extern int test_mutable_dispatch_pod_arguments(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
extern int test_mutable_dispatch_null_arguments(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements);
extern int test_mutable_dispatch_svm_arguments(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
extern int test_mutable_dispatch_out_of_order(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_mutable_dispatch_simultaneous_out_of_order(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_mutable_dispatch_global_size(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_mutable_dispatch_local_size(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_mutable_dispatch_global_offset(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
#endif /*_CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_PROCS_H*/

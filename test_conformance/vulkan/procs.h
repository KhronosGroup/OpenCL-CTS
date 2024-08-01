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

#include "harness/mt19937.h"

extern int test_vulkan_interop_buffer(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_vulkan_interop_image(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);
extern int test_consistency_external_buffer(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_consistency_external_image(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_consistency_external_for_3dimage(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements);
extern int test_consistency_external_for_1dimage(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements);
extern int test_consistency_external_semaphore(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
extern int test_platform_info(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements);
extern int test_device_info(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements);

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
#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

extern const int kVectorSizeCount;

extern int test_quick_1d_explicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_quick_2d_explicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_quick_3d_explicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_quick_1d_implicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_quick_2d_implicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_quick_3d_implicit_local(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);

extern int test_full_1d_explicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_full_2d_explicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_full_3d_explicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_full_1d_implicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_full_2d_implicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_full_3d_implicit_local(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);

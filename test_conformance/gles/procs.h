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
#include "testBase.h"
#include "harness/mt19937.h"


extern int test_buffers( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_buffers_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );
extern int test_images_create( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_images_read( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_images_2D_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );
extern int test_images_read_cube( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_images_cube_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );
extern int test_images_read_3D( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_images_3D_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );
extern int test_images_write( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_images_write_cube( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_renderbuffer_read( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_renderbuffer_write( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements );
extern int test_renderbuffer_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );
extern int test_fence_sync( cl_device_id device, cl_context context, cl_command_queue queue, int numElements );


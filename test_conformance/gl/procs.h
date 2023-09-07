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


#pragma mark -
#pragma Misc tests

extern int test_buffers(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements);

extern int test_fence_sync(cl_device_id device, cl_context context,
                           cl_command_queue queue, int numElements);


#pragma mark -
#pragma mark Tead tests

extern int test_images_read_2D(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);

extern int test_images_read_1D(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);

extern int test_images_read_texturebuffer(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);

extern int test_images_read_1Darray(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);

extern int test_images_read_2Darray(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);

extern int test_images_read_cube(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements);

extern int test_images_read_3D(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);

extern int test_renderbuffer_read(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements);

#pragma mark -
#pragma mark Write tests

// 2D tests are the ones with no suffix:

extern int test_images_write(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements);

extern int test_images_write_cube(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements);

extern int test_renderbuffer_write(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements);

// Here are the rest:

extern int test_images_write_1D(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);

extern int test_images_write_texturebuffer(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);

extern int test_images_write_1Darray(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);

extern int test_images_write_2Darray(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);

extern int test_images_write_3D(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);

#pragma mark -
#pragma mark Get info test entry points

extern int test_buffers_getinfo(cl_device_id device, cl_context context,
                                cl_command_queue queue, int numElements);

extern int test_images_1D_getinfo(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int numElements);

extern int test_images_texturebuffer_getinfo(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int numElements);

extern int test_images_1Darray_getinfo(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int numElements);

extern int test_images_2D_getinfo(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int numElements);

extern int test_images_2Darray_getinfo(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int numElements);

extern int test_images_cube_getinfo(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int numElements);

extern int test_images_3D_getinfo(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int numElements);

extern int test_images_read_2D_depth(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int numElements);

extern int test_images_write_2D_depth(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int numElements);

extern int test_images_read_2Darray_depth(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue, int);

extern int test_images_write_2Darray_depth(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int numElements);

extern int test_images_read_2D_multisample(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int numElements);

extern int test_images_read_2Darray_multisample(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue, int);

extern int test_image_methods_depth(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int);

extern int test_image_methods_multisample(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue, int);

extern int test_renderbuffer_getinfo(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int numElements);
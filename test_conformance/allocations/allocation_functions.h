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
#include "allocation_utils.h"

int do_allocation(cl_context context, cl_command_queue *queue, cl_device_id device_id, size_t size_to_allocate, int type, cl_mem *mem);
int allocate_buffer(cl_context context, cl_command_queue *queue, cl_device_id device_id, cl_mem *mem, size_t size_to_allocate);
int allocate_image2d_read(cl_context context, cl_command_queue *queue, cl_device_id device_id, cl_mem *mem, size_t size_to_allocate);
int allocate_image2d_write(cl_context context, cl_command_queue *queue, cl_device_id device_id, cl_mem *mem, size_t size_to_allocate);
int allocate_size(cl_context context, cl_command_queue *queue, cl_device_id device_id, int multiple_allocations, size_t size_to_allocate,
                  int type, cl_mem mems[], int *number_of_mems, size_t *final_size, int force_fill, MTdata d);

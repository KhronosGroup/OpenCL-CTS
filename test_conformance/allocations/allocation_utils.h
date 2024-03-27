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
#ifndef _allocation_utils_h
#define _allocation_utils_h

#include "testBase.h"

extern cl_uint checksum;

int check_allocation_error(cl_context context, cl_device_id device_id,
                           int error, cl_command_queue *queue,
                           cl_event *event = 0);
double toMB(cl_ulong size_in);
size_t get_actual_allocation_size(cl_mem mem);

#endif // _allocation_utils_h
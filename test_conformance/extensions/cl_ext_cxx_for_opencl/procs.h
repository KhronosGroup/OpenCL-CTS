//
// Copyright (c) 2021 The Khronos Group Inc.
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

#include "harness/typeWrappers.h"

extern int test_cxx_for_opencl_ext(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int);
extern int test_cxx_for_opencl_ver(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int);

#endif /*_procs_h*/

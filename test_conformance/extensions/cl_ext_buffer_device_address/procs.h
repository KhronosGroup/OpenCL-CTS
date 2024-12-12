// Copyright (c) 2024 The Khronos Group Inc.
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
#ifndef CL_EXT_BUFFER_DEVICE_ADDRESS_H
#define CL_EXT_BUFFER_DEVICE_ADDRESS_H

#include <CL/cl.h>

int test_private_address(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements);
int test_shared_address(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements);

#endif /* CL_EXT_BUFFER_DEVICE_ADDRESS_H */

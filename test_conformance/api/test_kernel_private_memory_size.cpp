//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include <iostream>

REGISTER_TEST(kernel_private_memory_size)
{
    const char* TEST_KERNEL =
        R"(__kernel void private_memory( __global uint *buffer ){
         volatile __private uint x[1];
         buffer[0] = x[0];
         })";

    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int err = create_single_kernel_helper(context, &program, &kernel, 1,
                                             &TEST_KERNEL, "private_memory");
    test_error(err, "create_single_kernel_helper");
    cl_ulong size = CL_ULONG_MAX;
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                   sizeof(cl_ulong), &size, nullptr);

    test_error(err, "clGetKernelWorkGroupInfo");

    return TEST_PASS;
}

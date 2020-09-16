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
#include "harness/testHarness.h"


const char *kernel_with_bool[] = {
    "__kernel void kernel_with_bool(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    bool myBool = (src[tid] < 0.5f) && (src[tid] > -0.5f);\n"
    "    if(myBool)\n"
    "    {\n"
    "        dst[tid] = (int)src[tid];\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "        dst[tid] = 0;\n"
    "    }\n"
    "\n"
    "}\n"
};

int test_bool_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{

    clProgramWrapper program;
    clKernelWrapper kernel;

    int err = create_single_kernel_helper(context,
                      &program,
                      &kernel,
                      1, kernel_with_bool,
                      "kernel_with_bool" );
    return err;
}


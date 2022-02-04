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
#include "procs.h"
#include "harness/parseParameters.h"

const char* pipe_kernel_code = {
    "__kernel void pipe_kernel(__write_only pipe int out_pipe)\n"
    "{}\n" };

int test_pipe_info( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    clMemWrapper pipe;
    cl_int err;
    cl_uint pipe_width = 512;
    cl_uint pipe_depth = 1024;
    cl_uint returnVal;
    clProgramWrapper program;
    clKernelWrapper kernel;

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, pipe_width, pipe_depth,
                        NULL, &err);
    test_error(err, "clCreatePipe failed.");

    err = clGetPipeInfo(pipe, CL_PIPE_PACKET_SIZE, sizeof(pipe_width),
                        (void *)&returnVal, NULL);
    test_error(err, "clGetPipeInfo failed.");

    if (pipe_width != returnVal)
    {
        test_fail("Error in clGetPipeInfo() check of pipe packet size\n");
    }
    else
    {
        log_info( " CL_PIPE_PACKET_SIZE passed.\n" );
    }

    err = clGetPipeInfo(pipe, CL_PIPE_MAX_PACKETS, sizeof(pipe_depth), (void *)&returnVal, NULL);
    test_error(err, "clGetPipeInfo failed.");

    if(pipe_depth != returnVal)
    {
        test_fail("Error in clGetPipeInfo() check of pipe max packets\n");
    }
    else
    {
        log_info( " CL_PIPE_MAX_PACKETS passed.\n" );
    }

    err = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &pipe_kernel_code, "pipe_kernel",
        "-cl-std=CL2.0 -cl-kernel-arg-info");
    test_error_fail(err, "Error creating program");

    cl_kernel_arg_type_qualifier arg_type_qualifier = 0;
    err = clGetKernelArgInfo(kernel, 0, CL_KERNEL_ARG_TYPE_QUALIFIER,
                             sizeof(arg_type_qualifier), &arg_type_qualifier,
                             NULL);
    if (gCompilationMode == kOnline)
    {
        test_error_fail(err, "clGetKernelArgInfo failed");
        if (arg_type_qualifier != CL_KERNEL_ARG_TYPE_PIPE)
        {
            test_fail("ERROR: Incorrect type qualifier: %i\n",
                      arg_type_qualifier);
        }
    }
    else
    {
        test_failure_error_ret(err, CL_KERNEL_ARG_INFO_NOT_AVAILABLE,
                               "clGetKernelArgInfo error not as expected",
                               TEST_FAIL);
    }

    return TEST_PASS;
}

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
#include <stdio.h>
#include <string.h>
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include "utils.h"

int run_single_kernel(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, void* results, size_t res_size)
{
    return run_single_kernel_args(context, queue, source, num_lines, kernel_name, results, res_size, 0, NULL);
}

int run_single_kernel_args(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, void* results, size_t res_size, cl_uint num_args, kernel_arg* args)
{
    return run_n_kernel_args(context, queue, source, num_lines, kernel_name, 1, 1, results, res_size, num_args, args);
}

int run_n_kernel_args(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, size_t local, size_t global, void* results, size_t res_size, cl_uint num_args, kernel_arg* args)
{
    cl_int err_ret, status;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper mem;
    clEventWrapper event;
    cl_uint i;
    size_t ret_len;

    err_ret = create_single_kernel_helper_with_build_options(context, &program, &kernel, num_lines, source, kernel_name, "-cl-std=CL2.0");
    if(check_error(err_ret, "Create single kernel failed")) return -1;

    mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, res_size, results, &err_ret);
    test_error(err_ret, "clCreateBuffer() failed");

    err_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);
    if(check_error(err_ret, "clSetKernelArg(%d, %d, %p) for kernel: '%s' failed: %d", 0, (int)sizeof(cl_mem), &mem, kernel_name, err_ret)) return err_ret;

    for(i = 0; i < num_args; ++i)
    {
        err_ret = clSetKernelArg(kernel, i+1, args[i].size, args[i].ptr);
        if(check_error(err_ret, "clSetKernelArg(%d, %d, %p) for kernel: '%s' failed: %d", (int)(i+1), (int)args[i].size, args[i].ptr, kernel_name, err_ret)) return err_ret;
    }

    err_ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, (local ? &local : NULL), 0, NULL, &event);
    if(check_error(err_ret, "clEnqueueNDRangeKernel('%s', gws=%d, lws=%d) failed", kernel_name, (int)global, (int)local)) return err_ret;

    err_ret = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, res_size, results, 0, NULL, NULL);
    test_error(err_ret, "clEnqueueReadBuffer() failed");

    err_ret = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &ret_len);
    test_error(err_ret, "clGetEventInfo() failed");

#if CL_COMPLETE != CL_SUCCESS
#error Fix me!
#endif

    // This hack is possible because CL_COMPLETE and CL_SUCCESS defined as 0x0
    if(check_error(status, "Kernel execution status %d", status)) return status;

    return 0;
}


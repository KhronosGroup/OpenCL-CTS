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
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#endif
#include "procs.h"


enum { SUCCESS, FAILURE };
typedef enum { NON_NULL_PATH, ADDROF_NULL_PATH, NULL_PATH } test_type;

#define NITEMS 4096

/* places the casted long value of the src ptr into each element of the output
 * array, to allow testing that the kernel actually _gets_ the NULL value */
const char *kernel_string =
"kernel void test_kernel(global float *src, global long *dst)\n"
"{\n"
"    uint tid = get_global_id(0);\n"
"    dst[tid] = (long)src;\n"
"}\n";

/*
 * The guts of the test:
 * call setKernelArgs with a regular buffer, &NULL, or NULL depending on
 * the value of 'test_type'
 */
static int test_setargs_and_execution(cl_command_queue queue, cl_kernel kernel,
    cl_mem test_buf, cl_mem result_buf, test_type type)
{
    unsigned int test_success = 0;

    unsigned int i;
    cl_int status;
    char *typestr;

    if (type == NON_NULL_PATH) {
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &test_buf);
        typestr = "non-NULL";
    } else if (type == ADDROF_NULL_PATH) {
        test_buf = NULL;
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &test_buf);
        typestr = "&NULL";
    } else if (type == NULL_PATH) {
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), NULL);
        typestr = "NULL";
    }

    log_info("Testing setKernelArgs with %s buffer.\n", typestr);

    if (status != CL_SUCCESS) {
        log_error("clSetKernelArg failed with status: %d\n", status);
        return FAILURE; // no point in continuing *this* test
    }

    size_t global = NITEMS;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global,
        NULL, 0, NULL, NULL);
    test_error(status, "NDRangeKernel failed.");

    cl_long* host_result = (cl_long*)malloc(NITEMS*sizeof(cl_long));
    status = clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0,
        sizeof(cl_long)*NITEMS, host_result, 0, NULL, NULL);
    test_error(status, "ReadBuffer failed.");

    // in the non-null case, we expect NONZERO values:
    if (type == NON_NULL_PATH) {
        for (i=0; i<NITEMS; i++) {
            if (host_result[i] == 0) {
                log_error("failure: item %d in the result buffer was unexpectedly NULL.\n", i);
                test_success = FAILURE; break;
            }
        }

    } else if (type == ADDROF_NULL_PATH || type == NULL_PATH) {
        for (i=0; i<NITEMS; i++) {
            if (host_result[i] != 0) {
                log_error("failure: item %d in the result buffer was unexpectedly non-NULL.\n", i);
                test_success = FAILURE; break;
            }
        }
    }

    free(host_result);

    if (test_success == SUCCESS) {
        log_info("\t%s ok.\n", typestr);
    }

    return test_success;
}

int test_null_buffer_arg(cl_device_id device, cl_context context,
    cl_command_queue queue, int num_elements)
{
    unsigned int test_success = 0;
    unsigned int i;
    cl_int status;
    cl_program program;
    cl_kernel kernel;

    // prep kernel:
    program = clCreateProgramWithSource(context, 1, &kernel_string, NULL, &status);
    test_error(status, "CreateProgramWithSource failed.");

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    test_error(status, "BuildProgram failed.");

    kernel = clCreateKernel(program, "test_kernel", &status);
    test_error(status, "CreateKernel failed.");

    cl_mem dev_src = clCreateBuffer(context, CL_MEM_READ_ONLY, NITEMS*sizeof(cl_float),
        NULL, NULL);

    cl_mem dev_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NITEMS*sizeof(cl_long),
        NULL, NULL);

    // set the destination buffer normally:
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_dst);
    test_error(status, "SetKernelArg failed.");

    //
    // we test three cases:
    //
    // - typical case, used everyday: non-null buffer
    // - the case of src as &NULL (the spec-compliance test)
    // - the case of src as NULL (the backwards-compatibility test, Apple only)
    //

    test_success  = test_setargs_and_execution(queue, kernel, dev_src, dev_dst, NON_NULL_PATH);
    test_success |= test_setargs_and_execution(queue, kernel, dev_src, dev_dst, ADDROF_NULL_PATH);

#ifdef __APPLE__
    test_success |= test_setargs_and_execution(queue, kernel, dev_src, dev_dst, NULL_PATH);
#endif

    // clean up:
    if (dev_src) clReleaseMemObject(dev_src);
    clReleaseMemObject(dev_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return test_success;
}

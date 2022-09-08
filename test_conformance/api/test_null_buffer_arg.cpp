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
#include "testBase.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"
#include "procs.h"


enum { SUCCESS, FAILURE };
typedef enum { NON_NULL_PATH, ADDROF_NULL_PATH, NULL_PATH } test_type;

#define NITEMS 4096

/* places the comparison result of value of the src ptr against 0 into each element of the output
 * array, to allow testing that the kernel actually _gets_ the NULL value */
const char *kernel_string_long =
"kernel void test_kernel(global float *src, global long *dst)\n"
"{\n"
"    uint tid = get_global_id(0);\n"
"    dst[tid] = (long)(src != 0);\n"
"}\n";

// For gIsEmbedded
const char *kernel_string =
"kernel void test_kernel(global float *src, global int *dst)\n"
"{\n"
"    uint tid = get_global_id(0);\n"
"    dst[tid] = (int)(src != 0);\n"
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
    const char *typestr;

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

    if (gIsEmbedded)
    {
        cl_int* host_result = (cl_int*)malloc(NITEMS*sizeof(cl_int));
        status = clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0,
                                     sizeof(cl_int)*NITEMS, host_result, 0, NULL, NULL);
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
    }
    else
    {
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
    }

    if (test_success == SUCCESS) {
        log_info("\t%s ok.\n", typestr);
    }

    return test_success;
}

int test_null_buffer_arg(cl_device_id device, cl_context context,
    cl_command_queue queue, int num_elements)
{
    unsigned int test_success = 0;
    unsigned int buffer_size;
    cl_int status;
    cl_program program;
    cl_kernel kernel;

    // prep kernel:
    if (gIsEmbedded)
        status = create_single_kernel_helper(context, &program, &kernel, 1,
                                             &kernel_string, "test_kernel");
    else
        status = create_single_kernel_helper(
            context, &program, &kernel, 1, &kernel_string_long, "test_kernel");

    test_error(status, "Unable to create kernel");

    cl_mem dev_src = clCreateBuffer(context, CL_MEM_READ_ONLY, NITEMS*sizeof(cl_float),
        NULL, NULL);

    if (gIsEmbedded)
        buffer_size = NITEMS*sizeof(cl_int);
    else
        buffer_size = NITEMS*sizeof(cl_long);

    cl_mem dev_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size,
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

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

const char *kernel_call_kernel_code[] = {
    "void test_function_to_call(__global int *output, __global int *input, int where);\n"
    "\n"
    "__kernel void test_kernel_to_call(__global int *output, __global int *input, int where) \n"
    "{\n"
    "  int b;\n"
    "  if (where == 0) {\n"
    "    output[get_global_id(0)] = 0;\n"
    "  }\n"
    "  for (b=0; b<where; b++)\n"
    "    output[get_global_id(0)] += input[b];  \n"
    "}\n"
    "\n"
    "__kernel void test_call_kernel(__global int *src, __global int *dst, int times) \n"
    "{\n"
    "  int tid = get_global_id(0);\n"
    "  int a;\n"
    "  dst[tid] = 1;\n"
    "  for (a=0; a<times; a++)\n"
    "    test_kernel_to_call(dst, src, tid);\n"
    "}\n"
    "void test_function_to_call(__global int *output, __global int *input, int where) \n"
    "{\n"
    "  int b;\n"
    "  if (where == 0) {\n"
    "    output[get_global_id(0)] = 0;\n"
    "  }\n"
    "  for (b=0; b<where; b++)\n"
    "    output[get_global_id(0)] += input[b];  \n"
    "}\n"
    "\n"
    "__kernel void test_call_function(__global int *src, __global int *dst, int times) \n"
    "{\n"
    "  int tid = get_global_id(0);\n"
    "  int a;\n"
    "  dst[tid] = 1;\n"
    "  for (a=0; a<times; a++)\n"
    "    test_function_to_call(dst, src, tid);\n"
    "}\n"
};



int test_kernel_call_kernel_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    num_elements = 256;

    int error, errors = 0;
    clProgramWrapper program;
    clKernelWrapper kernel1, kernel2, kernel_to_call;
    clMemWrapper    streams[2];

    size_t    threads[] = {num_elements,1,1};
    cl_int *input, *output, *expected;
    cl_int times = 4;
    int pass = 0;

    input = (cl_int*)malloc(sizeof(cl_int)*num_elements);
    output = (cl_int*)malloc(sizeof(cl_int)*num_elements);
    expected = (cl_int*)malloc(sizeof(cl_int)*num_elements);

    for (int i=0; i<num_elements; i++) {
        input[i] = i;
        output[i] = i;
        expected[i] = output[i];
    }
    // Calculate the expected results
    for (int tid=0; tid<num_elements; tid++) {
        expected[tid] = 1;
        for (int a=0; a<times; a++) {
            int where = tid;
            if (where == 0)
                expected[tid] = 0;
            for (int b=0; b<where; b++) {
                expected[tid] += input[b];
            }
        }
    }

    // Test kernel calling a kernel
    log_info("Testing kernel calling kernel...\n");
    // Create the kernel
    if( create_single_kernel_helper( context, &program, &kernel1, 1, kernel_call_kernel_code, "test_call_kernel" ) != 0 )
    {
        return -1;
    }

    kernel_to_call = clCreateKernel(program, "test_kernel_to_call", &error);
    test_error(error, "clCreateKernel failed");

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(cl_int)*num_elements, input, &error);
    test_error( error, "clCreateBuffer failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(cl_int)*num_elements, output, &error);
    test_error( error, "clCreateBuffer failed" );

    error = clSetKernelArg(kernel1, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel1, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel1, 2, sizeof( times ), &times);
    test_error( error, "clSetKernelArg failed" );

    error = clEnqueueNDRangeKernel( queue, kernel1, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "clEnqueueNDRangeKernel failed" );

    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(cl_int)*num_elements, output, 0, NULL, NULL );
    test_error( error, "clEnqueueReadBuffer failed" );

    // Compare the results
    pass = 1;
    for (int i=0; i<num_elements; i++) {
        if (output[i] != expected[i]) {
            if (errors > 10)
                continue;
            if (errors == 10) {
                log_error("Suppressing further results...\n");
                continue;
            }
            log_error("Results do not match: output[%d]=%d != expected[%d]=%d\n", i, output[i], i, expected[i]);
            errors++;
            pass = 0;
        }
    }
    if (pass) log_info("Passed kernel calling kernel...\n");



    // Test kernel calling a function
    log_info("Testing kernel calling function...\n");
    // Reset the inputs
    for (int i=0; i<num_elements; i++) {
        input[i] = i;
        output[i] = i;
    }
    error = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, input, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");
    error = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, sizeof(cl_int)*num_elements, output, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    kernel2 = clCreateKernel(program, "test_call_function", &error);
    test_error(error, "clCreateKernel failed");

    error = clSetKernelArg(kernel2, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel2, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel2, 2, sizeof( times ), &times);
    test_error( error, "clSetKernelArg failed" );

    error = clEnqueueNDRangeKernel( queue, kernel2, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "clEnqueueNDRangeKernel failed" );

    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(cl_int)*num_elements, output, 0, NULL, NULL );
    test_error( error, "clEnqueueReadBuffer failed" );

    // Compare the results
    pass = 1;
    for (int i=0; i<num_elements; i++) {
        if (output[i] != expected[i]) {
            if (errors > 10)
                continue;
            if (errors > 10) {
                log_error("Suppressing further results...\n");
                continue;
            }
            log_error("Results do not match: output[%d]=%d != expected[%d]=%d\n", i, output[i], i, expected[i]);
            errors++;
            pass = 0;
        }
    }
    if (pass) log_info("Passed kernel calling function...\n");


    // Test calling the kernel we called from another kernel
    log_info("Testing calling the kernel we called from another kernel before...\n");
    // Reset the inputs
    for (int i=0; i<num_elements; i++) {
        input[i] = i;
        output[i] = i;
        expected[i] = output[i];
    }
    error = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, input, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");
    error = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, sizeof(cl_int)*num_elements, output, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    // Calculate the expected results
    int where = times;
    for (int tid=0; tid<num_elements; tid++) {
        if (where == 0)
            expected[tid] = 0;
        for (int b=0; b<where; b++) {
            expected[tid] += input[b];
        }
    }


    error = clSetKernelArg(kernel_to_call, 0, sizeof( streams[1] ), &streams[1]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel_to_call, 1, sizeof( streams[0] ), &streams[0]);
    test_error( error, "clSetKernelArg failed" );
    error = clSetKernelArg(kernel_to_call, 2, sizeof( times ), &times);
    test_error( error, "clSetKernelArg failed" );

    error = clEnqueueNDRangeKernel( queue, kernel_to_call, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "clEnqueueNDRangeKernel failed" );

    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(cl_int)*num_elements, output, 0, NULL, NULL );
    test_error( error, "clEnqueueReadBuffer failed" );

    // Compare the results
    pass = 1;
    for (int i=0; i<num_elements; i++) {
        if (output[i] != expected[i]) {
            if (errors > 10)
                continue;
            if (errors > 10) {
                log_error("Suppressing further results...\n");
                continue;
            }
            log_error("Results do not match: output[%d]=%d != expected[%d]=%d\n", i, output[i], i, expected[i]);
            errors++;
            pass = 0;
        }
    }
    if (pass) log_info("Passed calling the kernel we called from another kernel before...\n");

    free( input );
    free( output );
    free( expected );

    return errors;
}



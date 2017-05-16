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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "../../test_common/harness/errorHelpers.h"

const char* pipe_readwrite_errors_kernel_code = {
    "__kernel void test_pipe_write_error(__global int *src, __write_only pipe int out_pipe, __global int *status)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    int reserve_idx;\n"
    "    reserve_id_t res_id;\n"
    "\n"
    "    res_id = reserve_write_pipe(out_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        write_pipe(out_pipe, res_id, 0, &src[gid]);\n"
    "        commit_write_pipe(out_pipe, res_id);\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "        *status = -1;\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_read_error(__read_only pipe int in_pipe, __global int *dst, __global int *status)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    int reserve_idx;\n"
    "    reserve_id_t res_id;\n"
    "\n"
    "    res_id = reserve_read_pipe(in_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        read_pipe(in_pipe, res_id, 0, &dst[gid]);\n"
    "        commit_read_pipe(in_pipe, res_id);\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "        *status = -1;\n"
    "    }\n"
    "}\n"
};


int test_pipe_readwrite_errors(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem        pipe;
    cl_mem      buffers[3];
    void        *outptr;
    cl_int        *inptr;
    cl_program  program;
    cl_kernel   kernel[2];
    size_t      global_work_size[3];
    cl_int      err;
    cl_int        size;
    cl_int      i;
    cl_int        status = 0;
    cl_event    producer_sync_event;
    cl_event    consumer_sync_event;
    MTdata      d = init_genrand( gRandomSeed );
    const char*    kernelName[] = {"test_pipe_write_error", "test_pipe_read_error"};

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = num_elements;

    size = num_elements * sizeof(cl_int);

    inptr = (cl_int *)align_malloc(size, min_alignment);

    for(i = 0; i < (cl_int)(size / sizeof(int)); i++){
        inptr[i] = (int)genrand_int32(d);
    }

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        print_error(err, " clCreateBuffer failed\n");
        return -1;
    }
    outptr = align_malloc(size, min_alignment);
    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size, outptr, &err);
    if ( err ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }
    buffers[2] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  sizeof(int), &status, &err);
    if ( err ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        align_free( outptr );
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }
    //Pipe created with max_packets less than global size
    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), num_elements - (num_elements/2), NULL, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        align_free( outptr );
        clReleaseMemObject(pipe);
        print_error(err, " clCreatePipe failed\n");
        return -1;
    }

    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, (const char**)&pipe_readwrite_errors_kernel_code, kernelName[0], "-cl-std=CL2.0");
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        print_error(err, "Error creating program\n");
        return -1;
    }
    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    if( kernel[1] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        print_error(err, "Error creating kernel\n");
        return -1;
    }

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void*)&buffers[2]);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void*)&buffers[2]);
    if ( err != CL_SUCCESS ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    // Launch Consumer kernel for empty pipe
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 0, NULL, &consumer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(status), &status, 1, &consumer_sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    if(status == 0){
        log_error("test_pipe_readwrite_errors failed\n");
        return -1;
    }
    else{
        status = 0;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(status), &status, 1, &producer_sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    if(status == 0){
        log_error("test_pipe_readwrite_errors failed\n");
        return -1;
    }
    else{
        status = 0;
    }

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(status), &status, 1, &consumer_sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    if(status == 0)
    {
        log_error("test_pipe_readwrite_errors failed\n");
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(buffers[2]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    log_info("test_pipe_readwrite_errors passed\n");
    //cleanup
    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[2]);
    clReleaseMemObject(pipe);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseEvent(producer_sync_event);
    clReleaseEvent(consumer_sync_event);
    clReleaseProgram(program);
    align_free(outptr);
    return 0;
}

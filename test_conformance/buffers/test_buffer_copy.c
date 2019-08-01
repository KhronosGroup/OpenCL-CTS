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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/errorHelpers.h"


static int verify_copy_buffer(int *inptr, int *outptr, int n)
{
    int         i;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int test_copy( cl_command_queue queue, cl_context context, int num_elements, MTdata d )
{
    cl_mem  buffers[2];
    cl_int  *int_input_ptr, *int_output_ptr;
    cl_int  err;
    int     i;
    int     src_flag_id, dst_flag_id;
    int     errors = 0;

    size_t  min_alignment = get_min_alignment(context);

    int_input_ptr = (cl_int*) align_malloc(sizeof(cl_int) * num_elements, min_alignment);
    int_output_ptr = (cl_int*)align_malloc(sizeof(cl_int) * num_elements, min_alignment);

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        for (dst_flag_id=0; dst_flag_id < NUM_FLAGS; dst_flag_id++) {
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            for (i=0; i<num_elements; i++){
                int_input_ptr[i] = (int)genrand_int32( d );
                int_output_ptr[i] = 0xdeaddead; // seed with incorrect data
            }

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  sizeof(cl_int) * num_elements, int_input_ptr, &err);
            else
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  sizeof(cl_int) * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" );
                align_free( (void *)int_input_ptr );
                align_free( (void *)int_output_ptr );
                return -1;
            }

            if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],  sizeof(cl_int) * num_elements, int_output_ptr, &err);
            else
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],  sizeof(cl_int) * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" );
                clReleaseMemObject( buffers[0] );
                align_free( (void *)int_input_ptr );
                align_free( (void *)int_output_ptr );
                return -1;
            }

            if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR)) {
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)int_input_ptr, 0, NULL, NULL);
                if ( err != CL_SUCCESS ){
                    print_error( err, "clEnqueueWriteBuffer failed" );
                    clReleaseMemObject( buffers[0] );
                    clReleaseMemObject( buffers[1] );
                    align_free( (void *)int_output_ptr );
                    align_free( (void *)int_input_ptr );
                    return -1;
                }
            }

            err = clEnqueueCopyBuffer(queue, buffers[0], buffers[1], 0, 0, sizeof(cl_int)*num_elements, 0, NULL, NULL);
            if ( err != CL_SUCCESS ){
                print_error( err, "clCopyArray failed" );
                clReleaseMemObject( buffers[0] );
                clReleaseMemObject( buffers[1] );
                align_free( (void *)int_output_ptr );
                align_free( (void *)int_input_ptr );
                return -1;
            }

            err = clEnqueueReadBuffer( queue, buffers[1], true, 0, sizeof(int)*num_elements, (void *)int_output_ptr, 0, NULL, NULL );
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueReadBuffer failed" );
                clReleaseMemObject( buffers[0] );
                clReleaseMemObject( buffers[1] );
                align_free( (void *)int_output_ptr );
                align_free( (void *)int_input_ptr );
                return -1;
            }

            if ( verify_copy_buffer(int_input_ptr, int_output_ptr, num_elements) ){
                log_error( " test failed\n" );
                errors++;
            }
            else{
                log_info( " test passed\n" );
            }
    // cleanup
            clReleaseMemObject( buffers[0] );
            clReleaseMemObject( buffers[1] );
        } // dst flags
    }  // src flags
    // cleanup
    align_free( (void *)int_output_ptr );
    align_free( (void *)int_input_ptr );

    return errors;

}   // end test_copy()


static int testPartialCopy( cl_command_queue queue, cl_context context, int num_elements, cl_uint srcStart, cl_uint dstStart, int size, MTdata d )
{
    cl_mem  buffers[2];
    int     *inptr, *outptr;
    cl_int  err;
    int     i;
    int     src_flag_id, dst_flag_id;
    int     errors = 0;

    size_t  min_alignment = get_min_alignment(context);

    inptr = (int *)align_malloc( sizeof(int) * num_elements, min_alignment);
    if ( ! inptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(int) * num_elements );
        return -1;
    }
    outptr = (int *)align_malloc( sizeof(int) * num_elements, min_alignment);
    if ( ! outptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(int) * num_elements );
        align_free( (void *)inptr );
        return -1;
    }

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        for (dst_flag_id=0; dst_flag_id < NUM_FLAGS; dst_flag_id++) {
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            for (i=0; i<num_elements; i++){
                inptr[i] = (int)genrand_int32( d );
                outptr[i] = (int)0xdeaddead;    // seed with incorrect data
            }

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  sizeof(cl_int) * num_elements, inptr, &err);
            else
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  sizeof(cl_int) * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" )
                align_free( (void *)outptr );
                align_free( (void *)inptr );
                return -1;
            }

            if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],  sizeof(cl_int) * num_elements, outptr, &err);
            else
                buffers[1] = clCreateBuffer(context, flag_set[dst_flag_id],  sizeof(cl_int) * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" );
                clReleaseMemObject( buffers[0] );
                align_free( (void *)outptr );
                align_free( (void *)inptr );
                return -1;
            }

            if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR)){
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)inptr, 0, NULL, NULL);
                if ( err != CL_SUCCESS ){
                    print_error( err, "clEnqueueWriteBuffer failed" );
                    clReleaseMemObject( buffers[1] );
                    clReleaseMemObject( buffers[0] );
                    align_free( (void *)outptr );
                    align_free( (void *)inptr );
                    return -1;
                }
            }

            err = clEnqueueCopyBuffer(queue, buffers[0], buffers[1], srcStart*sizeof(cl_int), dstStart*sizeof(cl_int), sizeof(cl_int)*size, 0, NULL, NULL);
            if ( err != CL_SUCCESS){
                print_error( err, "clEnqueueCopyBuffer failed" );
                clReleaseMemObject( buffers[1] );
                clReleaseMemObject( buffers[0] );
                align_free( (void *)outptr );
                align_free( (void *)inptr );
                return -1;
            }

            err = clEnqueueReadBuffer( queue, buffers[1], true, 0, sizeof(int)*num_elements, (void *)outptr, 0, NULL, NULL );
            if ( err != CL_SUCCESS){
                print_error( err, "clEnqueueReadBuffer failed" );
                clReleaseMemObject( buffers[1] );
                clReleaseMemObject( buffers[0] );
                align_free( (void *)outptr );
                align_free( (void *)inptr );
                return -1;
            }

            if ( verify_copy_buffer(inptr + srcStart, outptr + dstStart, size) ){
                log_error("buffer_COPY test failed\n");
                errors++;
            }
            else{
                log_info("buffer_COPY test passed\n");
            }
    // cleanup
            clReleaseMemObject( buffers[1] );
            clReleaseMemObject( buffers[0] );
        } // dst mem flags
    } // src mem flags
    // cleanup
    align_free( (void *)outptr );
    align_free( (void *)inptr );

    return errors;

}   // end testPartialCopy()


int test_buffer_copy( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int     i, err = 0;
    int     size;
    MTdata  d = init_genrand( gRandomSeed );

    // test the preset size
    log_info( "set size: %d: ", num_elements );
    if (test_copy( queue, context, num_elements, d ))
        err++;

    // now test random sizes
    for ( i = 0; i < 8; i++ ){
        size = (int)get_random_float(2.f,131072.f, d);
        log_info( "random size: %d: ", size );
        if (test_copy( queue, context, size, d ))
            err++;
    }

    free_mtdata(d);

    return err;

}   // end test_buffer_copy()


int test_buffer_partial_copy( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int     i, err = 0;
    int     size;
    cl_uint srcStart, dstStart;
    MTdata  d = init_genrand( gRandomSeed );

    // now test copy of partial sizes
    for ( i = 0; i < 8; i++ ){
        srcStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - 8), d );
        size = (int)get_random_float( 8.f, (float)(num_elements - srcStart), d );
        dstStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - size), d );
        log_info( "random partial copy from %d to %d, size: %d: ", (int)srcStart, (int)dstStart, size );
        if (testPartialCopy( queue, context, num_elements, srcStart, dstStart, size, d ))
            err++;
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_partial_copy()


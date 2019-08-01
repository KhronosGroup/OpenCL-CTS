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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"

//--- the code for the kernel executables
static const char *write_kernel_code =
"\n"
"__kernel void test_write(__global unsigned char *src, write_only image2d_t dstimg)\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    int            indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    float4         color;\n"
"\n"
"    indx *= 4;\n"
"    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);\n"
"    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


//--- the verify functions
static int verify_subimage( unsigned char *src, unsigned char *dst, size_t srcx, size_t srcy,
                           size_t dstx, size_t dsty, size_t subw, size_t subh, size_t pitch, size_t element_pitch )
{
    size_t        i, j, k;
    size_t        srcj, dstj;
    size_t        srcLoc, dstLoc;

    for( j = 0; j < subh; j++ ){
        srcj = ( j + srcy ) * pitch * element_pitch;
        dstj = ( j + dsty ) * pitch * element_pitch;
        for( i = 0; i < subw; i++ ){
            srcLoc = srcj + ( i + srcx ) * element_pitch;
            dstLoc = dstj + ( i + dstx ) * element_pitch;
            for( k = 0; k < element_pitch; k++ ){    // test each channel
                if( src[srcLoc+k] != dst[dstLoc+k] ){
                    return -1;
                }
            }
        }
    }

    return 0;
}


static int verify_copy_array( int *inptr, int *outptr, int n )
{
    int    i;

    for( i = 0; i < n; i++ ) {
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


//----- helper functions
static cl_uchar *generate_image( int n, MTdata d )
{
    cl_uchar   *ptr = (cl_uchar *)malloc( n );
    int i;

    for( i = 0; i < n; i++ )
        ptr[i] = (cl_uchar)genrand_int32(d);

    return ptr;
}


static int copy_size( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements, MTdata d )
{
    cl_mem                streams[2];
    cl_event            copyEvent;
    cl_ulong            queueStart, submitStart, writeStart, writeEnd;
    cl_int                *int_input_ptr, *int_output_ptr;
    int                    err = 0;
    int                    i;

    int_input_ptr = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    int_output_ptr = (cl_int*)malloc(sizeof(cl_int) * num_elements);

    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, &err );
    if( !streams[0] ){
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * num_elements, NULL, &err );
    if( !streams[1] ){
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    for (i=0; i<num_elements; i++){
        int_input_ptr[i] = (int)genrand_int32(d);
        int_output_ptr[i] = (int)genrand_int32(d) >> 30;    // seed with incorrect data
    }

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_int)*num_elements, (void *)int_input_ptr, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clWriteArray failed" );
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    err = clEnqueueCopyBuffer( queue, streams[0], streams[1], 0, 0, sizeof(cl_int)*num_elements, 0, NULL, &copyEvent );
    if( err != CL_SUCCESS ){
        print_error( err, "clCopyArray failed" );
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &copyEvent );
    if( err != CL_SUCCESS )
    {
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    // test profiling
    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int)*num_elements, (void *)int_output_ptr, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clEnqueueReadBuffer failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( (void *)int_output_ptr );
        free( (void *)int_input_ptr );
        return -1;
    }

    if( verify_copy_array(int_input_ptr, int_output_ptr, num_elements) ){
        log_error( "test failed\n" );
        err = -1;
    }
    else{
        log_info( "test passed\n" );
        err = 0;
    }

    // cleanup
    clReleaseEvent(copyEvent);
    clReleaseMemObject( streams[0] );
    clReleaseMemObject( streams[1] );
    free( (void *)int_output_ptr );
    free( (void *)int_input_ptr );

    if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
        err = -1;

    return err;

}    // end copy_size()


static int copy_partial_size( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements, cl_uint srcStart, cl_uint dstStart, int size, MTdata d )
{
    cl_mem                streams[2];
    cl_event            copyEvent;
    cl_ulong            queueStart, submitStart, writeStart, writeEnd;
    cl_int                *inptr, *outptr;
    int                    err = 0;
    int                    i;

    inptr = (cl_int *)malloc(sizeof(cl_int) * num_elements);
    outptr = (cl_int *)malloc(sizeof(cl_int) * num_elements);

    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, &err );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, &err );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    for (i=0; i<num_elements; i++){
        inptr[i] = (int)genrand_int32(d);
        outptr[i] = (int)get_random_float( -1.f, 1.f, d );    // seed with incorrect data
    }

    err = clEnqueueWriteBuffer(queue, streams[0], true, 0, sizeof(cl_int)*num_elements, (void *)inptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = clEnqueueCopyBuffer( queue, streams[0], streams[1], srcStart*sizeof(cl_int), dstStart*sizeof(cl_int),
                       sizeof(cl_int)*size, 0, NULL, &copyEvent );
    if( err != CL_SUCCESS){
        print_error( err, "clCopyArray failed" );
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &copyEvent );
    if( err != CL_SUCCESS )
    {
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }

    // test profiling
    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }


    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        free( outptr );
        free( inptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int)*num_elements, (void *)outptr, 0, NULL, NULL );
    if( err != CL_SUCCESS){
        log_error("clReadVariableStream failed\n");
        return -1;
    }

    if( verify_copy_array(inptr + srcStart, outptr + dstStart, size) ){
        log_error("test failed\n");
        err = -1;
    }
    else{
        log_info("test passed\n");
        err = 0;
    }

    // cleanup
    clReleaseEvent(copyEvent);
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    free(outptr);
    free(inptr);

    if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
        err = -1;

    return err;

}    // end copy_partial_size()


int test_copy_array( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int        i, err = 0;
    int        size;
    MTdata  d = init_genrand( gRandomSeed );

    // test the preset size
    log_info( "set size: %d: ", num_elements );
    err = copy_size( device, context, queue, num_elements, d );

    // now test random sizes
    for( i = 0; i < 8; i++ ){
        size = (int)get_random_float(2.f,131072.f, d);
        log_info( "random size: %d: ", size );
        err |= copy_size( device, context, queue, size, d );
    }

    free_mtdata(d);

    return err;

}    // end copy_array()


int test_copy_partial_array( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int        i, err = 0;
    int        size;
    cl_uint    srcStart, dstStart;
    MTdata  d = init_genrand( gRandomSeed );

    // now test copy of partial sizes
    for( i = 0; i < 8; i++ ){
        srcStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - 8), d );
        size = (int)get_random_float( 8.f, (float)(num_elements - srcStart), d );
        dstStart = (cl_uint)get_random_float( 0.f, (float)(num_elements - size), d );
        log_info( "random partial copy from %d to %d, size: %d: ", (int)srcStart, (int)dstStart, size );
        err |= copy_partial_size( device, context, queue, num_elements, srcStart, dstStart, size, d );
    }

    free_mtdata(d);
    return err;
}    // end copy_partial_array()


static int copy_image_size( cl_device_id device, cl_context context,
                                                        cl_command_queue queue, size_t srcx, size_t srcy,
                                                        size_t dstx, size_t dsty, size_t subw, size_t subh,
                                                        MTdata d )
{
    cl_mem                        memobjs[3];
    cl_program                program[1];
    cl_image_format        image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    cl_event                    copyEvent;
    cl_ulong                    queueStart, submitStart, writeStart, writeEnd;
    void                            *inptr;
    void                            *dst = NULL;
    cl_kernel                    kernel[1];
    size_t                        threads[2];
#ifdef USE_LOCAL_THREADS
    size_t                        localThreads[2];
#endif
    int                                err = 0;
    cl_mem_flags            flags;
    unsigned int            num_channels = 4;
    size_t                        w = 256, h = 256;
    size_t                        element_nbytes;
    size_t                        num_bytes;
    size_t                        channel_nbytes = sizeof( cl_char );


    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    element_nbytes = channel_nbytes * num_channels;
    num_bytes = w * h * element_nbytes;

    threads[0] = (size_t)w;
    threads[1] = (size_t)h;

#ifdef USE_LOCAL_THREADS
    err = clGetDeviceConfigInfo( id, CL_DEVICE_MAX_THREAD_GROUP_SIZE, localThreads, sizeof( cl_uint ), NULL );
    test_error( err, "Unable to get thread group max size" );
    localThreads[1] = localThreads[0];
    if( localThreads[0] > threads[0] )
        localThreads[0] = threads[0];
    if( localThreads[1] > threads[1] )
        localThreads[1] = threads[1];
#endif

    inptr = (void *)generate_image( (int)num_bytes, d );
    if( ! inptr ){
        log_error("unable to allocate inptr at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    dst = malloc( num_bytes );
    if( ! dst ){
        free( (void *)inptr );
        log_error("unable to allocate dst at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    // allocate the input image
    flags = (cl_mem_flags)(CL_MEM_READ_WRITE);
    memobjs[0] = create_image_2d(context, flags, &image_format_desc, w, h, 0, NULL, &err);
    if( memobjs[0] == (cl_mem)0 ) {
        free( dst );
        free( (void *)inptr );
        log_error("unable to create Image2D\n");
        return -1;
    }

    memobjs[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), num_bytes, NULL, &err );
    if( memobjs[1] == (cl_mem)0 ) {
        clReleaseMemObject(memobjs[0]);
        free( dst );
        free( (void *)inptr );
        log_error("unable to create array\n");
        return -1;
    }

    // allocate the input image
    memobjs[2] = create_image_2d(context, flags, &image_format_desc, w, h, 0, NULL, &err);
    if( memobjs[2] == (cl_mem)0 ) {
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( (void *)inptr );
        log_error("unable to create Image2D\n");
        return -1;
    }

    err = clEnqueueWriteBuffer( queue, memobjs[1], true, 0, num_bytes, inptr, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &write_kernel_code, "test_write" );
    if( err ){
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&memobjs[1] );
    err |= clSetKernelArg( kernel[0], 1, sizeof( cl_mem ), (void *)&memobjs[0] );
    if (err != CL_SUCCESS){
        log_error("clSetKernelArg failed\n");
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

#ifdef USE_LOCAL_THREADS
    err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, localThreads, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );
#endif
    if (err != CL_SUCCESS){
        print_error( err, "clEnqueueNDRangeKernel failed" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    // now do the copy
    size_t srcPt[3] = { srcx, srcy, 0 };
    size_t destPt[3] = { dstx, dsty, 0 };
    size_t region[3] = { subw, subh, 1 };
    err = clEnqueueCopyImage( queue, memobjs[0], memobjs[2], srcPt, destPt, region, 0, NULL, &copyEvent );
    if (err != CL_SUCCESS){
        print_error( err, "clCopyImage failed" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &copyEvent );
    if( err != CL_SUCCESS )
    {
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    // test profiling
    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( copyEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    size_t origin[3] = { 0, 0, 0 };
    size_t region2[3] = { w, h, 1 };
    err = clEnqueueReadImage( queue, memobjs[2], true, origin, region2, 0, 0, dst, 0, NULL, NULL );
    if (err != CL_SUCCESS){
        print_error( err, "clReadImage failed" );
        clReleaseEvent(copyEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[0] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[2] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = verify_subimage( (unsigned char *)inptr, (unsigned char *)dst, srcx, srcy,
                          dstx, dsty, subw, subh, w, 4 );
    //err = verify_image( (unsigned char *)inptr, (unsigned char *)dst, w * h * 4 );
    if( err ){
        log_error( "Image failed to verify.\n " );
    }
    else{
        log_info( "Image verified.\n" );
    }

    // cleanup
    clReleaseEvent(copyEvent);
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    clReleaseMemObject( memobjs[0] );
    clReleaseMemObject( memobjs[1] );
    clReleaseMemObject( memobjs[2] );
    free( dst );
    free( inptr );

    if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
        err = -1;

    return err;

}    // end copy_image_size()


int test_copy_image( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int            err = 0;
    int            i;
    size_t    srcx, srcy, dstx, dsty, subw, subh;
    MTdata    d;

    srcx = srcy = dstx = dsty = 0;
    subw = subh = 256;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    err = copy_image_size( device, context, queue, srcx, srcy, dstx, dsty, subw, subh, d );
    if( err ){
        log_error( "testing copy image, full size\n" );
    }
    else{
        log_info( "testing copy image, full size\n" );
    }

    // now test random sub images
    srcx = srcy = 0;
    subw = subh = 16;
    dstx = dsty = 0;
    err = copy_image_size( device, context, queue, srcx, srcy, dstx, dsty, subw, subh, d );
    if( err ){
        log_error( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                  (int)dstx, (int)dsty, (int)subw, (int)subh );
    }
    else{
        log_info( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                 (int)dstx, (int)dsty, (int)subw, (int)subh );
    }

    srcx = srcy = 8;
    subw = subh = 16;
    dstx = dsty = 32;
    err = copy_image_size( device, context, queue, srcx, srcy, dstx, dsty, subw, subh, d );
    if( err ){
        log_error( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                  (int)dstx, (int)dsty, (int)subw, (int)subh );
    }
    else{
        log_info( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                 (int)dstx, (int)dsty, (int)subw, (int)subh );
    }

    for( i = 0; i < 16; i++ ) {
        srcx = (size_t)get_random_float( 0.f, 248.f, d );
        srcy = (size_t)get_random_float( 0.f, 248.f, d );
        subw = (size_t)get_random_float( 8.f, (float)(256 - srcx), d );
        subh = (size_t)get_random_float( 8.f, (float)(256 - srcy), d );
        dstx = (size_t)get_random_float( 0.f, (float)(256 - subw), d );
        dsty = (size_t)get_random_float( 0.f, (float)(256 - subh), d );
        err = copy_image_size( device, context, queue, srcx, srcy, dstx, dsty, subw, subh, d );
        if( err ){
            log_error( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                      (int)dstx, (int)dsty, (int)subw, (int)subh );
        }
        else{
            log_info( "test copy of subimage size %d,%d  %d,%d  %d x %d\n", (int)srcx, (int)srcy,
                     (int)dstx, (int)dsty, (int)subw, (int)subh );
        }
    }

    free_mtdata(d);

    return err;

}    // end copy_image()


int test_copy_array_to_image( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem            memobjs[3];
    cl_image_format    image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    void            *inptr;
    void            *dst;
    int                err;
    cl_mem_flags    flags;
    unsigned int    num_channels = (unsigned int)get_format_channel_count( &image_format_desc );
    size_t            w = 256, h = 256;
    size_t            element_nbytes;
    size_t            num_bytes;
    size_t            channel_nbytes = sizeof( cl_char );
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    element_nbytes = channel_nbytes * num_channels;
    num_bytes = w * h * element_nbytes;
    d = init_genrand( gRandomSeed );
    inptr = (void *)generate_image( (int)num_bytes, d );
    free_mtdata(d); d = NULL;
    if( ! inptr ){
        log_error("unable to allocate inptr at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    dst = malloc( num_bytes );
    if( ! dst ){
        free( inptr );
        log_error( " unable to allocate dst at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    // allocate the input image
    flags = (cl_mem_flags)(CL_MEM_READ_WRITE);
    memobjs[0] = create_image_2d( context, flags, &image_format_desc, w, h, 0, NULL, &err );
    if( memobjs[0] == (cl_mem)0 ){
        free( dst );
        free( inptr );
        log_error( " unable to create Image2D\n" );
        return -1;
    }

    memobjs[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), channel_nbytes * num_channels*w*h, NULL, &err );
    if( memobjs[1] == (cl_mem)0 ) {
        clReleaseMemObject( memobjs[0] );
        free( dst );
        free( inptr );
        log_error( " unable to create array: " );
        return -1;
    }

    err = clEnqueueWriteBuffer( queue, memobjs[1], true, 0, num_bytes, (const void *)inptr, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clWriteArray failed" );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        free( dst );
        free( inptr );
        return -1;
    }

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { w, h, 1 };
    err = clEnqueueCopyBufferToImage( queue, memobjs[1], memobjs[0], 0, origin, region, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clCopyArrayToImage failed" );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = clEnqueueReadImage( queue, memobjs[0], true, origin, region, 0, 0, dst, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clReadImage failed" );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        free( dst );
        free( inptr );
        return -1;
    }

    err = verify_subimage( (cl_uchar *)inptr, (cl_uchar *)dst, 0, 0, 0, 0, w, h, w, num_channels );
    if( err ){
        log_error( " test failed: " );
    }
    else{
        log_info( " test passed: " );
    }

    // cleanup
    clReleaseMemObject( memobjs[1] );
    clReleaseMemObject( memobjs[0] );
    free( dst );
    free( inptr );

    return err;

}    // end copy_array_to_image()

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
#include "../../test_common/harness/testHarness.h"
#include "../../test_common/harness/parseParameters.h"

const char *sample_kernel_code_single_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n" };

size_t sample_single_line_lengths[1];

const char *sample_kernel_code_multi_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)",
"{",
"    int  tid = get_global_id(0);",
"",
"    dst[tid] = (int)src[tid];",
"",
"}" };

size_t sample_multi_line_lengths[ 7 ];

const char *sample_kernel_code_two_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n",
"__kernel void sample_test2(__global int *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)src[tid];\n"
"\n"
"}\n" };

size_t sample_two_line_lengths[1];

const char *sample_kernel_code_bad_multi_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)",
"{",
"    int  tid = get_global_id(0);thisisanerror",
"",
"    dst[tid] = (int)src[tid];",
"",
"}" };

size_t sample_bad_multi_line_lengths[ 7 ];


int test_load_program_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    size_t length;
    char *buffer;

    /* Preprocess: calc the length of each source file line */
    sample_single_line_lengths[ 0 ] = strlen( sample_kernel_code_single_line[ 0 ] );

    /* New OpenCL API only has one entry point, so go ahead and just try it */
    program = clCreateProgramWithSource( context, 1, sample_kernel_code_single_line, sample_single_line_lengths, &error);
    test_error( error, "Unable to create reference program" );

    /* Now get the source and compare against our original */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, NULL, NULL, &length );
    test_error( error, "Unable to get length of first program source" );

    // Note: according to spec section 5.4.5, the length returned should include the null terminator
    if( length != sample_single_line_lengths[0] + 1 )
    {
        log_error( "ERROR: Length of program (%ld) does not match reference length (%ld)!\n", length, sample_single_line_lengths[0] + 1  );
        return -1;
    }

    buffer = (char *)malloc( length );
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, length, buffer, NULL );
    test_error( error, "Unable to get buffer of first program source" );

    if( strcmp( (char *)buffer, sample_kernel_code_single_line[ 0 ] ) != 0 )
    {
        log_error( "ERROR: Program sources do not match!\n" );
        return -1;
    }

    /* All done */
    free( buffer );

    return 0;
}

int test_load_multistring_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;

    int i;


    /* Preprocess: calc the length of each source file line */
    for( i = 0; i < 7; i++ )
    {
        sample_multi_line_lengths[ i ] = strlen( sample_kernel_code_multi_line[ i ] );
    }

    /* Create another program using the macro function */
    program = clCreateProgramWithSource( context, 7, sample_kernel_code_multi_line, sample_multi_line_lengths, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build multi-line program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */

    return 0;
}

int test_load_two_kernel_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    int i;


    /* Preprocess: calc the length of each source file line */
    for( i = 0; i < 2; i++ )
    {
        sample_two_line_lengths[ i ] = strlen( sample_kernel_code_two_line[ i ] );
    }

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource( context, 2, sample_kernel_code_two_line, sample_two_line_lengths, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create two-kernel program!\n" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build two-kernel program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_load_null_terminated_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;


    /* Now create a program using the macro function */
    program = clCreateProgramWithSource( context, 1, sample_kernel_code_single_line, NULL, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_load_null_terminated_multi_line_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;


    /* Now create a program using the macro function */
    program = clCreateProgramWithSource( context, 7, sample_kernel_code_multi_line, NULL, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}


int test_load_discreet_length_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    int i;


    /* Preprocess: calc the length of each source file line */
    for( i = 0; i < 7; i++ )
    {
        sample_bad_multi_line_lengths[ i ] = strlen( sample_kernel_code_bad_multi_line[ i ] );
    }

    /* Now force the length of the third line to skip the actual error */
    sample_bad_multi_line_lengths[2] -= strlen("thisisanerror");

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource( context, 7, sample_kernel_code_bad_multi_line, sample_bad_multi_line_lengths, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_load_null_terminated_partial_multi_line_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    int i;

    /* Preprocess: calc the length of each source file line */
    for( i = 0; i < 7; i++ )
    {
        if( i & 0x01 )
            sample_multi_line_lengths[ i ] = 0; /* Should force for null-termination on this line only */
        else
            sample_multi_line_lengths[ i ] = strlen( sample_kernel_code_multi_line[ i ] );
    }

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource( context, 7, sample_kernel_code_multi_line, sample_multi_line_lengths, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_get_program_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_device_id device1;
    cl_context context1;
    size_t paramSize;
    cl_uint numInstances;


    error = create_single_kernel_helper_create_program(context, &program, 1, sample_kernel_code_single_line);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Test that getting the device works. */
    device1 = (cl_device_id)0xbaadfeed;
    error = clGetProgramInfo( program, CL_PROGRAM_DEVICES, sizeof( device1 ), &device1, NULL );
    test_error( error, "Unable to get device of program" );

  /* Since the device IDs are opaque types we check the CL_DEVICE_VENDOR_ID which is unique for identical hardware. */
  cl_uint device1_vid, deviceID_vid;
  error = clGetDeviceInfo(device1, CL_DEVICE_VENDOR_ID, sizeof(device1_vid), &device1_vid, NULL );
  test_error( error, "Unable to get device CL_DEVICE_VENDOR_ID" );
  error = clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR_ID, sizeof(deviceID_vid), &deviceID_vid, NULL );
  test_error( error, "Unable to get device CL_DEVICE_VENDOR_ID" );

    if( device1_vid != deviceID_vid )
    {
        log_error( "ERROR: Incorrect device returned for program! (Expected vendor ID 0x%x, got 0x%x)\n", deviceID_vid, device1_vid );
        return -1;
    }

    cl_uint devCount;
    error = clGetProgramInfo( program, CL_PROGRAM_NUM_DEVICES, sizeof( devCount ), &devCount, NULL );
    test_error( error, "Unable to get device count of program" );

    if( devCount != 1 )
    {
        log_error( "ERROR: Invalid device count returned for program! (Expected 1, got %d)\n", (int)devCount );
        return -1;
    }

    context1 = (cl_context)0xbaadfeed;
    error = clGetProgramInfo( program, CL_PROGRAM_CONTEXT, sizeof( context1 ), &context1, NULL );
    test_error( error, "Unable to get device of program" );

    if( context1 != context )
    {
        log_error( "ERROR: Invalid context returned for program! (Expected %p, got %p)\n", context, context1 );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_REFERENCE_COUNT, sizeof( numInstances ), &numInstances, NULL );
    test_error( error, "Unable to get instance count" );

    /* While we're at it, test the sizes of programInfo too */
    error = clGetProgramInfo( program, CL_PROGRAM_DEVICES, NULL, NULL, &paramSize );
    test_error( error, "Unable to get device param size" );
    if( paramSize != sizeof( cl_device_id ) )
    {
        log_error( "ERROR: Size returned for device is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_CONTEXT, NULL, NULL, &paramSize );
    test_error( error, "Unable to get context param size" );
    if( paramSize != sizeof( cl_context ) )
    {
        log_error( "ERROR: Size returned for context is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_REFERENCE_COUNT, NULL, NULL, &paramSize );
    test_error( error, "Unable to get instance param size" );
    if( paramSize != sizeof( cl_uint ) )
    {
        log_error( "ERROR: Size returned for num instances is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_NUM_DEVICES, NULL, NULL, &paramSize );
    test_error( error, "Unable to get device count param size" );
    if( paramSize != sizeof( cl_uint ) )
    {
        log_error( "ERROR: Size returned for device count is wrong!\n" );
        return -1;
    }

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_get_program_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_program program;
    int error;
    char buffer[10240];
    size_t length;


    error = create_single_kernel_helper_create_program(context, &program, 1, sample_kernel_code_single_line);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create test program!\n" );
        return -1;
    }

    /* Try getting the length */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, NULL, NULL, &length );
    test_error( error, "Unable to get program source length" );
    if (length != strlen(sample_kernel_code_single_line[0]) + 1 && !gOfflineCompiler)
    {
        log_error( "ERROR: Length returned for program source is incorrect!\n" );
        return -1;
    }

    /* Try normal source */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, sizeof( buffer ), buffer, NULL );
    test_error( error, "Unable to get program source" );
    if (strlen(buffer) != strlen(sample_kernel_code_single_line[0]) && !gOfflineCompiler)
    {
        log_error( "ERROR: Length of program source is incorrect!\n" );
        return -1;
    }

    /* Try both at once */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get program source" );
    if (strlen(buffer) != strlen(sample_kernel_code_single_line[0]) && !gOfflineCompiler)
    {
        log_error( "ERROR: Length of program source is incorrect!\n" );
        return -1;
    }
    if (length != strlen(sample_kernel_code_single_line[0]) + 1 && !gOfflineCompiler)
    {
        log_error( "ERROR: Returned length of program source is incorrect!\n" );
        return -1;
    }

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

int test_get_program_build_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_program program;
    int error;
    char *buffer;
    size_t length, newLength;
    cl_build_status status;


    error = create_single_kernel_helper_create_program(context, &program, 1, sample_kernel_code_single_line);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create test program!\n" );
        return -1;
    }

    /* Make sure getting the length works */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, 0, NULL, &length );
    test_error( error, "Unable to get program build status length" );
    if( length != sizeof( status ) )
    {
        log_error( "ERROR: Returned length of program build status is invalid! (Expected %d, got %d)\n", (int)sizeof( status ), (int)length );
        return -1;
    }

    /* Now actually build it and verify the status */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Unable to build program source" );

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
    test_error( error, "Unable to get program build status" );
    if( status != CL_BUILD_SUCCESS )
    {
        log_error( "ERROR: Getting built program build status did not return CL_BUILD_SUCCESS! (%d)\n", (int)status );
        return -1;
    }

    /***** Build log *****/

    /* Try getting the length */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &length );
    test_error( error, "Unable to get program build log length" );

    log_info("Build log is %ld long.\n", length);

    buffer = (char*)malloc(length);

    /* Try normal source */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_LOG, length, buffer, NULL );
    test_error( error, "Unable to get program build log" );

    if( buffer[length-1] != '\0' )
    {
        log_error( "clGetProgramBuildInfo overwrote allocated space for build log! '%c'\n", buffer[length-1]  );
        return -1;
    }

    /* Try both at once */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_LOG, length, buffer, &newLength );
    test_error( error, "Unable to get program build log" );

    free(buffer);

    /***** Build options *****/
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &length );
    test_error( error, "Unable to get program build options length" );

    buffer = (char*)malloc(length);

    /* Try normal source */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_OPTIONS, length, buffer, NULL );
    test_error( error, "Unable to get program build options" );

    /* Try both at once */
    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_OPTIONS, length, buffer, &newLength );
    test_error( error, "Unable to get program build options" );

    free(buffer);

    /* Try with a valid option */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    program = clCreateProgramWithSource( context, 1, sample_kernel_code_single_line, NULL, &error );
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create test program!\n" );
        return -1;
    }

    error = clBuildProgram( program, 1, &deviceID, "-cl-opt-disable", NULL, NULL );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Building with valid options failed!" );
        return -1;
    }

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_OPTIONS, NULL, NULL, &length );
    test_error( error, "Unable to get program build options" );

    buffer = (char*)malloc(length);

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_OPTIONS, length, buffer, NULL );
    test_error( error, "Unable to get program build options" );
    if( strcmp( (char *)buffer, "-cl-opt-disable" ) != 0 )
    {
        log_error( "ERROR: Getting program build options for program with -cl-opt-disable build options did not return expected value (got %s)\n", buffer );
        return -1;
    }

    /* All done */
    free( buffer );

    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

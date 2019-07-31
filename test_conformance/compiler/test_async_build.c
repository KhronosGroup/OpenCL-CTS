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

#if !defined(_WIN32)
#include <unistd.h>
#endif


const char *sample_async_kernel[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n" };

volatile int       buildNotificationSent;

void CL_CALLBACK test_notify_build_complete( cl_program program, void *userData )
{
    if( userData == NULL || strcmp( (char *)userData, "userData" ) != 0 )
    {
        log_error( "ERROR: User data passed in to build notify function was not correct!\n" );
        buildNotificationSent = -1;
    }
    else
        buildNotificationSent = 1;
    log_info( "\n   <-- program successfully built\n" );
}

int test_async_build(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_build_status status;


    buildNotificationSent = 0;

    /* First, test by doing the slow method of the individual calls */
    error = create_single_kernel_helper_create_program(context, &program, 1, sample_async_kernel);
    test_error(error, "Unable to create program from source");

    /* Compile the program */
    error = clBuildProgram( program, 1, &deviceID, NULL, test_notify_build_complete, (void *)"userData" );
    test_error( error, "Unable to build program source" );

    /* Wait for build to complete (just keep polling, since we're just a test */
    if( ( error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL ) ) != CL_SUCCESS )
    {
        print_error( error, "Unable to get program build status" );
        return -1;
    }
    while( (int)status == CL_BUILD_IN_PROGRESS )
    {
        log_info( "\n  -- still waiting for build... (status is %d)", status );
        sleep( 1 );
        error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
        test_error( error, "Unable to get program build status" );
    }

    if( status != CL_BUILD_SUCCESS )
    {
        log_error( "ERROR: build failed! (status: %d)\n", (int)status );
        return -1;
    }

    if( buildNotificationSent == 0 )
    {
        log_error( "ERROR: Async build completed, but build notification was not sent!\n" );
        return -1;
    }

    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

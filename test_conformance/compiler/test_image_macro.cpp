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
#if (defined( __APPLE__ ) || defined( __linux__ ))
#include <unistd.h>
#endif

const char * image_supported_source = "kernel void enabled(global int * buf) { \r\n" \
"int n = get_global_id(0); \r\n"\
"buf[n] = 0; \r\n "\
"#ifndef __IMAGE_SUPPORT__ \r\n" \
"ERROR; \r\n"\
"#endif \r\n"\
"\r\n } \r\n";


const char * image_not_supported_source = "kernel void not_enabled(global int * buf) { \r\n" \
"int n = get_global_id(0); \r\n"\
"buf[n] = 0; \r\n "\
"#ifdef __IMAGE_SUPPORT__ \r\n" \
"ERROR; \r\n"\
"#endif \r\n"\
"\r\n } \r\n";


int test_image_macro(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_bool image_support;
    char buf[256];
    int status;
    cl_program program;

    status = clGetDeviceInfo( deviceID, CL_DEVICE_NAME, sizeof( buf ), buf, NULL );
    if( status )
    {
      log_error( "getting device info (name): %d\n", status );
      exit(-1);
    }

    status = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof( image_support ), &image_support, NULL );
    if( status )
    {
      log_error( "getting device info (image support): %d\n", status );
      return status;
    }

    if (image_support == CL_TRUE)
    {
        status = create_single_kernel_helper_create_program(context, &program, 1, (const char**)&image_supported_source);

        if( status )
        {
            log_error ("Failure creating program, [%d] \n", status );
            return status;
        }

        status = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
        if( status )
            log_error("CL_DEVICE_IMAGE_SUPPORT is set, __IMAGE_SUPPORT__ macro not set \n");
        else
            log_info("CL_DEVICE_IMAGE_SUPPORT is set, __IMAGE_SUPPORT__ macro is set \n");
    }
    else
    {
        status = create_single_kernel_helper_create_program(context, &program, 1, (const char**)&image_not_supported_source);
        if( status )
        {
            log_error ("Failure creating program, [%d] \n", status );
            return status;
        }

        status = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
        if( status )
            log_error("CL_DEVICE_IMAGE_SUPPORT not set, __IMAGE_SUPPORT__ macro is set \n");
        else
            log_info("CL_DEVICE_IMAGE_SUPPORT not set, __IMAGE_SUPPORT__ macro not set \n");
    }

    status = clReleaseProgram( program );
    if( status )
    {
        log_error ("Unable to release program object, [%d] \n", status );
        return status;
    }

    return status;
}


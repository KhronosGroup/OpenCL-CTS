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
#include <limits.h>
#include <ctype.h>
#ifndef _WIN32
#include <unistd.h>
#endif




const char *known_extensions[] = {
    "cl_khr_byte_addressable_store",
    "cl_khr_3d_image_writes",
    "cl_khr_fp16",
    "cl_khr_fp64",
    "cl_khr_global_int32_base_atomics",
    "cl_khr_global_int32_extended_atomics",
    "cl_khr_local_int32_base_atomics",
    "cl_khr_local_int32_extended_atomics",
    "cl_khr_int64_base_atomics",
    "cl_khr_int64_extended_atomics",
    "cl_khr_select_fprounding_mode",
    "cl_khr_depth_images",
    "cl_khr_gl_depth_images",
    "cl_khr_gl_msaa_sharing",
    "cl_khr_device_enqueue_local_arg_types",
    "cl_khr_subgroups",
    "cl_khr_mipmap_image",
    "cl_khr_mipmap_image_writes",
    "cl_khr_srgb_image_writes",
    "cl_khr_subgroup_named_barrier",

    //API-only extensions after this point.  If you add above here, modify first_API_extension below.
    "cl_khr_icd",
    "cl_khr_gl_sharing",
    "cl_khr_gl_event",
    "cl_khr_d3d10_sharing",
    "cl_khr_d3d11_sharing",
    "cl_khr_dx9_media_sharing",
    "cl_khr_egl_event",
    "cl_khr_egl_image",
    "cl_khr_image2d_from_buffer",
    "cl_khr_spir",
    "cl_khr_il_program",
    "cl_khr_create_command_queue",
    "cl_khr_initialize_memory",
    "cl_khr_terminate_context",
    "cl_khr_priority_hints",
    "cl_khr_throttle_hints",
    "cl_khr_spirv_no_integer_wrap_decoration",
    "cl_khr_extended_versioning",
};

size_t num_known_extensions = sizeof(known_extensions)/sizeof(char*);
size_t first_API_extension = 20;

const char *known_embedded_extensions[] = {
    "cles_khr_int64",
    NULL
};

typedef enum
{
    kUnsupported_extension = -1,
    kVendor_extension = 0,
    kLanguage_extension = 1,
    kAPI_extension = 2
}Extension_Type;

const char *kernel_strings[] = {
    "kernel void test(global int *defines)\n{\n",
    "#pragma OPENCL EXTENSION %s : enable\n",
    "#ifdef %s\n"
    "  defines[%d] = 1;\n"
    "#else\n"
    "  defines[%d] = 0;\n"
    "#endif\n",
    "#pragma OPENCL EXTENSION %s : disable\n\n",
    "}\n"
};

int test_compiler_defines_for_extensions(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{

    int error;
    int total_errors = 0;


    // Get the extensions string for the device
    size_t size;
    error = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &size);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS size failed");

    char *extensions = (char*)malloc(sizeof(char)*(size + 1));
    if (extensions == 0) {
        log_error("Failed to allocate memory for extensions string.\n");
        return -1;
    }
    memset( extensions, CHAR_MIN, sizeof(char)*(size+1) );

    error = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(char)*size, extensions, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS failed");

    // Check to make sure the extension string is NUL terminated.
    if( extensions[size] != CHAR_MIN )
    {
        test_error( -1, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS wrote past the end of the array!" );
        return -1;
    }
    extensions[size] = '\0';    // set last char to NUL to avoid problems with string functions later

    // test for termination with '\0'
    size_t stringSize = strlen( extensions );
    if( stringSize == size )
    {
        test_error( -1, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS is not NUL terminated!" );
        return -1;
    }

    // Break up the extensions
    log_info("Device reports the following extensions:\n");
    char *extensions_supported[1024];
    Extension_Type extension_type[1024];
    int num_of_supported_extensions = 0;
    char *currentP = extensions;

    memset( extension_type, 0, sizeof( extension_type) );

    // loop over extension string
    while (currentP != extensions + stringSize)
    {
        // skip leading white space
        while( *currentP == ' ' )
            currentP++;

        // Exit if end of string
        if( *currentP == '\0' )
        {
            if( currentP != extensions + stringSize)
            {
                test_error( -1, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS contains a NUL in the middle of the string!" );
                return -1;
            }
            break;
        }

        // Not space, not end of string, so extension
        char *start = currentP;             // start of extension name

        // loop looking for the end
        while (*currentP != ' ' && currentP != extensions + stringSize)
        {
            // check for non-space white space in the extension name
            if( isspace(*currentP) )
            {
                test_error( -1, "clGetDeviceInfo for CL_DEVICE_EXTENSIONS contains a non-space whitespace in an extension name!" );
                return -1;
            }
            currentP++;
        }

        // record the extension name
        uintptr_t extension_length = (uintptr_t) currentP - (uintptr_t) start;
        extensions_supported[ num_of_supported_extensions ] = (char*) malloc( (extension_length + 1) * sizeof( char ) );
        if( NULL == extensions_supported[ num_of_supported_extensions ] )
        {
            log_error( "Error: unable to allocate memory to hold extension name: %ld chars\n", extension_length );
            return -1;
        }
        memcpy( extensions_supported[ num_of_supported_extensions ], start, extension_length * sizeof( char ) );
        extensions_supported[ num_of_supported_extensions ][extension_length] = '\0';

        // If the extension is a cl_khr extension, make sure it is an approved cl_khr extension -- looking for misspellings here
        if( extensions_supported[ num_of_supported_extensions ][0] == 'c'  &&
            extensions_supported[ num_of_supported_extensions ][1] == 'l'  &&
            extensions_supported[ num_of_supported_extensions ][2] == '_'  &&
            extensions_supported[ num_of_supported_extensions ][3] == 'k'  &&
            extensions_supported[ num_of_supported_extensions ][4] == 'h'  &&
            extensions_supported[ num_of_supported_extensions ][5] == 'r'  &&
            extensions_supported[ num_of_supported_extensions ][6] == '_' )
        {
            size_t ii;
            for( ii = 0; ii < num_known_extensions; ii++ )
            {
                if( 0 == strcmp( known_extensions[ii], extensions_supported[ num_of_supported_extensions ] ) )
                    break;
            }
            if( ii == num_known_extensions )
            {
                log_error( "FAIL: Extension %s is not in the list of approved Khronos extensions!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }
        }
        // Is it an embedded extension?
        else if( memcmp( extensions_supported[ num_of_supported_extensions ], "cles_khr_", 9 ) == 0 )
        {
            // Yes, but is it a known one?
            size_t ii;
            for( ii = 0; known_embedded_extensions[ ii ] != NULL; ii++ )
            {
                if( strcmp( known_embedded_extensions[ ii ], extensions_supported[ num_of_supported_extensions ] ) == 0 )
                    break;
            }
            if( known_embedded_extensions[ ii ] == NULL )
            {
                log_error( "FAIL: Extension %s is not in the list of approved Khronos embedded extensions!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }

            // It's approved, but are we even an embedded system?
            char profileStr[128] = "";
            error = clGetDeviceInfo( device, CL_DEVICE_PROFILE, sizeof( profileStr ), &profileStr, NULL );
            test_error( error, "Unable to get CL_DEVICE_PROFILE to validate embedded extension name" );

            if( strcmp( profileStr, "EMBEDDED_PROFILE" ) != 0 )
            {
                log_error( "FAIL: Extension %s is an approved embedded extension, but on a non-embedded profile!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }
        }
        else
        { // All other extensions must be of the form cl_<vendor_name>_<name>
            if( extensions_supported[ num_of_supported_extensions ][0] != 'c'  ||
                extensions_supported[ num_of_supported_extensions ][1] != 'l'  ||
                extensions_supported[ num_of_supported_extensions ][2] != '_' )
            {
                log_error( "FAIL:  Extension %s doesn't start with \"cl_\"!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }

            if( extensions_supported[ num_of_supported_extensions ][3] == '_' || extensions_supported[ num_of_supported_extensions ][3] == '\0' )
            {
                log_error( "FAIL:  Vendor name is missing in extension %s!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }

            // look for the second underscore for name
            char *p = extensions_supported[ num_of_supported_extensions ] + 4;
            while( *p != '\0' && *p != '_' )
                p++;

            if( *p != '_' || p[1] == '\0')
            {
                log_error( "FAIL:  extension name is missing in extension %s!", extensions_supported[ num_of_supported_extensions ] );
                return -1;
            }
        }


        num_of_supported_extensions++;
    }

    // Build a list of the known extensions that are not supported by the device
    char *extensions_not_supported[1024];
    int num_not_supported_extensions = 0;
    for( int i = 0; i < num_of_supported_extensions; i++ )
    {
        int is_supported = 0;
        for( size_t j = 0; j < num_known_extensions; j++ )
            {
            if( strcmp( extensions_supported[ i ], known_extensions[ j ] ) == 0 )
            {
                extension_type[ i ] = ( j < first_API_extension ) ? kLanguage_extension : kAPI_extension;
                is_supported = 1;
                break;
            }
        }
        if( !is_supported )
        {
            for( int j = 0; known_embedded_extensions[ j ] != NULL; j++ )
            {
                if( strcmp( extensions_supported[ i ], known_embedded_extensions[ j ] ) == 0 )
                {
                    extension_type[ i ] = kLanguage_extension;
                    is_supported = 1;
                    break;
                }
            }
        }
        if (!is_supported) {
            extensions_not_supported[num_not_supported_extensions] = (char*)malloc(strlen(extensions_supported[i])+1);
            strcpy(extensions_not_supported[num_not_supported_extensions], extensions_supported[i]);
            num_not_supported_extensions++;
        }
    }

    for (int i=0; i<num_of_supported_extensions; i++) {
        log_info("%40s -- Supported\n", extensions_supported[i]);
    }
    for (int i=0; i<num_not_supported_extensions; i++) {
        log_info("%40s -- Not Supported\n", extensions_not_supported[i]);
    }

    // Build the kernel
    char *kernel_code = (char*)malloc(1025*256*(num_not_supported_extensions+num_of_supported_extensions));
    memset(kernel_code, 0, 1025*256*(num_not_supported_extensions+num_of_supported_extensions));

    int i, index = 0;
    strcat(kernel_code, kernel_strings[0]);
    for (i=0; i<num_of_supported_extensions; i++, index++) {

        if (extension_type[i] == kLanguage_extension)
            sprintf(kernel_code + strlen(kernel_code), kernel_strings[1], extensions_supported[i]);

        sprintf(kernel_code + strlen(kernel_code), kernel_strings[2], extensions_supported[i], index, index );

        if (extension_type[i] == kLanguage_extension)
            sprintf(kernel_code + strlen(kernel_code), kernel_strings[3], extensions_supported[i] );
    }
    for ( i = 0; i<num_not_supported_extensions; i++, index++) {
        sprintf(kernel_code + strlen(kernel_code), kernel_strings[2], extensions_not_supported[i], index, index );
    }
    strcat(kernel_code, kernel_strings[4]);

    // Now we need to execute the kernel
    cl_mem defines;
    cl_int *data;
    cl_program program;
    cl_kernel kernel;

    Version version = get_device_cl_version(device);

    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&kernel_code, "test", version < Version(2,0) ? "" : "-cl-std=CL2.0");
    test_error(error, "create_single_kernel_helper failed");

    data = (cl_int*)malloc(sizeof(cl_int)*(num_not_supported_extensions+num_of_supported_extensions));
    memset(data, 0, sizeof(cl_int)*(num_not_supported_extensions+num_of_supported_extensions));
    defines = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                             sizeof(cl_int)*(num_not_supported_extensions+num_of_supported_extensions), data, &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(defines), &defines);
    test_error(error, "clSetKernelArg failed");

    size_t global_size = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, defines, CL_TRUE, 0, sizeof(cl_int)*(num_not_supported_extensions+num_of_supported_extensions),
                                data, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    // Report what the compiler reported
    log_info("\nCompiler reported the following extensions defined in the OpenCL C kernel environment:\n");
    index = 0;
    int total_supported = 0;
    for (int i=0; i<num_of_supported_extensions; i++, index++) {
        if (data[index] == 1) {
            log_info("\t%s\n", extensions_supported[i]);
            total_supported++;
        }
    }
    for (int i=0; i<num_not_supported_extensions; i++, index++) {
        if (data[index] == 1) {
            log_info("\t%s\n", extensions_not_supported[i]);
            total_supported++;
        }
    }
    if (total_supported == 0)
        log_info("\t(none)\n");

    // Count the errors
    index = 0;
    int unknown = 0;
    for ( i=0; i<num_of_supported_extensions; i++)
    {
        if (data[i] != 1)
        {
            switch( extension_type[i] )
            {
                case kLanguage_extension:
                    log_error("ERROR: Supported extension %s not defined in kernel.\n", extensions_supported[i]);
                    total_errors++;
                    break;
                case kVendor_extension:
                    unknown++;
                    break;
                case kAPI_extension:
                    break;
                default:
                    log_error( "ERROR: internal test error in extension detection.  This is probably a bug in the test.\n" );
                    break;
            }
        }
    }

    if(unknown)
    {
        log_info( "\nThe following non-KHR extensions are supported but do not add a preprocessor symbol to OpenCL C.\n" );
        for (int z=0; z<num_of_supported_extensions; z++)
        {
            if (data[z] != 1 && extension_type[z] == kVendor_extension )
                log_info( "\t%s\n", extensions_supported[z]);
        }
    }

    for ( ; i<num_not_supported_extensions; i++) {
        if (data[i] != 0) {
            log_error("ERROR: Unsupported extension %s is defined in kernel.\n", extensions_not_supported[i]);
            total_errors++;
        }
    }
    log_info("\n");

    // cleanup
    free(data);
    free(kernel_code);
    for(i=0; i<num_of_supported_extensions; i++) {
      free(extensions_supported[i]);
    }
    free(extensions);
    if( defines ) {
        error = clReleaseMemObject( defines );
        test_error( error, "Unable to release memory object" );
    }

    if (total_errors)
        return -1;
    return 0;
}

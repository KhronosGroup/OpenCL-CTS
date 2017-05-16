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
#include <ctype.h>

// Test __FILE__, __LINE__, __OPENCL_VERSION__, __OPENCL_C_VERSION__, __ENDIAN_LITTLE__, __ROUNDING_MODE__, __IMAGE_SUPPORT__, __FAST_RELAXED_MATH__
// __kernel_exec

const char *preprocessor_test = {
    "#line 2 \"%s\"\n"
    "__kernel void test( __global int *results, __global char *outFileString, __global char *outRoundingString )\n"
    "{\n"

    // Integer preprocessor macros
    "#ifdef __IMAGE_SUPPORT__\n"
    "    results[0] =    __IMAGE_SUPPORT__;\n"
    "#else\n"
    "    results[0] = 0xf00baa;\n"
    "#endif\n"

    "#ifdef __ENDIAN_LITTLE__\n"
    "    results[1] =    __ENDIAN_LITTLE__;\n"
    "#else\n"
    "    results[1] = 0xf00baa;\n"
    "#endif\n"

    "#ifdef __OPENCL_VERSION__\n"
    "    results[2] =    __OPENCL_VERSION__;\n"
    "#else\n"
    "    results[2] = 0xf00baa;\n"
    "#endif\n"

    "#ifdef __OPENCL_C_VERSION__\n"
    "    results[3] =    __OPENCL_C_VERSION__;\n"
    "#else\n"
    "    results[3] = 0xf00baa;\n"
    "#endif\n"

    "#ifdef __LINE__\n"
    "    results[4] =    __LINE__;\n"
    "#else\n"
    "    results[4] = 0xf00baa;\n"
    "#endif\n"

#if 0 // Removed by Affie's request 2/24
    "#ifdef __FAST_RELAXED_MATH__\n"
    "    results[5] =    __FAST_RELAXED_MATH__;\n"
    "#else\n"
    "    results[5] = 0xf00baa;\n"
    "#endif\n"
#endif

    "#ifdef __kernel_exec\n"
    "    results[6] = 1;\n"    // By spec, we can only really evaluate that it is defined, not what it expands to
    "#else\n"
    "    results[6] = 0xf00baa;\n"
    "#endif\n"

    // String preprocessor macros. Technically, there are strings in OpenCL, but not really.
    "#ifdef __FILE__\n"
    "    int i;\n"
    "    constant char *f = \"\" __FILE__;\n"
    "   for( i = 0; f[ i ] != 0 && i < 512; i++ )\n"
    "        outFileString[ i ] = f[ i ];\n"
    "    outFileString[ i ] = 0;\n"
    "#else\n"
    "    outFileString[ 0 ] = 0;\n"
    "#endif\n"

    "}\n"
    };

int test_kernel_preprocessor_macros(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 3 ];

    int error;
    size_t    threads[] = {1,1,1};

    cl_int results[ 7 ];
    cl_char fileString[ 512 ] = "", roundingString[ 128 ] = "";
    char programSource[4096];
    char curFileName[512];
    char *programPtr = programSource;
    int i = 0;
    snprintf(curFileName, 512, "%s", __FILE__);
#ifdef _WIN32
    // Replace "\" with "\\"
    while(curFileName[i] != '\0') {
        if (curFileName[i] == '\\') {
            int j = i + 1;
            char prev = '\\';
            while (curFileName[j - 1] != '\0') {
                char tmp = curFileName[j];
                curFileName[j] = prev;
                prev = tmp;
                j++;
            }
            i++;
        }
        i++;
    }
#endif
    sprintf(programSource,preprocessor_test,curFileName);

    // Create the kernel
    if( create_single_kernel_helper( context, &program, &kernel, 1,  (const char **)&programPtr, "test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(results), NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(fileString), NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(roundingString), NULL, &error);
    test_error( error, "Creating test array failed" );

    // Set up and run
    for( int i = 0; i < 3; i++ )
    {
        error = clSetKernelArg( kernel, i, sizeof( streams[i] ), &streams[i] );
        test_error( error, "Unable to set indexed kernel arguments" );
    }

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(results), results, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );
    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(fileString), fileString, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );
    error = clEnqueueReadBuffer( queue, streams[2], CL_TRUE, 0, sizeof(roundingString), roundingString, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );


    /////// Check the integer results

    // We need to check these values against what we know is supported on the device
    if( checkForImageSupport( deviceID ) == 0 )
    {
        // If images are supported, the constant should have been defined to the value 1
        if( results[ 0 ] == 0xf00baa )
        {
            log_error( "ERROR: __IMAGE_SUPPORT__ undefined even though images are supported\n" );
            return -1;
        }
        else if( results[ 0 ] != 1 )
        {
            log_error( "ERROR: __IMAGE_SUPPORT__ defined, but to the wrong value (defined as %d, spec states it should be 1)\n", (int)results[ 0 ] );
            return -1;
        }
    }
    else
    {
        // If images aren't supported, the constant should be undefined
        if( results[ 0 ] != 0xf00baa )
        {
            log_error( "ERROR: __IMAGE_SUPPORT__ defined to value %d even though images aren't supported", (int)results[ 0 ] );
            return -1;
        }
    }

    // __ENDIAN_LITTLE__ is similar to __IMAGE_SUPPORT__: 1 if it's true, undefined if it isn't
    cl_bool deviceIsLittleEndian;
    error = clGetDeviceInfo( deviceID, CL_DEVICE_ENDIAN_LITTLE, sizeof( deviceIsLittleEndian ), &deviceIsLittleEndian, NULL );
    test_error( error, "Unable to get endian property of device to validate against" );

    if( deviceIsLittleEndian )
    {
        if( results[ 1 ] == 0xf00baa )
        {
            log_error( "ERROR: __ENDIAN_LITTLE__ undefined even though the device is little endian\n" );
            return -1;
        }
        else if( results[ 1 ] != 1 )
        {
            log_error( "ERROR: __ENDIAN_LITTLE__ defined, but to the wrong value (defined as %d, spec states it should be 1)\n", (int)results[ 1 ] );
            return -1;
        }
    }
    else
    {
        if( results[ 1 ] != 0xf00baa )
        {
            log_error( "ERROR: __ENDIAN_LITTLE__ defined to value %d even though the device is not little endian (should be undefined per spec)", (int)results[ 1 ] );
            return -1;
        }
    }

    // __OPENCL_VERSION__
    if( results[ 2 ] == 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __OPENCL_VERSION__ undefined!" );
        return -1;
    }

    // The OpenCL version reported by the macro reports the feature level supported by the compiler. Since
    // this doesn't directly match any property we can query, we just check to see if it's a sane value
    char versionBuffer[ 128 ];
    error = clGetDeviceInfo( deviceID, CL_DEVICE_VERSION, sizeof( versionBuffer ), versionBuffer, NULL );
    test_error( error, "Unable to get device's version to validate against" );

    // We need to parse to get the version number to compare against
    char *p1, *p2, *p3;
    for( p1 = versionBuffer; ( *p1 != 0 ) && !isdigit( *p1 ); p1++ )
        ;
    for( p2 = p1; ( *p2 != 0 ) && ( *p2 != '.' ); p2++ )
        ;
    for( p3 = p2; ( *p3 != 0 ) && ( *p3 != ' ' ); p3++ )
        ;

    if( p2 == p3 )
    {
        log_error( "ERROR: Unable to verify OpenCL version string (platform string is incorrect format)\n" );
        return -1;
    }
    *p2 = 0;
    *p3 = 0;
    int major = atoi( p1 );
    int minor = atoi( p2 + 1 );
    int realVersion = ( major * 100 ) + ( minor * 10 );
    if( ( results[ 2 ] < 100 ) || ( results[ 2 ] > realVersion ) )
    {
        log_error( "ERROR: Kernel preprocessor __OPENCL_VERSION__ does not make sense w.r.t. device's version string! "
                  "(preprocessor states %d, real version is %d (%d.%d))\n", results[ 2 ], realVersion, major, minor );
        return -1;
    }

    // __OPENCL_C_VERSION__
    if( results[ 3 ] == 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __OPENCL_C_VERSION__ undefined!\n" );
        return -1;
    }

    // The OpenCL C version reported by the macro reports the OpenCL C supported by the compiler for this OpenCL device.
    char cVersionBuffer[ 128 ];
    error = clGetDeviceInfo( deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof( cVersionBuffer ), cVersionBuffer, NULL );
    test_error( error, "Unable to get device's OpenCL C version to validate against" );

    // We need to parse to get the version number to compare against
    for( p1 = cVersionBuffer; ( *p1 != 0 ) && !isdigit( *p1 ); p1++ )
        ;
    for( p2 = p1; ( *p2 != 0 ) && ( *p2 != '.' ); p2++ )
        ;
    for( p3 = p2; ( *p3 != 0 ) && ( *p3 != ' ' ); p3++ )
        ;

    if( p2 == p3 )
    {
        log_error( "ERROR: Unable to verify OpenCL C version string (platform string is incorrect format)\n" );
        return -1;
    }
    *p2 = 0;
    *p3 = 0;
    major = atoi( p1 );
    minor = atoi( p2 + 1 );
    realVersion = ( major * 100 ) + ( minor * 10 );
    if( ( results[ 3 ] < 100 ) || ( results[ 3 ] > realVersion ) )
    {
        log_error( "ERROR: Kernel preprocessor __OPENCL_C_VERSION__ does not make sense w.r.t. device's version string! "
                  "(preprocessor states %d, real version is %d (%d.%d))\n", results[ 2 ], realVersion, major, minor );
        return -1;
    }

    // __LINE__
    if( results[ 4 ] == 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __LINE__ undefined!" );
        return -1;
    }

    // This is fun--we get to search for where __LINE__ actually is so we know what line it should define to!
    // Note: it shows up twice, once for the #ifdef, and the other for the actual result output
    const char *linePtr = strstr( preprocessor_test, "__LINE__" );
    if( linePtr == NULL )
    {
        log_error( "ERROR: Nonsensical NULL pointer encountered!" );
        return -2;
    }
    linePtr = strstr( linePtr + strlen( "__LINE__" ), "__LINE__" );
    if( linePtr == NULL )
    {
        log_error( "ERROR: Nonsensical NULL pointer encountered!" );
        return -2;
    }

    // Now count how many carriage returns are before the string
    const char *retPtr = strchr( preprocessor_test, '\n' );
    int retCount = 1;
    for( ; ( retPtr < linePtr ) && ( retPtr != NULL ); retPtr = strchr( retPtr + 1, '\n' ) )
        retCount++;

    if( retCount != results[ 4 ] )
    {
        log_error( "ERROR: Kernel preprocessor __LINE__ does not expand to the actual line number! (expanded to %d, but was on line %d)\n",
                  results[ 4 ], retCount );
        return -1;
    }

#if 0 // Removed by Affie's request 2/24
    // __FAST_RELAXED_MATH__
    // Since create_single_kernel_helper does NOT define -cl-fast-relaxed-math, this should be undefined
    if( results[ 5 ] != 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __FAST_RELAXED_MATH__ defined even though build option was not used (should be undefined)\n" );
        return -1;
    }
#endif

    // __kernel_exec
    // We can ONLY check to verify that it is defined
    if( results[ 6 ] == 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __kernel_exec must be defined\n" );
        return -1;
    }

    //// String preprocessors

    // Since we provided the program directly, __FILE__ should compile to "<program source>".
    if( fileString[ 0 ] == 0 )
    {
        log_error( "ERROR: Kernel preprocessor __FILE__ undefined!\n" );
        return -1;
    }
    else if( strncmp( (char *)fileString, __FILE__, 512 ) != 0 )
    {
        log_info( "WARNING: __FILE__ defined, but to an unexpected value (%s)\n\tShould be: \"%s\"", fileString, __FILE__ );
        return -1;
    }


#if 0 // Removed by Affie's request 2/24
    // One more try through: try with -cl-fast-relaxed-math to make sure the appropriate preprocessor gets defined
    clProgramWrapper programB = clCreateProgramWithSource( context, 1, preprocessor_test, NULL, &error );
    test_error( error, "Unable to create test program" );

    // Try compiling
    error = clBuildProgram( programB, 1, &deviceID, "-cl-fast-relaxed-math", NULL, NULL );
    test_error( error, "Unable to build program" );

    // Create a kernel again to run against
    clKernelWrapper kernelB = clCreateKernel( programB, "test", &error );
    test_error( error, "Unable to create testing kernel" );

    // Set up and run
    for( int i = 0; i < 3; i++ )
    {
        error = clSetKernelArg( kernelB, i, sizeof( streams[i] ), &streams[i] );
        test_error( error, "Unable to set indexed kernel arguments" );
    }

    error = clEnqueueNDRangeKernel( queue, kernelB, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    // Only need the one read
    error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(results), results, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );

    // We only need to check the one result this time
    if( results[ 5 ] == 0xf00baa )
    {
        log_error( "ERROR: Kernel preprocessor __FAST_RELAXED_MATH__ not defined!\n" );
        return -1;
    }
    else if( results[ 5 ] != 1 )
    {
        log_error( "ERROR: Kernel preprocessor __FAST_RELAXED_MATH__ not defined to 1 (was %d)\n", results[ 5 ] );
        return -1;
    }
#endif

    return 0;
}


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
#include "harness/os_helpers.h"

const char *preprocessor_test_kernel[] = {
"__kernel void sample_test(__global int *dst)\n"
"{\n"
"    dst[0] = TEST_MACRO;\n"
"\n"
"}\n" };

const char *preprocessor_existence_test_kernel[] = {
    "__kernel void sample_test(__global int *dst)\n"
    "{\n"
    "#ifdef TEST_MACRO\n"
    "    dst[0] = 42;\n"
    "#else\n"
    "    dst[0] = 24;\n"
    "#endif\n"
    "\n"
    "}\n" };

const char *include_test_kernel[] = {
"#include \"./testIncludeFile.h\"\n"
"__kernel void sample_test(__global int *dst)\n"
"{\n"
"    dst[0] = HEADER_FOUND;\n"
"\n"
"}\n" };

const char *options_test_kernel[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    size_t tid = get_global_id(0);\n"
"    dst[tid] = src[tid];\n"
"}\n" };

const char *optimization_options[] = {
    "-cl-single-precision-constant",
    "-cl-denorms-are-zero",
    "-cl-opt-disable",
    "-cl-mad-enable",
    "-cl-no-signed-zeros",
    "-cl-unsafe-math-optimizations",
    "-cl-finite-math-only",
    "-cl-fast-relaxed-math",
    "-w",
    "-Werror",
#if defined( __APPLE__ )
    "-cl-opt-enable",
    "-cl-auto-vectorize-enable"
#endif
    };

cl_int get_result_from_program( cl_context context, cl_command_queue queue, cl_program program, cl_int *outValue )
{
    cl_int error;
    clKernelWrapper kernel = clCreateKernel( program, "sample_test", &error );
    test_error( error, "Unable to create kernel from program" );

    clMemWrapper outStream;
    outStream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int), NULL, &error );
    test_error( error, "Unable to create test buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( outStream ), &outStream );
    test_error( error, "Unable to set kernel argument" );

    size_t threads[1] = { 1 };

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    error = clEnqueueReadBuffer( queue, outStream, true, 0, sizeof( cl_int ), outValue, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    return CL_SUCCESS;
}

int test_options_build_optimizations(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_build_status status;

    for(size_t i = 0; i < sizeof(optimization_options) / (sizeof(char*)); i++) {

        clProgramWrapper program;
        error = create_single_kernel_helper_create_program(context, &program, 1, options_test_kernel, optimization_options[i]);
        if( program == NULL || error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create reference program!\n" );
            return -1;
        }

        /* Build with the macro defined */
        log_info("Testing optimization option '%s'\n", optimization_options[i]);
        error = clBuildProgram( program, 1, &deviceID, optimization_options[i], NULL, NULL );
        test_error( error, "Test program did not properly build" );

        error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
        test_error( error, "Unable to get program build status" );

        if( (int)status != CL_BUILD_SUCCESS )
        {
            log_info("Building with optimization option '%s' failed to compile!\n", optimization_options[i]);
            print_error( error, "Failed to build with optimization defined")
            return -1;
        }
    }
    return 0;
}

int test_options_build_macro(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    cl_build_status status;


    program = clCreateProgramWithSource( context, 1, preprocessor_test_kernel, NULL, &error );
    if( program == NULL || error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Build with the macro defined */
    error = clBuildProgram( program, 1, &deviceID, "-DTEST_MACRO=1 ", NULL, NULL );
    test_error( error, "Test program did not properly build" );

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
    test_error( error, "Unable to get program build status" );

    if( (int)status != CL_BUILD_SUCCESS )
    {
        print_error( error, "Failed to build with macro defined" );
        return -1;
    }


    // Go ahead and run the program to verify results
    cl_int firstResult, secondResult;

    error = get_result_from_program( context, queue, program, &firstResult );
    test_error( error, "Unable to get result from first program" );

    if( firstResult != 1 )
    {
        log_error( "ERROR: Result from first program did not validate! (Expected 1, got %d)\n", firstResult );
        return -1;
    }

    // Rebuild with a different value for the define macro, to make sure caching behaves properly
    error = clBuildProgram( program, 1, &deviceID, "-DTEST_MACRO=5 ", NULL, NULL );
    test_error( error, "Test program did not properly rebuild" );

    error = get_result_from_program( context, queue, program, &secondResult );
    test_error( error, "Unable to get result from second program" );

    if( secondResult != 5 )
    {
        if( secondResult == firstResult )
            log_error( "ERROR: Program result did not change with device macro change (program was not recompiled)!\n" );
        else
            log_error( "ERROR: Result from second program did not validate! (Expected 5, got %d)\n", secondResult );
        return -1;
    }

    return 0;
}

int test_options_build_macro_existence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;


    // In this case, the program should still run without the macro, but it should give a different result
    program = clCreateProgramWithSource( context, 1, preprocessor_existence_test_kernel, NULL, &error );
    if( program == NULL || error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Build without the macro defined */
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Test program did not properly build" );

    // Go ahead and run the program to verify results
    cl_int firstResult, secondResult;

    error = get_result_from_program( context, queue, program, &firstResult );
    test_error( error, "Unable to get result from first program" );

    if( firstResult != 24 )
    {
        log_error( "ERROR: Result from first program did not validate! (Expected 24, got %d)\n", firstResult );
        return -1;
    }

    // Now compile again with the macro defined and verify a change in results
    error = clBuildProgram( program, 1, &deviceID, "-DTEST_MACRO", NULL, NULL );
    test_error( error, "Test program did not properly build" );

    error = get_result_from_program( context, queue, program, &secondResult );
    test_error( error, "Unable to get result from second program" );

    if( secondResult != 42 )
    {
        if( secondResult == firstResult )
            log_error( "ERROR: Program result did not change with device macro addition (program was not recompiled)!\n" );
        else
            log_error( "ERROR: Result from second program did not validate! (Expected 42, got %d)\n", secondResult );
        return -1;
    }

    return 0;
}

int test_options_include_directory(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;

    std::string sep  = dir_sep();
    std::string path = exe_dir();    // Directory where test executable is located.
    std::string include_dir;

    clProgramWrapper program;
    cl_build_status status;

    /* Try compiling the program first without the directory included Should fail. */
    program = clCreateProgramWithSource( context, 1, include_test_kernel, NULL, &error );
    if( program == NULL || error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Build with the include directory defined */
    include_dir = "-I " + path + sep + "includeTestDirectory";

//    log_info("%s\n", include_dir);
    error = clBuildProgram( program, 1, &deviceID, include_dir.c_str(), NULL, NULL );
    test_error( error, "Test program did not properly build" );

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
    test_error( error, "Unable to get program build status" );

    if( (int)status != CL_BUILD_SUCCESS )
    {
        print_error( error, "Failed to build with include directory" );
        return -1;
    }

    // Go ahead and run the program to verify results
    cl_int firstResult, secondResult;

    error = get_result_from_program( context, queue, program, &firstResult );
    test_error( error, "Unable to get result from first program" );

    if( firstResult != 12 )
    {
        log_error( "ERROR: Result from first program did not validate! (Expected 12, got %d)\n", firstResult );
        return -1;
    }

    // Rebuild with a different include directory
    include_dir = "-I " + path + sep + "secondIncludeTestDirectory";
    error = clBuildProgram( program, 1, &deviceID, include_dir.c_str(), NULL, NULL );
    test_error( error, "Test program did not properly rebuild" );

    error = get_result_from_program( context, queue, program, &secondResult );
    test_error( error, "Unable to get result from second program" );

    if( secondResult != 42 )
    {
        if( secondResult == firstResult )
            log_error( "ERROR: Program result did not change with include path change (program was not recompiled)!\n" );
        else
            log_error( "ERROR: Result from second program did not validate! (Expected 42, got %d)\n", secondResult );
        return -1;
    }

    return 0;
}

const char *denorm_test_kernel[] = {
    "__kernel void sample_test( float src1, float src2, __global float *dst)\n"
    "{\n"
    "    dst[ 0 ] = src1 + src2;\n"
    "\n"
    "}\n" };

cl_int get_float_result_from_program( cl_context context, cl_command_queue queue, cl_program program, cl_float inA, cl_float inB, cl_float *outValue )
{
    cl_int error;

    clKernelWrapper kernel = clCreateKernel( program, "sample_test", &error );
    test_error( error, "Unable to create kernel from program" );

    clMemWrapper outStream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_float), NULL, &error );
    test_error( error, "Unable to create test buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( cl_float ), &inA );
    test_error( error, "Unable to set kernel argument" );

    error = clSetKernelArg( kernel, 1, sizeof( cl_float ), &inB );
    test_error( error, "Unable to set kernel argument" );

    error = clSetKernelArg( kernel, 2, sizeof( outStream ), &outStream );
    test_error( error, "Unable to set kernel argument" );

    size_t threads[1] = { 1 };

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    error = clEnqueueReadBuffer( queue, outStream, true, 0, sizeof( cl_float ), outValue, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    return CL_SUCCESS;
}

int test_options_denorm_cache(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;

    clProgramWrapper program;
    cl_build_status status;


    // If denorms aren't even supported, testing this flag is pointless
    cl_device_fp_config floatCaps = 0;
    error = clGetDeviceInfo( deviceID, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(floatCaps), &floatCaps,  NULL);
    test_error( error, "Unable to get device FP config" );
    if( ( floatCaps & CL_FP_DENORM ) == 0 )
    {
        log_info( "Device does not support denormalized single-precision floats; skipping test.\n" );
        return 0;
    }

    program = clCreateProgramWithSource( context, 1, denorm_test_kernel, NULL, &error );
    test_error( error, "Unable to create test program" );

    // Build first WITH the denorm flush flag
    error = clBuildProgram( program, 1, &deviceID, "-cl-denorms-are-zero", NULL, NULL );
    test_error( error, "Test program did not properly build" );

    error = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof( status ), &status, NULL );
    test_error( error, "Unable to get program build status" );

    if( (int)status != CL_BUILD_SUCCESS )
    {
        print_error( error, "Failed to build with include directory" );
        return -1;
    }

    // Note: the following in floating point is a subnormal number, thus adding two of them together
    // should give us a subnormalized result. If denormals are flushed to zero, however, it'll give us zero instead
    uint32_t intSubnormal = 0x00000001;
    cl_float *input = (cl_float *)&intSubnormal;
    cl_float firstResult, secondResult;

    error = get_float_result_from_program( context, queue, program, *input, *input, &firstResult );
    test_error( error, "Unable to get result from first program" );

    // Note: since -cl-denorms-are-zero is a HINT, not a requirement, the result we got could
    // either be subnormal (hint ignored) or zero (hint respected). Since either is technically
    // valid, there isn't anything we can to do validate results for now

    // Rebuild without flushing flag set
    error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
    test_error( error, "Test program did not properly rebuild" );

    error = get_float_result_from_program( context, queue, program, *input, *input, &secondResult );
    test_error( error, "Unable to get result from second program" );

    // Now, there are three possiblities here:
    // 1. The denorms-are-zero hint is not respected, in which case the first and second result will be identical
    // 2. The hint is respected, and the program was properly rebuilt, in which case the first result will be zero and the second non-zero
    // 3. The hint is respected, but the program was not properly rebuilt, in which case both results will be zero
    // 3 is the only error condition we need to look for
    uint32_t *fPtr = (uint32_t *)&firstResult;
    uint32_t *sPtr = (uint32_t *)&secondResult;

    if( ( *fPtr == 0 ) && ( *sPtr == 0 ) )
    {
        log_error( "ERROR: Program result didn't change when -cl-denorms-are-zero flag was removed.\n"
                  "First result (should be zero): 0x%08x, Second result (should be non-zero): 0x%08x\n",
                  *fPtr, *sPtr );
        return -1;
    }

    return 0;
}


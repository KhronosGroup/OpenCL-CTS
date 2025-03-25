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
#include "harness/testHarness.h"
#include "harness/parseParameters.h"

#include <array>
#include <memory>
#include <vector>

const char *sample_kernel_code_single_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n" };

const char *sample_kernel_code_multi_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)",
"{",
"    int  tid = get_global_id(0);",
"",
"    dst[tid] = (int)src[tid];",
"",
"}" };

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

const char *sample_kernel_code_bad_multi_line[] = {
"__kernel void sample_test(__global float *src, __global int *dst)",
"{",
"    int  tid = get_global_id(0);thisisanerror",
"",
"    dst[tid] = (int)src[tid];",
"",
"}" };

const char *sample_multi_kernel_code_with_macro = R"(
__kernel void sample_test_A(__global float *src, __global int *dst)
{
    size_t  tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}

#ifdef USE_SAMPLE_TEST_B
__kernel void sample_test_B(__global float *src, __global int *dst)
{
    size_t  tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
#endif

__kernel void sample_test_C(__global float *src, __global int *dst)
{
    size_t  tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
)";

const char *sample_multi_kernel_code_AB_with_macro = R"(
__kernel void sample_test_A(__global float *src, __global int *dst)
{
    size_t tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
#ifdef USE_SAMPLE_TEST_B
__kernel void sample_test_B(__global float *src, __global int *dst)
{
    size_t tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
#endif
)";

const char *sample_multi_kernel_code_CD_with_macro = R"(
__kernel void sample_test_C(__global float *src, __global int *dst)
{
    size_t tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
#ifdef USE_SAMPLE_TEST_D
__kernel void sample_test_D(__global float *src, __global int *dst)
{
    size_t tid = get_global_id(0);
    dst[tid] = (int)src[tid];
}
#endif
)";

REGISTER_TEST(load_program_source)
{
    int error;
    clProgramWrapper program;
    size_t length;
    char *buffer;

    /* Preprocess: calc the length of the source file line */
    size_t line_length = strlen(sample_kernel_code_single_line[0]);

    /* New OpenCL API only has one entry point, so go ahead and just try it */
    program = clCreateProgramWithSource(
        context, 1, sample_kernel_code_single_line, &line_length, &error);
    test_error( error, "Unable to create reference program" );

    /* Now get the source and compare against our original */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, 0, NULL, &length );
    test_error( error, "Unable to get length of first program source" );

    // Note: according to spec section 5.4.5, the length returned should include the null terminator
    if (length != line_length + 1)
    {
        log_error("ERROR: Length of program (%zu) does not match reference "
                  "length (%zu)!\n",
                  length, line_length + 1);
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

REGISTER_TEST(load_multistring_source)
{
    int error;
    clProgramWrapper program;

    int i;

    constexpr int num_lines = ARRAY_SIZE(sample_kernel_code_multi_line);

    /* Preprocess: calc the length of each source file line */
    size_t line_lengths[num_lines];
    for (i = 0; i < num_lines; i++)
    {
        line_lengths[i] = strlen(sample_kernel_code_multi_line[i]);
    }

    /* Create another program using the macro function */
    program = clCreateProgramWithSource(context, num_lines,
                                        sample_kernel_code_multi_line,
                                        line_lengths, &error);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build multi-line program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */

    return 0;
}

REGISTER_TEST(load_two_kernel_source)
{
    int error;
    cl_program program;

    int i;

    constexpr int num_lines = ARRAY_SIZE(sample_kernel_code_two_line);

    /* Preprocess: calc the length of each source file line */
    size_t line_lengths[num_lines];
    for (i = 0; i < num_lines; i++)
    {
        line_lengths[i] = strlen(sample_kernel_code_two_line[i]);
    }

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource(
        context, num_lines, sample_kernel_code_two_line, line_lengths, &error);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create two-kernel program!\n" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build two-kernel program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(load_null_terminated_source)
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
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(load_null_terminated_multi_line_source)
{
    int error;
    cl_program program;


    int num_lines = ARRAY_SIZE(sample_kernel_code_multi_line);

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource(
        context, num_lines, sample_kernel_code_multi_line, NULL, &error);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(load_discreet_length_source)
{
    int error;
    cl_program program;

    int i;

    constexpr int num_lines = ARRAY_SIZE(sample_kernel_code_bad_multi_line);

    /* Preprocess: calc the length of each source file line */
    size_t line_lengths[num_lines];
    for (i = 0; i < num_lines; i++)
    {
        line_lengths[i] = strlen(sample_kernel_code_bad_multi_line[i]);
    }

    /* Now force the length of the third line to skip the actual error */
    static_assert(num_lines >= 3, "expected at least 3 lines in source");
    line_lengths[2] -= strlen("thisisanerror");

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource(context, num_lines,
                                        sample_kernel_code_bad_multi_line,
                                        line_lengths, &error);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(load_null_terminated_partial_multi_line_source)
{
    int error;
    cl_program program;
    int i;

    constexpr int num_lines = ARRAY_SIZE(sample_kernel_code_multi_line);

    /* Preprocess: calc the length of each source file line */
    size_t line_lengths[num_lines];
    for (i = 0; i < num_lines; i++)
    {
        if( i & 0x01 )
            line_lengths[i] =
                0; /* Should force for null-termination on this line only */
        else
            line_lengths[i] = strlen(sample_kernel_code_multi_line[i]);
    }

    /* Now create a program using the macro function */
    program = clCreateProgramWithSource(context, num_lines,
                                        sample_kernel_code_multi_line,
                                        line_lengths, &error);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create null-terminated program!" );
        return -1;
    }

    /* Try compiling */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build null-terminated program source" );

    /* Should probably check binary here to verify the same results... */

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(get_program_info)
{
    int error;
    cl_program program;
    cl_device_id device1;
    cl_context context1;
    size_t paramSize;
    cl_uint numInstances;

    error = create_single_kernel_helper_create_program(context, &program, 1, sample_kernel_code_single_line);
    test_error(error, "create_single_kernel_helper_create_program failed");

    if( program == NULL )
    {
        log_error( "ERROR: Unable to create reference program!\n" );
        return -1;
    }

    /* Test that getting the device works. */
    device1 = (cl_device_id)0xbaadfeed;
    error = clGetProgramInfo( program, CL_PROGRAM_DEVICES, sizeof( device1 ), &device1, NULL );
    test_error( error, "Unable to get device of program" );

    /* Object comparability test. */
    test_assert_error(device1 == device,
                      "Unexpected result returned by CL_PROGRAM_DEVICES query");

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
    error = clGetProgramInfo( program, CL_PROGRAM_DEVICES, 0, NULL, &paramSize );
    test_error( error, "Unable to get device param size" );
    if( paramSize != sizeof( cl_device_id ) )
    {
        log_error( "ERROR: Size returned for device is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_CONTEXT, 0, NULL, &paramSize );
    test_error( error, "Unable to get context param size" );
    if( paramSize != sizeof( cl_context ) )
    {
        log_error( "ERROR: Size returned for context is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_REFERENCE_COUNT, 0, NULL, &paramSize );
    test_error( error, "Unable to get instance param size" );
    if( paramSize != sizeof( cl_uint ) )
    {
        log_error( "ERROR: Size returned for num instances is wrong!\n" );
        return -1;
    }

    error = clGetProgramInfo( program, CL_PROGRAM_NUM_DEVICES, 0, NULL, &paramSize );
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

REGISTER_TEST(get_program_info_kernel_names)
{
    int error = CL_SUCCESS;
    size_t total_kernels = 0;
    size_t kernel_names_len = 0;

    clProgramWrapper program = nullptr;

    // 1) Program without build call. Query CL_PROGRAM_NUM_KERNELS and check
    // that it fails with CL_INVALID_PROGRAM_EXECUTABLE. Query
    // CL_PROGRAM_KERNEL_NAMES and check that it fails with
    // CL_INVALID_PROGRAM_EXECUTABLE.
    {
        program = clCreateProgramWithSource(
            context, 1, &sample_multi_kernel_code_with_macro, nullptr, &error);
        test_error(error, "clCreateProgramWithSource failed");

        error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
                                 sizeof(size_t), &total_kernels, nullptr);
        test_failure_error(error, CL_INVALID_PROGRAM_EXECUTABLE,
                           "Unexpected clGetProgramInfo result");

        error = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                 &kernel_names_len);
        test_failure_error(error, CL_INVALID_PROGRAM_EXECUTABLE,
                           "Unexpected clGetProgramInfo result");
    }

    // 2) Build the program with the preprocessor macro undefined.
    //    Query CL_PROGRAM_NUM_KERNELS and check that the correct number is
    //    returned. Query CL_PROGRAM_KERNEL_NAMES and check that the right
    //    kernel names are returned.
    {
        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "clBuildProgram failed");

        error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
                                 sizeof(size_t), &total_kernels, nullptr);
        test_error(error, "clGetProgramInfo failed");

        test_assert_error(total_kernels == 2,
                          "Unexpected clGetProgramInfo result");

        error = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                 &kernel_names_len);
        test_error(error, "clGetProgramInfo failed");

        std::vector<std::string> actual_names = { "sample_test_A",
                                                  "sample_test_C" };

        const size_t len = kernel_names_len + 1;
        std::vector<char> kernel_names(len, '\0');
        error =
            clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, kernel_names_len,
                             kernel_names.data(), &kernel_names_len);
        test_error(error, "Unable to get kernel names list.");

        std::string program_names = kernel_names.data();
        for (const auto &name : actual_names)
        {
            test_assert_error(program_names.find(name) != std::string::npos,
                              "Unexpected kernel name");
        }

        test_assert_error(program_names.find("sample_test_B")
                              == std::string::npos,
                          "sample_test_B should not be present");
    }

    // 3) Build the program again with the preprocessor macro defined.
    //    Query CL_PROGRAM_NUM_KERNELS and check that the correct number is
    //    returned. Query CL_PROGRAM_KERNEL_NAMES and check that the right
    //    kernel names are returned.
    {
        const char *build_options = "-DUSE_SAMPLE_TEST_B";
        error = clBuildProgram(program, 1, &device, build_options, nullptr,
                               nullptr);
        test_error(error, "clBuildProgram failed");

        error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
                                 sizeof(size_t), &total_kernels, nullptr);
        test_error(error, "clGetProgramInfo failed");

        test_assert_error(total_kernels == 3,
                          "Unexpected clGetProgramInfo result");

        error = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                 &kernel_names_len);
        test_error(error, "clGetProgramInfo failed");

        std::vector<std::string> actual_names = { "sample_test_A",
                                                  "sample_test_B",
                                                  "sample_test_C" };

        const size_t len = kernel_names_len + 1;
        std::vector<char> kernel_names(len, '\0');
        error =
            clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, kernel_names_len,
                             kernel_names.data(), &kernel_names_len);
        test_error(error, "Unable to get kernel names list.");

        std::string program_names = kernel_names.data();
        for (const auto &name : actual_names)
        {
            test_assert_error(program_names.find(name) != std::string::npos,
                              "Unexpected kernel name");
        }
    }
    return CL_SUCCESS;
}

REGISTER_TEST(get_linked_program_info_kernel_names)
{
    int error = CL_SUCCESS;
    size_t total_kernels = 0;
    size_t kernel_names_len = 0;

    clProgramWrapper program_AB = clCreateProgramWithSource(
        context, 1, &sample_multi_kernel_code_AB_with_macro, nullptr, &error);
    test_error(error, "clCreateProgramWithSource failed");

    clProgramWrapper program_CD = clCreateProgramWithSource(
        context, 1, &sample_multi_kernel_code_CD_with_macro, nullptr, &error);
    test_error(error, "clCreateProgramWithSource failed");

    clProgramWrapper program = nullptr;

    // 1) Compile and link the programs with the preprocessor macro undefined.
    //    Query CL_PROGRAM_NUM_KERNELS and check that the correct number is
    //    returned. Query CL_PROGRAM_KERNEL_NAMES and check that the right
    //    kernel names are returned.
    {
        error =
            clCompileProgram(program_AB, 1, &device, nullptr, 0, 0, 0, 0, 0);
        test_error(error, "clCompileProgram failed");

        error =
            clCompileProgram(program_CD, 1, &device, nullptr, 0, 0, 0, 0, 0);
        test_error(error, "clCompileProgram failed");

        cl_program progs[] = { program_AB, program_CD };
        program =
            clLinkProgram(context, 1, &device, "", 2, progs, 0, 0, &error);
        test_error(error, "clLinkProgram failed");

        error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
                                 sizeof(size_t), &total_kernels, nullptr);
        test_error(error, "clGetProgramInfo failed");

        test_assert_error(total_kernels == 2,
                          "Unexpected clGetProgramInfo result");

        error = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                 &kernel_names_len);
        test_error(error, "clGetProgramInfo failed");

        const size_t len = kernel_names_len + 1;
        std::vector<char> kernel_names(len, '\0');
        error =
            clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, kernel_names_len,
                             kernel_names.data(), &kernel_names_len);
        test_error(error, "Unable to get kernel names list.");
        std::string program_names = kernel_names.data();

        std::vector<std::string> expected_names = { "sample_test_A",
                                                    "sample_test_C" };
        for (const auto &name : expected_names)
        {
            test_assert_error(program_names.find(name) != std::string::npos,
                              "Unexpected kernel name");
        }

        std::vector<std::string> unexpected_names = { "sample_test_B",
                                                      "sample_test_D" };
        for (const auto &name : unexpected_names)
        {
            test_assert_error(program_names.find(name) == std::string::npos,
                              "Unexpected kernel name");
        }
    }

    // 2) Compile and link the programs with the preprocessor macro defined.
    //    Query CL_PROGRAM_NUM_KERNELS and check that the correct number is
    //    returned. Query CL_PROGRAM_KERNEL_NAMES and check that the right
    //    kernel names are returned.
    {
        const char *build_options_B = "-DUSE_SAMPLE_TEST_B";
        error = clCompileProgram(program_AB, 1, &device, build_options_B, 0, 0,
                                 0, 0, 0);
        test_error(error, "clCompileProgram failed");

        const char *build_options_D = "-DUSE_SAMPLE_TEST_D";
        error = clCompileProgram(program_CD, 1, &device, build_options_D, 0, 0,
                                 0, 0, 0);
        test_error(error, "clCompileProgram failed");

        cl_program progs[] = { program_AB, program_CD };
        program =
            clLinkProgram(context, 1, &device, "", 2, progs, 0, 0, &error);
        test_error(error, "clLinkProgram failed");

        error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS,
                                 sizeof(size_t), &total_kernels, nullptr);
        test_error(error, "clGetProgramInfo failed");

        test_assert_error(total_kernels == 4,
                          "Unexpected clGetProgramInfo result");

        error = clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, 0, nullptr,
                                 &kernel_names_len);
        test_error(error, "clGetProgramInfo failed");

        std::vector<std::string> expected_names = {
            "sample_test_A", "sample_test_B", "sample_test_C", "sample_test_D"
        };

        const size_t len = kernel_names_len + 1;
        std::vector<char> kernel_names(len, '\0');
        error =
            clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, kernel_names_len,
                             kernel_names.data(), &kernel_names_len);
        test_error(error, "Could not find expected kernel name");

        std::string program_names = kernel_names.data();
        for (const auto &name : expected_names)
        {
            test_assert_error(program_names.find(name) != std::string::npos,
                              "Unexpected kernel name");
        }
    }
    return TEST_PASS;
}

REGISTER_TEST(get_program_info_mult_devices)
{
    size_t size = 0;

    // query multi-device context and perform objects comparability test
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, 0,
                                 nullptr, &size);
    test_error_fail(err, "clGetDeviceInfo failed");

    if (size == 0)
    {
        log_info("Can't partition device, test not supported\n");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_device_partition_property> supported_props(
        size / sizeof(cl_device_partition_property), 0);
    err = clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES,
                          supported_props.size()
                              * sizeof(cl_device_partition_property),
                          supported_props.data(), &size);
    test_error_fail(err, "clGetDeviceInfo failed");

    if (supported_props.empty() || supported_props.front() == 0)
    {
        log_info("Can't partition device, test not supported\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_uint maxComputeUnits = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(maxComputeUnits), &maxComputeUnits, nullptr);
    test_error_ret(err, "Unable to get maximal number of compute units",
                   TEST_FAIL);

    std::vector<std::array<cl_device_partition_property, 5>> partition_props = {
        { CL_DEVICE_PARTITION_EQUALLY, (cl_int)maxComputeUnits / 2, 0, 0, 0 },
        { CL_DEVICE_PARTITION_BY_COUNTS, 1, (cl_int)maxComputeUnits - 1,
          CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 },
        { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
          CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0, 0, 0 }
    };

    std::unique_ptr<SubDevicesScopeGuarded> scope_guard;
    cl_uint num_devices = 0;
    for (auto &sup_prop : supported_props)
    {
        for (auto &prop : partition_props)
        {
            if (sup_prop == prop[0])
            {
                // how many sub-devices can we create?
                err = clCreateSubDevices(device, prop.data(), 0, nullptr,
                                         &num_devices);
                test_error_fail(err, "clCreateSubDevices failed");
                if (num_devices < 2) continue;

                // get the list of subDevices
                scope_guard.reset(new SubDevicesScopeGuarded(num_devices));
                err = clCreateSubDevices(device, prop.data(), num_devices,
                                         scope_guard->sub_devices.data(),
                                         &num_devices);
                test_error_fail(err, "clCreateSubDevices failed");
                break;
            }
        }
        if (scope_guard.get() != nullptr) break;
    }

    if (scope_guard.get() == nullptr)
    {
        log_info("Can't partition device, test not supported\n");
        return TEST_SKIPPED_ITSELF;
    }

    /* Create a multi device context */
    clContextWrapper multi_device_context = clCreateContext(
        nullptr, (cl_uint)num_devices, scope_guard->sub_devices.data(), nullptr,
        nullptr, &err);
    test_error_ret(err, "Unable to create testing context",
                   TEST_SKIPPED_ITSELF);

    clProgramWrapper program = nullptr;
    err = create_single_kernel_helper_create_program(
        multi_device_context, &program, 1, sample_kernel_code_single_line);
    test_error_ret(err, "create_single_kernel_helper_create_program failed",
                   TEST_FAIL);

    if (program == nullptr)
    {
        log_error("ERROR: Unable to create reference program!\n");
        return -1;
    }

    err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices),
                           &num_devices, nullptr);
    test_error_ret(err, "Unable to get device count of program", TEST_FAIL);

    test_assert_error_ret(
        num_devices == scope_guard->sub_devices.size(),
        "Program must be associated to exact number of devices\n", TEST_FAIL);

    std::vector<cl_device_id> devices(num_devices);
    err = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                           num_devices * sizeof(cl_device_id), devices.data(),
                           nullptr);
    test_error_ret(err, "Unable to get devices of program", TEST_FAIL);

    for (cl_uint i = 0; i < devices.size(); i++)
    {
        bool found = false;
        for (auto &it : scope_guard->sub_devices)
        {
            if (it == devices[i])
            {
                found = true;
                break;
            }
        }
        test_error_fail(
            !found, "Unexpected result returned by CL_CONTEXT_DEVICES query");
    }

    return TEST_PASS;
}

REGISTER_TEST(get_program_source)
{
    cl_program program;
    int error;
    char buffer[10240];
    size_t length;
    size_t line_length = strlen(sample_kernel_code_single_line[0]);
    bool online_compilation = (gCompilationMode == kOnline);

    error = create_single_kernel_helper_create_program(context, &program, 1, sample_kernel_code_single_line);
    if( program == NULL )
    {
        log_error( "ERROR: Unable to create test program!\n" );
        return -1;
    }

    /* Try getting the length */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, 0, NULL, &length );
    test_error( error, "Unable to get program source length" );
    if (length != line_length + 1 && online_compilation)
    {
        log_error( "ERROR: Length returned for program source is incorrect!\n" );
        return -1;
    }

    /* Try normal source */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, sizeof( buffer ), buffer, NULL );
    test_error( error, "Unable to get program source" );
    if (strlen(buffer) != line_length && online_compilation)
    {
        log_error( "ERROR: Length of program source is incorrect!\n" );
        return -1;
    }

    /* Try both at once */
    error = clGetProgramInfo( program, CL_PROGRAM_SOURCE, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get program source" );
    if (strlen(buffer) != line_length && online_compilation)
    {
        log_error( "ERROR: Length of program source is incorrect!\n" );
        return -1;
    }
    if (length != line_length + 1 && online_compilation)
    {
        log_error( "ERROR: Returned length of program source is incorrect!\n" );
        return -1;
    }

    /* All done! */
    error = clReleaseProgram( program );
    test_error( error, "Unable to release program object" );

    return 0;
}

REGISTER_TEST(get_program_build_info)
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
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 0,
                                  NULL, &length);
    test_error( error, "Unable to get program build status length" );
    if( length != sizeof( status ) )
    {
        log_error( "ERROR: Returned length of program build status is invalid! (Expected %d, got %d)\n", (int)sizeof( status ), (int)length );
        return -1;
    }

    /* Now actually build it and verify the status */
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error( error, "Unable to build program source" );

    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS,
                                  sizeof(status), &status, NULL);
    test_error( error, "Unable to get program build status" );
    if( status != CL_BUILD_SUCCESS )
    {
        log_error( "ERROR: Getting built program build status did not return CL_BUILD_SUCCESS! (%d)\n", (int)status );
        return -1;
    }

    /***** Build log *****/

    /* Try getting the length */
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                  NULL, &length);
    test_error( error, "Unable to get program build log length" );

    log_info("Build log is %zu long.\n", length);

    buffer = (char*)malloc(length);

    /* Try normal source */
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length,
                                  buffer, NULL);
    test_error( error, "Unable to get program build log" );

    if( buffer[length-1] != '\0' )
    {
        log_error( "clGetProgramBuildInfo overwrote allocated space for build log! '%c'\n", buffer[length-1]  );
        return -1;
    }

    /* Try both at once */
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length,
                                  buffer, &newLength);
    test_error( error, "Unable to get program build log" );

    free(buffer);

    /***** Build options *****/
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, 0,
                                  NULL, &length);
    test_error( error, "Unable to get program build options length" );

    buffer = (char*)malloc(length);

    /* Try normal source */
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS,
                                  length, buffer, NULL);
    test_error( error, "Unable to get program build options" );

    /* Try both at once */
    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS,
                                  length, buffer, &newLength);
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

    error = clBuildProgram(program, 1, &device, "-cl-opt-disable", NULL, NULL);
    if( error != CL_SUCCESS )
    {
        print_error( error, "Building with valid options failed!" );
        return -1;
    }

    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, 0,
                                  NULL, &length);
    test_error( error, "Unable to get program build options" );

    buffer = (char*)malloc(length);

    error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS,
                                  length, buffer, NULL);
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

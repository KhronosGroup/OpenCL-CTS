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
#ifndef _WIN32
#include <unistd.h>
#endif

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"

// For global, local, and constant
const char *parameter_kernel_long =
"%s\n" // optional pragma
"kernel void test(global ulong *results, %s %s *mem0, %s %s2 *mem2, %s %s3 *mem3, %s %s4 *mem4, %s %s8 *mem8, %s %s16 *mem16)\n"
"{\n"
"   results[0] = (ulong)&mem0[0];\n"
"   results[1] = (ulong)&mem2[0];\n"
"   results[2] = (ulong)&mem3[0];\n"
"   results[3] = (ulong)&mem4[0];\n"
"   results[4] = (ulong)&mem8[0];\n"
"   results[5] = (ulong)&mem16[0];\n"
"}\n";

// For private and local
const char *local_kernel_long =
"%s\n" // optional pragma
"kernel void test(global ulong *results)\n"
"{\n"
"   %s %s mem0[3];\n"
"   %s %s2 mem2[3];\n"
"   %s %s3 mem3[3];\n"
"   %s %s4 mem4[3];\n"
"   %s %s8 mem8[3];\n"
"   %s %s16 mem16[3];\n"
"   results[0] = (ulong)&mem0[0];\n"
"   results[1] = (ulong)&mem2[0];\n"
"   results[2] = (ulong)&mem3[0];\n"
"   results[3] = (ulong)&mem4[0];\n"
"   results[4] = (ulong)&mem8[0];\n"
"   results[5] = (ulong)&mem16[0];\n"
"}\n";

// For constant
const char *constant_kernel_long =
"%s\n" // optional pragma
"  constant %s mem0[3]    = {0};\n"
"  constant %s2 mem2[3]   = {(%s2)(0)};\n"
"  constant %s3 mem3[3]   = {(%s3)(0)};\n"
"  constant %s4 mem4[3]   = {(%s4)(0)};\n"
"  constant %s8 mem8[3]   = {(%s8)(0)};\n"
"  constant %s16 mem16[3] = {(%s16)(0)};\n"
"\n"
"kernel void test(global ulong *results)\n"
"{\n"
"   results[0] = (ulong)&mem0;\n"
"   results[1] = (ulong)&mem2;\n"
"   results[2] = (ulong)&mem3;\n"
"   results[3] = (ulong)&mem4;\n"
"   results[4] = (ulong)&mem8;\n"
"   results[5] = (ulong)&mem16;\n"
"}\n";


// For global, local, and constant
const char *parameter_kernel_no_long =
"%s\n" // optional pragma
"kernel void test(global uint *results, %s %s *mem0, %s %s2 *mem2, %s %s3 *mem3, %s %s4 *mem4, %s %s8 *mem8, %s %s16 *mem16)\n"
"{\n"
"   results[0] = (uint)&mem0[0];\n"
"   results[1] = (uint)&mem2[0];\n"
"   results[2] = (uint)&mem3[0];\n"
"   results[3] = (uint)&mem4[0];\n"
"   results[4] = (uint)&mem8[0];\n"
"   results[5] = (uint)&mem16[0];\n"
"}\n";

// For private and local
const char *local_kernel_no_long =
"%s\n" // optional pragma
"kernel void test(global uint *results)\n"
"{\n"
"   %s %s mem0[3];\n"
"   %s %s2 mem2[3];\n"
"   %s %s3 mem3[3];\n"
"   %s %s4 mem4[3];\n"
"   %s %s8 mem8[3];\n"
"   %s %s16 mem16[3];\n"
"   results[0] = (uint)&mem0[0];\n"
"   results[1] = (uint)&mem2[0];\n"
"   results[2] = (uint)&mem3[0];\n"
"   results[3] = (uint)&mem4[0];\n"
"   results[4] = (uint)&mem8[0];\n"
"   results[5] = (uint)&mem16[0];\n"
"}\n";

// For constant
const char *constant_kernel_no_long =
"%s\n" // optional pragma
"  constant %s mem0[3]    = {0};\n"
"  constant %s2 mem2[3]   = {(%s2)(0)};\n"
"  constant %s3 mem3[3]   = {(%s3)(0)};\n"
"  constant %s4 mem4[3]   = {(%s4)(0)};\n"
"  constant %s8 mem8[3]   = {(%s8)(0)};\n"
"  constant %s16 mem16[3] = {(%s16)(0)};\n"
"\n"
"kernel void test(global uint *results)\n"
"{\n"
"   results[0] = (uint)&mem0;\n"
"   results[1] = (uint)&mem2;\n"
"   results[2] = (uint)&mem3;\n"
"   results[3] = (uint)&mem4;\n"
"   results[4] = (uint)&mem8;\n"
"   results[5] = (uint)&mem16;\n"
"}\n";

enum AddressSpaces
{
    kGlobal        = 0,
    kLocal,
    kConstant,
    kPrivate
};

typedef enum AddressSpaces    AddressSpaces;

#define DEBUG 0

const char * get_explicit_address_name( AddressSpaces address )
{
    /* Quick method to avoid branching: make sure the following array matches the Enum order */
    static const char *sExplicitAddressNames[] = { "global", "local", "constant", "private"};

    return sExplicitAddressNames[ address ];
}


int test_kernel_memory_alignment(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems, AddressSpaces address )
{
    const char *constant_kernel;
    const char *parameter_kernel;
    const char *local_kernel;

    if ( gHasLong )
    {
        constant_kernel  = constant_kernel_long;
        parameter_kernel = parameter_kernel_long;
        local_kernel     = local_kernel_long;
    }
    else
    {
        constant_kernel  = constant_kernel_no_long;
        parameter_kernel = parameter_kernel_no_long;
        local_kernel     = local_kernel_no_long;
    }

    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble };
    char *kernel_code = (char*)malloc(4096);
    cl_kernel kernel;
    cl_program program;
    int error;
    int total_errors = 0;
    cl_mem results;
    cl_ulong *results_data;
    cl_mem mem0, mem2, mem3, mem4, mem8, mem16;

    results_data = (cl_ulong*)malloc(sizeof(cl_ulong)*6);
    results = clCreateBuffer(context, 0, sizeof(cl_ulong)*6, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    mem0 = clCreateBuffer(context, 0, sizeof(cl_long), NULL, &error);
    test_error(error, "clCreateBuffer failed");
    mem2 = clCreateBuffer(context, 0, sizeof(cl_long)*2, NULL, &error);
    test_error(error, "clCreateBuffer failed");
    mem3 = clCreateBuffer(context, 0, sizeof(cl_long)*4, NULL, &error);
    test_error(error, "clCreateBuffer failed");
    mem4 = clCreateBuffer(context, 0, sizeof(cl_long)*4, NULL, &error);
    test_error(error, "clCreateBuffer failed");
    mem8 = clCreateBuffer(context, 0, sizeof(cl_long)*8, NULL, &error);
    test_error(error, "clCreateBuffer failed");
    mem16 = clCreateBuffer(context, 0, sizeof(cl_long)*16, NULL, &error);
    test_error(error, "clCreateBuffer failed");


    // For each type

    // Calculate alignment mask for each size

    // For global, local, constant, private

    // If global, local or constant -- do parameter_kernel
    // If private or local -- do local_kernel
    // If constant -- do constant kernel

    int numConstantArgs;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(numConstantArgs), &numConstantArgs, NULL);

    int typeIndex;
    for (typeIndex = 0; typeIndex < 10; typeIndex++) {
        // Skip double tests if we don't support doubles
        if (vecType[typeIndex] == kDouble && !is_extension_available(device, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            continue;
        }

        if (( vecType[ typeIndex ] == kLong || vecType[ typeIndex ] == kULong ) && !gHasLong )
            continue;

        log_info("Testing %s...\n", get_explicit_type_name(vecType[typeIndex]));

        // Determine the expected alignment masks.
        // E.g., if it is supposed to be 4 byte aligned, we should get 4-1=3 = ... 000011
        // We can then and the returned address with that and we should have 0.
        cl_ulong alignments[6];
        alignments[0] = get_explicit_type_size(vecType[typeIndex])-1;
        alignments[1] = (get_explicit_type_size(vecType[typeIndex])<<1)-1;
        alignments[2] = (get_explicit_type_size(vecType[typeIndex])<<2)-1;
        alignments[3] = (get_explicit_type_size(vecType[typeIndex])<<2)-1;
        alignments[4] = (get_explicit_type_size(vecType[typeIndex])<<3)-1;
        alignments[5] = (get_explicit_type_size(vecType[typeIndex])<<4)-1;

        // Parameter kernel
        if (address == kGlobal || address == kLocal || address == kConstant) {
            log_info("\tTesting parameter kernel...\n");

            if ( (gIsEmbedded) && (address == kConstant) && (numConstantArgs < 6)) {
                sprintf(kernel_code, parameter_kernel,
                    vecType[typeIndex] == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex])
                );
            }
            else {
                sprintf(kernel_code, parameter_kernel,
                    vecType[typeIndex] == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex])
                );
            }
            //printf("Kernel is: \n%s\n", kernel_code);

            // Create the kernel
            error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&kernel_code, "test");
            test_error(error, "create_single_kernel_helper failed");

            // Initialize the results
            memset(results_data, 0, sizeof(cl_long)*5);
            error = clEnqueueWriteBuffer(queue, results, CL_TRUE, 0, sizeof(cl_long)*6, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueWriteBuffer failed");

            // Set the arguments
            error = clSetKernelArg(kernel, 0, sizeof(results), &results);
            test_error(error, "clSetKernelArg failed");
            if (address != kLocal) {
                error = clSetKernelArg(kernel, 1, sizeof(mem0), &mem0);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 2, sizeof(mem2), &mem2);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 3, sizeof(mem3), &mem3);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 4, sizeof(mem4), &mem4);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 5, sizeof(mem8), &mem8);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 6, sizeof(mem16), &mem16);
                test_error(error, "clSetKernelArg failed");
            } else {
                error = clSetKernelArg(kernel, 1, get_explicit_type_size(vecType[typeIndex]), NULL);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 2, get_explicit_type_size(vecType[typeIndex])*2, NULL);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 3, get_explicit_type_size(vecType[typeIndex])*4, NULL);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 4, get_explicit_type_size(vecType[typeIndex])*4, NULL);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 5, get_explicit_type_size(vecType[typeIndex])*8, NULL);
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel, 6, get_explicit_type_size(vecType[typeIndex])*16, NULL);
                test_error(error, "clSetKernelArg failed");
            }

            // Enqueue the kernel
            size_t global_size = 1;
            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
            test_error(error, "clEnqueueNDRangeKernel failed");

            // Read back the results
            error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0, sizeof(cl_ulong)*6, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueReadBuffer failed");

            // Verify the results
            if (gHasLong) {
                for (int i = 0; i < 6; i++) {
                    if ((results_data[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data[i]);
                    }
                }
            }
            // Verify the results on devices that do not support longs
            else {
                cl_uint *results_data_no_long = (cl_uint *)results_data;

                for (int i = 0; i < 6; i++) {
                    if ((results_data_no_long[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data_no_long[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data_no_long[i]);
                    }
                }
            }

            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }




        // Local kernel
        if (address == kLocal || address == kPrivate) {
            log_info("\tTesting local kernel...\n");
            sprintf(kernel_code, local_kernel,
                    vecType[typeIndex] == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_address_name(address), get_explicit_type_name(vecType[typeIndex])
                    );
            //printf("Kernel is: \n%s\n", kernel_code);

            // Create the kernel
            error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&kernel_code, "test");
            test_error(error, "create_single_kernel_helper failed");

            // Initialize the results
            memset(results_data, 0, sizeof(cl_long)*5);
            error = clEnqueueWriteBuffer(queue, results, CL_TRUE, 0, sizeof(cl_long)*5, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueWriteBuffer failed");

            // Set the arguments
            error = clSetKernelArg(kernel, 0, sizeof(results), &results);
            test_error(error, "clSetKernelArg failed");

            // Enqueue the kernel
            size_t global_size = 1;
            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
            test_error(error, "clEnqueueNDRangeKernel failed");

            // Read back the results
            error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0, sizeof(cl_ulong)*5, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueReadBuffer failed");

            // Verify the results
            if (gHasLong) {
                for (int i = 0; i < 5; i++) {
                    if ((results_data[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data[i]);
                    }
                }
            }
            // Verify the results on devices that do not support longs
            else {
                cl_uint *results_data_no_long = (cl_uint *)results_data;

                for (int i = 0; i < 5; i++) {
                    if ((results_data_no_long[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data_no_long[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data_no_long[i]);
                    }
                }
            }
            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }



        // Constant kernel
        if (address == kConstant) {
            log_info("\tTesting constant kernel...\n");
            sprintf(kernel_code, constant_kernel,
                    vecType[typeIndex] == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex]),
                    get_explicit_type_name(vecType[typeIndex])
                    );
            //printf("Kernel is: \n%s\n", kernel_code);

            // Create the kernel
            error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&kernel_code, "test");
            test_error(error, "create_single_kernel_helper failed");

            // Initialize the results
            memset(results_data, 0, sizeof(cl_long)*5);
            error = clEnqueueWriteBuffer(queue, results, CL_TRUE, 0, sizeof(cl_long)*5, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueWriteBuffer failed");

            // Set the arguments
            error = clSetKernelArg(kernel, 0, sizeof(results), &results);
            test_error(error, "clSetKernelArg failed");

            // Enqueue the kernel
            size_t global_size = 1;
            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
            test_error(error, "clEnqueueNDRangeKernel failed");

            // Read back the results
            error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0, sizeof(cl_ulong)*5, results_data, 0, NULL, NULL);
            test_error(error, "clEnqueueReadBuffer failed");

            // Verify the results
            if (gHasLong) {
                for (int i = 0; i < 5; i++) {
                    if ((results_data[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data[i]);
                    }
                }
            }
            // Verify the results on devices that do not support longs
            else {
                cl_uint *results_data_no_long = (cl_uint *)results_data;

                for (int i = 0; i < 5; i++) {
                    if ((results_data_no_long[i] & alignments[i]) != 0) {
                        total_errors++;
                        log_error("\tVector size %d failed: 0x%llx is not properly aligned.\n", 1 << i, results_data_no_long[i]);
                    } else {
                        if (DEBUG) log_info("\tVector size %d passed: 0x%llx is properly aligned.\n", 1 << i, results_data_no_long[i]);
                    }
                }
            }
            clReleaseKernel(kernel);
            clReleaseProgram(program);
        }
    }

    clReleaseMemObject(results);
    clReleaseMemObject(mem0);
    clReleaseMemObject(mem2);
    clReleaseMemObject(mem3);
    clReleaseMemObject(mem4);
    clReleaseMemObject(mem8);
    clReleaseMemObject(mem16);
    free( kernel_code );
    free( results_data );

    if (total_errors != 0)
        return -1;
    return 0;

}


int test_kernel_memory_alignment_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_kernel_memory_alignment( device,  context,  queue,  n_elems, kLocal );
}

int test_kernel_memory_alignment_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_kernel_memory_alignment( device,  context,  queue,  n_elems, kGlobal );
}

int test_kernel_memory_alignment_constant(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // There is a class of approved OpenCL 1.0 conformant devices out there that in some circumstances
    // are unable to meaningfully take (or more precisely use) the address of constant data by virtue
    // of limitations in their ISA design. This feature was not tested in 1.0, so they were declared
    // conformant by Khronos. The failure is however caught here.
    //
    // Unfortunately, determining whether or not these devices are 1.0 conformant is not the jurisdiction
    // of the 1.1 tests -- We can't fail them from 1.1 conformance here because they are not 1.1
    // devices. They are merely 1.0 conformant devices that interop with 1.1 devices in a 1.1 platform.
    // To add new binding tests now to conformant 1.0 devices would violate the workingroup requirement
    // of no new tests for 1.0 devices.  So certain allowances have to be made in intractable cases
    // such as this one.
    //
    // There is some precedent. Similar allowances are made for other 1.0 hardware features such as
    // local memory size.  The minimum required local memory size grew from 16 kB to 32 kB in OpenCL 1.1.

    // Detect 1.0 devices
    // Get CL_DEVICE_VERSION size
    size_t string_size = 0;
    int err;
    if( (err = clGetDeviceInfo( device, CL_DEVICE_VERSION, 0, NULL, &string_size ) ) )
    {
        log_error( "FAILURE: Unable to get size of CL_DEVICE_VERSION string!" );
        return -1;
    }

    //Allocate storage to hold the version string
    char *version_string = (char*) malloc(string_size);
    if( NULL == version_string )
    {
        log_error( "FAILURE: Unable to allocate memory to hold CL_DEVICE_VERSION string!" );
        return -1;
    }

    // Get CL_DEVICE_VERSION string
    if( (err = clGetDeviceInfo( device, CL_DEVICE_VERSION, string_size, version_string, NULL ) ) )
    {
        log_error( "FAILURE: Unable to read CL_DEVICE_VERSION string!" );
        return -1;
    }

    // easy out for 1.0 devices
    const char *string_1_0 = "OpenCL 1.0 ";
    if( 0 == strncmp( version_string, string_1_0, strlen(string_1_0)) )
    {
        log_info( "WARNING: Allowing device to escape testing of difficult constant memory alignment case.\n\tDevice is not a OpenCL 1.1 device. CL_DEVICE_VERSION: \"%s\"\n", version_string );
        free(version_string);
        return 0;
    }
    log_info( "Device version string: \"%s\"\n", version_string );
    free(version_string);

    // Everyone else is to be ground mercilessly under the wheels of progress
    return test_kernel_memory_alignment( device,  context,  queue,  n_elems, kConstant );
}

int test_kernel_memory_alignment_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_kernel_memory_alignment( device,  context,  queue,  n_elems, kPrivate );
}



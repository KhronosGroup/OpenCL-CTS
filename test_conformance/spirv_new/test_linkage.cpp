//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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
#include "types.hpp"

#include <sstream>
#include <string>

static int test_linkage_compile(cl_device_id deviceID,
                                cl_context context,
                                cl_command_queue queue,
                                const char *fname,
                                clProgramWrapper &prog)
{
    cl_int err = CL_SUCCESS;
    std::vector<unsigned char> buffer_vec = readSPIRV(fname);

    int file_bytes = buffer_vec.size();
    if (file_bytes == 0) {
        log_error("File not found\n");
        return -1;
    }
    unsigned char *buffer = &buffer_vec[0];

    if (gCoreILProgram)
    {
        prog = clCreateProgramWithIL(context, buffer, file_bytes, &err);
        SPIRV_CHECK_ERROR(
            err, "Failed to create program with clCreateProgramWithIL");
    }
    else
    {
        cl_platform_id platform;
        err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM,
                              sizeof(cl_platform_id), &platform, NULL);
        SPIRV_CHECK_ERROR(err,
                          "Failed to get platform info with clGetDeviceInfo");
        clCreateProgramWithILKHR_fn clCreateProgramWithILKHR = NULL;

        clCreateProgramWithILKHR = (clCreateProgramWithILKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clCreateProgramWithILKHR");
        if (clCreateProgramWithILKHR == NULL)
        {
            log_error(
                "ERROR: clGetExtensionFunctionAddressForPlatform failed\n");
            return -1;
        }
        prog = clCreateProgramWithILKHR(context, buffer, file_bytes, &err);
        SPIRV_CHECK_ERROR(
            err, "Failed to create program with clCreateProgramWithILKHR");
    }

    err = clCompileProgram(prog, 1, &deviceID,
                           NULL, // options
                           0, // num headers
                           NULL, // input headers
                           NULL, // header include names
                           NULL, // callback
                           NULL // User data
    );
    SPIRV_CHECK_ERROR(err, "Failed to compile spv program");
    return 0;
}

TEST_SPIRV_FUNC(linkage_export_function_compile)
{
    clProgramWrapper prog;
    return test_linkage_compile(deviceID, context, queue, "linkage_export", prog);
}

TEST_SPIRV_FUNC(linkage_import_function_compile)
{
    clProgramWrapper prog;
    return test_linkage_compile(deviceID, context, queue, "linkage_import", prog);
}

TEST_SPIRV_FUNC(linkage_import_function_link)
{
    int err = 0;

    clProgramWrapper prog_export;
    err = test_linkage_compile(deviceID, context, queue, "linkage_export", prog_export);
    SPIRV_CHECK_ERROR(err, "Failed to compile export program");

    clProgramWrapper prog_import;
    err = test_linkage_compile(deviceID, context, queue, "linkage_import", prog_import);
    SPIRV_CHECK_ERROR(err, "Failed to compile import program");

    cl_program progs[] = {prog_export, prog_import};

    clProgramWrapper prog = clLinkProgram(context, 1, &deviceID, NULL, 2, progs, NULL, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to link programs");

    clKernelWrapper kernel = clCreateKernel(prog, "test_linkage", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    const int num = 1 << 20;
    std::vector<cl_float> h_in(num);
    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num; i++) {
        h_in[i] = genrand<cl_float>(seed);
    }

    size_t bytes = sizeof(cl_float) * num;
    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");


    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<cl_float> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != -h_in[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }

    return 0;
}

static int test_linkonce_odr_helper(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue,
                                    const char *main_module_filename)
{
    cl_int err = 0;

    clProgramWrapper prog_obj;
    err = test_linkage_compile(deviceID, context, queue,
                               "linkage_linkonce_odr_obj", prog_obj);
    SPIRV_CHECK_ERROR(err, "Failed to compile export program");

    clProgramWrapper prog_main;
    err = test_linkage_compile(deviceID, context, queue, main_module_filename,
                               prog_main);
    SPIRV_CHECK_ERROR(err, "Failed to compile import program");

    cl_program progs[] = { prog_obj, prog_main };

    clProgramWrapper prog =
        clLinkProgram(context, 1, &deviceID, NULL, 2, progs, NULL, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to link programs");

    clKernelWrapper kernel = clCreateKernel(prog, "test_linkonce_odr", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    const int num = 256;
    std::vector<cl_int> h_in(num);
    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num; i++)
    {
        h_in[i] = genrand<cl_int>(seed) % 2048;
    }

    size_t bytes = sizeof(cl_int) * num;
    clMemWrapper in =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL,
                               NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL,
                                 NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<cl_int> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL,
                              NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++)
    {
        if (h_out[i] != 5)
        {
            log_error("Incorrect values at location %d\n", i);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

TEST_SPIRV_FUNC(linkage_linkonce_odr)
{
    if (!is_extension_available(deviceID, "cl_khr_spirv_linkonce_odr"))
    {
        log_info("Extension cl_khr_spirv_linkonce_odr not supported; skipping "
                 "tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int result = TEST_PASS;

    // For this test, use the default main module, which has an "a" function
    // with the linkonce_odr linkage type.  This ensures that having two "a"
    // functions with linkonce_odr works properly.
    result |= test_linkonce_odr_helper(deviceID, context, queue,
                                       "linkage_linkonce_odr_main");

    // For this test, use a main module without the "a" function.  This ensures
    // that the "a" function is properly exported with the linkonce_odr linkage
    // type.
    result |= test_linkonce_odr_helper(deviceID, context, queue,
                                       "linkage_linkonce_odr_noa_main");

    return result;
}

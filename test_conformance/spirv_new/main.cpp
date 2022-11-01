/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
in whole or in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include <stdio.h>
#include <string.h>
#include "procs.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#if defined(_WIN32)
const std::string slash = "\\";
#else
const std::string slash = "/";
#endif

const std::string spvExt = ".spv";
std::string gAddrWidth = "";
std::string spvBinariesPath = "spirv_bin";
std::string spvBinariesPathArg = "--spirv-binaries-path";

std::vector<unsigned char> readBinary(const char *file_name)
{
    using namespace std;

    ifstream file(file_name, ios::in | ios::binary | ios::ate);

    std::vector<char> tmpBuffer(0);

    if (file.is_open()) {
        size_t size = file.tellg();
        tmpBuffer.resize(size);
        file.seekg(0, ios::beg);
        file.read(&tmpBuffer[0], size);
        file.close();
    } else {
        log_error("File %s not found\n", file_name);
    }

    std::vector<unsigned char> result(tmpBuffer.begin(), tmpBuffer.end());

    return result;
}


std::vector<unsigned char> readSPIRV(const char *file_name)
{
    std::string full_name_str = spvBinariesPath + slash + file_name + spvExt + gAddrWidth;
    return readBinary(full_name_str.c_str());
}

test_definition *spirvTestsRegistry::getTestDefinitions()
{
    return &testDefinitions[0];
}

size_t spirvTestsRegistry::getNumTests()
{
    return testDefinitions.size();
}

void spirvTestsRegistry::addTestClass(baseTestClass *test, const char *testName,
                                      Version version)
{

    testClasses.push_back(test);
    test_definition testDef;
    testDef.func = test->getFunction();
    testDef.name = testName;
    testDef.min_version = version;
    testDefinitions.push_back(testDef);
}

spirvTestsRegistry& spirvTestsRegistry::getInstance()
{
    static spirvTestsRegistry instance;
    return instance;
}

static int offline_get_program_with_il(clProgramWrapper &prog,
                                       const cl_device_id deviceID,
                                       const cl_context context,
                                       const char *prog_name)
{
    cl_int err = 0;
    std::string outputTypeStr = "binary";
    std::string defaultScript = std::string("..") + slash + std::string("spv_to_binary.py");
    std::string outputFilename = spvBinariesPath + slash + std::string(prog_name);
    std::string sourceFilename = outputFilename +  spvExt;

    std::string scriptArgs =
        sourceFilename + " " +
        outputFilename + " " +
        gAddrWidth + " " +
        outputTypeStr + " " +
        "-cl-std=CL2.0";

    std::string scriptToRunString = defaultScript + scriptArgs;

    // execute script
    log_info("Executing command: %s\n", scriptToRunString.c_str());
    fflush(stdout);
    int returnCode = system(scriptToRunString.c_str());
    if (returnCode != 0) {
        log_error("Command finished with error: 0x%x\n", returnCode);
        return CL_COMPILE_PROGRAM_FAILURE;
    }

    // read output file
    std::vector<unsigned char> buffer_vec = readBinary(outputFilename.c_str());
    size_t file_bytes = buffer_vec.size();
    if (file_bytes == 0) {
        log_error("OfflinerCompiler: Failed to open binary file: %s", outputFilename.c_str());
        return -1;
    }

    const unsigned char *buffer = &buffer_vec[0];
    cl_int status = 0;
    prog = clCreateProgramWithBinary(context, 1, &deviceID, &file_bytes, &buffer, &status, &err);
    SPIRV_CHECK_ERROR((err || status), "Failed to create program with clCreateProgramWithBinary");
    return err;
}

int get_program_with_il(clProgramWrapper &prog, const cl_device_id deviceID,
                        const cl_context context, const char *prog_name,
                        spec_const spec_const_def)
{
    cl_int err = 0;
    if (gCompilationMode == kBinary)
    {
        return offline_get_program_with_il(prog, deviceID, context, prog_name);
    }

    std::vector<unsigned char> buffer_vec = readSPIRV(prog_name);

    int file_bytes = buffer_vec.size();
    if (file_bytes == 0)
    {
        log_error("File %s not found\n", prog_name);
        return -1;
    }

    unsigned char *buffer = &buffer_vec[0];
    if (gCoreILProgram)
    {
        prog = clCreateProgramWithIL(context, buffer, file_bytes, &err);
        SPIRV_CHECK_ERROR(
            err, "Failed to create program with clCreateProgramWithIL");

        if (spec_const_def.spec_value != NULL)
        {
            err = clSetProgramSpecializationConstant(
                prog, spec_const_def.spec_id, spec_const_def.spec_size,
                spec_const_def.spec_value);
            SPIRV_CHECK_ERROR(
                err, "Failed to run clSetProgramSpecializationConstant");
        }
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

    err = clBuildProgram(prog, 1, &deviceID, NULL, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    return err;
}

test_status InitCL(cl_device_id id)
{
    test_status spirv_status;
    spirv_status = check_spirv_compilation_readiness(id);
    if (spirv_status != TEST_PASS)
    {
        return spirv_status;
    }

    cl_uint address_bits;
    cl_uint err = clGetDeviceInfo(id, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
                                  &address_bits, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed to get address bits!");
        return TEST_FAIL;
    }

    gAddrWidth = address_bits == 32 ? "32" : "64";
    return TEST_PASS;
}

void printUsage() {
    log_info("Reading SPIR-V files from default '%s' path.\n", spvBinariesPath.c_str());
    log_info("In case you want to set other directory use '%s' argument.\n", spvBinariesPathArg.c_str());
}

int main(int argc, const char *argv[])
{
    gReSeed = 1;
    bool modifiedSpvBinariesPath = false;
    for (int i = 0; i < argc; ++i) {
        int argsRemoveNum = 0;
        if (argv[i] == spvBinariesPathArg) {
            if (i + 1 == argc) {
                log_error("Missing value for '%s' argument.\n", spvBinariesPathArg.c_str());
                return TEST_FAIL;
            } else {
                spvBinariesPath = std::string(argv[i + 1]);
                argsRemoveNum += 2;
                modifiedSpvBinariesPath = true;
            }
        }

        if (argsRemoveNum > 0) {
            for (int j = i; j < (argc - argsRemoveNum); ++j)
                argv[j] = argv[j + argsRemoveNum];

            argc -= argsRemoveNum;
            --i;
        }
    }
    if (modifiedSpvBinariesPath == false) {
       printUsage();
    }

    return runTestHarnessWithCheck(
        argc, argv, spirvTestsRegistry::getInstance().getNumTests(),
        spirvTestsRegistry::getInstance().getTestDefinitions(), false, 0,
        InitCL);
}

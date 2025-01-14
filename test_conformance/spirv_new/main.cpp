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
bool gVersionSkip = false;
std::string gAddrWidth = "";
std::string spvBinariesPath = "spirv_bin";

const std::string spvBinariesPathArg = "--spirv-binaries-path";
const std::string spvVersionSkipArg = "--skip-spirv-version-check";

std::vector<unsigned char> readBinary(const char *file_name)
{
    std::ifstream file(file_name,
                       std::ios::in | std::ios::binary | std::ios::ate);

    std::vector<char> tmpBuffer(0);

    if (file.is_open()) {
        size_t size = file.tellg();
        tmpBuffer.resize(size);
        file.seekg(0, std::ios::beg);
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
    log_info("In case you want to set other directory use '%s' argument.\n",
             spvBinariesPathArg.c_str());
    log_info("To skip the SPIR-V version check use the '%s' argument.\n",
             spvVersionSkipArg.c_str());
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
        if (argv[i] == spvVersionSkipArg)
        {
            gVersionSkip = true;
            argsRemoveNum++;
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
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

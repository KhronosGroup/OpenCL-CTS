//
// Copyright (c) 2026 The Khronos Group Inc.
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

#include "spirvTools.h"

#include "errorHelpers.h"
#include "parseParameters.h"
#include "testHarness.h"

#include <spirv-tools/libspirv.hpp>

cl_int
emit_spirv_assembly_preamble(std::ostream &output, cl_device_id device,
                             cl_uint &address_bits,
                             const std::vector<std::string> &capabilities,
                             const std::vector<std::string> &extensions)
{
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits),
                        &address_bits, nullptr);
    if (error != CL_SUCCESS) return error;

    output << "OpCapability Addresses\n"
           << "OpCapability Kernel\n";
    if (address_bits == 64) output << "OpCapability Int64\n";
    for (const auto &capability : capabilities)
        output << "OpCapability " << capability << "\n";
    for (const auto &extension : extensions)
        output << "OpExtension \"" << extension << "\"\n";
    output << "OpMemoryModel Physical" << address_bits << " OpenCL\n";
    return CL_SUCCESS;
}

bool assemble_spirv_text(const std::string &spirv_text,
                         std::vector<uint32_t> &spirv_binary)
{
    auto message_consumer = [](spv_message_level_t, const char *,
                               const spv_position_t &position,
                               const char *message) {
        log_error("SPIR-V Tools at line %zu, column %zu: %s\n", position.line,
                  position.column, message);
    };

    spvtools::SpirvTools spirv_tools(SPV_ENV_UNIVERSAL_1_0);
    spirv_tools.SetMessageConsumer(message_consumer);

    if (!spirv_tools.Assemble(spirv_text, &spirv_binary)) return false;
    if (gDisableSPIRVValidation) return true;
    return spirv_tools.Validate(spirv_binary.data(), spirv_binary.size());
}

cl_int
create_program_from_spirv_binary(clProgramWrapper &program, cl_device_id device,
                                 cl_context context,
                                 const std::vector<uint32_t> &spirv_binary)
{
    cl_int error = CL_SUCCESS;
    const size_t binary_size = spirv_binary.size() * sizeof(spirv_binary[0]);
    if (gCoreILProgram)
    {
        program = clCreateProgramWithIL(context, spirv_binary.data(),
                                        binary_size, &error);
    }
    else
    {
        cl_platform_id platform = nullptr;
        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                &platform, nullptr);
        if (error != CL_SUCCESS)
        {
            print_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");
            return error;
        }

        auto clCreateProgramWithILKHR =
            reinterpret_cast<clCreateProgramWithILKHR_fn>(
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clCreateProgramWithILKHR"));
        if (clCreateProgramWithILKHR == nullptr)
        {
            log_error("Failed to get clCreateProgramWithILKHR\n");
            return CL_INVALID_OPERATION;
        }
        program = clCreateProgramWithILKHR(context, spirv_binary.data(),
                                           binary_size, &error);
    }
    if (error != CL_SUCCESS)
    {
        print_error(error,
                    gCoreILProgram ? "clCreateProgramWithIL failed"
                                   : "clCreateProgramWithILKHR failed");
        return error;
    }

    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) OutputBuildLog(program, device);
    return error;
}

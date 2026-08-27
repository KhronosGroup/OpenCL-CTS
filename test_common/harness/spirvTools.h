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

#ifndef SPIRV_TOOLS_H
#define SPIRV_TOOLS_H

#include "typeWrappers.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

// Emit an OpenCL SPIR-V preamble and return the device address width.
cl_int
emit_spirv_assembly_preamble(std::ostream &output, cl_device_id device,
                             cl_uint &address_bits,
                             const std::vector<std::string> &capabilities = {},
                             const std::vector<std::string> &extensions = {});

// Assemble and (unless disabled) validate textual SPIR-V 1.0.
bool assemble_spirv_text(const std::string &spirv_text,
                         std::vector<uint32_t> &spirv_binary);

// Create an IL program from assembled SPIR-V and build it for the device.
cl_int
create_program_from_spirv_binary(clProgramWrapper &program, cl_device_id device,
                                 cl_context context,
                                 const std::vector<uint32_t> &spirv_binary);

#endif // SPIRV_TOOLS_H

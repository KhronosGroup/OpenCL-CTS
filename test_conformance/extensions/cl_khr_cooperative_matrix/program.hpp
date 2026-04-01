// Copyright (c) 2024-2026 The Khronos Group Inc.
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

#ifndef COOPERATIVE_MATRIX_PROGRAM_HPP
#define COOPERATIVE_MATRIX_PROGRAM_HPP

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "cooperative_matrix.hpp"

// Program binary and kernel argument data.
struct Program
{
    // Number of kernel arguments.
    static constexpr size_t numKernelArgs = 4;

    // SPIR-V binary.
    std::vector<uint32_t> spirvBinary;

    // Kernel argument types given as u32/f16/...
    std::string argType[numKernelArgs];

    // Kernel buffer sizes in bytes.
    size_t bufferSize[numKernelArgs];
};

// Program generator for a given variant and operation.
class ProgramGenerator {
public:
    ProgramGenerator(const Variant &variant, CoopMatOp op)
        : variant(variant), op(op), muladdOperand("")
    {}

    // Generate a SPIR-V binary.
    bool generateSpirv(Program *spirv_out);

private:
    void genTypeDecls();
    void genConstants();
    void genVariables();
    void genBody();

    void emitConversionTypes();
    std::string emitConversion();

    std::ostringstream spirv_text;
    const Variant &variant;
    CoopMatOp op;
    std::string muladdOperand; // Includes leading space unless empty
};

#endif // COOPERATIVE_MATRIX_PROGRAM_HPP

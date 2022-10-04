//
// Copyright (c) 2022 The Khronos Group Inc.
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

#include "common.h"

#include "utility.h" // for sizeNames and sizeValues.

#include <sstream>
#include <string>

namespace {

const char *GetTypeName(ParameterType type)
{
    switch (type)
    {
        case ParameterType::Float: return "float";
        case ParameterType::Double: return "double";
    }
    return nullptr;
}

const char *GetUndefValue(ParameterType type)
{
    switch (type)
    {
        case ParameterType::Float:
        case ParameterType::Double: return "NAN";
    }
    return nullptr;
}

void EmitDefineType(std::ostringstream &kernel, const char *name,
                    ParameterType type, int vector_size_index)
{
    kernel << "#define " << name << " " << GetTypeName(type)
           << sizeNames[vector_size_index] << '\n';
    kernel << "#define " << name << "_SCALAR " << GetTypeName(type) << '\n';
}

void EmitDefineUndef(std::ostringstream &kernel, const char *name,
                     ParameterType type)
{
    kernel << "#define " << name << " " << GetUndefValue(type) << '\n';
}

void EmitEnableExtension(std::ostringstream &kernel, ParameterType type)
{
    switch (type)
    {
        case ParameterType::Double:
            kernel << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            break;

        case ParameterType::Float:
            // No extension required.
            break;
    }
}

} // anonymous namespace

std::string GetKernelName(int vector_size_index)
{
    return std::string("math_kernel") + sizeNames[vector_size_index];
}

std::string GetTernaryKernel(const std::string &kernel_name,
                             const char *builtin, ParameterType retType,
                             ParameterType type1, ParameterType type2,
                             ParameterType type3, int vector_size_index)
{
    // To keep the kernel code readable, use macros for types and undef values.
    std::ostringstream kernel;
    EmitDefineType(kernel, "RETTYPE", retType, vector_size_index);
    EmitDefineType(kernel, "TYPE1", type1, vector_size_index);
    EmitDefineType(kernel, "TYPE2", type2, vector_size_index);
    EmitDefineType(kernel, "TYPE3", type3, vector_size_index);
    EmitDefineUndef(kernel, "UNDEF1", type1);
    EmitDefineUndef(kernel, "UNDEF2", type2);
    EmitDefineUndef(kernel, "UNDEF3", type3);
    EmitEnableExtension(kernel, type1);

    // clang-format off
    const char *kernel_nonvec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE* out,
                          __global TYPE1* in1,
                          __global TYPE2* in2,
                          __global TYPE3* in3)
{
    size_t i = get_global_id(0);
    out[i] = )", builtin, R"((in1[i], in2[i], in3[i]);
}
)" };

    const char *kernel_vec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE_SCALAR* out,
                          __global TYPE1_SCALAR* in1,
                          __global TYPE2_SCALAR* in2,
                          __global TYPE3_SCALAR* in3)
{
    size_t i = get_global_id(0);

    if (i + 1 < get_global_size(0))
    {
        TYPE1 a = vload3(0, in1 + 3 * i);
        TYPE2 b = vload3(0, in2 + 3 * i);
        TYPE3 c = vload3(0, in3 + 3 * i);
        RETTYPE res = )", builtin, R"((a, b, c);
        vstore3(res, 0, out + 3 * i);
    }
    else
    {
        // Figure out how many elements are left over after
        // BUFFER_SIZE % (3 * sizeof(type)).
        // Assume power of two buffer size.
        size_t parity = i & 1;
        TYPE1 a = (TYPE1)(UNDEF1, UNDEF1, UNDEF1);
        TYPE2 b = (TYPE2)(UNDEF2, UNDEF2, UNDEF2);
        TYPE3 c = (TYPE3)(UNDEF3, UNDEF3, UNDEF3);
        switch (parity)
        {
            case 0:
                a.y = in1[3 * i + 1];
                b.y = in2[3 * i + 1];
                c.y = in3[3 * i + 1];
                // fall through
            case 1:
                a.x = in1[3 * i];
                b.x = in2[3 * i];
                c.x = in3[3 * i];
                break;
        }

        RETTYPE res = )", builtin, R"((a, b, c);

        switch (parity)
        {
            case 0:
                out[3 * i + 1] = res.y;
                // fall through
            case 1:
                out[3 * i] = res.x;
                break;
        }
    }
}
)" };
    // clang-format on

    if (sizeValues[vector_size_index] != 3)
        for (const auto &chunk : kernel_nonvec3) kernel << chunk;
    else
        for (const auto &chunk : kernel_vec3) kernel << chunk;

    return kernel.str();
}

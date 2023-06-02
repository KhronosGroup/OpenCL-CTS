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
        case ParameterType::Int: return "int";
        case ParameterType::UInt: return "uint";
        case ParameterType::Long: return "long";
        case ParameterType::ULong: return "ulong";
    }
    return nullptr;
}

const char *GetUndefValue(ParameterType type)
{
    switch (type)
    {
        case ParameterType::Float:
        case ParameterType::Double: return "NAN";

        case ParameterType::Int:
        case ParameterType::UInt: return "0x12345678";

        case ParameterType::Long:
        case ParameterType::ULong: return "0x0ddf00dbadc0ffee";
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

void EmitEnableExtension(std::ostringstream &kernel,
                         const std::initializer_list<ParameterType> &types)
{
    bool needsFp64 = false;

    for (const auto &type : types)
    {
        switch (type)
        {
            case ParameterType::Double: needsFp64 = true; break;

            case ParameterType::Float:
            case ParameterType::Int:
            case ParameterType::UInt:
            case ParameterType::Long:
            case ParameterType::ULong:
                // No extension required.
                break;
        }
    }

    if (needsFp64) kernel << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
}

std::string GetBuildOptions(bool relaxed_mode)
{
    std::ostringstream options;

    if (gForceFTZ)
    {
        options << " -cl-denorms-are-zero";
    }

    if (gFloatCapabilities & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    {
        options << " -cl-fp32-correctly-rounded-divide-sqrt";
    }

    if (relaxed_mode)
    {
        options << " -cl-fast-relaxed-math";
    }

    return options.str();
}

} // anonymous namespace

std::string GetKernelName(int vector_size_index)
{
    return std::string("math_kernel") + sizeNames[vector_size_index];
}

std::string GetUnaryKernel(const std::string &kernel_name, const char *builtin,
                           ParameterType retType, ParameterType type1,
                           int vector_size_index)
{
    // To keep the kernel code readable, use macros for types and undef values.
    std::ostringstream kernel;
    EmitDefineType(kernel, "RETTYPE", retType, vector_size_index);
    EmitDefineType(kernel, "TYPE1", type1, vector_size_index);
    EmitDefineUndef(kernel, "UNDEF1", type1);
    EmitEnableExtension(kernel, { retType, type1 });

    // clang-format off
    const char *kernel_nonvec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE* out,
                          __global TYPE1* in1)
{
    size_t i = get_global_id(0);
    out[i] = )", builtin, R"((in1[i]);
}
)" };

    const char *kernel_vec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE_SCALAR* out,
                          __global TYPE1_SCALAR* in1)
{
    size_t i = get_global_id(0);

    if (i + 1 < get_global_size(0))
    {
        TYPE1 a = vload3(0, in1 + 3 * i);
        RETTYPE res = )", builtin, R"((a);
        vstore3(res, 0, out + 3 * i);
    }
    else
    {
        // Figure out how many elements are left over after
        // BUFFER_SIZE % (3 * sizeof(type)).
        // Assume power of two buffer size.
        size_t parity = i & 1;
        TYPE1 a = (TYPE1)(UNDEF1, UNDEF1, UNDEF1);
        switch (parity)
        {
            case 0:
                a.y = in1[3 * i + 1];
                // fall through
            case 1:
                a.x = in1[3 * i];
                break;
        }

        RETTYPE res = )", builtin, R"((a);

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

std::string GetUnaryKernel(const std::string &kernel_name, const char *builtin,
                           ParameterType retType1, ParameterType retType2,
                           ParameterType type1, int vector_size_index)
{
    // To keep the kernel code readable, use macros for types and undef values.
    std::ostringstream kernel;
    EmitDefineType(kernel, "RETTYPE1", retType1, vector_size_index);
    EmitDefineType(kernel, "RETTYPE2", retType2, vector_size_index);
    EmitDefineType(kernel, "TYPE1", type1, vector_size_index);
    EmitDefineUndef(kernel, "UNDEF1", type1);
    EmitDefineUndef(kernel, "UNDEFR2", retType2);
    EmitEnableExtension(kernel, { retType1, retType2, type1 });

    // clang-format off
    const char *kernel_nonvec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE1* out1,
                          __global RETTYPE2* out2,
                          __global TYPE1* in1)
{
    size_t i = get_global_id(0);
    out1[i] = )", builtin, R"((in1[i], out2 + i);
}
)" };

    const char *kernel_vec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE1_SCALAR* out1,
                          __global RETTYPE2_SCALAR* out2,
                          __global TYPE1_SCALAR* in1)
{
    size_t i = get_global_id(0);

    if (i + 1 < get_global_size(0))
    {
        TYPE1 a = vload3(0, in1 + 3 * i);
        RETTYPE2 res2 = UNDEFR2;
        RETTYPE1 res1 = )", builtin, R"((a, &res2);
        vstore3(res1, 0, out1 + 3 * i);
        vstore3(res2, 0, out2 + 3 * i);
    }
    else
    {
        // Figure out how many elements are left over after
        // BUFFER_SIZE % (3 * sizeof(type)).
        // Assume power of two buffer size.
        size_t parity = i & 1;
        TYPE1 a = (TYPE1)(UNDEF1, UNDEF1, UNDEF1);
        switch (parity)
        {
            case 0:
                a.y = in1[3 * i + 1];
                // fall through
            case 1:
                a.x = in1[3 * i];
                break;
        }

        RETTYPE2 res2 = UNDEFR2;
        RETTYPE1 res1 = )", builtin, R"((a, &res2);

        switch (parity)
        {
            case 0:
                out1[3 * i + 1] = res1.y;
                out2[3 * i + 1] = res2.y;
                // fall through
            case 1:
                out1[3 * i] = res1.x;
                out2[3 * i] = res2.x;
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

std::string GetBinaryKernel(const std::string &kernel_name, const char *builtin,
                            ParameterType retType, ParameterType type1,
                            ParameterType type2, int vector_size_index)
{
    // To keep the kernel code readable, use macros for types and undef values.
    std::ostringstream kernel;
    EmitDefineType(kernel, "RETTYPE", retType, vector_size_index);
    EmitDefineType(kernel, "TYPE1", type1, vector_size_index);
    EmitDefineType(kernel, "TYPE2", type2, vector_size_index);
    EmitDefineUndef(kernel, "UNDEF1", type1);
    EmitDefineUndef(kernel, "UNDEF2", type2);
    EmitEnableExtension(kernel, { retType, type1, type2 });

    const bool is_vec3 = sizeValues[vector_size_index] == 3;

    std::string invocation;
    if (strlen(builtin) == 1)
    {
        // Assume a single-character builtin is an operator (e.g., +, *, ...).
        invocation = is_vec3 ? "a" : "in1[i] ";
        invocation += builtin;
        invocation += is_vec3 ? "b" : " in2[i]";
    }
    else
    {
        // Otherwise call the builtin as a function with two arguments.
        invocation = builtin;
        invocation += is_vec3 ? "(a, b)" : "(in1[i], in2[i])";
    }

    // clang-format off
    const char *kernel_nonvec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE* out,
                          __global TYPE1* in1,
                          __global TYPE2* in2)
{
    size_t i = get_global_id(0);
    out[i] = )", invocation.c_str(), R"(;
}
)" };

    const char *kernel_vec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE_SCALAR* out,
                          __global TYPE1_SCALAR* in1,
                          __global TYPE2_SCALAR* in2)
{
    size_t i = get_global_id(0);

    if (i + 1 < get_global_size(0))
    {
        TYPE1 a = vload3(0, in1 + 3 * i);
        TYPE2 b = vload3(0, in2 + 3 * i);
        RETTYPE res = )", invocation.c_str(), R"(;
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
        switch (parity)
        {
            case 0:
                a.y = in1[3 * i + 1];
                b.y = in2[3 * i + 1];
                // fall through
            case 1:
                a.x = in1[3 * i];
                b.x = in2[3 * i];
                break;
        }

        RETTYPE res = )", invocation.c_str(), R"(;

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

    if (!is_vec3)
        for (const auto &chunk : kernel_nonvec3) kernel << chunk;
    else
        for (const auto &chunk : kernel_vec3) kernel << chunk;

    return kernel.str();
}

std::string GetBinaryKernel(const std::string &kernel_name, const char *builtin,
                            ParameterType retType1, ParameterType retType2,
                            ParameterType type1, ParameterType type2,
                            int vector_size_index)
{
    // To keep the kernel code readable, use macros for types and undef values.
    std::ostringstream kernel;
    EmitDefineType(kernel, "RETTYPE1", retType1, vector_size_index);
    EmitDefineType(kernel, "RETTYPE2", retType2, vector_size_index);
    EmitDefineType(kernel, "TYPE1", type1, vector_size_index);
    EmitDefineType(kernel, "TYPE2", type2, vector_size_index);
    EmitDefineUndef(kernel, "UNDEF1", type1);
    EmitDefineUndef(kernel, "UNDEF2", type2);
    EmitDefineUndef(kernel, "UNDEFR2", retType2);
    EmitEnableExtension(kernel, { retType1, retType2, type1, type2 });

    // clang-format off
    const char *kernel_nonvec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE1* out1,
                          __global RETTYPE2* out2,
                          __global TYPE1* in1,
                          __global TYPE2* in2)
{
    size_t i = get_global_id(0);
    out1[i] = )", builtin, R"((in1[i], in2[i], out2 + i);
}
)" };

    const char *kernel_vec3[] = { R"(
__kernel void )", kernel_name.c_str(), R"((__global RETTYPE1_SCALAR* out1,
                          __global RETTYPE2_SCALAR* out2,
                          __global TYPE1_SCALAR* in1,
                          __global TYPE2_SCALAR* in2)
{
    size_t i = get_global_id(0);

    if (i + 1 < get_global_size(0))
    {
        TYPE1 a = vload3(0, in1 + 3 * i);
        TYPE2 b = vload3(0, in2 + 3 * i);
        RETTYPE2 res2 = UNDEFR2;
        RETTYPE1 res1 = )", builtin, R"((a, b, &res2);
        vstore3(res1, 0, out1 + 3 * i);
        vstore3(res2, 0, out2 + 3 * i);
    }
    else
    {
        // Figure out how many elements are left over after
        // BUFFER_SIZE % (3 * sizeof(type)).
        // Assume power of two buffer size.
        size_t parity = i & 1;
        TYPE1 a = (TYPE1)(UNDEF1, UNDEF1, UNDEF1);
        TYPE2 b = (TYPE2)(UNDEF2, UNDEF2, UNDEF2);
        switch (parity)
        {
            case 0:
                a.y = in1[3 * i + 1];
                b.y = in2[3 * i + 1];
                // fall through
            case 1:
                a.x = in1[3 * i];
                b.x = in2[3 * i];
                break;
        }

        RETTYPE2 res2 = UNDEFR2;
        RETTYPE1 res1 = )", builtin, R"((a, b, &res2);

        switch (parity)
        {
            case 0:
                out1[3 * i + 1] = res1.y;
                out2[3 * i + 1] = res2.y;
                // fall through
            case 1:
                out1[3 * i] = res1.x;
                out2[3 * i] = res2.x;
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
    EmitEnableExtension(kernel, { retType, type1, type2, type3 });

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

cl_int BuildKernels(BuildKernelInfo &info, cl_uint job_id,
                    SourceGenerator generator)
{
    // Generate the kernel code.
    cl_uint vector_size_index = gMinVectorSizeIndex + job_id;
    auto kernel_name = GetKernelName(vector_size_index);
    auto source = generator(kernel_name, info.nameInCode, vector_size_index);
    std::array<const char *, 1> sources{ source.c_str() };

    // Create the program.
    clProgramWrapper &program = info.programs[vector_size_index];
    auto options = GetBuildOptions(info.relaxedMode);
    int error =
        create_single_kernel_helper(gContext, &program, nullptr, sources.size(),
                                    sources.data(), nullptr, options.c_str());
    if (error != CL_SUCCESS)
    {
        vlog_error("\t\tFAILED -- Failed to create program. (%d)\n", error);
        return error;
    }

    // Create a kernel for each thread. cl_kernels aren't thread safe, so make
    // one for every thread
    auto &kernels = info.kernels[vector_size_index];
    assert(kernels.empty() && "Dirty BuildKernelInfo");
    kernels.resize(info.threadCount);
    for (auto &kernel : kernels)
    {
        kernel = clCreateKernel(program, kernel_name.c_str(), &error);
        if (!kernel || error != CL_SUCCESS)
        {
            vlog_error("\t\tFAILED -- clCreateKernel() failed: (%d)\n", error);
            return error;
        }
    }

    return CL_SUCCESS;
}

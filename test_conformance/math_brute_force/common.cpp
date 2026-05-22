//
// Copyright (c) 2022-2024 The Khronos Group Inc.
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

#include <climits>
#include <vector>
#include <sstream>
#include <string>

namespace {

const char *GetTypeName(ParameterType type)
{
    switch (type)
    {
        case ParameterType::Half: return "half";
        case ParameterType::Float: return "float";
        case ParameterType::Double: return "double";
        case ParameterType::Short: return "short";
        case ParameterType::UShort: return "ushort";
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
        case ParameterType::Half:
        case ParameterType::Float:
        case ParameterType::Double: return "NAN";

        case ParameterType::Short:
        case ParameterType::UShort: return "0x5678";

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
    bool needsFp16 = false;

    for (const auto &type : types)
    {
        switch (type)
        {
            case ParameterType::Double: needsFp64 = true; break;
            case ParameterType::Half: needsFp16 = true; break;
            case ParameterType::Float:
            case ParameterType::Short:
            case ParameterType::UShort:
            case ParameterType::Int:
            case ParameterType::UInt:
            case ParameterType::Long:
            case ParameterType::ULong:
                // No extension required.
                break;
        }
    }

    if (needsFp64) kernel << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    if (needsFp16) kernel << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
}

std::string GetBuildOptions(const BuildKernelInfo &info)
{
    std::ostringstream options;

    if (gForceFTZ)
    {
        options << " -cl-denorms-are-zero";
    }

    if (info.relaxedMode)
    {
        options << " -cl-fast-relaxed-math";
    }

    if (info.correctlyRounded)
    {
        options << " -cl-fp32-correctly-rounded-divide-sqrt";
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
    auto options = GetBuildOptions(info);
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

static const std::vector<double> doubleSpecialValues = {
    -NAN,
    -INFINITY,
    -DBL_MAX,
    MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8),
    MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22),
    -1000.0,
    -100.0,
    -4.0,
    -3.5,
    -3.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51),
    -2.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51),
    -2.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52),
    -1.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    -1.0,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53),
    -0.5,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54),
    -0.25,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074),
    -DBL_MIN,
    MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074),
    -0.0,

    +NAN,
    +INFINITY,
    +DBL_MAX,
    MAKE_HEX_DOUBLE(+0x1.0000000000001p64, +0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(+0x1.0p64, +0x1LL, 64),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8),
    MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22),
    +1000.0,
    +100.0,
    +4.0,
    +3.5,
    +3.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51),
    +2.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51),
    +2.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52),
    +1.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p0, +0x10000000000001LL, -52),
    +1.0,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53),
    +0.5,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54),
    +0.25,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074),
    +DBL_MIN,
    MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074),
    +0.0,
};

static const std::vector<float> floatSpecialValues = {
    -NAN,
    -INFINITY,
    -FLT_MAX,
    MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40),
    MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64),
    MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),
    MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39),
    MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63),
    MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8),
    MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32),
    MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7),
    MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7),
    MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31),
    MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6),
    -1000.f,
    -100.f,
    -4.0f,
    -3.5f,
    -3.0f,
    MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23),
    -2.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23),
    -2.0f,
    MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24),
    -1.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24),
    MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24),
    -1.0f,
    MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25),
    -0.5f,
    MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26),
    MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26),
    -0.25f,
    MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150),
    -FLT_MIN,
    MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150),
    MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150),
    MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150),
    MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150),
    MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150),
    MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150),
    MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150),
    -0.0f,

    +NAN,
    +INFINITY,
    +FLT_MAX,
    MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40),
    MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64),
    MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39),
    MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39),
    MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63),
    MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8),
    MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32),
    MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7),
    MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7),
    MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31),
    MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6),
    +1000.f,
    +100.f,
    +4.0f,
    +3.5f,
    +3.0f,
    MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23),
    2.5f,
    MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),
    +2.0f,
    MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24),
    1.5f,
    MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24),
    MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24),
    +1.0f,
    MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25),
    +0.5f,
    MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26),
    MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26),
    +0.25f,
    MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(+0x1.000002p-126f, +0x1000002L, -150),
    +FLT_MIN,
    MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150),
    MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150),
    MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150),
    MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150),
    MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150),
    MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150),
    MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150),
    +0.0f,
};

static const std::vector<cl_half> halfSpecialValues = {
    0xffff, 0x0000, 0x0001, 0x7c00, /*INFINITY*/
    0xfc00, /*-INFINITY*/
    0x8000, /*-0*/
    0x7bff, /*HALF_MAX*/
    0x0400, /*HALF_MIN*/
    0x03ff, /* Largest denormal */
    0x3c00, /* 1 */
    0xbc00, /* -1 */
    0x3555, /*nearest value to 1/3*/
    0x3bff, /*largest number less than one*/
    0xc000, /* -2 */
    0xfbff, /* -HALF_MAX */
    0x8400, /* -HALF_MIN */
    0x4248, /* M_PI_H */
    0xc248, /* -M_PI_H */
    0xbbff, /* Largest negative fraction */
};

static const std::vector<int> intSpecialValues = {
    0,          1,           2,           3,           126,         127,
    128,        1022,        1023,        1024,        0x02000001,  0x04000001,
    1465264071, 1488522147,  INT_MIN,     INT_MAX,     -1,          -2,
    -3,         -126,        -127,        -128,        -1022,       -1023,
    -1024,      -0x02000001, -0x04000001, -1465264071, -1488522147, -INT_MAX,
};

static uint32_t getExponentDeBruijn(uint32_t number)
{
    // Lookup table mapping 5-bit De Bruijn hashes to exponents
    static const int MultiplyDeBruijnBitPosition[32] = {
        0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9
    };

    // Multiply by the magic De Bruijn sequence and shift down
    return MultiplyDeBruijnBitPosition[(uint32_t)(number * 0x077CB531U) >> 27];
}

static uint32_t input_count_power_of_two = 27;
void initInputCount(int wimpyReductionFactor)
{
    if (gTestAll)
    {
        input_count_power_of_two = 32;
    }
    if (gWimpyMode)
    {
        input_count_power_of_two -= getExponentDeBruijn(wimpyReductionFactor);
    }
    if (gIsEmbedded)
    {
        input_count_power_of_two -=
            getExponentDeBruijn(EMBEDDED_REDUCTION_FACTOR);
    }
}
const size_t getInputCount() { return 1LL << input_count_power_of_two; }

static void fillIntUnaryInput(int *data, size_t num_elems, size_t base_elem,
                              MTdata d)
{
    size_t idx = 0;
    size_t specialValuesCount = intSpecialValues.size();
    for (; idx < num_elems && ((idx + base_elem) < specialValuesCount); idx++)
    {
        data[idx] = intSpecialValues[idx + base_elem];
    }
    for (; idx < num_elems; idx++)
    {
        data[idx] = genrand_int32(d);
    }
}

void fillHalfUnaryInput(cl_half *data, size_t num_elems, size_t base_elem,
                        MTdata d, bool testAll)
{
    cl_ushort *data_short = (cl_ushort *)data;
    if (testAll)
    {
        for (uint32_t i = 0; i < num_elems; i++)
        {
            data_short[i] = (cl_ushort)(base_elem + i);
        }
        return;
    }

    size_t idx = 0;
    size_t specialValuesCount = halfSpecialValues.size();
    for (; idx < num_elems && ((idx + base_elem) < specialValuesCount); idx++)
    {
        data[idx] = halfSpecialValues[idx + base_elem];
    }
    for (; (idx + 1) < num_elems; idx += 2)
    {
        cl_uint gen = genrand_int32(d);
        data_short[idx] = (cl_ushort)(gen & 0xffff);
        data_short[idx + 1] = (cl_ushort)(gen >> 16);
    }
    if (idx < num_elems)
    {
        data_short[idx] = (cl_ushort)(genrand_int32(d) & 0xffff);
    }
}
void fillFloatUnaryInput(float *data, size_t num_elems, size_t base_elem,
                         MTdata d, bool testAll)
{
    cl_uint *data_int = (cl_uint *)data;
    if (testAll)
    {
        for (uint32_t i = 0; i < num_elems; i++)
        {
            data_int[i] = base_elem + i;
        }
        return;
    }

    size_t idx = 0;
    size_t specialValuesCount = floatSpecialValues.size();
    for (; idx < num_elems && ((idx + base_elem) < specialValuesCount); idx++)
    {
        data[idx] = floatSpecialValues[idx + base_elem];
    }
    for (; idx < num_elems; idx++)
    {
        data_int[idx] = genrand_int32(d);
    }
}
void fillDoubleUnaryInput(double *data, size_t num_elems, size_t base_elem,
                          MTdata d)
{
    size_t idx = 0;
    size_t specialValuesCount = doubleSpecialValues.size();
    for (; idx < num_elems && ((idx + base_elem) < specialValuesCount); idx++)
    {
        data[idx] = doubleSpecialValues[idx + base_elem];
    }
    cl_uint *data_int = (cl_uint *)data;
    for (; idx < num_elems; idx++)
    {
        data_int[idx] = genrand_int64(d);
    }
}

#define MASK(n) ((1 << (n)) - 1)

void fillHalfBinaryInput(cl_half *data1, cl_half *data2, size_t num_elems,
                         size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillHalfUnaryInput(data1, num_elems, base_elem & MASK(shift), d, gTestAll);
    fillHalfUnaryInput(data2, num_elems, base_elem >> shift, d, gTestAll);
}
void fillFloatBinaryInput(float *data1, float *data2, size_t num_elems,
                          size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillFloatUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillFloatUnaryInput(data2, num_elems, base_elem >> shift, d);
}
void fillDoubleBinaryInput(double *data1, double *data2, size_t num_elems,
                           size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillDoubleUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillDoubleUnaryInput(data2, num_elems, base_elem >> shift, d);
}

void fillIntHalfBinaryInput(int *data1, cl_half *data2, size_t num_elems,
                            size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillIntUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillHalfUnaryInput(data2, num_elems, base_elem >> shift, d, gTestAll);
}
void fillIntFloatBinaryInput(int *data1, float *data2, size_t num_elems,
                             size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillIntUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillFloatUnaryInput(data2, num_elems, base_elem >> shift, d);
}
void fillIntDoubleBinaryInput(int *data1, double *data2, size_t num_elems,
                              size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 2;
    fillIntUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillDoubleUnaryInput(data2, num_elems, base_elem >> shift, d);
}

void fillHalfTernaryInput(cl_half *data1, cl_half *data2, cl_half *data3,
                          size_t num_elems, size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 3;
    fillHalfUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillHalfUnaryInput(data2, num_elems, (base_elem >> shift) & MASK(shift), d);
    fillHalfUnaryInput(data3, num_elems, base_elem >> (shift * 2), d);
}
void fillFloatTernaryInput(float *data1, float *data2, float *data3,
                           size_t num_elems, size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 3;
    fillFloatUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillFloatUnaryInput(data2, num_elems, (base_elem >> shift) & MASK(shift),
                        d);
    fillFloatUnaryInput(data3, num_elems, base_elem >> (shift * 2), d);
}
void fillDoubleTernaryInput(double *data1, double *data2, double *data3,
                            size_t num_elems, size_t base_elem, MTdata d)
{
    uint32_t shift = input_count_power_of_two / 3;
    fillDoubleUnaryInput(data1, num_elems, base_elem & MASK(shift), d);
    fillDoubleUnaryInput(data2, num_elems, (base_elem >> shift) & MASK(shift),
                         d);
    fillDoubleUnaryInput(data3, num_elems, base_elem >> (shift * 2), d);
}

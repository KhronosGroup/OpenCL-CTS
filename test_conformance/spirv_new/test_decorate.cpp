/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed h_in whole or h_in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited h_in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
h_in whole or h_in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include "testBase.h"
#include "types.hpp"
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

#include <CL/cl_half.h>

long double reference_remainderl(long double x, long double y);
int gIsInRTZMode = 0;
int gDeviceILogb0 = 1;
int gDeviceILogbNaN = 1;
int gCheckTininessBeforeRounding = 1;

static int verify_results(cl_device_id deviceID,
                          cl_context context,
                          cl_command_queue queue,
                          const char *kname,
                          const clProgramWrapper &prog)
{
    const int num = 1 << 20;
    std::vector<cl_int> h_lhs(num);
    std::vector<cl_int> h_rhs(num);

    cl_int err = 0;

    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num; i++)
    {
        h_lhs[i] = genrand<cl_int>(seed);
        h_rhs[i] = genrand<cl_int>(seed);
    }

    clKernelWrapper kernel = clCreateKernel(prog, kname, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    size_t bytes = sizeof(cl_int) * num;

    clMemWrapper lhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, bytes, &h_lhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    clMemWrapper rhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, bytes, &h_rhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 3");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<cl_int> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++)
    {
        if (h_res[i] != (h_lhs[i] + h_rhs[i]))
        {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

int test_decorate_full(cl_device_id deviceID,
                       cl_context context,
                       cl_command_queue queue,
                       const char *name)
{
    clProgramWrapper prog;
    cl_int err = 0;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    return verify_results(deviceID, context, queue, name, prog);
}

TEST_SPIRV_FUNC(decorate_restrict)
{
    return test_decorate_full(deviceID, context, queue, "decorate_restrict");
}

TEST_SPIRV_FUNC(decorate_aliased)
{
    return test_decorate_full(deviceID, context, queue, "decorate_aliased");
}

TEST_SPIRV_FUNC(decorate_alignment)
{
    //TODO: Check for results ? How to ensure buffers are aligned
    clProgramWrapper prog;
    return get_program_with_il(prog, deviceID, context, "decorate_alignment");
}

TEST_SPIRV_FUNC(decorate_constant)
{
    return test_decorate_full(deviceID, context, queue, "decorate_constant");
}

TEST_SPIRV_FUNC(decorate_cpacked)
{
    PACKED(struct packed_struct_t {
        cl_int ival;
        cl_char cval;
    });

    typedef struct packed_struct_t packed_t;

    const int num = 1 << 20;

    std::vector<packed_t> packed(num);
    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, "decorate_cpacked");

    clKernelWrapper kernel = clCreateKernel(prog, "decorate_cpacked", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    size_t bytes = sizeof(packed_t) * num;

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 3");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<packed_t> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++)
    {
        if (h_res[i].ival != 2100483600 || h_res[i].cval != 127)
        {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }

    return 0;
}

template <typename Ti, typename Tl, typename To>
static inline Ti generate_saturated_lhs_input(RandomSeed &seed)
{
    constexpr auto loVal = std::numeric_limits<To>::min();
    constexpr auto hiVal = std::numeric_limits<To>::max();
    constexpr Tl range = (Tl)(hiVal) - (Tl)(loVal);

    if (std::is_same<cl_half, Ti>::value)
    {
        return cl_half_from_float(genrand<float>(seed) * range, CL_HALF_RTE);
    }

    return genrand<Ti>(seed) * range;
}

template <typename Ti, typename Tl, typename To>
static inline Ti generate_saturated_rhs_input(RandomSeed &seed)
{
    constexpr auto hiVal = std::numeric_limits<To>::max();

    Tl val = genrand<Tl>(seed) % hiVal;
    if (std::is_same<cl_half, Ti>::value)
    {
        if (val > 0 && val * 20 < hiVal)
        {
            return cl_half_from_float(NAN, CL_HALF_RTE);
        }
        return cl_half_from_float(val, CL_HALF_RTE);
    }

    if (val > 0 && val * 20 < hiVal)
    {
        return (Ti)NAN;
    }
    return val;
}

template <typename Ti, typename Tl, typename To>
static inline To compute_saturated_output(Ti lhs, Ti rhs)
{
    constexpr auto loVal = std::numeric_limits<To>::min();
    constexpr auto hiVal = std::numeric_limits<To>::max();

    if (std::is_same<Ti, cl_half>::value)
    {
        cl_float f = cl_half_to_float(lhs) * cl_half_to_float(rhs);

        // Quantize to fp16:
        f = cl_half_to_float(cl_half_from_float(f, CL_HALF_RTE));

        To val = (To)std::min<float>(std::max<float>(f, loVal), hiVal);
        if (std::isnan(cl_half_from_float(rhs, CL_HALF_RTE)))
        {
            val = 0;
        }
        return val;
    }

    Tl ival = (Tl)(lhs * rhs);
    To val = (To)std::min<Ti>(std::max<Ti>(ival, loVal), hiVal);

    if (std::isnan(rhs))
    {
        val = 0;
    }
    return val;
}

template <typename Ti, typename Tl, typename To>
int verify_saturated_results(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, const char *kname,
                             const clProgramWrapper &prog)
{
    cl_int err = 0;

    const int num = 1 << 20;

    clKernelWrapper kernel = clCreateKernel(prog, kname, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    size_t in_bytes = sizeof(Ti) * num;
    size_t out_bytes = sizeof(To) * num;

    std::vector<Ti> h_lhs(num);
    std::vector<Ti> h_rhs(num);

    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num; i++)
    {
        h_lhs[i] = generate_saturated_lhs_input<Ti, Tl, To>(seed);
        h_rhs[i] = generate_saturated_rhs_input<Ti, Tl, To>(seed);
    }

    clMemWrapper lhs = clCreateBuffer(context, CL_MEM_READ_ONLY, in_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, in_bytes, &h_lhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    clMemWrapper rhs = clCreateBuffer(context, CL_MEM_READ_ONLY, in_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, in_bytes, &h_rhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 3");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<To> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, out_bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++)
    {
        To val = compute_saturated_output<Ti, Tl, To>(h_lhs[i], h_rhs[i]);

        if (val != h_res[i])
        {
            log_error("Value error at %d: got %d, want %d\n", i, val, h_res[i]);
            return -1;
        }
    }

    return 0;
}


template<typename Ti, typename Tl, typename To>
int test_saturate_full(cl_device_id deviceID,
                       cl_context context,
                       cl_command_queue queue,
                       const char *name,
                       const char *types)
{
    if (std::string(types).find("double") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp64"))
        {
            log_info("Extension cl_khr_fp64 not supported; skipping double "
                     "tests.\n");
            return 0;
        }
    }

    if (std::string(types).find("half") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp16"))
        {
            log_info(
                "Extension cl_khr_fp16 not supported; skipping half tests.\n");
            return 0;
        }
    }

    clProgramWrapper prog;
    cl_int err = 0;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");
    return verify_saturated_results<Ti, Tl, To>(deviceID, context, queue, name,
                                                prog);
}

#define TEST_SATURATED_CONVERSION(Ti, Tl, To)                                  \
    TEST_SPIRV_FUNC(decorate_saturated_conversion_##Ti##_to_##To)              \
    {                                                                          \
        typedef cl_##Ti cl_Ti;                                                 \
        typedef cl_##Tl cl_Tl;                                                 \
        typedef cl_##To cl_To;                                                 \
        const char *name = "decorate_saturated_conversion_" #Ti "_to_" #To;    \
        return test_saturate_full<cl_Ti, cl_Tl, cl_To>(                        \
            deviceID, context, queue, name, #Ti #Tl #To);                      \
    }

TEST_SATURATED_CONVERSION(half, short, char)
TEST_SATURATED_CONVERSION(half, ushort, uchar)
TEST_SATURATED_CONVERSION(float, int, char)
TEST_SATURATED_CONVERSION(float, uint, uchar)
TEST_SATURATED_CONVERSION(float, int, short)
TEST_SATURATED_CONVERSION(float, uint, ushort)
TEST_SATURATED_CONVERSION(double, long, int)
TEST_SATURATED_CONVERSION(double, ulong, uint)

template<typename Ti, typename To>
int test_fp_rounding(cl_device_id deviceID,
                     cl_context context,
                     cl_command_queue queue,
                     const char *name,
                     std::vector<Ti> &h_in,
                     std::vector<To> &h_out)
{
    if (std::string(name).find("double") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp64"))
        {
            log_info("Extension cl_khr_fp64 not supported; skipping double "
                     "tests.\n");
            return 0;
        }
    }

    if (std::string(name).find("half") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp16"))
        {
            log_info(
                "Extension cl_khr_fp16 not supported; skipping half tests.\n");
            return 0;
        }
    }

    const int num = h_in.size();
    const size_t in_bytes = num * sizeof(Ti);
    const size_t out_bytes = num * sizeof(To);
    cl_int err = 0;

    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_ONLY, in_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create input buffer");

    clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_ONLY, out_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create output buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, in_bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to write to input");

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

    std::vector<To> h_res(num);
    err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, out_bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from output");

    for (int i = 0; i < num; i++)
    {
        if (h_res[i] != h_out[i])
        {
            log_error("Values do not match at location %d. Original :%lf, "
                      "Expected: %ld, Found %ld\n",
                      i, h_in[i], h_out[i], h_res[i]);
            return -1;
        }
    }

    return 0;
}

template <typename T> static inline double to_double(T in) { return in; }

template <> inline double to_double(cl_half in) { return cl_half_to_float(in); }

template <typename Ti, typename To> static inline To round_to_zero(Ti in)
{
    return (To)to_double(in);
}

template <typename T> static inline int sign(T val)
{
    if (val < 0) return -1;
    if (val > 0) return 1;
    return 0;
}

template <typename Ti, typename To> static inline To round_to_even(Ti in)
{
    double din = to_double(in);
    return std::floor(din + 0.5) - 1
        + std::abs(sign(reference_remainderl((long double)din, 2) - 0.5));
}

template <typename Ti, typename To> static inline To round_to_posinf(Ti in)
{
    return std::ceil(to_double(in));
}

template <typename Ti, typename To> static inline To round_to_neginf(Ti in)
{
    return std::floor(to_double(in));
}

template <typename Ti, typename To>
static inline Ti generate_fprounding_input(RandomSeed &seed)
{
    constexpr auto minVal = std::numeric_limits<To>::min() / 2;
    constexpr auto maxVal = std::numeric_limits<To>::max() / 2;

    if (std::is_same<cl_half, Ti>::value)
    {
        cl_float f = genrandReal_range<cl_float>(minVal, maxVal, seed);
        return cl_half_from_float(f, CL_HALF_RTE);
    }

    return genrandReal_range<Ti>(minVal, maxVal, seed);
}

#define TEST_SPIRV_FP_ROUNDING_DECORATE(name, func, Ti, To)                    \
    TEST_SPIRV_FUNC(decorate_fp_rounding_mode_##name##_##Ti##_##To)            \
    {                                                                          \
        typedef cl_##Ti clTi;                                                  \
        typedef cl_##To clTo;                                                  \
        const int num = 1 << 16;                                               \
        std::vector<clTi> in(num);                                             \
        std::vector<clTo> out(num);                                            \
        RandomSeed seed(gRandomSeed);                                          \
                                                                               \
        for (int i = 0; i < num; i++)                                          \
        {                                                                      \
            in[i] = generate_fprounding_input<clTi, clTo>(seed);               \
            out[i] = func<clTi, clTo>(in[i]);                                  \
        }                                                                      \
        const char *name = "decorate_rounding_" #name "_" #Ti "_" #To;         \
        return test_fp_rounding(deviceID, context, queue, name, in, out);      \
    }

TEST_SPIRV_FP_ROUNDING_DECORATE(rte, round_to_even, half, short);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtz, round_to_zero, half, short);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtp, round_to_posinf, half, short);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtn, round_to_neginf, half, short);

TEST_SPIRV_FP_ROUNDING_DECORATE(rte, round_to_even, float, int);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtz, round_to_zero, float, int);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtp, round_to_posinf, float, int);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtn, round_to_neginf, float, int);

TEST_SPIRV_FP_ROUNDING_DECORATE(rte, round_to_even, double, long);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtz, round_to_zero, double, long);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtp, round_to_posinf, double, long);
TEST_SPIRV_FP_ROUNDING_DECORATE(rtn, round_to_neginf, double, long);

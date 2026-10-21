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

#include "CL/cl.h"
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/specialFloatValues.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include "utility.h"

#include <array>
#include <cmath>
#include <cinttypes>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t CL_VALUE_MAX_BYTES = 64;

struct AnyValue
{
    std::string cl_type;
    std::size_t byte_size = 0;
    std::array<uint8_t, CL_VALUE_MAX_BYTES> data{};

    template <typename T> static AnyValue make(const T &val)
    {
        static_assert(sizeof(T) <= CL_VALUE_MAX_BYTES,
                      "CLValue: type exceeds CL_VALUE_MAX_BYTES");
        AnyValue v;
        v.cl_type = "half";
        if constexpr (std::is_same_v<T, cl_float>)
            v.cl_type = "float";
        else if constexpr (std::is_same_v<T, cl_double>)
            v.cl_type = "double";
        else if constexpr (std::is_same_v<T, cl_int>)
            v.cl_type = "int";

        v.byte_size = sizeof(T);
        std::memcpy(v.data.data(), &val, sizeof(T));
        return v;
    }

    template <typename T> bool all_elements_nan() const
    {
        if constexpr (std::is_same_v<T, cl_half>)
        {
            for (std::size_t off = 0; off + 2 <= byte_size; off += 2)
            {
                uint16_t bits;
                std::memcpy(&bits, data.data() + off, 2);
                if ((bits & 0x7C00u) != 0x7C00u || (bits & 0x03FFu) == 0u)
                    return false;
            }
            return true;
        }
        if constexpr (std::is_same_v<T, cl_double>)
        {
            for (std::size_t off = 0; off + 8 <= byte_size; off += 8)
            {
                uint64_t bits;
                std::memcpy(&bits, data.data() + off, 8);
                if ((bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull
                    || (bits & 0x000FFFFFFFFFFFFFull) == 0ull)
                    return false;
            }
            return true;
        }

        for (std::size_t off = 0; off + 4 <= byte_size; off += 4)
        {
            uint32_t bits;
            std::memcpy(&bits, data.data() + off, 4);
            if ((bits & 0x7F800000u) != 0x7F800000u
                || (bits & 0x007FFFFFu) == 0u)
                return false;
        }
        return true;
    }
};

struct EdgeCaseSpec
{
    const char *func_name;
    std::vector<AnyValue> inputs;
    AnyValue expected;
    bool expect_nan = false;
};

using special_float_values::AbstractValue;

const AbstractValue POS_ZERO = special_float_values::pos_zero();
const AbstractValue NEG_ZERO = special_float_values::neg_zero();
const AbstractValue POS_INF = special_float_values::pos_inf();
const AbstractValue NEG_INF = special_float_values::neg_inf();
const AbstractValue NAN_V = special_float_values::nan();
const AbstractValue ONE = special_float_values::finite(1.0);
const AbstractValue NEG_ONE = special_float_values::finite(-1.0);
const AbstractValue TWO = special_float_values::finite(2.0);
const AbstractValue NEG_TWO = special_float_values::finite(-2.0);

inline AbstractValue AV_F(double v) { return special_float_values::finite(v); }
inline AbstractValue AV_INT(int v) { return special_float_values::integer(v); }
inline AbstractValue AV_SMALLEST_POS_DENORM()
{
    return special_float_values::smallest_pos_denorm();
}
inline AbstractValue AV_SMALLEST_NEG_DENORM()
{
    return special_float_values::smallest_neg_denorm();
}

struct AbstractEdgeCase
{
    const char *func_name;
    std::vector<AbstractValue> inputs;
    AbstractValue expected;

    bool expect_nan = false;
    bool requires_rte = false; // CL_FP_ROUND_TO_NEAREST

    bool areValuesSupportedBy(cl_device_fp_config fp_config) const
    {
        if (!expected.isSupportedBy(fp_config)) return false;
        for (const auto &input : inputs)
            if (!input.isSupportedBy(fp_config)) return false;
        return true;
    }
};

// Taken from OpenCL C Section 7.5.1. Additional Requirements Beyond C99 TC2

const AbstractEdgeCase edge_case_table[] = {

    { "acospi", { ONE }, POS_ZERO },
    { "acospi", { AV_F(2) }, NAN_V, true },
    { "acospi", { AV_F(-2) }, NAN_V, true },
    { "asinpi", { POS_ZERO }, POS_ZERO },
    { "asinpi", { NEG_ZERO }, NEG_ZERO },
    { "asinpi", { AV_F(2) }, NAN_V, true },
    { "asinpi", { AV_F(-2) }, NAN_V, true },
    { "atanpi", { POS_ZERO }, POS_ZERO },
    { "atanpi", { NEG_ZERO }, NEG_ZERO },
    { "atanpi", { POS_INF }, AV_F(0.5) },
    { "atanpi", { NEG_INF }, AV_F(-0.5) },
    { "atan2pi", { POS_ZERO, NEG_ZERO }, AV_F(1.0) },
    { "atan2pi", { NEG_ZERO, NEG_ZERO }, AV_F(-1.0) },
    { "atan2pi", { POS_ZERO, POS_ZERO }, POS_ZERO },
    { "atan2pi", { NEG_ZERO, POS_ZERO }, NEG_ZERO },
    { "atan2pi", { POS_ZERO, NEG_ONE }, AV_F(1.0) },
    { "atan2pi", { NEG_ZERO, NEG_ONE }, AV_F(-1.0) },
    { "atan2pi", { POS_ZERO, ONE }, POS_ZERO },
    { "atan2pi", { NEG_ZERO, ONE }, NEG_ZERO },
    { "atan2pi", { NEG_ONE, POS_ZERO }, AV_F(-0.5) },
    { "atan2pi", { NEG_ONE, NEG_ZERO }, AV_F(-0.5) },
    { "atan2pi", { ONE, POS_ZERO }, AV_F(0.5) },
    { "atan2pi", { ONE, NEG_ZERO }, AV_F(0.5) },
    { "atan2pi", { ONE, NEG_INF }, AV_F(1.0) },
    { "atan2pi", { NEG_ONE, NEG_INF }, AV_F(-1.0) },
    { "atan2pi", { ONE, POS_INF }, POS_ZERO },
    { "atan2pi", { NEG_ONE, POS_INF }, NEG_ZERO },
    { "atan2pi", { POS_INF, ONE }, AV_F(0.5) },
    { "atan2pi", { NEG_INF, ONE }, AV_F(-0.5) },
    { "atan2pi", { POS_INF, NEG_INF }, AV_F(0.75) },
    { "atan2pi", { NEG_INF, NEG_INF }, AV_F(-0.75) },
    { "atan2pi", { POS_INF, POS_INF }, AV_F(0.25) },
    { "atan2pi", { NEG_INF, POS_INF }, AV_F(-0.25) },
    { "ceil", { AV_F(-0.5) }, NEG_ZERO, false, true },
    { "ceil", { AV_F(-0.25) }, NEG_ZERO, false, true },
    { "cospi", { POS_ZERO }, ONE },
    { "cospi", { NEG_ZERO }, ONE },
    { "cospi", { AV_F(0.5) }, POS_ZERO },
    { "cospi", { AV_F(1.5) }, POS_ZERO },
    { "cospi", { AV_F(2.5) }, POS_ZERO },
    { "cospi", { AV_F(-0.5) }, POS_ZERO },
    { "cospi", { AV_F(-1.5) }, POS_ZERO },
    { "cospi", { POS_INF }, NAN_V, true },
    { "cospi", { NEG_INF }, NAN_V, true },
    { "exp10", { NEG_INF }, POS_ZERO },
    { "exp10", { POS_INF }, POS_INF },
    { "fdim", { ONE, NAN_V }, NAN_V, true },
    { "fdim", { NAN_V, ONE }, NAN_V, true },
    { "fdim", { POS_INF, NAN_V }, NAN_V, true },
    { "fdim", { NAN_V, POS_INF }, NAN_V, true },
    { "fmod", { POS_ZERO, NAN_V }, NAN_V, true },
    { "fmod", { NEG_ZERO, NAN_V }, NAN_V, true },
    { "nextafter", { NEG_ZERO, ONE }, AV_SMALLEST_POS_DENORM() },
    { "nextafter", { POS_ZERO, NEG_ONE }, AV_SMALLEST_NEG_DENORM() },
    { "pow", { POS_ZERO, NEG_INF }, POS_INF },
    { "pow", { NEG_ZERO, NEG_INF }, POS_INF },
    { "pown", { POS_ZERO, AV_INT(0) }, ONE },
    { "pown", { NEG_ZERO, AV_INT(0) }, ONE },
    { "pown", { POS_INF, AV_INT(0) }, ONE },
    { "pown", { NEG_INF, AV_INT(0) }, ONE },
    { "pown", { NAN_V, AV_INT(0) }, ONE },
    { "pown", { POS_ZERO, AV_INT(-1) }, POS_INF },
    { "pown", { NEG_ZERO, AV_INT(-1) }, NEG_INF },
    { "pown", { POS_ZERO, AV_INT(-3) }, POS_INF },
    { "pown", { NEG_ZERO, AV_INT(-3) }, NEG_INF },
    { "pown", { POS_ZERO, AV_INT(-2) }, POS_INF },
    { "pown", { NEG_ZERO, AV_INT(-2) }, POS_INF },
    { "pown", { POS_ZERO, AV_INT(2) }, POS_ZERO },
    { "pown", { NEG_ZERO, AV_INT(2) }, POS_ZERO },
    { "pown", { POS_ZERO, AV_INT(1) }, POS_ZERO },
    { "pown", { NEG_ZERO, AV_INT(1) }, NEG_ZERO },
    { "pown", { POS_ZERO, AV_INT(3) }, POS_ZERO },
    { "pown", { NEG_ZERO, AV_INT(3) }, NEG_ZERO },
    { "powr", { ONE, POS_ZERO }, ONE },
    { "powr", { TWO, NEG_ZERO }, ONE },
    { "powr", { POS_ZERO, NEG_ONE }, POS_INF },
    { "powr", { NEG_ZERO, NEG_ONE }, POS_INF },
    { "powr", { POS_ZERO, NEG_INF }, POS_INF },
    { "powr", { NEG_ZERO, NEG_INF }, POS_INF },
    { "powr", { POS_ZERO, ONE }, POS_ZERO },
    { "powr", { NEG_ZERO, ONE }, POS_ZERO },
    { "powr", { ONE, TWO }, ONE },
    { "powr", { ONE, NEG_ONE }, ONE },
    { "powr", { NEG_ONE, TWO }, NAN_V, true },
    { "powr", { POS_ZERO, POS_ZERO }, NAN_V, true },
    { "powr", { NEG_ZERO, NEG_ZERO }, NAN_V, true },
    { "powr", { POS_INF, POS_ZERO }, NAN_V, true },
    { "powr", { POS_INF, NEG_ZERO }, NAN_V, true },
    { "powr", { ONE, POS_INF }, NAN_V, true },
    { "powr", { ONE, NEG_INF }, NAN_V, true },
    { "rint", { AV_F(-0.5) }, NEG_ZERO, false, true },
    { "rootn", { POS_ZERO, AV_INT(-1) }, POS_INF },
    { "rootn", { NEG_ZERO, AV_INT(-1) }, NEG_INF },
    { "rootn", { POS_ZERO, AV_INT(-3) }, POS_INF },
    { "rootn", { NEG_ZERO, AV_INT(-3) }, NEG_INF },
    { "rootn", { POS_ZERO, AV_INT(-2) }, POS_INF },
    { "rootn", { NEG_ZERO, AV_INT(-2) }, POS_INF },
    { "rootn", { POS_ZERO, AV_INT(2) }, POS_ZERO },
    { "rootn", { NEG_ZERO, AV_INT(2) }, POS_ZERO },
    { "rootn", { POS_ZERO, AV_INT(1) }, POS_ZERO },
    { "rootn", { NEG_ZERO, AV_INT(1) }, NEG_ZERO },
    { "rootn", { POS_ZERO, AV_INT(3) }, POS_ZERO },
    { "rootn", { NEG_ZERO, AV_INT(3) }, NEG_ZERO },
    { "rootn", { NEG_ONE, AV_INT(2) }, NAN_V, true },
    { "rootn", { NEG_ONE, AV_INT(4) }, NAN_V, true },
    { "rootn", { ONE, AV_INT(0) }, NAN_V, true },
    { "rootn", { POS_ZERO, AV_INT(0) }, NAN_V, true },
    { "round", { AV_F(-0.25) }, NEG_ZERO, false, true },
    { "sinpi", { POS_ZERO }, POS_ZERO },
    { "sinpi", { NEG_ZERO }, NEG_ZERO },
    { "sinpi", { ONE }, POS_ZERO },
    { "sinpi", { TWO }, POS_ZERO },
    { "sinpi", { AV_F(4.0) }, POS_ZERO },
    { "sinpi", { NEG_ONE }, NEG_ZERO },
    { "sinpi", { NEG_TWO }, NEG_ZERO },
    { "sinpi", { AV_F(-4.0) }, NEG_ZERO },
    { "sinpi", { POS_INF }, NAN_V, true },
    { "sinpi", { NEG_INF }, NAN_V, true },
    { "tanpi", { POS_ZERO }, POS_ZERO },
    { "tanpi", { NEG_ZERO }, NEG_ZERO },
    { "tanpi", { POS_INF }, NAN_V, true },
    { "tanpi", { NEG_INF }, NAN_V, true },
    { "tanpi", { AV_F(0.0) }, POS_ZERO },
    { "tanpi", { AV_F(2.0) }, POS_ZERO },
    { "tanpi", { AV_F(-2.0) }, NEG_ZERO },
    { "tanpi", { ONE }, NEG_ZERO },
    { "tanpi", { NEG_ONE }, POS_ZERO },
    { "tanpi", { AV_F(3.0) }, NEG_ZERO },
    { "tanpi", { AV_F(-3.0) }, POS_ZERO },
    { "tanpi", { AV_F(0.5) }, POS_INF },
    { "tanpi", { AV_F(2.5) }, POS_INF },
    { "tanpi", { AV_F(1.5) }, NEG_INF },
    { "tanpi", { AV_F(-0.5) }, NEG_INF },
    { "trunc", { AV_F(-0.5) }, NEG_ZERO, false, true },
    { "trunc", { AV_F(-0.25) }, NEG_ZERO, false, true },
};

struct EdgeCasesTest
{
    std::vector<EdgeCaseSpec> batch_cases;
    std::string kernel_src;
    std::vector<uint8_t> inData;
    std::vector<uint8_t> result;

    void log_anyvalue(const AnyValue &v)
    {
        const std::size_t elem = [&] {
            if (v.cl_type == "double") return std::size_t(8);
            if (v.cl_type == "float") return std::size_t(4);
            if (v.cl_type == "int") return std::size_t(4);
            return std::size_t(2); // half
        }();

        for (std::size_t off = 0; off < v.byte_size; off += elem)
        {
            switch (elem)
            {
                case 8: {
                    uint64_t bits;
                    std::memcpy(&bits, v.data.data() + off, 8);
                    log_error("0x%016" PRIx64, bits);
                    break;
                }
                case 4: {
                    uint32_t bits;
                    std::memcpy(&bits, v.data.data() + off, 4);
                    log_error("0x%08" PRIx32, bits);
                    break;
                }
                case 2: {
                    uint16_t bits;
                    std::memcpy(&bits, v.data.data() + off, 2);
                    log_error("0x%04" PRIx16, bits);
                    break;
                }
            }
        }
    }

    inline void accumulate_edge_case(const EdgeCaseSpec &ec)
    {
        std::string &src = kernel_src;
        std::size_t ind = batch_cases.size();
        // Build kernel
        if (src.empty())
        {
            src += "__kernel void test_edge_case(\n";
            src += "    __global ";
            src += ec.expected.cl_type;
            src += " *out";
            for (std::size_t i = 0; i < ec.inputs.size(); ++i)
            {
                src += ",\n    __global const ";
                src += ec.inputs[i].cl_type;
                src += " *in";
                src += std::to_string(i);
            }
            src += ")\n{";
        }
        else
        {
            assert(batch_cases.front().inputs.size() == ec.inputs.size());
        }

        src += "\n    out[";
        src += std::to_string(ind);
        src += "] = ";
        src += ec.func_name;
        src += "(";

        for (std::size_t i = 0; i < ec.inputs.size(); ++i)
        {
            if (i) src += ", ";
            src += "in";
            src += std::to_string(i);
            src += "[";
            src += std::to_string(ind);
            src += "]";
        }
        src += ");";

        batch_cases.push_back(ec);
    }

    template <typename T>
    inline cl_int run_accumulated_cases(cl_context context,
                                        cl_command_queue queue)
    {
        std::string &src = kernel_src;
        src += "\n}\n";
        if constexpr (std::is_same_v<T, cl_half>)
            src = std::string("#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n")
                + src;
        else if constexpr (std::is_same_v<T, cl_double>)
            src = std::string("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                + src;
        const char *src_ptr = src.c_str();
        clProgramWrapper program;
        clKernelWrapper kernel;

        if (create_single_kernel_helper(context, &program, &kernel, 1, &src_ptr,
                                        "test_edge_case"))
        {
            log_error("ERROR: Failed to build kernel for '%s'\nSource:\n%s\n",
                      batch_cases.front().func_name, src.c_str());
            return TEST_FAIL;
        }

        cl_int err = CL_SUCCESS;

        clMemWrapper out_buf = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY,
            batch_cases.front().expected.byte_size * batch_cases.size(),
            nullptr, &err);
        if (err != CL_SUCCESS)
        {
            log_error("ERROR: clCreateBuffer (out) failed for '%s': %d\n",
                      batch_cases.front().func_name, err);
            return TEST_FAIL;
        }

        std::vector<clMemWrapper> in_bufs;
        in_bufs.reserve(batch_cases.front().inputs.size());

        for (std::size_t i = 0; i < batch_cases.front().inputs.size(); ++i)
        {
            cl_mem buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        batch_cases.front().inputs[i].byte_size
                                            * batch_cases.size(),
                                        nullptr, &err);
            if (err != CL_SUCCESS)
            {
                log_error("ERROR: clCreateBuffer (in%zu) failed for '%s': %d\n",
                          i, batch_cases.front().func_name, err);
                return TEST_FAIL;
            }
            in_bufs.push_back(buf);

            inData.resize(batch_cases.front().inputs[i].byte_size
                          * batch_cases.size());

            size_t byte_offset = 0;
            for (const auto &elem : batch_cases)
            {
                std::memcpy(&inData[byte_offset], elem.inputs[i].data.data(),
                            elem.inputs[i].byte_size);
                byte_offset += elem.inputs[i].byte_size;
            }

            err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, inData.size(),
                                       inData.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                log_error("ERROR: clEnqueueWriteBuffer (in%zu) failed for"
                          " '%s': %d\n",
                          i, batch_cases.front().func_name, err);
                return TEST_FAIL;
            }
        }

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_buf);
        for (cl_uint i = 0; i < static_cast<cl_uint>(in_bufs.size()); ++i)
            err |= clSetKernelArg(kernel, i + 1, sizeof(cl_mem), &in_bufs[i]);
        if (err != CL_SUCCESS)
        {
            log_error("ERROR: clSetKernelArg failed for '%s': %d\n",
                      batch_cases.front().func_name, err);
            return TEST_FAIL;
        }

        {
            const std::size_t gws = 1;
            err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws,
                                         nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                log_error("ERROR: clEnqueueNDRangeKernel failed for '%s': %d\n",
                          batch_cases.front().func_name, err);
                return TEST_FAIL;
            }
        }

        {
            result.resize(batch_cases.front().expected.byte_size
                          * batch_cases.size());
            err = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0,
                                      batch_cases.front().expected.byte_size
                                          * batch_cases.size(),
                                      result.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                log_error("ERROR: clEnqueueReadBuffer failed for '%s': %d\n",
                          batch_cases.front().func_name, err);
                return TEST_FAIL;
            }

            size_t byte_offset = 0;
            for (const auto &ec : batch_cases)
            {
                if (ec.expect_nan)
                {
                    AnyValue got;
                    got.byte_size = ec.expected.byte_size;
                    std::memcpy(got.data.data(), &result[byte_offset],
                                got.byte_size);
                    if (!got.all_elements_nan<T>())
                    {
                        log_error("FAIL: %s(", ec.func_name);
                        for (std::size_t i = 0; i < ec.inputs.size(); ++i)
                        {
                            if (i) log_error(", ");
                            log_anyvalue(ec.inputs[i]);
                        }
                        log_error(") - expected NaN, got 0x");

                        for (std::size_t i = ec.expected.byte_size; i != 0; --i)
                            log_error("%02x", result[byte_offset + i - 1]);
                        log_error("\n");
                        err = -1;
                    }
                }
                else
                {
                    if (std::memcmp(&result[byte_offset],
                                    ec.expected.data.data(),
                                    ec.expected.byte_size)
                        != 0)
                    {
                        log_error("FAIL: %s(", ec.func_name);
                        for (std::size_t i = 0; i < ec.inputs.size(); ++i)
                        {
                            if (i) log_error(", ");
                            log_anyvalue(ec.inputs[i]);
                        }

                        log_error(") - expected ");
                        log_anyvalue(ec.expected);

                        log_error(", got 0x");
                        for (std::size_t i = ec.expected.byte_size; i != 0; --i)
                            log_error("%02x", result[byte_offset + i - 1]);
                        log_error("\n");
                        err = -1;
                    }
                }

                byte_offset += ec.expected.byte_size;
            }
        }

        return err == -1 ? TEST_FAIL : TEST_PASS;
    }

    template <typename T> AnyValue abstract_to_anyvalue(const AbstractValue &av)
    {
        if (av.kind == AbstractValue::Kind::Int)
            return AnyValue::make<cl_int>(av.i);

        return AnyValue::make<T>(av.toValue<T>());
    }

    template <typename T>
    EdgeCaseSpec make_edge_case(const AbstractEdgeCase &aec)
    {
        EdgeCaseSpec ec;
        ec.expect_nan = aec.expect_nan;
        ec.expected = abstract_to_anyvalue<T>(aec.expected);
        ec.inputs.reserve(aec.inputs.size());
        for (const auto &av : aec.inputs)
            ec.inputs.push_back(abstract_to_anyvalue<T>(av));
        ec.func_name = aec.func_name;
        return ec;
    }

    template <typename T>
    cl_int flush_group(const AbstractEdgeCase *cases, std::size_t count,
                       cl_context context, cl_command_queue queue,
                       std::size_t i)
    {
        cl_int ret = 0;
        if (!batch_cases.empty()
            && ((i == count - 1)
                || std::strcmp(cases[i].func_name, cases[i + 1].func_name)
                    != 0))
        {
            if (run_accumulated_cases<T>(context, queue) != CL_SUCCESS)
                ret = -1;
            batch_cases.clear();
            kernel_src.clear();
        }
        return ret;
    }

    inline cl_int run_edge_cases(const AbstractEdgeCase *cases,
                                 std::size_t count, cl_context context,
                                 cl_command_queue queue)
    {
        cl_int overall = CL_SUCCESS;
        if (gTestFloat)
        {
            log_info("float\n");

            // Iterate over edge cases, grouping those with the same function
            // name into a single kernel call to avoid per-case build overhead.
            // The same pattern is applied for all three floating point
            // precisions.

            for (std::size_t i = 0; i < count; ++i)
            {
                const auto &aec = cases[i];
                bool skip = false;
                cl_device_fp_config fp_config = gFloatCapabilities;
                // Full Profile requires Inf and NaN, so still run these cases
                // if the mandatory capability bit is absent.
                if (!gIsEmbedded) fp_config |= CL_FP_INF_NAN;

                if (!aec.areValuesSupportedBy(fp_config))
                {
                    log_info("SKIP (unsupported special values): %s\n",
                             aec.func_name);
                    skip = true;
                }

                if (gIsEmbedded)
                {
                    if (aec.requires_rte
                        && !(gFloatCapabilities & CL_FP_ROUND_TO_NEAREST))
                    {
                        log_info("SKIP (no CL_FP_ROUND_TO_NEAREST): %s\n",
                                 aec.func_name);
                        skip = true;
                    }
                }

                if (!skip)
                {
                    const EdgeCaseSpec ec = make_edge_case<cl_float>(aec);
                    accumulate_edge_case(ec);
                }

                if (flush_group<cl_float>(cases, count, context, queue, i)
                    != CL_SUCCESS)
                    overall = -1;
            }
        }
        else
            log_info("skipping float test\n");


        if (gHasHalf)
        {
            log_info("half\n");
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto &aec = cases[i];
                bool skip = false;
                if (!aec.areValuesSupportedBy(gHalfCapabilities))
                {
                    log_info("SKIP fp16 (unsupported special values): %s\n",
                             aec.func_name);
                    skip = true;
                }

                if (aec.requires_rte
                    && !(gHalfCapabilities & CL_FP_ROUND_TO_NEAREST))
                {
                    log_info("SKIP fp16 (no CL_FP_ROUND_TO_NEAREST): %s\n",
                             aec.func_name);
                    skip = true;
                }

                if (!skip)
                {
                    const EdgeCaseSpec ec = make_edge_case<cl_half>(aec);
                    accumulate_edge_case(ec);
                }

                if (flush_group<cl_half>(cases, count, context, queue, i)
                    != CL_SUCCESS)
                    overall = -1;
            }
        }
        else
            log_info("skipping half test\n");

        if (gHasDouble)
        {
            log_info("double\n");
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto &aec = cases[i];
                bool skip = false;
                if (!aec.areValuesSupportedBy(gDoubleCapabilities))
                {
                    log_info("SKIP fp64 (unsupported special values): %s\n",
                             aec.func_name);
                    skip = true;
                }

                if (aec.requires_rte
                    && !(gDoubleCapabilities & CL_FP_ROUND_TO_NEAREST))
                {
                    log_info("SKIP fp64 (no CL_FP_ROUND_TO_NEAREST): %s\n",
                             aec.func_name);
                    skip = true;
                }

                if (!skip)
                {
                    const EdgeCaseSpec ec = make_edge_case<cl_double>(aec);
                    accumulate_edge_case(ec);
                }

                if (flush_group<cl_double>(cases, count, context, queue, i)
                    != CL_SUCCESS)
                    overall = -1;
            }
        }
        else
            log_info("skipping double test\n");

        return overall;
    }
};

} // anonymous namespace

REGISTER_TEST(math_edge_cases)
{
    if (gSkipCorrectnessTesting)
    {
        log_info("Skipping math_edge_cases test\n");
        return TEST_SKIPPED_ITSELF;
    }
    EdgeCasesTest edge_cases_test;
    return edge_cases_test.run_edge_cases(
        edge_case_table, sizeof(edge_case_table) / sizeof((edge_case_table)[0]),
        gContext, gQueue);
}

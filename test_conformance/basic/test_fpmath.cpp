//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "harness/compat.h"
#include "harness/rounding_mode.h"
#include "harness/stringHelpers.h"

#include <CL/cl_half.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "procs.h"

extern cl_half_rounding_mode halfRoundingMode;

namespace {

const char *fp_kernel_code = R"(
%s
__kernel void test_fp(__global TYPE *srcA, __global TYPE *srcB, __global TYPE *dst)
{
    int  tid = get_global_id(0);

    dst[tid] = srcA[tid] OP srcB[tid];
})";

#define HFF(num) cl_half_from_float(num, halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

template <typename T> double toDouble(T val)
{
    if (std::is_same<cl_half, T>::value)
        return HTF(val);
    else
        return val;
}

bool isHalfNan(cl_half v)
{
    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (v >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = v & 0x3FF;

    // NaN test
    return (h_exp == 0x1F && h_mant != 0);
}

cl_half half_plus(cl_half a, cl_half b)
{
    return HFF(std::plus<float>()(HTF(a), HTF(b)));
}

cl_half half_minus(cl_half a, cl_half b)
{
    return HFF(std::minus<float>()(HTF(a), HTF(b)));
}

cl_half half_mult(cl_half a, cl_half b)
{
    return HFF(std::multiplies<float>()(HTF(a), HTF(b)));
}

template <typename T> struct TestDef
{
    const char op;
    std::function<T(T, T)> ref;
    std::string type_str;
    size_t vec_size;
};

template <typename T>
int verify_fp(std::vector<T> (&input)[2], std::vector<T> &output,
              const TestDef<T> &test)
{
    auto &inA = input[0];
    auto &inB = input[1];
    for (size_t i = 0; i < output.size(); i++)
    {
        bool nan_test = false;

        T r = test.ref(inA[i], inB[i]);

        if (std::is_same<T, cl_half>::value)
            nan_test = !(isHalfNan(r) && isHalfNan(output[i]));

        if (r != output[i] && nan_test)
        {
            log_error("FP math test for type: %s, vec size: %zu, failed at "
                      "index %zu, %a '%c' %a, expected %a, get %a\n",
                      test.type_str.c_str(), test.vec_size, i, toDouble(inA[i]),
                      test.op, toDouble(inB[i]), toDouble(r),
                      toDouble(output[i]));
            return -1;
        }
    }

    return 0;
}

template <typename T> void generate_random_inputs(std::vector<T> (&input)[2])
{
    RandomSeed seed(gRandomSeed);

    if (std::is_same<T, float>::value)
    {
        auto random_generator = [&seed]() {
            return get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31),
                                    MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), seed);
        };
        for (auto &v : input)
            std::generate(v.begin(), v.end(), random_generator);
    }
    else if (std::is_same<T, double>::value)
    {
        auto random_generator = [&seed]() {
            return get_random_double(-MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63),
                                     MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63),
                                     seed);
        };
        for (auto &v : input)
            std::generate(v.begin(), v.end(), random_generator);
    }
    else
    {
        auto random_generator = [&seed]() {
            return HFF(get_random_float(-MAKE_HEX_FLOAT(0x1.0p8f, 0x1, 8),
                                        MAKE_HEX_FLOAT(0x1.0p8f, 0x1, 8),
                                        seed));
        };
        for (auto &v : input)
            std::generate(v.begin(), v.end(), random_generator);
    }
}

struct TypesIterator
{
    using TypeIter = std::tuple<cl_float, cl_half, cl_double>;

    TypesIterator(cl_device_id deviceID, cl_context context,
                  cl_command_queue queue, int num_elems)
        : context(context), queue(queue), fpConfigHalf(0), fpConfigFloat(0),
          num_elements(num_elems)
    {
        // typeid().name one day
        type2name[sizeof(cl_half)] = "half";
        type2name[sizeof(cl_float)] = "float";
        type2name[sizeof(cl_double)] = "double";

        fp16Support = is_extension_available(deviceID, "cl_khr_fp16");
        fp64Support = is_extension_available(deviceID, "cl_khr_fp64");

        fpConfigFloat = get_default_rounding_mode(deviceID);

        if (fp16Support)
            fpConfigHalf =
                get_default_rounding_mode(deviceID, CL_DEVICE_HALF_FP_CONFIG);

        for_each_elem(it);
    }

    template <typename T> int test_fpmath(TestDef<T> &test)
    {
        constexpr size_t vecSizes[] = { 1, 2, 4, 8, 16 };
        cl_int err = CL_SUCCESS;

        std::ostringstream sstr;
        if (std::is_same<T, double>::value)
            sstr << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        if (std::is_same<T, cl_half>::value)
            sstr << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

        std::string program_source =
            str_sprintf(std::string(fp_kernel_code), sstr.str().c_str());

        for (unsigned i = 0; i < ARRAY_SIZE(vecSizes); i++)
        {
            test.vec_size = vecSizes[i];

            std::ostringstream vecNameStr;
            vecNameStr << test.type_str;
            if (test.vec_size != 1) vecNameStr << test.vec_size;

            clMemWrapper streams[3];
            clProgramWrapper program;
            clKernelWrapper kernel;

            size_t length = sizeof(T) * num_elements * test.vec_size;

            bool isRTZ = false;
            RoundingMode oldMode = kDefaultRoundingMode;


            // If we only support rtz mode
            if (std::is_same<T, cl_half>::value)
            {
                if (CL_FP_ROUND_TO_ZERO == fpConfigHalf)
                {
                    isRTZ = true;
                    oldMode = get_round();
                }
            }
            else if (std::is_same<T, float>::value)
            {
                if (CL_FP_ROUND_TO_ZERO == fpConfigFloat)
                {
                    isRTZ = true;
                    oldMode = get_round();
                }
            }

            std::vector<T> inputs[]{
                std::vector<T>(test.vec_size * num_elements),
                std::vector<T>(test.vec_size * num_elements)
            };
            std::vector<T> output =
                std::vector<T>(test.vec_size * num_elements);

            generate_random_inputs<T>(inputs);

            for (size_t i = 0; i < ARRAY_SIZE(streams); i++)
            {
                streams[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, length,
                                            NULL, &err);
                test_error(err, "clCreateBuffer failed.");
            }
            for (size_t i = 0; i < ARRAY_SIZE(inputs); i++)
            {
                err =
                    clEnqueueWriteBuffer(queue, streams[i], CL_TRUE, 0, length,
                                         inputs[i].data(), 0, NULL, NULL);
                test_error(err, "clEnqueueWriteBuffer failed.");
            }

            std::string build_options = "-DTYPE=";
            build_options.append(vecNameStr.str())
                .append(" -DOP=")
                .append(1, test.op);

            const char *ptr = program_source.c_str();
            err =
                create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                            "test_fp", build_options.c_str());

            test_error(err, "create_single_kernel_helper failed");

            for (size_t i = 0; i < ARRAY_SIZE(streams); i++)
            {
                err =
                    clSetKernelArg(kernel, i, sizeof(streams[i]), &streams[i]);
                test_error(err, "clSetKernelArgs failed.");
            }

            size_t threads[] = { static_cast<size_t>(num_elements) };
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL,
                                         0, NULL, NULL);
            test_error(err, "clEnqueueNDRangeKernel failed.");

            err = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, length,
                                      output.data(), 0, NULL, NULL);
            test_error(err, "clEnqueueReadBuffer failed.");

            if (isRTZ) set_round(kRoundTowardZero, kfloat);

            err = verify_fp(inputs, output, test);

            if (isRTZ) set_round(oldMode, kfloat);

            test_error(err, "test verification failed");
            log_info("FP '%c' '%s' test passed\n", test.op,
                     vecNameStr.str().c_str());
        }

        return err;
    }

    template <typename T> int test_fpmath_common()
    {
        int err = TEST_PASS;
        if (std::is_same<cl_half, T>::value)
        {
            TestDef<T> tests[] = { { '+', half_plus, type2name[sizeof(T)] },
                                   { '-', half_minus, type2name[sizeof(T)] },
                                   { '*', half_mult, type2name[sizeof(T)] } };
            for (auto &test : tests) err |= test_fpmath<T>(test);
        }
        else
        {
            TestDef<T> tests[] = {
                { '+', std::plus<T>(), type2name[sizeof(T)] },
                { '-', std::minus<T>(), type2name[sizeof(T)] },
                { '*', std::multiplies<T>(), type2name[sizeof(T)] }
            };
            for (auto &test : tests) err |= test_fpmath<T>(test);
        }

        return err;
    }

    template <typename T> bool skip_type()
    {
        if (std::is_same<double, T>::value && !fp64Support)
            return true;
        else if (std::is_same<cl_half, T>::value && !fp16Support)
            return true;
        return false;
    }

    template <std::size_t Cnt = 0, typename Type>
    void iterate_type(const Type &t)
    {
        bool doTest = !skip_type<Type>();

        if (doTest)
        {
            if (test_fpmath_common<Type>())
            {
                throw std::runtime_error("test_fpmath_common failed\n");
            }
        }
    }

    template <std::size_t Cnt = 0, typename... Tp>
    inline typename std::enable_if<Cnt == sizeof...(Tp), void>::type
    for_each_elem(
        const std::tuple<Tp...> &) // Unused arguments are given no names.
    {}

    template <std::size_t Cnt = 0, typename... Tp>
        inline typename std::enable_if < Cnt<sizeof...(Tp), void>::type
        for_each_elem(const std::tuple<Tp...> &t)
    {
        iterate_type<Cnt>(std::get<Cnt>(t));
        for_each_elem<Cnt + 1, Tp...>(t);
    }

protected:
    TypeIter it;

    cl_context context;
    cl_command_queue queue;

    cl_device_fp_config fpConfigHalf;
    cl_device_fp_config fpConfigFloat;

    bool fp16Support;
    bool fp64Support;

    int num_elements;
    std::map<size_t, std::string> type2name;
};

} // anonymous namespace

int test_fpmath(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elements)
{
    try
    {
        TypesIterator(device, context, queue, num_elements);
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

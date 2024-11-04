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
#include "CL/cl_half.h"
#include "harness/compat.h"
#include "harness/errorHelpers.h"
#include "harness/stringHelpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

#include "procs.h"

extern cl_half_rounding_mode halfRoundingMode;

#define HFF(num) cl_half_from_float(num, halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

namespace {
const char *int2float_kernel_code = R"(
%s
__kernel void test_X2Y(__global TYPE_X *src, __global TYPE_Y *dst)
{
    int  tid = get_global_id(0);

    dst[tid] = (TYPE_Y)src[tid];

})";

template <bool int2fp> struct TypesIterator
{
    TypesIterator(cl_device_id deviceID, cl_context context,
                  cl_command_queue queue, int num_elems, const char *test_name)
        : context(context), queue(queue), test_name(test_name),
          num_elements(num_elems)
    {
        fp16Support = is_extension_available(deviceID, "cl_khr_fp16");
        fp64Support = is_extension_available(deviceID, "cl_khr_fp64");

        type2name[sizeof(cl_half)] = std::make_pair("half", "short");
        type2name[sizeof(cl_float)] = std::make_pair("float", "int");
        type2name[sizeof(cl_double)] = std::make_pair("double", "long");

        std::tuple<cl_float, cl_half, cl_double> it;
        for_each_elem(it);
    }

    template <typename T> void generate_random_inputs(std::vector<T> &v)
    {
        RandomSeed seed(gRandomSeed);

        if (sizeof(T) == sizeof(cl_half))
        {
            // Bound generated half values to 0x1.ffcp+14(32752.0) which is the
            // largest cl_half value smaller than the max value of cl_short,
            // 32767.
            if (int2fp)
            {
                auto random_generator = [&seed]() {
                    return (cl_short)get_random_float(
                        -MAKE_HEX_FLOAT(0x1.ffcp+14, 1.9990234375f, 14),
                        MAKE_HEX_FLOAT(0x1.ffcp+14, 1.9990234375f, 14), seed);
                };
                std::generate(v.begin(), v.end(), random_generator);
            }
            else
            {
                auto random_generator = [&seed]() {
                    return HFF(get_random_float(
                        -MAKE_HEX_FLOAT(0x1.ffcp+14, 1.9990234375f, 14),
                        MAKE_HEX_FLOAT(0x1.ffcp+14, 1.9990234375f, 14), seed));
                };
                std::generate(v.begin(), v.end(), random_generator);
            }
        }
        else if (sizeof(T) == sizeof(cl_float))
        {
            auto random_generator = [&seed]() {
                return get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31),
                                        MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31),
                                        seed);
            };
            std::generate(v.begin(), v.end(), random_generator);
        }
        else if (sizeof(T) == sizeof(cl_double))
        {
            auto random_generator = [&seed]() {
                return get_random_double(-MAKE_HEX_DOUBLE(0x1.0p63, 0x1, 63),
                                         MAKE_HEX_DOUBLE(0x1.0p63, 0x1, 63),
                                         seed);
            };
            std::generate(v.begin(), v.end(), random_generator);
        }
    }

    template <typename Tx, typename Ty> static bool equal_value(Tx a, Ty b)
    {
        return a == (Tx)b;
    }

    static bool equal_value_from_half(cl_short a, cl_half b)
    {
        return a == (cl_short)HTF(b);
    }

    static bool equal_value_to_half(cl_half a, cl_short b)
    {
        return a == HFF((float)b);
    }


    template <typename Tx, typename Ty>
    int verify_X2Y(std::vector<Tx> input, std::vector<Ty> output)
    {
        if (std::is_same<Tx, cl_half>::value
            || std::is_same<Ty, cl_half>::value)
        {
            bool res = true;
            if (int2fp)
                res = std::equal(output.begin(), output.end(), input.begin(),
                                 equal_value_to_half);
            else
                res = std::equal(output.begin(), output.end(), input.begin(),
                                 equal_value_from_half);

            if (!res)
            {
                log_error("%s test failed\n", test_name.c_str());
                return -1;
            }
        }
        else
        {
            if (!std::equal(output.begin(), output.end(), input.begin(),
                            equal_value<Tx, Ty>))
            {
                log_error("%s test failed\n", test_name.c_str());
                return -1;
            }
        }

        log_info("%s test passed\n", test_name.c_str());
        return 0;
    }

    template <typename Tx, typename Ty> int test_X2Y()
    {
        clMemWrapper streams[2];
        clProgramWrapper program;
        clKernelWrapper kernel;
        int err;

        std::vector<Tx> input(num_elements);
        std::vector<Ty> output(num_elements);

        streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(Tx) * num_elements, nullptr, &err);
        test_error(err, "clCreateBuffer failed.");
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(Ty) * num_elements, nullptr, &err);
        test_error(err, "clCreateBuffer failed.");

        generate_random_inputs(input);

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                                   sizeof(Tx) * num_elements, input.data(), 0,
                                   nullptr, nullptr);
        test_error(err, "clEnqueueWriteBuffer failed.");

        std::string src_name = type2name[sizeof(Tx)].first;
        std::string dst_name = type2name[sizeof(Tx)].second;
        if (int2fp) std::swap(src_name, dst_name);

        std::string build_options;
        build_options.append("-DTYPE_X=").append(src_name.c_str());
        build_options.append(" -DTYPE_Y=").append(dst_name.c_str());

        std::string extension;
        if (sizeof(Tx) == sizeof(cl_double))
            extension = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        if (sizeof(Tx) == sizeof(cl_half))
            extension = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

        std::string kernelSource =
            str_sprintf(int2float_kernel_code, extension.c_str());
        const char *ptr = kernelSource.c_str();

        err = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                          "test_X2Y", build_options.c_str());
        test_error(err, "create_single_kernel_helper failed.");

        err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
        test_error(err, "clSetKernelArg failed.");

        size_t threads[] = { (size_t)num_elements };
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, threads,
                                     nullptr, 0, nullptr, nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed.");

        err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                  sizeof(Ty) * num_elements, output.data(), 0,
                                  nullptr, nullptr);
        test_error(err, "clEnqueueReadBuffer failed.");

        err = verify_X2Y(input, output);

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

    template <std::size_t Cnt = 0, typename T> void iterate_type(const T &t)
    {
        bool doTest = !skip_type<T>();

        if (doTest)
        {
            typedef typename std::conditional<
                (sizeof(T) == sizeof(std::int16_t)), std::int16_t,
                typename std::conditional<(sizeof(T) == sizeof(std::int32_t)),
                                          std::int32_t,
                                          std::int64_t>::type>::type U;
            if (int2fp)
            {
                if (test_X2Y<U, T>())
                    throw std::runtime_error("test_X2Y failed\n");
            }
            else
            {
                if (test_X2Y<T, U>())
                    throw std::runtime_error("test_X2Y failed\n");
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
    cl_context context;
    cl_command_queue queue;

    cl_device_fp_config fpConfigHalf;
    cl_device_fp_config fpConfigFloat;

    bool fp16Support;
    bool fp64Support;

    std::map<size_t, std::pair<std::string, std::string>> type2name;

    std::string test_name;
    int num_elements;
};

}

int test_int2fp(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elements)
{
    try
    {
        TypesIterator<true>(device, context, queue, num_elements, "INT2FP");
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_fp2int(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elements)
{
    try
    {
        TypesIterator<false>(device, context, queue, num_elements, "FP2INT");
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

//
// Copyright (c) 2017 The Khronos Group Inc.
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
#ifndef TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_BITWISE_HPP
#define TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_BITWISE_HPP

#include "common.hpp"
#include <type_traits>

template<class IN1, class OUT1>
struct int_func_popcount : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "popcount";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(IN1 x)
    {
        OUT1 count = 0;
        for (count = 0; x != 0; count++)
        {
            x &= x - 1;
        }
        return count;
    }
};

template<class IN1, class OUT1>
struct int_func_clz : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "clz";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(IN1 x)
    {
        OUT1 count = 0;
        if(std::is_unsigned<IN1>::value)
        {
            cl_ulong value = x;
            value <<= 8 * sizeof(value) - (8 * sizeof(x));
            for(count = 0; 0 == (value & (CL_LONG_MIN)); count++)
            {
                value <<= 1;
            }
        }
        else
        {            
            cl_long value = x;
            value <<= 8 * sizeof(value) - (8 * sizeof(x));
            for(count = 0; 0 == (value & (CL_LONG_MIN)); count++)
            {
                value <<= 1;
            }
        }
        return count;
    }
};

template<class IN1, class OUT1>
struct int_func_ctz : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "ctz";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(IN1 x)
    {
        if(x == 0)
            return sizeof(x);

        OUT1 count = 0;
        IN1 value = x;
        for(count = 0; 0 == (value & 0x1); count++)
        {
            value >>= 1;
        }
        return count;
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_rotate : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "rotate";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(IN1 value, IN2 shift)
    {
        static_assert(
            std::is_unsigned<IN1>::value,
            "Only unsigned integers are supported"
        );
        if ((shift &= sizeof(value)*8 - 1) == 0)
            return value;
        return (value << shift) | (value >> (sizeof(value)*8 - shift));
    }

    IN2 min2()
    {
        return 0;
    }

    IN2 max2()
    {
        return sizeof(IN1) * 8;
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_upsample : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "upsample";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(IN1 hi, IN2 lo)
    {
        static_assert(
            sizeof(IN1) == sizeof(IN2),
            "sizeof(IN1) != sizeof(IN2)"
        );
        static_assert(
            sizeof(OUT1) == 2 * sizeof(IN1),
            "sizeof(OUT1) != 2 * sizeof(IN1)"
        );
        static_assert(
            std::is_unsigned<IN2>::value,
            "IN2 type must be unsigned"
        );
        return (static_cast<OUT1>(hi) << (8*sizeof(IN1))) | lo;
    }

    IN2 min2()
    {
        return 0;
    }

    IN2 max2()
    {
        return sizeof(IN1) * 8;
    }
};

AUTO_TEST_CASE(test_int_bitwise_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;
    
    TEST_UNARY_FUNC_MACRO((int_func_popcount<cl_int, cl_int>()))
    TEST_UNARY_FUNC_MACRO((int_func_popcount<cl_uint, cl_uint>()))
    TEST_UNARY_FUNC_MACRO((int_func_popcount<cl_long, cl_long>()))
    TEST_UNARY_FUNC_MACRO((int_func_popcount<cl_ulong, cl_ulong>()))

    TEST_UNARY_FUNC_MACRO((int_func_clz<cl_int, cl_int>()))
    TEST_UNARY_FUNC_MACRO((int_func_clz<cl_uint, cl_uint>()))
    TEST_UNARY_FUNC_MACRO((int_func_clz<cl_long, cl_long>()))
    TEST_UNARY_FUNC_MACRO((int_func_clz<cl_ulong, cl_ulong>()))

    TEST_UNARY_FUNC_MACRO((int_func_ctz<cl_int, cl_int>()))
    TEST_UNARY_FUNC_MACRO((int_func_ctz<cl_uint, cl_uint>()))
    TEST_UNARY_FUNC_MACRO((int_func_ctz<cl_long, cl_long>()))
    TEST_UNARY_FUNC_MACRO((int_func_ctz<cl_ulong, cl_ulong>()))

    TEST_BINARY_FUNC_MACRO((int_func_rotate<cl_uint, cl_uint, cl_uint>()))
    TEST_BINARY_FUNC_MACRO((int_func_rotate<cl_ulong, cl_ulong, cl_ulong>()))

    // shortn upsample(charn hi, ucharn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_char, cl_uchar, cl_short>()))
    // ushortn upsample(ucharn hi, ucharn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_uchar, cl_uchar, cl_ushort>()))
    // intn upsample(shortn hi, ushortn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_short, cl_ushort, cl_int>()))
    // uintn upsample(ushortn hi, ushortn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_ushort, cl_ushort, cl_uint>()))
    // longn upsample(intn hi, uintn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_int, cl_uint, cl_long>()))
    // ulongn upsample(uintn hi, uintn lo);
    TEST_BINARY_FUNC_MACRO((int_func_upsample<cl_uint, cl_uint, cl_ulong>()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_BITWISE_HPP

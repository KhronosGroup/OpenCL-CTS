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
#ifndef TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_24BIT_HPP
#define TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_24BIT_HPP

#include "common.hpp"
#include <type_traits>

template<class IN1, class IN2, class IN3, class OUT1>
struct int_func_mad24 : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "mad24";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y, const IN3& z)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "All types must be the same"
        );
        static_assert(
            std::is_same<cl_uint, IN1>::value || std::is_same<cl_int, IN1>::value,
            "Function takes only signed/unsigned integers."
        );
        return (x * y) + z;
    }

    IN1 min1()
    {
        return 0;
    }

    IN1 max1()
    {
        return (std::numeric_limits<IN1>::max)() & IN1(0x00FFFF);
    }

    IN2 min2()
    {
        return 0;
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() & IN2(0x00FFFF);
    }
};

template<class IN1, class IN2, class OUT1>
struct int_func_mul24 : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "mul24";
    }

    std::string headers()
    {
        return "#include <opencl_integer>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, OUT1>::value,
            "All types must be the same"
        );
        static_assert(
            std::is_same<cl_uint, IN1>::value || std::is_same<cl_int, IN1>::value,
            "Function takes only signed/unsigned integers."
        );
        return x * y;
    }

    IN1 min1()
    {
        return 0;
    }

    IN1 max1()
    {
        return (std::numeric_limits<IN1>::max)() & IN1(0x00FFFF);
    }

    IN2 min2()
    {
        return 0;
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() & IN2(0x00FFFF);
    }
};

AUTO_TEST_CASE(test_int_24bit_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;
    
    // intn mad24(intn x, intn y, intn z);
    // uintn mad24(uintn x, uintn y, uintn z);
    TEST_TERNARY_FUNC_MACRO((int_func_mad24<cl_int, cl_int, cl_int, cl_int>()))
    TEST_TERNARY_FUNC_MACRO((int_func_mad24<cl_uint, cl_uint, cl_uint, cl_uint>()))

    // intn mul24(intn x, intn y);
    // uintn mul24(uintn x, uintn y);
    TEST_BINARY_FUNC_MACRO((int_func_mul24<cl_int, cl_int, cl_int>()))
    TEST_BINARY_FUNC_MACRO((int_func_mul24<cl_uint, cl_uint, cl_uint>()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_INTEGER_FUNCS_24BIT_HPP

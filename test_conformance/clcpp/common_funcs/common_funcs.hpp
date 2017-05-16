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
#ifndef TEST_CONFORMANCE_CLCPP_COMMON_FUNCS_COMMON_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_COMMON_FUNCS_COMMON_FUNCS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <type_traits>
#include <algorithm>

// floatn clamp(floatn x, floatn min, floatn max) (only scalars)
template<class IN1, class IN2, class IN3, class OUT1>
struct common_func_clamp : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "clamp";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& minval, const IN3& maxval)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "All types must be the same"
        );
        return (std::min)((std::max)(x, minval), maxval);
    }

    IN2 min2()
    {
        return (std::numeric_limits<IN2>::min)();
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() / IN2(4000.0f);
    }

    IN3 min3()
    {
        return IN3(1) + ((std::numeric_limits<IN3>::max)() / IN3(4000.0f));
    }

    IN3 max3()
    {
        return (std::numeric_limits<IN3>::max)() / IN3(2000.0f);
    }

    float ulp()
    {
        return 0.0f;
    }
};

// floatn degrees(floatn t)
template<class IN1, class OUT1, class REFERENCE>
struct common_func_degrees : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "degrees";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    REFERENCE operator()(const IN1& x)
    {
        static_assert(
            std::is_same<IN1, OUT1>::value,
            "All types must be the same"
        );
        return (REFERENCE(180.0) / CL_M_PI) * static_cast<REFERENCE>(x);
    }

    float ulp()
    {
        return 2.5f;
    }
};

// floatn max(floatn x, floatn y)
template<class IN1, class IN2, class OUT1>
struct common_func_max : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "max";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "All types must be the same"
        );
        return (std::max)(x, y);
    }

    float ulp()
    {
        return 0.0f;
    }
};

// floatn min(floatn x, floatn y)
template<class IN1, class IN2, class OUT1>
struct common_func_min : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "min";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "All types must be the same"
        );
        return (std::min)(x, y);
    }

    float ulp()
    {
        return 0.0f;
    }
};

// floatn mix(floatn x, floatn y, floatn a);
template<class IN1, class IN2, class IN3, class OUT1>
struct common_func_mix : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "mix";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& x, const IN2& y, const IN3& a)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "All types must be the same"
        );
        return static_cast<double>(x) + ((static_cast<double>(y) - static_cast<double>(x)) * static_cast<double>(a));
    }

    IN3 min3()
    {
        return IN3(0.0f + CL_FLT_EPSILON);
    }

    IN3 max3()
    {
        return IN3(1.0f - CL_FLT_EPSILON);
    }

    bool use_ulp()
    {
        return false;
    }
};

// floatn radians(floatn t)
template<class IN1, class OUT1, class REFERENCE>
struct common_func_radians : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "radians";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    REFERENCE operator()(const IN1& x)
    {
        static_assert(
            std::is_same<IN1, OUT1>::value,
            "All types must be the same"
        );
        return (CL_M_PI / REFERENCE(180.0)) * static_cast<REFERENCE>(x);
    }

    float ulp()
    {
        return 2.5f;
    }
};

// floatn step(floatn edge, floatn x)
template<class IN1, class IN2, class OUT1>
struct common_func_step : public binary_func<IN1, IN2, OUT1>
{
    std::string str()
    {
        return "step";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& edge, const IN2& x)
    {
        static_assert(
            std::is_same<IN1, IN2>::value && std::is_same<IN2, OUT1>::value,
            "All types must be the same"
        );
        if(x < edge)
            return OUT1(0.0f);
        return OUT1(1.0f);
    }

    float ulp()
    {
        return 0.0f;
    }
};

// floatn smoothstep(floatn edge0, floatn edge1, floatn x);
template<class IN1, class IN2, class IN3, class OUT1>
struct common_func_smoothstep : public ternary_func<IN1, IN2, IN3, OUT1>
{
    std::string str()
    {
        return "smoothstep";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& edge0, const IN2& edge1, const IN3& x)
    {
        static_assert(
            std::is_same<IN1, IN2>::value
                && std::is_same<IN2, IN3>::value
                && std::is_same<IN3, OUT1>::value,
            "All types must be the same"
        );
        if(x <= edge0)
        {
            return OUT1(0.0f);
        }
        if(x >= edge1)
        {
            return OUT1(1.0f);
        }
        OUT1 t = (x - edge0) / (edge1 - edge0);
        t = t * t * (3.0f - 2.0f * t);
        return t;
    }

    // edge0 must be < edge1
    IN1 min1()
    {
        return (std::numeric_limits<IN1>::min)();
    }

    IN1 max1()
    {
        return (std::numeric_limits<IN1>::max)() / IN1(8000.0f);
    }

    IN2 min2()
    {
        return IN3(1) + ((std::numeric_limits<IN2>::max)() / IN2(4000.0f));
    }

    IN2 max2()
    {
        return (std::numeric_limits<IN2>::max)() / IN2(2000.0f);
    }

    bool use_ulp()
    {
        return false;
    }
};

// floatn sign(floatn t)
template<class IN1, class OUT1>
struct common_func_sign : public unary_func<IN1, OUT1>
{
    std::string str()
    {
        return "sign";
    }

    std::string headers()
    {
        return "#include <opencl_common>\n";
    }

    OUT1 operator()(const IN1& x)
    {
        static_assert(
            std::is_same<IN1, OUT1>::value,
            "All types must be the same"
        );
        if(x == IN1(-0.0f))
        {
            return IN1(-0.0f);
        }
        if(x == IN1(+0.0f))
        {
            return IN1(+0.0f);
        }
        if(x > IN1(0.0f))
        {
            return IN1(1.0f);
        }
        return IN1(-1.0f);
    }

    bool use_ulp()
    {
        return false;
    }

    float ulp()
    {
        return 0.0f;
    }

    std::vector<IN1> in_special_cases()
    {
        return { -0.0f, +0.0f };
    }
};

AUTO_TEST_CASE(test_common_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // floatn clamp(floatn x, floatn min, floatn max)
    TEST_TERNARY_FUNC_MACRO((common_func_clamp<cl_float, cl_float, cl_float, cl_float>()))  

    // floatn degrees(floatn t)
    TEST_UNARY_FUNC_MACRO((common_func_degrees<cl_float, cl_float, cl_double>()))  
        
    // floatn max(floatn x, floatn y);
    TEST_BINARY_FUNC_MACRO((common_func_max<cl_float, cl_float, cl_float>()))

    // floatn min(floatn x, floatn y);
    TEST_BINARY_FUNC_MACRO((common_func_min<cl_float, cl_float, cl_float>()))
   
    // floatn mix(floatn x, floatn y, floatn a);
    TEST_TERNARY_FUNC_MACRO((common_func_mix<cl_float, cl_float, cl_float, cl_float>()))

    // floatn radians(floatn t)
    TEST_UNARY_FUNC_MACRO((common_func_radians<cl_float, cl_float, cl_double>()))

    // floatn step(floatn edge, floatn x)
    TEST_BINARY_FUNC_MACRO((common_func_step<cl_float, cl_float, cl_float>()))

    // floatn smoothstep(floatn edge0, floatn edge1, floatn x)
    TEST_TERNARY_FUNC_MACRO((common_func_smoothstep<cl_float, cl_float, cl_float, cl_float>()))

    // floatn sign(floatn t);
    TEST_UNARY_FUNC_MACRO((common_func_sign<cl_float, cl_float>()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_COMMON_FUNCS_COMMON_FUNCS_HPP

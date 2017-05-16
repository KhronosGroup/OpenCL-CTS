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
#ifndef TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_FAST_GEOMETRIC_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_FAST_GEOMETRIC_FUNCS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <type_traits>

// float fast_distance(float4 p0, float4 p1);
struct geometric_func_fast_distance : public binary_func<cl_float4, cl_float4, cl_float>
{

    std::string str()
    {
        return "fast_distance";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float operator()(const cl_float4& p0, const cl_float4& p1)
    {
        cl_double r = 0.0f;
        cl_double t;
        for(size_t i = 0; i < 4; i++)
        {
            t = static_cast<cl_double>(p0.s[i]) - static_cast<cl_double>(p1.s[i]);
            r += t * t;
        }
        return std::sqrt(r);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-512.0f);
    }

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(512.0f);
    }

    cl_float4 min2()
    {
        return detail::def_limit<cl_float4>(-512.0f);
    }

    cl_float4 max2()
    {
        return detail::def_limit<cl_float4>(512.0f);
    }

    cl_double delta(const cl_float4& p0, const cl_float4& p1, const cl_float& expected)
    {
        (void) p0; (void) p1;
        return 0.01f * expected;
    }

    float ulp()
    {
        return
            8192.0f + // error in sqrt
            (1.5f * 4.0f) + // cumulative error for multiplications
            (0.5f * 3.0f);  // cumulative error for additions
    }
};

// float fast_length(float4 p);
struct geometric_func_fast_length : public unary_func<cl_float4,cl_float>
{
    std::string str()
    {
        return "fast_length";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float operator()(const cl_float4& p)
    {
        cl_double r = 0.0f;
        for(size_t i = 0; i < 4; i++)
        {
            r += static_cast<cl_double>(p.s[i]) * static_cast<cl_double>(p.s[i]);
        }
        return std::sqrt(r);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-512.0f);
    }

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(512.0f);
    }

    cl_double delta(const cl_float4& p, const cl_float& expected)
    {
        (void) p;
        return 0.01f * expected;
    }

    float ulp()
    {
        return
            8192.0f + // error in sqrt
            0.5f * // effect on e of taking sqrt( x + e )
                ((0.5f * 4.0f) + // cumulative error for multiplications
                (0.5f * 3.0f));  // cumulative error for additions
    }
};

// float4 fast_normalize(float4 p);
struct geometric_func_fast_normalize : public unary_func<cl_float4,cl_float4>
{
    std::string str()
    {
        return "fast_normalize";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float4 operator()(const cl_float4& p)
    {
        cl_double t = 0.0f;
        cl_float4 r;
        for(size_t i = 0; i < 4; i++)
        {
            t += static_cast<cl_double>(p.s[i]) * static_cast<cl_double>(p.s[i]);
        }

        if(t == 0.0f)
        {
            for(size_t i = 0; i < 4; i++)
            {
                r.s[i] = 0.0f;
            }
            return r;
        }

        t = std::sqrt(t);
        for(size_t i = 0; i < 4; i++)
        {
            r.s[i] = static_cast<cl_double>(p.s[i]) / t;
        }
        return r;
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-512.0f);
    }

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(512.0f);
    }

    std::vector<cl_float4> in_special_cases()
    {
        return {
            {0.0f, 0.0f, 0.0f, 0.0f}
        };
    }


    cl_double4 delta(const cl_float4& p, const cl_float4& expected)
    {
        (void) p;
        auto e = detail::make_value<cl_double4>(0.01f);
        return detail::multiply<cl_double4>(e, expected);
    }

    float ulp()
    {
        return
            8192.5f + // error in rsqrt + error in multiply
            (0.5f * 4.0f) + // cumulative error for multiplications
            (0.5f * 3.0f);  // cumulative error for additions
    }
};

AUTO_TEST_CASE(test_fast_geometric_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // float fast_distance(float4 p0, float4 p1)
    TEST_BINARY_FUNC_MACRO((geometric_func_fast_distance()))

    // float fast_length(float4 p)
    TEST_UNARY_FUNC_MACRO((geometric_func_fast_length()))

    // float4 fast_normalize(float4 p)
    TEST_UNARY_FUNC_MACRO((geometric_func_fast_normalize()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_FAST_GEOMETRIC_FUNCS_HPP

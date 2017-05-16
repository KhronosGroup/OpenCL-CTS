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
#ifndef TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_GEOMETRIC_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_GEOMETRIC_FUNCS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <type_traits>

// float4 cross(float4 p0, float4 p1)
struct geometric_func_cross : public binary_func<cl_float4, cl_float4, cl_float4>
{
    geometric_func_cross(cl_device_id device)
    {
        // On an embedded device w/ round-to-zero, 3 ulps is the worst-case tolerance for cross product
        this->m_delta = 3.0f * CL_FLT_EPSILON;
        // RTZ devices accrue approximately double the amount of error per operation.  Allow for that.
        if(get_default_rounding_mode(device) == CL_FP_ROUND_TO_ZERO)
        {
            this->m_delta *= 2.0f;
        }
    }

    std::string str()
    {
        return "cross";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float4 operator()(const cl_float4& p0, const cl_float4& p1)
    {
        cl_float4 r;
        r.s[0] = (p0.s[1] * p1.s[2]) - (p0.s[2] * p1.s[1]);
        r.s[1] = (p0.s[2] * p1.s[0]) - (p0.s[0] * p1.s[2]);
        r.s[2] = (p0.s[0] * p1.s[1]) - (p0.s[1] * p1.s[0]);
        r.s[3] = 0.0f;
        return r;
    }

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 max2()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    cl_float4 min2()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    bool use_ulp()
    {
        return false;
    }

    cl_double4 delta(const cl_float4& p0, const cl_float4& p1, const cl_float4& expected)
    {
        (void) p0; (void) p1;
        auto e = detail::make_value<cl_double4>(m_delta);
        return detail::multiply<cl_double4>(e, expected);
    }

private:
    cl_double m_delta;
};

// float dot(float4 p0, float4 p1);
struct geometric_func_dot : public binary_func<cl_float4, cl_float4, cl_float>
{

    std::string str()
    {
        return "dot";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float operator()(const cl_float4& p0, const cl_float4& p1)
    {
        cl_float r;
        r = p0.s[0] * p1.s[0];
        r += p0.s[1] * p1.s[1];
        r += p0.s[2] * p1.s[2];
        r += p0.s[3] * p1.s[3];
        return r;
    }

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 max2()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    cl_float4 min2()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    bool use_ulp()
    {
        return false;
    }

    cl_double delta(const cl_float4& p0, const cl_float4& p1, cl_float expected)
    {
        (void) p0; (void) p1;
        return expected * ((4.0f + (4.0f - 1.0f)) * CL_FLT_EPSILON);
    }
};

// float distance(float4 p0, float4 p1);
struct geometric_func_distance : public binary_func<cl_float4, cl_float4, cl_float>
{

    std::string str()
    {
        return "distance";
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

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 max2()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    cl_float4 min2()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    float ulp()
    {
        return
            3.0f + // error in sqrt
            (1.5f * 4.0f) + // cumulative error for multiplications
            (0.5f * 3.0f);  // cumulative error for additions
    }
};

// float length(float4 p);
struct geometric_func_length : public unary_func<cl_float4,cl_float>
{

    std::string str()
    {
        return "length";
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

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    float ulp()
    {
        return
            3.0f + // error in sqrt
            0.5f * // effect on e of taking sqrt( x + e )
                ((0.5f * 4.0f) + // cumulative error for multiplications
                (0.5f * 3.0f));  // cumulative error for additions
    }
};

// float4 normalize(float4 p);
struct geometric_func_normalize : public unary_func<cl_float4,cl_float4>
{
    std::string str()
    {
        return "normalize";
    }

    std::string headers()
    {
        return "#include <opencl_geometric>\n";
    }

    cl_float4 operator()(const cl_float4& p)
    {
        cl_double t = 0.0f;
        cl_float4 r;

        // normalize( v ) returns a vector full of NaNs if any element is a NaN.
        for(size_t i = 0; i < 4; i++)
        {
            if((std::isnan)(p.s[i]))
            {
                for(size_t j = 0; j < 4; j++)
                {
                    r.s[j] = p.s[i];
                }
                return r;
            }
        }

        // normalize( v ) for which any element in v is infinite shall proceed as
        // if the elements in v were replaced as follows:
        // for( i = 0; i < sizeof(v) / sizeof(v[0] ); i++ )
        //     v[i] = isinf(v[i]) ? copysign(1.0, v[i]) : 0.0 * v [i];
        for(size_t i = 0; i < 4; i++)
        {
            if((std::isinf)(p.s[i]))
            {
                for(size_t j = 0; j < 4; j++)
                {
                    r.s[j] = (std::isinf)(p.s[j]) ? (std::copysign)(1.0, p.s[j]) : 0.0 * p.s[j];
                }
                r = (*this)(r);
                return r;
            }
        }

        for(size_t i = 0; i < 4; i++)
        {
            t += static_cast<cl_double>(p.s[i]) * static_cast<cl_double>(p.s[i]);
        }

        // normalize( v ) returns v if all elements of v are zero.
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

    cl_float4 max1()
    {
        return detail::def_limit<cl_float4>(1000.0f);
    }

    cl_float4 min1()
    {
        return detail::def_limit<cl_float4>(-1000.0f);
    }

    std::vector<cl_float4> in_special_cases()
    {
        return {
            {0.0f, 0.0f, 0.0f, 0.0f},
            {std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f},
            {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()
            },
            {
                std::numeric_limits<float>::infinity(),
                1.0f,
                0.0f,
                std::numeric_limits<float>::quiet_NaN()
            },
            {-1.0f, -1.0f, 0.0f,-300.0f}
        };
    }

    float ulp()
    {
        return
            2.5f + // error in rsqrt + error in multiply
            (0.5f * 4.0f) + // cumulative error for multiplications
            (0.5f * 3.0f);  // cumulative error for additions
    }
};

AUTO_TEST_CASE(test_geometric_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // float4 cross(float4 p0, float4 p1)
    TEST_BINARY_FUNC_MACRO((geometric_func_cross(device)))

    // float dot(float4 p0, float4 p1)
    TEST_BINARY_FUNC_MACRO((geometric_func_dot()))

    // float distance(float4 p0, float4 p1)
    TEST_BINARY_FUNC_MACRO((geometric_func_distance()))

    // float length(float4 p)
    TEST_UNARY_FUNC_MACRO((geometric_func_length()))

    // float4 normalize(float4 p)
    TEST_UNARY_FUNC_MACRO((geometric_func_normalize()))

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_GEOMETRIC_FUNCS_GEOMETRIC_FUNCS_HPP

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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_COMPARE_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_COMPARE_HPP

#include <random>
#include <limits>
#include <type_traits>
#include <algorithm>

#include <cmath>

#include "../common.hpp"

// Checks if x is equal to y.
template<class type, class delta_type, class op_type>
inline bool are_equal(const type& x,
                      const type& y,
                      const delta_type& delta,
                      op_type op,
                      typename std::enable_if<
                        is_vector_type<type>::value
                        && std::is_integral<typename scalar_type<type>::type>::value
                      >::type* = 0)
{
    (void) delta;
    for(size_t i = 0; i < vector_size<type>::value; i++)
    {
        if(op.is_out_bool())
        {
            if(!((x.s[i] != 0) == (y.s[i] != 0)))
            {
                return false;
            }
        }
        else if(!(x.s[i] == y.s[i]))
        {
            return false;
        }
    }
    return true;
}

template<class type, class delta_type, class op_type>
inline bool are_equal(const type& x,
                      const type& y,
                      const delta_type& delta,
                      op_type op,
                      typename std::enable_if<
                        !is_vector_type<type>::value
                        && std::is_integral<type>::value
                      >::type* = 0)
{
    (void) delta;
    if(op.is_out_bool())
    {
        if(!((x != 0) == (y != 0)))
        {
            return false;
        }
    }
    return x == y;
}

template<class type, class type1, class type2, class op_type>
inline bool are_equal(const type& x,
                      const type1& y,
                      const type2& delta,
                      op_type op,
                      typename std::enable_if<
                        !is_vector_type<type>::value
                        && std::is_floating_point<type>::value
                      >::type* = 0)
{
    // x - expected
    // y - result

    // INFO:
    // Whe don't care about subnormal values in OpenCL C++ tests
    if(std::fpclassify(static_cast<type1>(x)) == FP_SUBNORMAL || std::fpclassify(y) == FP_SUBNORMAL)
    {
        return true;
    }

    // both are NaN
    if((std::isnan)(static_cast<type1>(x)) && (std::isnan)(y))
    {
        return true;
    }
    // one is NaN
    else if((std::isnan)(static_cast<type1>(x)) || (std::isnan)(y))
    {
        return false;
    }

    // Check for perfect match, it also covers inf, -inf
    if(static_cast<type1>(x) != y)
    {
        // Check if values are close
        if(std::abs(static_cast<type1>(x) - y) > (std::max)(std::numeric_limits<type2>::epsilon(), std::abs(delta)))
        {
            return false;
        }
        // Check ulp
        if(op.use_ulp())
        {
            return !(std::abs(Ulp_Error(x, y)) > op.ulp());
        }
    }
    return true;
}

template<class type, class type1, class type2, class op_type>
inline bool are_equal(const type& x,
                      const type1& y,
                      const type2& delta,
                      op_type op,
                      typename std::enable_if<
                        is_vector_type<type>::value
                        && std::is_floating_point<typename scalar_type<type>::type>::value
                      >::type* = 0)
{
    // x - expected
    // y - result
    for(size_t i = 0; i < vector_size<type>::value; i++)
    {
        if(!are_equal(x.s[i], y.s[i], delta.s[i], op))
        {
            return false;
        }
    }
    return true;
}

template<class type, class type1, class func>
inline void print_error_msg(const type& expected, const type1& result, size_t i, func op)
{
    log_error(
        "ERROR: test_%s %s failed. Error at %lu: Expected: %s, got: %s\n",
        op.str().c_str(),
        op.decl_str().c_str(),
        i,
        format_value(expected).c_str(),
        format_value(result).c_str()
    );
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_COMPARE_HPP

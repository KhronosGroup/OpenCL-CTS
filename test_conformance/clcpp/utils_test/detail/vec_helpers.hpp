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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_VEC_HELPERS_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_VEC_HELPERS_HPP

#include <random>
#include <limits>
#include <type_traits>
#include <algorithm>

#include <cmath>

#include "../../common.hpp"

namespace detail
{

template<class T>
T make_value(typename scalar_type<T>::type x, typename std::enable_if<is_vector_type<T>::value>::type* = 0)
{
    T value;
    for(size_t i = 0; i < vector_size<T>::value; i++)
    {
        value.s[i] = x;
    }
    return value;
}

template<class T>
T make_value(T x, typename std::enable_if<!is_vector_type<T>::value>::type* = 0)
{
    return x;
}

template<class result_type, class IN1, class IN2>
result_type multiply(const IN1& x, const IN2& y, typename std::enable_if<is_vector_type<result_type>::value>::type* = 0)
{
    static_assert(
        (vector_size<IN1>::value == vector_size<IN2>::value)
            && (vector_size<IN2>::value == vector_size<result_type>::value),
        "Vector sizes must be the same."
    );
    typedef typename scalar_type<result_type>::type SCALAR;
    result_type value;
    for(size_t i = 0; i < vector_size<result_type>::value; i++)
    {
        value.s[i] = static_cast<SCALAR>(x.s[i]) * static_cast<SCALAR>(y.s[i]);
    }
    return value;
}

template<class result_type, class IN1, class IN2>
result_type multiply(const IN1& x, const IN2& y, typename std::enable_if<!is_vector_type<result_type>::value>::type* = 0)
{
    static_assert(
        !is_vector_type<IN1>::value && !is_vector_type<IN2>::value,
        "IN1 and IN2 must be scalar types"
    );
    return static_cast<result_type>(x) * static_cast<result_type>(y);
}

template<class T>
T get_min()
{
    typedef typename scalar_type<T>::type SCALAR;
    return make_value<T>((std::numeric_limits<SCALAR>::min)());
}

template<class T>
T get_max()
{
    typedef typename scalar_type<T>::type SCALAR;
    return make_value<T>((std::numeric_limits<SCALAR>::max)());
}

template<class T>
T get_part_max(typename scalar_type<T>::type x)
{
    typedef typename scalar_type<T>::type SCALAR;
    return make_value<T>((std::numeric_limits<SCALAR>::max)() / x);
}

template<class T>
T def_limit(typename scalar_type<T>::type x)
{
    return make_value<T>(x);
}

} // detail namespace

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_DETAIL_VEC_HELPERS_HPP

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
#ifndef TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMMON_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <type_traits>
#include <cmath>

template<class IN1, class IN2, class IN3, class OUT1, class F>
OUT1 perform_function(const IN1& in1, const IN2& in2, const IN3& in3, F func, typename std::enable_if<is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result;
    for(size_t i = 0; i < vector_size<OUT1>::value; i++)
    {
        result.s[i] = func(in1.s[i], in2.s[i], in3.s[i]);
    }
    return result;
}

template<class IN1, class IN2, class IN3, class OUT1, class F>
OUT1 perform_function(const IN1& in1, const IN2& in2, const IN3& in3, F func, typename std::enable_if<!is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result = func(in1, in2, in3);
    return result;
}


template<class IN1, class IN2, class OUT1, class F>
OUT1 perform_function(const IN1& in1, const IN2& in2, F func, typename std::enable_if<is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result;
    for(size_t i = 0; i < vector_size<OUT1>::value; i++)
    {
        result.s[i] = func(in1.s[i], in2.s[i]);
    }
    return result;
}

template<class IN1, class IN2, class OUT1, class F>
OUT1 perform_function(const IN1& in1, const IN2& in2, F func, typename std::enable_if<!is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result = func(in1, in2);
    return result;
}

template<class IN1, class OUT1, class F>
OUT1 perform_function(const IN1& in1, F func, typename std::enable_if<is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result;
    for(size_t i = 0; i < vector_size<OUT1>::value; i++)
    {
        result.s[i] = func(in1.s[i]);
    }
    return result;
}

template<class IN1, class OUT1, class F>
OUT1 perform_function(const IN1& in1, F func, typename std::enable_if<!is_vector_type<OUT1>::value>::type* = 0)
{
    OUT1 result = func(in1);
    return result;
}

template<class IN1>
cl_int perform_all_function(const IN1& in1, typename std::enable_if<is_vector_type<IN1>::value>::type* = 0)
{
    cl_int result = 1;
    for(size_t i = 0; i < vector_size<IN1>::value; i++)
    {
        result = (in1.s[i] != 0) ? result : cl_int(0);
    }
    return result;
}

cl_int perform_all_function(const cl_int& in1, typename std::enable_if<!is_vector_type<cl_int>::value>::type* = 0)
{
    return (in1 != 0) ? cl_int(1) : cl_int(0);
}

template<class IN1>
cl_int perform_any_function(const IN1& in1, typename std::enable_if<is_vector_type<IN1>::value>::type* = 0)
{
    cl_int result = 0;
    for(size_t i = 0; i < vector_size<IN1>::value; i++)
    {
        result = (in1.s[i] != 0) ? cl_int(1) : result;
    }
    return result;
}

cl_int perform_any_function(const cl_int& in1, typename std::enable_if<!is_vector_type<cl_int>::value>::type* = 0)
{
    return (in1 != 0) ? cl_int(1) : cl_int(0);
}

#endif // TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMMON_HPP

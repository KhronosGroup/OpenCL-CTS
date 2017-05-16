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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_GENERATE_INPUTS_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_GENERATE_INPUTS_HPP

#include <random>
#include <limits>
#include <type_traits>
#include <algorithm>

#include <cmath>

#include "../common.hpp"

template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    is_vector_type<type>::value
                                    && std::is_integral<typename scalar_type<type>::type>::value
                                    // std::uniform_int_distribution<> does not work in VS2015 for cl_uchar and cl_char,
                                    // because VS2015 thinks that use cl_int, because VS2015 thinks cl_uchar cl_char are
                                    // not int types
                                    && !(std::is_same<typename scalar_type<type>::type, cl_uchar>::value
                                         || std::is_same<typename scalar_type<type>::type, cl_char>::value)
                                 >::type* = 0)
{
    typedef typename scalar_type<type>::type SCALAR;
    const size_t vec_size = vector_size<type>::value;

    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::uniform_int_distribution<SCALAR>> dists(vec_size);
    for(size_t i = 0; i < vec_size; i++)
    {
        dists[i] = std::uniform_int_distribution<SCALAR>(min.s[i], max.s[i]);
    }
    for(auto& i : input)
    {
        for(size_t j = 0; j < vec_size; j++)
        {
            i.s[j] = dists[j](gen);
        }
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    is_vector_type<type>::value
                                    && std::is_integral<typename scalar_type<type>::type>::value
                                    // std::uniform_int_distribution<> does not work in VS2015 for cl_uchar and cl_char,
                                    // because VS2015 thinks that use cl_int, because VS2015 thinks cl_uchar cl_char are
                                    // not int types
                                    && (std::is_same<typename scalar_type<type>::type, cl_uchar>::value
                                        || std::is_same<typename scalar_type<type>::type, cl_char>::value)
                                 >::type* = 0)
{
    typedef typename scalar_type<type>::type SCALAR;
    const size_t vec_size = vector_size<type>::value;

    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::uniform_int_distribution<cl_int>> dists(vec_size);
    for(size_t i = 0; i < vec_size; i++)
    {
        dists[i] = std::uniform_int_distribution<cl_int>(
            static_cast<cl_int>(min.s[i]),
            static_cast<cl_int>(max.s[i])
        );
    }
    for(auto& i : input)
    {
        for(size_t j = 0; j < vec_size; j++)
        {
            i.s[j] = static_cast<SCALAR>(dists[j](gen));
        }
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}


template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    !is_vector_type<type>::value
                                    && std::is_integral<type>::value
                                    // std::uniform_int_distribution<> does not work in VS2015 for cl_uchar and cl_char,
                                    // because VS2015 thinks that use cl_int, because VS2015 thinks cl_uchar cl_char are
                                    // not int types
                                    && !(std::is_same<type, cl_uchar>::value || std::is_same<type, cl_char>::value)
                                 >::type* = 0)
{
    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<type> dis(min, max);
    for(auto& i : input)
    {
        i = dis(gen);
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    !is_vector_type<type>::value
                                    && std::is_integral<type>::value
                                    // std::uniform_int_distribution<> does not work in VS2015 for cl_uchar and cl_char,
                                    // because VS2015 thinks that use cl_int, because VS2015 thinks cl_uchar cl_char are
                                    // not int types
                                    && (std::is_same<type, cl_uchar>::value || std::is_same<type, cl_char>::value)
                                 >::type* = 0)
{
    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<cl_int> dis(
        static_cast<cl_int>(min), static_cast<cl_int>(max)
    );
    for(auto& i : input)
    {
        i = static_cast<type>(dis(gen));
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    is_vector_type<type>::value
                                    && std::is_floating_point<typename scalar_type<type>::type>::value
                                 >::type* = 0)
{
    typedef typename scalar_type<type>::type SCALAR;
    const size_t vec_size = vector_size<type>::value;

    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::uniform_real_distribution<SCALAR>> dists(vec_size);
    for(size_t i = 0; i < vec_size; i++)
    {
        // Fatal error
        if(std::fpclassify(max.s[i]) == FP_SUBNORMAL || std::fpclassify(min.s[i]) == FP_SUBNORMAL)
        {
            log_error("ERROR: min and max value for input generation CAN NOT BE subnormal\n");
        }
        dists[i] = std::uniform_real_distribution<SCALAR>(min.s[i], max.s[i]);
    }
    for(auto& i : input)
    {
        for(size_t j = 0; j < vec_size; j++)
        {
            SCALAR x = dists[j](gen);
            while(std::fpclassify(x) == FP_SUBNORMAL)
            {
                x = dists[j](gen);
            }
            i.s[j] = x;
        }
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

template <class type>
std::vector<type> generate_input(size_t count,
                                 const type& min,
                                 const type& max,
                                 const std::vector<type> special_cases,
                                 typename std::enable_if<
                                    !is_vector_type<type>::value
                                    && std::is_floating_point<type>::value
                                 >::type* = 0)
{
    // Fatal error
    if(std::fpclassify(max) == FP_SUBNORMAL || std::fpclassify(min) == FP_SUBNORMAL)
    {
        log_error("ERROR: min and max value for input generation CAN NOT BE subnormal\n");
    }
    std::vector<type> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<type> dis(min, max);
    for(auto& i : input)
    {
        type x = dis(gen);
        while(std::fpclassify(x) == FP_SUBNORMAL)
        {
            x = dis(gen);
        }
        i = x;
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

template <class type>
std::vector<type> generate_output(size_t count,
                                  typename scalar_type<type>::type svalue = typename scalar_type<type>::type(0),
                                  typename std::enable_if<is_vector_type<type>::value>::type* = 0)
{
    type value;
    for(size_t i = 0; i < vector_size<type>::value; i++)
        value.s[i] = svalue;
    return std::vector<type>(count, value);
}

template <class type>
std::vector<type> generate_output(size_t count,
                                  type svalue = type(0),
                                  typename std::enable_if<!is_vector_type<type>::value>::type* = 0)
{
    return std::vector<type>(count, svalue);
}

template<class T, class K>
void prepare_special_cases(std::vector<T>& in1_spec_cases, std::vector<K>& in2_spec_cases)
{
    if(in1_spec_cases.empty() || in2_spec_cases.empty())
    {
        return;
    }

    size_t new_size = in1_spec_cases.size() * in2_spec_cases.size();
    std::vector<T> new_in1(new_size);
    std::vector<K> new_in2(new_size);
    for(size_t i = 0; i < in1_spec_cases.size(); i++)
    {
        for(size_t j = 0; j < in2_spec_cases.size(); j++)
        {
            new_in1[(i * in2_spec_cases.size()) + j] = in1_spec_cases[i];
            new_in2[(i * in2_spec_cases.size()) + j] = in2_spec_cases[j];
        }
    }
    in1_spec_cases = new_in1;
    in2_spec_cases = new_in2;
}

template<class T, class K, class M>
void prepare_special_cases(std::vector<T>& in1_spec_cases,
                           std::vector<K>& in2_spec_cases,
                           std::vector<M>& in3_spec_cases)
{
    if(in3_spec_cases.empty())
    {
        return prepare_special_cases(in1_spec_cases, in2_spec_cases);
    }
    else if (in2_spec_cases.empty())
    {
        return prepare_special_cases(in1_spec_cases, in3_spec_cases);
    }
    else if (in1_spec_cases.empty())
    {
        return prepare_special_cases(in2_spec_cases, in3_spec_cases);
    }

    size_t new_size = in1_spec_cases.size() * in2_spec_cases.size() * in3_spec_cases.size();
    std::vector<T> new_in1(new_size);
    std::vector<K> new_in2(new_size);
    std::vector<M> new_in3(new_size);
    for(size_t i = 0; i < in1_spec_cases.size(); i++)
    {
        for(size_t j = 0; j < in2_spec_cases.size(); j++)
        {
            for(size_t k = 0; k < in3_spec_cases.size(); k++)
            {
                size_t idx =
                    (i * in2_spec_cases.size() * in3_spec_cases.size())
                    + (j * in3_spec_cases.size())
                    + k;
                new_in1[idx] = in1_spec_cases[i];
                new_in2[idx] = in2_spec_cases[j];
                new_in3[idx] = in3_spec_cases[k];
            }
        }
    }
    in1_spec_cases = new_in1;
    in2_spec_cases = new_in2;
    in3_spec_cases = new_in3;
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_GENERATE_INPUTS_HPP

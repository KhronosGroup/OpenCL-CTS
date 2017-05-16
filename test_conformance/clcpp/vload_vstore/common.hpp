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

#include <type_traits>
#include <cmath>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include "half_utils.hpp"

// Generates cl_half input
std::vector<cl_half> generate_half_input(size_t count,
                                         const cl_float& min,
                                         const cl_float& max,
                                         const std::vector<cl_half> special_cases)
{
    std::vector<cl_half> input(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<cl_float> dis(min, max);
    for(auto& i : input)
    {
        i = float2half_rte(dis(gen));
    }

    input.insert(input.begin(), special_cases.begin(), special_cases.end());
    input.resize(count);
    return input;
}

// Generates input for vload_vstore tests, we can't just simply use function
// generate_input<type>(...), because cl_half is typedef of cl_short (but generating
// cl_shorts and generating cl_halfs are different operations).
template <class type>
std::vector<type> vload_vstore_generate_input(size_t count,
                                              const type& min,
                                              const type& max, 
                                              const std::vector<type> special_cases,
                                              const bool generate_half,
                                              typename std::enable_if<
                                                  std::is_same<type, cl_half>::value
                                              >::type* = 0)
{
    if(!generate_half)
    {
        return generate_input<type>(count, min, max, special_cases);
    }
    return generate_half_input(count, -(CL_HALF_MAX/4.f), (CL_HALF_MAX/4.f), special_cases);
}

// If !std::is_same<type, cl_half>::value, we can just use generate_input<type>(...).
template <class type>
std::vector<type> vload_vstore_generate_input(size_t count,
                                              const type& min,
                                              const type& max, 
                                              const std::vector<type> special_cases,
                                              const bool generate_half,
                                              typename std::enable_if<
                                                  !std::is_same<type, cl_half>::value
                                              >::type* = 0)
{
    return generate_input<type>(count, min, max, special_cases);
}

#endif // TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMMON_HPP

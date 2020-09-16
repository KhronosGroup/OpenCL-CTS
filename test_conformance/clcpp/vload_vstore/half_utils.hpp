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
#ifndef TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP
#define TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <cmath>

namespace detail 
{

template<class INT_TYPE>
inline int clz(INT_TYPE x)
{
    int count = 0;
    if(std::is_unsigned<INT_TYPE>::value)
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

} // namespace detail 

#endif // TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP

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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_STRING_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_STRING_HPP


#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits>

#include "is_vector_type.hpp"
#include "scalar_type.hpp"
#include "type_name.hpp"

#include "../common.hpp"


template<class type>
std::string format_value(const type& value,
                         typename std::enable_if<is_vector_type<type>::value>::type* = 0)
{
    std::stringstream s;
    s << type_name<type>() << "{ ";
    s << std::scientific << std::setprecision(6);
    for (size_t j = 0; j < vector_size<type>::value; j++)
    {
        if (j > 0)
            s << ", ";
        s << value.s[j];
    }
    s << " }";
    return s.str();
}

template<class type>
std::string format_value(const type& value,
                         typename std::enable_if<!is_vector_type<type>::value>::type* = 0)
{
    std::stringstream s;
    s << type_name<type>() << "{ ";
    s << std::scientific << std::setprecision(6);
    s << value;
    s << " }";
    return s.str();
}

void replace_all(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_STRING_HPP

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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_NAME_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_NAME_HPP

#include "../common.hpp"

// Returns type name (in OpenCL device). 
// cl_uint - "uint", cl_float2 -> "float2"
template<class Type>
std::string type_name()
{
    return "unknown";
}

#define ADD_TYPE_NAME(Type, str) \
    template<> \
    std::string type_name<Type>() \
    { \
        return #str; \
    }

#define ADD_TYPE_NAME2(Type) \
    ADD_TYPE_NAME(cl_ ## Type, Type)

#define ADD_TYPE_NAME3(Type, x) \
    ADD_TYPE_NAME2(Type ## x)

#define ADD_TYPE_NAMES(Type) \
    ADD_TYPE_NAME2(Type) \
    ADD_TYPE_NAME3(Type, 2) \
    ADD_TYPE_NAME3(Type, 4) \
    ADD_TYPE_NAME3(Type, 8) \
    ADD_TYPE_NAME3(Type, 16)

ADD_TYPE_NAMES(char)
ADD_TYPE_NAMES(uchar)
ADD_TYPE_NAMES(short)
ADD_TYPE_NAMES(ushort)
ADD_TYPE_NAMES(int)
ADD_TYPE_NAMES(uint)
ADD_TYPE_NAMES(long)
ADD_TYPE_NAMES(ulong)
ADD_TYPE_NAMES(float)
ADD_TYPE_NAMES(double)

#undef ADD_TYPE_NAMES
#undef ADD_TYPE_NAME3
#undef ADD_TYPE_NAME2
#undef ADD_TYPE_NAME

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_NAME_HPP

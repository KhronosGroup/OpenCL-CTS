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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_SCALAR_TYPE_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_SCALAR_TYPE_HPP

#include "../common.hpp"

// scalar_type<Type>::type returns scalar type of Type.
//
// Examples:
// * scalar_type<cl_float>::type is cl_float
// * scalar_type<cl_float4>::types is cl_float
template<class Type>
struct scalar_type
{
    typedef void type;
};

#define ADD_VECTOR_TYPE(Type, n) \
    template<> \
    struct scalar_type<Type ## n> \
    { \
        typedef Type type; \
    };

#define ADD_VECTOR_TYPES(Type) \
    template<> \
    struct scalar_type<Type> \
    { \
        typedef Type type; \
    }; \
    ADD_VECTOR_TYPE(Type, 2) \
    ADD_VECTOR_TYPE(Type, 4) \
    ADD_VECTOR_TYPE(Type, 8) \
    ADD_VECTOR_TYPE(Type, 16)

ADD_VECTOR_TYPES(cl_char)
ADD_VECTOR_TYPES(cl_uchar)
ADD_VECTOR_TYPES(cl_short)
ADD_VECTOR_TYPES(cl_ushort)
ADD_VECTOR_TYPES(cl_int)
ADD_VECTOR_TYPES(cl_uint)
ADD_VECTOR_TYPES(cl_long)
ADD_VECTOR_TYPES(cl_ulong)
ADD_VECTOR_TYPES(cl_float)
ADD_VECTOR_TYPES(cl_double)

#undef ADD_VECTOR_TYPES
#undef ADD_VECTOR_TYPE

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_SCALAR_TYPE_HPP

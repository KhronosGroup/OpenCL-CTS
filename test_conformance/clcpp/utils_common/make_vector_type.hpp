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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_MAKE_VECTOR_TYPE_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_MAKE_VECTOR_TYPE_HPP

#include "../common.hpp"

// Using scalar_type and i creates a type scalar_typei.
// 
// Example:
// * make_vector_type<cl_uint, 8>::type is cl_uint8
// * make_vector_type<cl_uint, 1>::type is cl_uint
template<class scalar_type, size_t i>
struct make_vector_type
{
    typedef void type;
};

#define ADD_MAKE_VECTOR_TYPE(Type, n) \
    template<> \
    struct make_vector_type<Type, n> \
    { \
        typedef Type ## n type; \
    };

#define ADD_MAKE_VECTOR_TYPES(Type) \
    template<> \
    struct make_vector_type<Type, 1> \
    { \
        typedef Type type; \
    }; \
    ADD_MAKE_VECTOR_TYPE(Type, 2) \
    ADD_MAKE_VECTOR_TYPE(Type, 3) \
    ADD_MAKE_VECTOR_TYPE(Type, 4) \
    ADD_MAKE_VECTOR_TYPE(Type, 8) \
    ADD_MAKE_VECTOR_TYPE(Type, 16)

ADD_MAKE_VECTOR_TYPES(cl_char)
ADD_MAKE_VECTOR_TYPES(cl_uchar)
ADD_MAKE_VECTOR_TYPES(cl_short)
ADD_MAKE_VECTOR_TYPES(cl_ushort)
ADD_MAKE_VECTOR_TYPES(cl_int)
ADD_MAKE_VECTOR_TYPES(cl_uint)
ADD_MAKE_VECTOR_TYPES(cl_long)
ADD_MAKE_VECTOR_TYPES(cl_ulong)
ADD_MAKE_VECTOR_TYPES(cl_float)
ADD_MAKE_VECTOR_TYPES(cl_double)

#undef ADD_MAKE_VECTOR_TYPES
#undef ADD_MAKE_VECTOR_TYPE

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_MAKE_VECTOR_TYPE_HPP

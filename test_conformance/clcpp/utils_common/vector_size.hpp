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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_VECTOR_SIZE_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_VECTOR_SIZE_HPP

#include "../common.hpp"

// Returns 1 if Type is a scalar type; otherwise if it's a vector type, 
// it returns number of components in that Type. 
template<class Type>
struct vector_size
{
    const static size_t value = 1;
};

#define ADD_VECTOR_SIZE_TYPE(Type, n) \
    template<> \
    struct vector_size<Type ## n> \
    { \
        const static size_t value = n; \
    };

#define ADD_VECTOR_SIZE_TYPES(Type) \
    template<> \
    struct vector_size<Type> \
    { \
        const static size_t value = 1; \
    }; \
    ADD_VECTOR_SIZE_TYPE(Type, 2) \
    ADD_VECTOR_SIZE_TYPE(Type, 4) \
    ADD_VECTOR_SIZE_TYPE(Type, 8) \
    ADD_VECTOR_SIZE_TYPE(Type, 16)

ADD_VECTOR_SIZE_TYPES(cl_char)
ADD_VECTOR_SIZE_TYPES(cl_uchar)
ADD_VECTOR_SIZE_TYPES(cl_short)
ADD_VECTOR_SIZE_TYPES(cl_ushort)
ADD_VECTOR_SIZE_TYPES(cl_int)
ADD_VECTOR_SIZE_TYPES(cl_uint)
ADD_VECTOR_SIZE_TYPES(cl_long)
ADD_VECTOR_SIZE_TYPES(cl_ulong)
ADD_VECTOR_SIZE_TYPES(cl_float)
ADD_VECTOR_SIZE_TYPES(cl_double)

#undef ADD_VECTOR_SIZE_TYPES
#undef ADD_VECTOR_SIZE_TYPE

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_VECTOR_SIZE_HPP

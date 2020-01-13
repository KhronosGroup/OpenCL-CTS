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
#ifndef SUBHELPERS_H
#define SUBHELPERS_H

#include "testHarness.h"
#include "kernelHelpers.h"
#include "typeWrappers.h"

#include <limits>
#include <vector>
#undef min;
#undef max;
#define NON_UNIFORM 4
// Some template helpers
namespace subgroups {
    struct cl_char3 {
        ::cl_char3 data;
    };
    struct cl_uchar3 {
        ::cl_uchar3 data;
    };
    struct cl_short3 {
        ::cl_short3 data;
    };
    struct cl_ushort3 {
        ::cl_ushort3 data;
    };
    struct cl_int3 {
        ::cl_int3 data;
    };
    struct cl_uint3 {
        ::cl_uint3 data;
    };
    struct cl_long3 {
        ::cl_long3 data;
    };
    struct cl_ulong3 {
        ::cl_ulong3 data;
    };
    struct cl_float3 {
        ::cl_float3 data;
    };
    struct cl_double3 {
        ::cl_double3 data;
    };
    struct cl_half {
        ::cl_half data;
    };
    struct cl_half2 {
        ::cl_half2 data;
    };
    struct cl_half3 {
        ::cl_half3 data;
    };
    struct cl_half4 {
        ::cl_half4 data;
    };
    struct cl_half8 {
        ::cl_half8 data;
    };
    struct cl_half16 {
        ::cl_half16 data;
    };
};

template<typename Ty>
struct scalar_type {};

template<>
struct scalar_type<cl_uint2> { using type = cl_uint; };
template<>
struct scalar_type<subgroups::cl_uint3> { using type = cl_uint; };
template<>
struct scalar_type<cl_uint4> { using type = cl_uint; };
template<>
struct scalar_type<cl_uint8> { using type = cl_uint; };
template<>
struct scalar_type<cl_uint16> { using type = cl_uint; };

template<>
struct scalar_type<cl_int2> { using type = cl_int; };
template<>
struct scalar_type<subgroups::cl_int3> { using type = cl_int; };
template<>
struct scalar_type<cl_int4> { using type = cl_int; };
template<>
struct scalar_type<cl_int8> { using type = cl_int; };
template<>
struct scalar_type<cl_int16> { using type = cl_int; };

template<>
struct scalar_type<cl_ulong2> { using type = cl_ulong; };
template<>
struct scalar_type<subgroups::cl_ulong3> { using type = cl_ulong; };
template<>
struct scalar_type<cl_ulong4> { using type = cl_ulong; };
template<>
struct scalar_type<cl_ulong8> { using type = cl_ulong; };
template<>
struct scalar_type<cl_ulong16> { using type = cl_ulong; };

template<>
struct scalar_type<cl_long2> { using type = cl_long; };
template<>
struct scalar_type<subgroups::cl_long3> { using type = cl_long; };
template<>
struct scalar_type<cl_long4> { using type = cl_long; };
template<>
struct scalar_type<cl_long8> { using type = cl_long; };
template<>
struct scalar_type<cl_long16> { using type = cl_long; };

template<>
struct scalar_type<cl_ushort2> { using type = cl_ushort; };
template<>
struct scalar_type<subgroups::cl_ushort3> { using type = cl_ushort; };
template<>
struct scalar_type<cl_ushort4> { using type = cl_ushort; };
template<>
struct scalar_type<cl_ushort8> { using type = cl_ushort; };
template<>
struct scalar_type<cl_ushort16> { using type = cl_ushort; };

template<>
struct scalar_type<cl_short2> { using type = cl_short; };
template<>
struct scalar_type<subgroups::cl_short3> { using type = cl_short; };
template<>
struct scalar_type<cl_short4> { using type = cl_short; };
template<>
struct scalar_type<cl_short8> { using type = cl_short; };
template<>
struct scalar_type<cl_short16> { using type = cl_short; };

template<>
struct scalar_type<cl_uchar2> { using type = cl_uchar; };
template<>
struct scalar_type<subgroups::cl_uchar3> { using type = cl_uchar; };
template<>
struct scalar_type<cl_uchar4> { using type = cl_uchar; };
template<>
struct scalar_type<cl_uchar8> { using type = cl_uchar; };
template<>
struct scalar_type<cl_uchar16> { using type = cl_uchar; };

template<>
struct scalar_type<cl_char2> { using type = cl_char; };
template<>
struct scalar_type<subgroups::cl_char3> { using type = cl_char; };
template<>
struct scalar_type<cl_char4> { using type = cl_char; };
template<>
struct scalar_type<cl_char8> { using type = cl_char; };
template<>
struct scalar_type<cl_char16> { using type = cl_char; };

template<>
struct scalar_type<cl_float2> { using type = cl_float; };
template<>
struct scalar_type<subgroups::cl_float3> { using type = cl_float; };
template<>
struct scalar_type<cl_float4> { using type = cl_float; };
template<>
struct scalar_type<cl_float8> { using type = cl_float; };
template<>
struct scalar_type<cl_float16> { using type = cl_float; };

template<>
struct scalar_type<subgroups::cl_half2> { using type = cl_half; };
template<>
struct scalar_type<subgroups::cl_half3> { using type = cl_half; };
template<>
struct scalar_type<subgroups::cl_half4> { using type = cl_half; };
template<>
struct scalar_type<subgroups::cl_half8> { using type = cl_half; };
template<>
struct scalar_type<subgroups::cl_half16> { using type = cl_half; };

template<>
struct scalar_type<cl_double2> { using type = cl_double; };
template<>
struct scalar_type<subgroups::cl_double3> { using type = cl_double; };
template<>
struct scalar_type<cl_double4> { using type = cl_double; };
template<>
struct scalar_type<cl_double8> { using type = cl_double; };
template<>
struct scalar_type<cl_double16> { using type = cl_double; };

template<typename Ty>
struct is_vector_type {};

template<typename Ty>
struct is_vector_type3 {};
template<typename Ty>
struct is_vector_type_half {};

template<>
struct is_vector_type<cl_int> : std::false_type {};
template<>
struct is_vector_type<cl_uint> : std::false_type {};
template<>
struct is_vector_type<cl_long> : std::false_type {};
template<>
struct is_vector_type<cl_ulong> : std::false_type {};
template<>
struct is_vector_type<cl_float> : std::false_type {};
template<>
struct is_vector_type<cl_double> : std::false_type {};
template<>
struct is_vector_type<cl_short> : std::false_type {};
template<>
struct is_vector_type<cl_ushort> : std::false_type {};
template<>
struct is_vector_type<cl_char> : std::false_type {};
template<>
struct is_vector_type<cl_uchar> : std::false_type {};
template<>
struct is_vector_type_half<subgroups::cl_half> : std::false_type {};

template<>
struct is_vector_type<cl_uint2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_uint3> : std::true_type {};
template<>
struct is_vector_type<cl_uint4> : std::true_type {};
template<>
struct is_vector_type<cl_uint8> : std::true_type {};
template<>
struct is_vector_type<cl_uint16> : std::true_type {};
template<>
struct is_vector_type<cl_int2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_int3> : std::true_type {};
template<>
struct is_vector_type<cl_int4> : std::true_type {};
template<>
struct is_vector_type<cl_int8> : std::true_type {};
template<>
struct is_vector_type<cl_int16> : std::true_type {};

template<>
struct is_vector_type<cl_ulong2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_ulong3> : std::true_type {};
template<>
struct is_vector_type<cl_ulong4> : std::true_type {};
template<>
struct is_vector_type<cl_ulong8> : std::true_type {};
template<>
struct is_vector_type<cl_ulong16> : std::true_type {};

template<>
struct is_vector_type<cl_long2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_long3> : std::true_type {};
template<>
struct is_vector_type<cl_long4> : std::true_type {};
template<>
struct is_vector_type<cl_long8> : std::true_type {};
template<>
struct is_vector_type<cl_long16> : std::true_type {};

template<>
struct is_vector_type<cl_ushort2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_ushort3> : std::true_type {};
template<>
struct is_vector_type<cl_ushort4> : std::true_type {};
template<>
struct is_vector_type<cl_ushort8> : std::true_type {};
template<>
struct is_vector_type<cl_ushort16> : std::true_type {};

template<>
struct is_vector_type<cl_short2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_short3> : std::true_type {};
template<>
struct is_vector_type<cl_short4> : std::true_type {};
template<>
struct is_vector_type<cl_short8> : std::true_type {};
template<>
struct is_vector_type<cl_short16> : std::true_type {};

template<>
struct is_vector_type<cl_uchar2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_uchar3> : std::true_type {};
template<>
struct is_vector_type<cl_uchar4> : std::true_type {};
template<>
struct is_vector_type<cl_uchar8> : std::true_type {};
template<>
struct is_vector_type<cl_uchar16> : std::true_type {};

template<>
struct is_vector_type<cl_char2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_char3> : std::true_type {};
template<>
struct is_vector_type<cl_char4> : std::true_type {};
template<>
struct is_vector_type<cl_char8> : std::true_type {};
template<>
struct is_vector_type<cl_char16> : std::true_type {};

template<>
struct is_vector_type<cl_float2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_float3> : std::true_type {};
template<>
struct is_vector_type<cl_float4> : std::true_type {};
template<>
struct is_vector_type<cl_float8> : std::true_type {};
template<>
struct is_vector_type<cl_float16> : std::true_type {};

template<>
struct is_vector_type_half<subgroups::cl_half2> : std::true_type {};
template<>
struct is_vector_type_half<subgroups::cl_half3> : std::true_type {};
template<>
struct is_vector_type_half<subgroups::cl_half4> : std::true_type {};
template<>
struct is_vector_type_half<subgroups::cl_half8> : std::true_type {};
template<>
struct is_vector_type_half<subgroups::cl_half16> : std::true_type {};

template<>
struct is_vector_type<cl_double2> : std::true_type {};
template<>
struct is_vector_type3<subgroups::cl_double3> : std::true_type {};
template<>
struct is_vector_type<cl_double4> : std::true_type {};
template<>
struct is_vector_type<cl_double8> : std::true_type {};
template<>
struct is_vector_type<cl_double16> : std::true_type {};

template <typename Ty>
typename std::enable_if<is_vector_type<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        if (lhs.s[i] != rhs.s[i]) {
            return false;
        }
    }
    return true;
}

template <typename Ty>
typename std::enable_if<is_vector_type3<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        if (lhs.data.s[i] != rhs.data.s[i]) {
            return false;
        }
    }
    return true;
}

template <typename Ty>
typename std::enable_if<is_vector_type_half<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        if (lhs.data.s[i] != rhs.data.s[i]) {
            return false;
        }
    }
    return true;
}

template <typename Ty>
typename std::enable_if<is_vector_type<Ty>::value, bool>::type
set_value(Ty &lhs, const cl_uint &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        lhs.s[i] = rhs;
    }
    return true;
}


template <typename Ty>
typename std::enable_if<is_vector_type<Ty>::value, bool>::type
set_value(Ty &lhs, const Ty &rhs) {
    lhs = rhs;
    return true;
}


template <typename Ty, int N = 0 >
typename std::enable_if<is_vector_type3<Ty>::value, bool>::type
set_value(Ty &lhs, const cl_uint &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        lhs.data.s[i] = rhs;
    }
    return true;
}

template <typename Ty, int N = 0 >
typename std::enable_if<is_vector_type_half<Ty>::value, bool>::type
set_value(Ty &lhs, const cl_uint &rhs) {
    const int size = sizeof(Ty) / sizeof(typename scalar_type<Ty>::type);
    for (auto i = 0; i < size; ++i) {
        lhs.data.s[i] = rhs;
    }
    return true;
}

template <typename Ty>
typename std::enable_if<!is_vector_type<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs) {
    return (lhs == rhs) ? true : false;
}

template <typename Ty>
typename std::enable_if<!is_vector_type<Ty>::value, bool>::type
set_value(Ty &lhs, const cl_uint &rhs) {
    return lhs = rhs;
}

template <typename Ty>
typename std::enable_if<!is_vector_type_half<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs) {
    return (lhs.data == rhs.data) ? true : false;
}

template <typename Ty>
typename std::enable_if<!is_vector_type_half<Ty>::value, bool>::type
set_value(Ty &lhs, const cl_uint &rhs) {
    return lhs.data = rhs;
}

template <typename Ty> struct TypeName;
template <> struct TypeName<cl_int> { static const char * val() { return "int"; } };
template <> struct TypeName<cl_int2> { static const char * val() { return "int2"; } };
template <> struct TypeName<subgroups::cl_int3> { static const char * val() { return "int3"; } };
template <> struct TypeName<cl_int4> { static const char * val() { return "int4"; } };
template <> struct TypeName<cl_int8> { static const char * val() { return "int8"; } };
template <> struct TypeName<cl_int16> { static const char * val() { return "int16"; } };
template <> struct TypeName<cl_uint> { static const char * val() { return "uint"; } };
template <> struct TypeName<cl_uint2> { static const char * val() { return "uint2"; } };
template <> struct TypeName<subgroups::cl_uint3> { static const char * val() { return "uint3"; } };
template <> struct TypeName<cl_uint4> { static const char * val() { return "uint4"; } };
template <> struct TypeName<cl_uint8> { static const char * val() { return "uint8"; } };
template <> struct TypeName<cl_uint16> { static const char * val() { return "uint16"; } };

template <> struct TypeName<cl_long> { static const char * val() { return "long"; } };
template <> struct TypeName<cl_long2> { static const char * val() { return "long2"; } };
template <> struct TypeName<subgroups::cl_long3> { static const char * val() { return "long3"; } };
template <> struct TypeName<cl_long4> { static const char * val() { return "long4"; } };
template <> struct TypeName<cl_long8> { static const char * val() { return "long8"; } };
template <> struct TypeName<cl_long16> { static const char * val() { return "long16"; } };

template <> struct TypeName<cl_ulong> { static const char * val() { return "ulong"; } };
template <> struct TypeName<cl_ulong2> { static const char * val() { return "ulong2"; } };
template <> struct TypeName<subgroups::cl_ulong3> { static const char * val() { return "ulong3"; } };
template <> struct TypeName<cl_ulong4> { static const char * val() { return "ulong4"; } };
template <> struct TypeName<cl_ulong8> { static const char * val() { return "ulong8"; } };
template <> struct TypeName<cl_ulong16> { static const char * val() { return "ulong16"; } };

template <> struct TypeName<cl_float> { static const char * val() { return "float"; } };
template <> struct TypeName<cl_float2> { static const char * val() { return "float2"; } };
template <> struct TypeName<subgroups::cl_float3> { static const char * val() { return "float3"; } };
template <> struct TypeName<cl_float4> { static const char * val() { return "float4"; } };
template <> struct TypeName<cl_float8> { static const char * val() { return "float8"; } };
template <> struct TypeName<cl_float16> { static const char * val() { return "float16"; } };

template <> struct TypeName<subgroups::cl_half> { static const char * val() { return "half"; } };
template <> struct TypeName<subgroups::cl_half2> { static const char * val() { return "half2"; } };
template <> struct TypeName<subgroups::cl_half3> { static const char * val() { return "half3"; } };
template <> struct TypeName<subgroups::cl_half4> { static const char * val() { return "half4"; } };
template <> struct TypeName<subgroups::cl_half8> { static const char * val() { return "half8"; } };
template <> struct TypeName<subgroups::cl_half16> { static const char * val() { return "half16"; } };

template <> struct TypeName<cl_double> { static const char * val() { return "double"; } };
template <> struct TypeName<cl_double2> { static const char * val() { return "double2"; } };
template <> struct TypeName<subgroups::cl_double3> { static const char * val() { return "double3"; } };
template <> struct TypeName<cl_double4> { static const char * val() { return "double4"; } };
template <> struct TypeName<cl_double8> { static const char * val() { return "double8"; } };
template <> struct TypeName<cl_double16> { static const char * val() { return "double16"; } };

template <> struct TypeName<cl_short> { static const char * val() { return "short"; } };
template <> struct TypeName<cl_short2> { static const char * val() { return "short2"; } };
template <> struct TypeName<subgroups::cl_short3> { static const char * val() { return "short3"; } };
template <> struct TypeName<cl_short4> { static const char * val() { return "short4"; } };
template <> struct TypeName<cl_short8> { static const char * val() { return "short8"; } };
template <> struct TypeName<cl_short16> { static const char * val() { return "short16"; } };

template <> struct TypeName<cl_ushort> { static const char * val() { return "ushort"; } };
template <> struct TypeName<cl_ushort2> { static const char * val() { return "ushort2"; } };
template <> struct TypeName<subgroups::cl_ushort3> { static const char * val() { return "ushort3"; } };
template <> struct TypeName<cl_ushort4> { static const char * val() { return "ushort4"; } };
template <> struct TypeName<cl_ushort8> { static const char * val() { return "ushort8"; } };
template <> struct TypeName<cl_ushort16> { static const char * val() { return "ushort16"; } };

template <> struct TypeName<cl_char> { static const char * val() { return "char"; } };
template <> struct TypeName<cl_char2> { static const char * val() { return "char2"; } };
template <> struct TypeName<subgroups::cl_char3> { static const char * val() { return "char3"; } };
template <> struct TypeName<cl_char4> { static const char * val() { return "char4"; } };
template <> struct TypeName<cl_char8> { static const char * val() { return "char8"; } };
template <> struct TypeName<cl_char16> { static const char * val() { return "char16"; } };

template <> struct TypeName<cl_uchar> { static const char * val() { return "uchar"; } };
template <> struct TypeName<cl_uchar2> { static const char * val() { return "uchar2"; } };
template <> struct TypeName<subgroups::cl_uchar3> { static const char * val() { return "uchar3"; } };
template <> struct TypeName<cl_uchar4> { static const char * val() { return "uchar4"; } };
template <> struct TypeName<cl_uchar8> { static const char * val() { return "uchar8"; } };
template <> struct TypeName<cl_uchar16> { static const char * val() { return "uchar16"; } };

//template <> struct TypeName<cl_half> { static const char * val() { return "half"; } };
template <typename Ty> struct TypeDef;
template <> struct TypeDef<cl_int> { static const char * val() { return "typedef int Type;\n"; } };
template <> struct TypeDef<cl_int2> { static const char * val() { return "typedef int2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_int3> { static const char * val() { return "typedef int3 Type;\n"; } };
template <> struct TypeDef<cl_int4> { static const char * val() { return "typedef int4 Type;\n"; } };
template <> struct TypeDef<cl_int8> { static const char * val() { return "typedef int8 Type;\n"; } };
template <> struct TypeDef<cl_int16> { static const char * val() { return "typedef int16 Type;\n"; } };
template <> struct TypeDef<cl_uint> { static const char * val() { return "typedef uint Type;\n"; } };
template <> struct TypeDef<cl_uint2> { static const char * val() { return "typedef uint2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_uint3> { static const char * val() { return "typedef uint3 Type;\n"; } };
template <> struct TypeDef<cl_uint4> { static const char * val() { return "typedef uint4 Type;\n"; } };
template <> struct TypeDef<cl_uint8> { static const char * val() { return "typedef uint8 Type;\n"; } };
template <> struct TypeDef<cl_uint16> { static const char * val() { return "typedef uint16 Type;\n"; } };

template <> struct TypeDef<cl_long> { static const char * val() { return "typedef long Type;\n"; } };
template <> struct TypeDef<cl_long2> { static const char * val() { return "typedef long2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_long3> { static const char * val() { return "typedef long3 Type;\n"; } };
template <> struct TypeDef<cl_long4> { static const char * val() { return "typedef long4 Type;\n"; } };
template <> struct TypeDef<cl_long8> { static const char * val() { return "typedef long8 Type;\n"; } };
template <> struct TypeDef<cl_long16> { static const char * val() { return "typedef long16 Type;\n"; } };
template <> struct TypeDef<cl_ulong> { static const char * val() { return "typedef ulong Type;\n"; } };
template <> struct TypeDef<cl_ulong2> { static const char * val() { return "typedef ulong2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_ulong3> { static const char * val() { return "typedef ulong3 Type;\n"; } };
template <> struct TypeDef<cl_ulong4> { static const char * val() { return "typedef ulong4 Type;\n"; } };
template <> struct TypeDef<cl_ulong8> { static const char * val() { return "typedef ulong8 Type;\n"; } };
template <> struct TypeDef<cl_ulong16> { static const char * val() { return "typedef ulong16 Type;\n"; } };

template <> struct TypeDef<cl_float> { static const char * val() { return "typedef float Type;\n"; } };
template <> struct TypeDef<cl_float2> { static const char * val() { return "typedef float2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_float3> { static const char * val() { return "typedef float3 Type;\n"; } };
template <> struct TypeDef<cl_float4> { static const char * val() { return "typedef float4 Type;\n"; } };
template <> struct TypeDef<cl_float8> { static const char * val() { return "typedef float8 Type;\n"; } };
template <> struct TypeDef<cl_float16> { static const char * val() { return "typedef float16 Type;\n"; } };

template <> struct TypeDef<subgroups::cl_half> { static const char * val() { return "typedef half Type;\n"; } };
template <> struct TypeDef<subgroups::cl_half2> { static const char * val() { return "typedef half2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_half3> { static const char * val() { return "typedef half3 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_half4> { static const char * val() { return "typedef half4 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_half8> { static const char * val() { return "typedef half8 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_half16> { static const char * val() { return "typedef half16 Type;\n"; } };

template <> struct TypeDef<cl_double> { static const char * val() { return "typedef double Type;\n"; } };
template <> struct TypeDef<cl_double2> { static const char * val() { return "typedef double2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_double3> { static const char * val() { return "typedef double3 Type;\n"; } };
template <> struct TypeDef<cl_double4> { static const char * val() { return "typedef double4 Type;\n"; } };
template <> struct TypeDef<cl_double8> { static const char * val() { return "typedef double8 Type;\n"; } };
template <> struct TypeDef<cl_double16> { static const char * val() { return "typedef double16 Type;\n"; } };

template <> struct TypeDef<cl_short> { static const char * val() { return "typedef short Type;\n"; } };
template <> struct TypeDef<cl_short2> { static const char * val() { return "typedef short2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_short3> { static const char * val() { return "typedef short3 Type;\n"; } };
template <> struct TypeDef<cl_short4> { static const char * val() { return "typedef short4 Type;\n"; } };
template <> struct TypeDef<cl_short8> { static const char * val() { return "typedef short8 Type;\n"; } };
template <> struct TypeDef<cl_short16> { static const char * val() { return "typedef short16 Type;\n"; } };
template <> struct TypeDef<cl_ushort> { static const char * val() { return "typedef ushort Type;\n"; } };
template <> struct TypeDef<cl_ushort2> { static const char * val() { return "typedef ushort2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_ushort3> { static const char * val() { return "typedef ushort3 Type;\n"; } };
template <> struct TypeDef<cl_ushort4> { static const char * val() { return "typedef ushort4 Type;\n"; } };
template <> struct TypeDef<cl_ushort8> { static const char * val() { return "typedef ushort8 Type;\n"; } };
template <> struct TypeDef<cl_ushort16> { static const char * val() { return "typedef ushort16 Type;\n"; } };

template <> struct TypeDef<cl_char> { static const char * val() { return "typedef char Type;\n"; } };
template <> struct TypeDef<cl_char2> { static const char * val() { return "typedef char2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_char3> { static const char * val() { return "typedef char3 Type;\n"; } };
template <> struct TypeDef<cl_char4> { static const char * val() { return "typedef char4 Type;\n"; } };
template <> struct TypeDef<cl_char8> { static const char * val() { return "typedef char8 Type;\n"; } };
template <> struct TypeDef<cl_char16> { static const char * val() { return "typedef char16 Type;\n"; } };
template <> struct TypeDef<cl_uchar> { static const char * val() { return "typedef uchar Type;\n"; } };
template <> struct TypeDef<cl_uchar2> { static const char * val() { return "typedef uchar2 Type;\n"; } };
template <> struct TypeDef<subgroups::cl_uchar3> { static const char * val() { return "typedef uchar3 Type;\n"; } };
template <> struct TypeDef<cl_uchar4> { static const char * val() { return "typedef uchar4 Type;\n"; } };
template <> struct TypeDef<cl_uchar8> { static const char * val() { return "typedef uchar8 Type;\n"; } };
template <> struct TypeDef<cl_uchar16> { static const char * val() { return "typedef uchar16 Type;\n"; } };

//template <> struct TypeDef<cl_half> { static const char * val() { return "typedef short Type;\n"; } };

template <typename Ty, int Which> struct TypeIdentity;
template <> struct TypeIdentity<subgroups::cl_half, 0> { static cl_half val() { return (cl_half)0.0; } };
template <> struct TypeIdentity<subgroups::cl_half, 1> { static cl_half val() { return std::numeric_limits<cl_half>::min(); } };
template <> struct TypeIdentity<subgroups::cl_half, 2> { static cl_half val() { return std::numeric_limits<cl_half>::max(); } };
template <> struct TypeIdentity<subgroups::cl_half, 3> { static cl_half val() { return (cl_half)1; } };     //mul
template <> struct TypeIdentity<subgroups::cl_half, 4> { static cl_half val() { return (cl_half)~0; } };    //and
template <> struct TypeIdentity<subgroups::cl_half, 5> { static cl_half val() { return (cl_half)0; } };     //or
template <> struct TypeIdentity<subgroups::cl_half, 6> { static cl_half val() { return (cl_half)0; } };     //xor

template <> struct TypeIdentity<cl_uchar, 0> { static cl_uchar val() { return (cl_uchar)0; } };     //add
template <> struct TypeIdentity<cl_uchar, 1> { static cl_uchar val() { return std::numeric_limits<cl_uchar>::min(); } }; //max
template <> struct TypeIdentity<cl_uchar, 2> { static cl_uchar val() { return std::numeric_limits<cl_uchar>::max(); } }; //min
template <> struct TypeIdentity<cl_uchar, 3> { static cl_uchar val() { return (cl_uchar)1; } };     //mul
template <> struct TypeIdentity<cl_uchar, 4> { static cl_uchar val() { return (cl_uchar)~0; } };    //and
template <> struct TypeIdentity<cl_uchar, 5> { static cl_uchar val() { return (cl_uchar)0; } };     //or
template <> struct TypeIdentity<cl_uchar, 6> { static cl_uchar val() { return (cl_uchar)0; } };     //xor

template <> struct TypeIdentity<cl_char, 0> { static cl_char val() { return (cl_char)0; } };     //add
template <> struct TypeIdentity<cl_char, 1> { static cl_char val() { return std::numeric_limits<cl_char>::min(); } }; //max
template <> struct TypeIdentity<cl_char, 2> { static cl_char val() { return std::numeric_limits<cl_char>::max(); } }; //min
template <> struct TypeIdentity<cl_char, 3> { static cl_char val() { return (cl_char)1; } };     //mul
template <> struct TypeIdentity<cl_char, 4> { static cl_char val() { return (cl_char)~0; } };    //and
template <> struct TypeIdentity<cl_char, 5> { static cl_char val() { return (cl_char)0; } };     //or
template <> struct TypeIdentity<cl_char, 6> { static cl_char val() { return (cl_char)0; } };     //xor

template <> struct TypeIdentity<cl_uint, 0> { static cl_uint val() { return (cl_uint)0; } };     //add
template <> struct TypeIdentity<cl_uint, 1> { static cl_uint val() { return std::numeric_limits<cl_uint>::min(); } }; //max
template <> struct TypeIdentity<cl_uint, 2> { static cl_uint val() { return std::numeric_limits<cl_uint>::max(); } }; //min
template <> struct TypeIdentity<cl_uint, 3> { static cl_uint val() { return (cl_uint)1; } };     //mul
template <> struct TypeIdentity<cl_uint, 4> { static cl_uint val() { return (cl_uint)~0; } };    //and
template <> struct TypeIdentity<cl_uint, 5> { static cl_uint val() { return (cl_uint)0; } };     //or
template <> struct TypeIdentity<cl_uint, 6> { static cl_uint val() { return (cl_uint)0; } };     //xor

template <> struct TypeIdentity<cl_int, 0> { static cl_int val() { return (cl_int)0 ; } };       //add
template <> struct TypeIdentity<cl_int, 1> { static cl_int val() { return std::numeric_limits<int>::min(); } }; //max
template <> struct TypeIdentity<cl_int, 2> { static cl_int val() { return std::numeric_limits<int>::max(); } }; //min
template <> struct TypeIdentity<cl_int, 3> { static cl_int val() { return (cl_int)1; } };      //mul
template <> struct TypeIdentity<cl_int, 4> { static cl_int val() { return (cl_int)~0; } };     //and
template <> struct TypeIdentity<cl_int, 5> { static cl_int val() { return (cl_int)0; } };      //or
template <> struct TypeIdentity<cl_int, 6> { static cl_int val() { return (cl_int)0; } };      //xor

template <> struct TypeIdentity<cl_short, 0> { static cl_short val() { return (cl_short)0; } };     //add
template <> struct TypeIdentity<cl_short, 1> { static cl_short val() { return std::numeric_limits<cl_short>::min(); } }; //max
template <> struct TypeIdentity<cl_short, 2> { static cl_short val() { return std::numeric_limits<cl_short>::max(); } }; //min
template <> struct TypeIdentity<cl_short, 3> { static cl_short val() { return (cl_short)1; } };     //mul
template <> struct TypeIdentity<cl_short, 4> { static cl_short val() { return (cl_short)~0; } };    //and
template <> struct TypeIdentity<cl_short, 5> { static cl_short val() { return (cl_short)0; } };     //or
template <> struct TypeIdentity<cl_short, 6> { static cl_short val() { return (cl_short)0; } };     //xor

template <> struct TypeIdentity<cl_ushort, 0> { static cl_ushort val() { return (cl_ushort)0; } };     //add
template <> struct TypeIdentity<cl_ushort, 1> { static cl_ushort val() { return std::numeric_limits<cl_ushort>::min(); } }; //max
template <> struct TypeIdentity<cl_ushort, 2> { static cl_ushort val() { return std::numeric_limits<cl_ushort>::max(); } }; //min
template <> struct TypeIdentity<cl_ushort, 3> { static cl_ushort val() { return (cl_ushort)1; } };     //mul
template <> struct TypeIdentity<cl_ushort, 4> { static cl_ushort val() { return (cl_ushort)~0; } };    //and
template <> struct TypeIdentity<cl_ushort, 5> { static cl_ushort val() { return (cl_ushort)0; } };     //or
template <> struct TypeIdentity<cl_ushort, 6> { static cl_ushort val() { return (cl_ushort)0; } };     //xor

template <> struct TypeIdentity<cl_ulong, 0> { static cl_ulong val() { return (cl_ulong)0 ; } };
template <> struct TypeIdentity<cl_ulong, 1> { static cl_ulong val() { return std::numeric_limits<cl_ulong>::min(); } };
template <> struct TypeIdentity<cl_ulong, 2> { static cl_ulong val() { return std::numeric_limits<cl_ulong>::max(); } };
template <> struct TypeIdentity<cl_ulong, 3> { static cl_ulong val() { return (cl_ulong)1; } };      //mul
template <> struct TypeIdentity<cl_ulong, 4> { static cl_ulong val() { return (cl_ulong)~0; } };     //and
template <> struct TypeIdentity<cl_ulong, 5> { static cl_ulong val() { return (cl_ulong)0; } };      //or
template <> struct TypeIdentity<cl_ulong, 6> { static cl_ulong val() { return (cl_ulong)0; } };      //xor

template <> struct TypeIdentity<cl_long, 0> { static cl_long val() { return (cl_long)0; } };
template <> struct TypeIdentity<cl_long, 1> { static cl_long val() { return std::numeric_limits<cl_long>::min(); } };
template <> struct TypeIdentity<cl_long, 2> { static cl_long val() { return std::numeric_limits<cl_long>::max(); } };
template <> struct TypeIdentity<cl_long, 3> { static cl_long val() { return (cl_long)1; } };      //mul
template <> struct TypeIdentity<cl_long, 4> { static cl_long val() { return (cl_long)~0; } };     //and
template <> struct TypeIdentity<cl_long, 5> { static cl_long val() { return (cl_long)0; } };      //or
template <> struct TypeIdentity<cl_long, 6> { static cl_long val() { return (cl_long)0; } };      //xor

template <> struct TypeIdentity<cl_float, 0> { static cl_float val() { return 0.F; } };
template <> struct TypeIdentity<cl_float, 1> { static cl_float val() { return -std::numeric_limits<float>::infinity(); } };
template <> struct TypeIdentity<cl_float, 2> { static cl_float val() { return std::numeric_limits<float>::infinity(); } };
template <> struct TypeIdentity<cl_float, 3> { static cl_float val() { return (cl_float)1; } };      //mul
template <> struct TypeIdentity<cl_float, 4> { static cl_float val() { return (cl_float)~0; } };     //and
template <> struct TypeIdentity<cl_float, 5> { static cl_float val() { return (cl_float)0; } };      //or
template <> struct TypeIdentity<cl_float, 6> { static cl_float val() { return (cl_float)0; } };      //xor

template <> struct TypeIdentity<cl_double, 0> { static cl_double val() { return 0.L; } };
template <> struct TypeIdentity<cl_double, 1> { static cl_double val() { return -std::numeric_limits<double>::infinity(); } };
template <> struct TypeIdentity<cl_double, 2> { static cl_double val() { return std::numeric_limits<double>::infinity(); } };
template <> struct TypeIdentity<cl_double, 3> { static cl_double val() { return (cl_double)1; } };      //mul
template <> struct TypeIdentity<cl_double, 4> { static cl_double val() { return (cl_double)~0; } };     //and
template <> struct TypeIdentity<cl_double, 5> { static cl_double val() { return (cl_double)0; } };      //or
template <> struct TypeIdentity<cl_double, 6> { static cl_double val() { return (cl_double)0; } };      //xor

template <typename Ty> struct TypeCheck;
template <> struct TypeCheck<cl_uint> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uint2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_uint3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uint4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uint8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uint16> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_int> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_int2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_int3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_int4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_int8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_int16> { static bool val(cl_device_id) { return true; } };

template <> struct TypeCheck<cl_ushort> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_ushort2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_ushort3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_ushort4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_ushort8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_ushort16> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_short> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_short2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_short3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_short4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_short8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_short16> { static bool val(cl_device_id) { return true; } };

template <> struct TypeCheck<cl_uchar> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uchar2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_uchar3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uchar4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uchar8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_uchar16> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_char> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_char2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_char3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_char4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_char8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_char16> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_float> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_float2> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<subgroups::cl_float3> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_float4> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_float8> { static bool val(cl_device_id) { return true; } };
template <> struct TypeCheck<cl_float16> { static bool val(cl_device_id) { return true; } };

static bool
int64_ok(cl_device_id device)
{
    char profile[128];
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), (void *)&profile, NULL);
    if (error) {
        log_info("clGetDeviceInfo failed with CL_DEVICE_PROFILE\n");
    return false;
    }

    if (strcmp(profile, "EMBEDDED_PROFILE") == 0)
     return is_extension_available(device, "cles_khr_int64");

    return true;
}

static bool
double_ok(cl_device_id device) {
    int error;
    cl_device_fp_config c;
    error = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(c), (void *)&c, NULL);
    if (error) {
        log_info("clGetDeviceInfo failed with CL_DEVICE_DOUBLE_FP_CONFIG\n");
        return false;
    }
    return c != 0;
}

static bool
half_ok(cl_device_id device) {
    int error;
    cl_device_fp_config c;
    error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(c), (void *)&c, NULL);
    if (error) {
        log_info("clGetDeviceInfo failed with CL_DEVICE_HALF_FP_CONFIG\n");
        return false;
    }
    return c != 0;
}

template <> struct TypeCheck<cl_ulong> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_ulong2> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<subgroups::cl_ulong3> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_ulong4> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_ulong8> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_ulong16> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_long> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_long2> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<subgroups::cl_long3> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_long4> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_long8> { static bool val(cl_device_id device) { return int64_ok(device); } };
template <> struct TypeCheck<cl_long16> { static bool val(cl_device_id device) { return int64_ok(device); } };

template <> struct TypeCheck<subgroups::cl_half> { static bool val(cl_device_id device) { return half_ok(device); } };
template <> struct TypeCheck<subgroups::cl_half2> { static bool val(cl_device_id device) { return half_ok(device); } };
template <> struct TypeCheck<subgroups::cl_half3> { static bool val(cl_device_id device) { return half_ok(device); } };
template <> struct TypeCheck<subgroups::cl_half4> { static bool val(cl_device_id device) { return half_ok(device); } };
template <> struct TypeCheck<subgroups::cl_half8> { static bool val(cl_device_id device) { return half_ok(device); } };
template <> struct TypeCheck<subgroups::cl_half16> { static bool val(cl_device_id device) { return half_ok(device); } };

template <> struct TypeCheck<cl_double> { static bool val(cl_device_id device) { return double_ok(device); } };
template <> struct TypeCheck<cl_double2> { static bool val(cl_device_id device) { return double_ok(device); } };
template <> struct TypeCheck<subgroups::cl_double3> { static bool val(cl_device_id device) { return double_ok(device); } };
template <> struct TypeCheck<cl_double4> { static bool val(cl_device_id device) { return double_ok(device); } };
template <> struct TypeCheck<cl_double8> { static bool val(cl_device_id device) { return double_ok(device); } };
template <> struct TypeCheck<cl_double16> { static bool val(cl_device_id device) { return double_ok(device); } };



// Run a test kernel to compute the result of a built-in on an input
static int
run_kernel(cl_context context, cl_command_queue queue, cl_kernel kernel, size_t global, size_t local,
           void *idata, size_t isize, void *mdata, size_t msize,
       void *odata, size_t osize, size_t tsize=0)
{
    clMemWrapper in;
    clMemWrapper xy;
    clMemWrapper out;
    clMemWrapper tmp;
    int error;

    in = clCreateBuffer(context, CL_MEM_READ_ONLY, isize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    xy = clCreateBuffer(context, CL_MEM_WRITE_ONLY, msize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, osize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    if (tsize) {
        tmp = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, tsize, NULL, &error);
        test_error(error, "clCreateBuffer failed");
    }

    error = clSetKernelArg(kernel, 0, sizeof(in), (void *)&in);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 1, sizeof(xy), (void *)&xy);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 2, sizeof(out), (void *)&out);
    test_error(error, "clSetKernelArg failed");

    if (tsize) {
        error = clSetKernelArg(kernel, 3, sizeof(tmp), (void *)&tmp);
        test_error(error, "clSetKernelArg failed");
    }

    error = clEnqueueWriteBuffer(queue, in, CL_FALSE, 0, isize, idata, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    error = clEnqueueWriteBuffer(queue, xy, CL_FALSE, 0, msize, mdata, 0, NULL, NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, xy, CL_FALSE, 0, msize, mdata, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clEnqueueReadBuffer(queue, out, CL_FALSE, 0, osize, odata, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    return error;
}

// Driver for testing a single built in function
template <typename Ty, typename Fns, size_t GSIZE, size_t LSIZE, size_t TSIZE=0>
struct test {
    static int
    run(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements, const char *kname, const char *src, int dynscl = 0, std::vector<std::string> required_extensions = {})
    {
        size_t tmp;
        int error;
        int subgroup_size, num_subgroups;
        size_t realSize;
        size_t global;
        size_t local;
        const char *kstrings[3];
        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_platform_id platform;
        cl_int sgmap[4*GSIZE];
        Ty mapin[LSIZE];
        Ty mapout[LSIZE];

    // Make sure a test of type Ty is supported by the device
        std::string build_kernel_code = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";

        if (!TypeCheck<Ty>::val(device)) {
            log_info("Data type not supported : %s\n", TypeName<Ty>::val());
            return 0;
        }
        else {
            if (strstr(TypeDef<Ty>::val(), "double")) {
                build_kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
            }
            else if (strstr(TypeDef<Ty>::val(), "half")) {
                build_kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp16: enable\n";
            }
        }

        for (std::string extension : required_extensions) {
            if (!is_extension_available(device, extension.c_str())) {
                log_info("The extension %s not supported on this device. SKIP testing - kernel %s data type %s\n", extension.c_str(), kname, TypeName<Ty>::val());
                return 0;
            }
            build_kernel_code += "#pragma OPENCL EXTENSION " + extension + ": enable\n";
        }
        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), (void *)&platform, NULL);
        test_error(error, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");

        build_kernel_code += "#define NON_UNIFORM "+ std::to_string(NON_UNIFORM) +" \n";
        build_kernel_code += "#define XY(M,I) M[I].x = get_sub_group_local_id(); M[I].y = get_sub_group_id();\n";
        kstrings[0] = build_kernel_code.c_str();
        kstrings[1] = TypeDef<Ty>::val();
        kstrings[2] = src;
        error = create_single_kernel_helper_with_build_options(context, &program, &kernel, 3, kstrings, kname, "-cl-std=CL2.0");
        if (error != 0)
            return error;

        // Determine some local dimensions to use for the test.
        global = GSIZE;
        error = get_max_common_work_group_size(context, kernel, GSIZE, &local);
        test_error(error, "get_max_common_work_group_size failed");

        // Limit it a bit so we have muliple work groups
        // Ideally this will still be large enough to give us multiple subgroups
        if (local > LSIZE)
            local = LSIZE;

    // Get the sub group info
        clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfoKHR_ptr;
        clGetKernelSubGroupInfoKHR_ptr = (clGetKernelSubGroupInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(platform,
        "clGetKernelSubGroupInfoKHR");
        if (clGetKernelSubGroupInfoKHR_ptr == NULL) {
            log_error("ERROR: clGetKernelSubGroupInfoKHR function not available");
            return -1;
        }

        error = clGetKernelSubGroupInfoKHR_ptr(kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                                           sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        test_error(error, "clGetKernelSubGroupInfoKHR failed for CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR");
        subgroup_size = (int)tmp;

        error = clGetKernelSubGroupInfoKHR_ptr(kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
                                           sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        test_error(error, "clGetKernelSubGroupInfoKHR failed for CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR");
        num_subgroups = (int)tmp;

    // Make sure the number of sub groups is what we expect
        if (num_subgroups != (local + subgroup_size - 1)/ subgroup_size) {
            log_error("ERROR: unexpected number of subgroups (%d) returned by clGetKernelSubGroupInfoKHR\n", num_subgroups);
            return -1;
        }

        std::vector<Ty> idata;
        std::vector<Ty> odata;
        size_t input_array_size = GSIZE;
        size_t output_array_size = GSIZE;

        if (dynscl != 0) {
          input_array_size = (int)global / (int)local * num_subgroups * dynscl;
          output_array_size = (int)global / (int)local * dynscl;
        }

        idata.resize(input_array_size);
        odata.resize(output_array_size);

    // Run the kernel once on zeroes to get the map
    memset(&idata[0], 0, input_array_size * sizeof(Ty));
        error = run_kernel(context, queue, kernel, global, local,
                           &idata[0], input_array_size * sizeof(Ty),
               sgmap, global*sizeof(cl_int4),
               &odata[0], output_array_size * sizeof(Ty),
               TSIZE*sizeof(Ty));
    if (error)
        return error;

    // Generate the desired input for the kernel
        Fns::gen(&idata[0], mapin, sgmap, subgroup_size, (int)local, (int)global / (int)local);

        error = run_kernel(context, queue, kernel, global, local,
                           &idata[0], input_array_size * sizeof(Ty),
               sgmap, global*sizeof(cl_int4),
               &odata[0], output_array_size * sizeof(Ty),
               TSIZE*sizeof(Ty));
    if (error)
        return error;


    // Check the result
    return Fns::chk(&idata[0], &odata[0], mapin, mapout, sgmap, subgroup_size, (int)local, (int)global / (int)local);
    }
};

#endif

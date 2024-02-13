//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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

#pragma once
#include <CL/cl.h>

#if defined(_MSC_VER) || defined(_WIN32)
#define PACKED(__STRUCT__) __pragma(pack(push, 1)) __STRUCT__ __pragma(pack(pop))
#elif defined(__GNUC__) || defined(__clang__)
#define PACKED(__STRUCT__) __STRUCT__ __attribute__((packed))
#endif

template<typename T, int n>
inline bool isVectorNotEqual(const T &lhs, const T &rhs)
{
    bool result = false;
    for (int i = 0; !result && i < n; i++) {
        result |= lhs.s[i] != rhs.s[i];
    }
    return result;
}

#define VEC_NOT_EQ_FUNC(TYPE, N)                                    \
    inline bool operator!=(const TYPE##N &lhs, const TYPE##N &rhs)  \
    {                                                               \
        return isVectorNotEqual<TYPE##N, N>(lhs, rhs);                    \
    }                                                               \

VEC_NOT_EQ_FUNC(cl_int, 2)
VEC_NOT_EQ_FUNC(cl_int, 4)
VEC_NOT_EQ_FUNC(cl_uint, 4)
VEC_NOT_EQ_FUNC(cl_float, 2)
VEC_NOT_EQ_FUNC(cl_float, 4)
VEC_NOT_EQ_FUNC(cl_double, 2)
VEC_NOT_EQ_FUNC(cl_double, 4)
VEC_NOT_EQ_FUNC(cl_half, 2)
VEC_NOT_EQ_FUNC(cl_half, 4)

template<typename T>
bool isNotEqual(const T &lhs, const T &rhs)
{
    return lhs != rhs;
}

// Can replace the following with tuples if c++11 can be used
template<typename T>
struct AbstractStruct1
{
    T val;
};

template<typename T>
inline bool operator != (const AbstractStruct1<T> &lhs, const AbstractStruct1<T> &rhs)
{
    return lhs.val != rhs.val;
}

template<typename T0, typename T1>
struct AbstractStruct2
{
    T0 val0;
    T1 val1;
};


template<typename T0, typename T1>
inline bool operator != (const AbstractStruct2<T0, T1> &lhs,
                         const AbstractStruct2<T0, T1> &rhs)
{
    return lhs.val0 != rhs.val0 || lhs.val1 != rhs.val1;
}


template<typename T> struct is_double { static const bool value = false; };
template<> struct  is_double<cl_double> { static const bool value = true; };
template<> struct  is_double<cl_double2> { static const bool value = true; };

template<typename T>
T genrandReal(RandomSeed &seed)
{
    return genrand_real1(seed);
}

// Longer-term this could be refactored out and replace random_float():
template <typename T> T genrandReal_range(T low, T high, RandomSeed &seed)
{
    T t = genrand_real1(seed);
    return (1.0 - t) * low + t * high;
}

template<typename T, int N>
T genrandRealVec(RandomSeed &seed)
{
    T res;
    for (int i = 0; i < N; i++) {
        res.s[i] = genrand_real1(seed);
    }
    return res;
}

#define GENRAND_REAL_FUNC(TYPE, N)                                      \
    template<> inline TYPE##N genrandReal<TYPE##N>(RandomSeed &seed)    \
    {                                                                   \
        return genrandRealVec<TYPE##N, N>(seed);                        \
    }                                                                   \

GENRAND_REAL_FUNC(cl_float, 2)
GENRAND_REAL_FUNC(cl_float, 4)
GENRAND_REAL_FUNC(cl_double, 2)
GENRAND_REAL_FUNC(cl_double, 4)
GENRAND_REAL_FUNC(cl_half, 2)
GENRAND_REAL_FUNC(cl_half, 4)
GENRAND_REAL_FUNC(cl_half, 8)

template<> inline cl_half genrandReal<cl_half>(RandomSeed &seed)
{
    return (cl_half)(genrand_int32(seed) % 2048);
}

template<typename T>
T genrand(RandomSeed &seed)
{
    return genrandReal<T>(seed);
}

template<> inline cl_int genrand<cl_int>(RandomSeed &seed)
{
    return genrand_int32(seed);
}

template<> inline cl_long genrand<cl_long>(RandomSeed &seed)
{
    return genrand_int32(seed);
}

template<> inline cl_short genrand<cl_short>(RandomSeed &seed)
{
    return genrand_int32(seed);
}

#define GENRAND_INT_VEC(T, N)                               \
    template<> inline T##N genrand<T##N>(RandomSeed &seed)  \
    {                                                       \
        T##N res;                                           \
        for (int i = 0; i < N; i++) {                       \
            res.s[i] = (T)genrand_int32(seed);              \
        }                                                   \
        return res;                                         \
    }                                                       \

GENRAND_INT_VEC(cl_int, 4)
GENRAND_INT_VEC(cl_uint, 4)
GENRAND_INT_VEC(cl_long, 2)
GENRAND_INT_VEC(cl_char, 16)

template<typename Tv>
Tv negOp(Tv in)
{
    return -in;
}

inline cl_half negOpHalf(cl_half v) { return v ^ 0x8000; }

template<typename Tv>
Tv notOp(Tv in)
{
    return ~in;
}

template<typename Tv, int N>
Tv negOpVec(Tv in)
{
    Tv out;
    for (int i = 0; i < N; i++) {
        out.s[i] = -in.s[i];
    }
    return out;
}

template<typename Tv, int N>
Tv notOpVec(Tv in)
{
    Tv out;
    for (int i = 0; i < N; i++) {
        out.s[i] = ~in.s[i];
    }
    return out;
}

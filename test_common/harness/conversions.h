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
#ifndef _conversions_h
#define _conversions_h

#include "compat.h"

#include "errorHelpers.h"
#include "mt19937.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <CL/cl_half.h>

#include <vector>
#include <unordered_set>
#include <cstring>
#include <mutex>

/* Note: the next three all have to match in size and order!! */

enum ExplicitTypes
{
    kBool = 0,
    kChar,
    kUChar,
    kUnsignedChar,
    kShort,
    kUShort,
    kUnsignedShort,
    kInt,
    kUInt,
    kUnsignedInt,
    kLong,
    kULong,
    kUnsignedLong,
    kFloat,
    kHalf,
    kDouble,
    kNumExplicitTypes
};

typedef enum ExplicitTypes ExplicitType;

enum RoundingTypes
{
    kRoundToEven = 0,
    kRoundToZero,
    kRoundToPosInf,
    kRoundToNegInf,
    kRoundToNearest,

    kNumRoundingTypes,

    kDefaultRoundingType = kRoundToNearest
};

typedef enum RoundingTypes RoundingType;

extern void print_type_to_string(ExplicitType type, void *data, char *string);
extern size_t get_explicit_type_size(ExplicitType type);
extern const char *get_explicit_type_name(ExplicitType type);
extern void convert_explicit_value(void *inRaw, void *outRaw,
                                   ExplicitType inType, bool saturate,
                                   RoundingType roundType,
                                   cl_half_rounding_mode halfRoundingMode,
                                   ExplicitType outType);

extern void generate_random_data(ExplicitType type, size_t count, MTdata d,
                                 void *outData);
extern void *create_random_data(ExplicitType type, MTdata d, size_t count);

extern cl_long read_upscale_signed(void *inRaw, ExplicitType inType);
extern cl_ulong read_upscale_unsigned(void *inRaw, ExplicitType inType);
extern float read_as_float(void *inRaw, ExplicitType inType);

extern float get_random_float(float low, float high, MTdata d);
extern double get_random_double(double low, double high, MTdata d);
extern float any_float(MTdata d);
extern double any_double(MTdata d);

extern int random_in_range(int minV, int maxV, MTdata d);

size_t get_random_size_t(size_t low, size_t high, MTdata d);

// Note: though this takes a double, this is for use with single precision tests
static inline int IsFloatSubnormal(float x)
{
#if 2 == FLT_RADIX
    // Do this in integer to avoid problems with FTZ behavior
    union {
        float d;
        uint32_t u;
    } u;
    u.d = fabsf(x);
    return (u.u - 1) < 0x007fffffU;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware --
    // will fail if you flush subnormals to zero
    return fabs(x) < (double)FLT_MIN && x != 0.0;
#endif
}

static inline int IsDoubleSubnormal(double x)
{
#if 2 == FLT_RADIX
    // Do this in integer to avoid problems with FTZ behavior
    union {
        double d;
        uint64_t u;
    } u;
    u.d = fabs(x);
    return (u.u - 1) < 0x000fffffffffffffULL;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware --
    // will fail if you flush subnormals to zero
    return fabs(x) < (double)DBL_MIN && x != 0.0;
#endif
}

static inline int IsHalfSubnormal(cl_half x)
{
    // this relies on interger overflow to exclude 0 as a subnormal
    return ((x & 0x7fffU) - 1U) < 0x03ffU;
}

extern const std::vector<float> specialValuesFloat;
extern const std::vector<double> specialValuesDouble;

template <typename InType, typename OutType> OutType bitcast(InType in)
{
    OutType out;
    assert(sizeof(InType) == sizeof(OutType));
    std::memcpy(&out, &in, sizeof(InType));
    return out;
}

// Adds a value to a vector if it is not already present in the set.
// The set is used to check for duplicates, and the vector is used to
// store the unique values in the order they were added.
template <typename Type, typename IntegerType>
void push_unique(std::vector<Type> &vec, std::unordered_set<IntegerType> &set,
                 Type val)
{
    IntegerType set_val;
    if constexpr (std::is_same<Type, IntegerType>::value)
    {
        set_val = val;
    }
    else
    {
        set_val = bitcast<Type, IntegerType>(val);
    }
    if (set.count(set_val) == 0)
    {
        set.insert(set_val);
        vec.push_back(val);
    }
};

template <typename T>
std::vector<T> GetIntSpecialValues(std::recursive_mutex &mutex,
                                   int offset_limit = 3, bool wimpy = false)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    static std::vector<T> vec;
    if (!vec.empty())
    {
        return vec;
    }
    std::unordered_set<T> set;
    const T sign = static_cast<T>(1) << (sizeof(T) * 8 - 1);

    // For each value, add it and its neighbors to the vector. And for each
    // of those values, add the bitwise not and the XOR with the sign to the
    // vector.
    std::vector<int> offsets = { 0 };
    if (!wimpy)
        for (int i = 1; i <= offset_limit; i++)
        {
            offsets.push_back(i);
            offsets.push_back(-i);
        }
    auto push = [&set, &sign, &wimpy, &offsets](T val) {
        for (const int offset : offsets)
        {
            T v = val + offset;
            push_unique<T, T>(vec, set, v);
            push_unique<T, T>(vec, set, ~v);
            v = v ^ sign;
            push_unique<T, T>(vec, set, v);
            if (!wimpy) push_unique<T, T>(vec, set, ~v);
        }
    };

    // Add all the values close to 0, MIN, and MAX.
    push((T)0);
    push((T)1);

    // Add powers of 2, 3, 5, 7, 10.
    std::vector<T> wimpy_powers = { 2 };
    std::vector<T> all_powers = { 2, 3, 5, 7, 10 };
    for (const T base : wimpy ? wimpy_powers : all_powers)
    {
        T val = base;
        T next = val * base;
        do
        {
            push(val);
            val = next;
            next *= base;
        } while (next > val && next < sign);
    }

    // Generate patterns and masks.
    // For uint16_t:
    // patterns: 0x1111, 0x2222, ..., 0xEEEE
    // masks: 0x0F0F, 0xF0F0, 0x00FF, 0xFF00, 0xFFFF
    std::vector<T> patterns;
    std::vector<T> masks;
    for (T i = 1; i < 15; i += (wimpy ? 2 : 1))
    {
        T pattern = i;
        for (unsigned j = 0; j < sizeof(T) * 2; j++)
        {
            pattern = pattern << 4 | i;
        }
        patterns.push_back(pattern);
    }
    for (unsigned chunk_size = 4; chunk_size < (sizeof(T) * 8); chunk_size *= 2)
    {
        T mask = (static_cast<T>(1) << chunk_size) - 1;
        for (unsigned shift = chunk_size * 2; shift <= ((sizeof(T) * 8) / 2);
             shift *= 2)
        {
            mask |= (mask << shift);
        }
        masks.push_back(mask);
        masks.push_back(~mask);
    }
    masks.push_back(~static_cast<T>(0));

    // Add all the combinations of patterns and masks.
    for (const auto &pattern : patterns)
    {
        for (const auto &mask : masks)
        {
            push(pattern & mask);
        }
    }
    return vec;
}

template <typename InType, typename InIntegerType, typename OutType>
std::vector<InType> GetFpSpecialValues(std::recursive_mutex &mutex)
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    static std::vector<InType> vec;
    if (!vec.empty())
    {
        return vec;
    }
    std::unordered_set<InIntegerType> set;

    // Adds a value and its neighbors (values +/- 1 ULP away) to the vector.
    auto push = [&set](InType val) {
        push_unique<InType, InIntegerType>(vec, set, val);
        push_unique<InType, InIntegerType>(
            vec, set,
            bitcast<InIntegerType, InType>(bitcast<InType, InIntegerType>(val)
                                           + 1));
        push_unique<InType, InIntegerType>(
            vec, set,
            bitcast<InIntegerType, InType>(bitcast<InType, InIntegerType>(val)
                                           - 1));
    };

    // Add all the values from the special values list.
    if constexpr (std::is_same<InType, cl_float>::value)
    {
        for (const auto &val : specialValuesFloat)
        {
            push(val);
        }
    }
    else if constexpr (std::is_same<InType, cl_double>::value)
    {
        for (const auto &val : specialValuesDouble)
        {
            push(val);
        }
    }

    // Add values to the input test set that correspond to values used as
    // input for the output type.
    if constexpr (std::is_same<OutType, cl_uchar>::value)
    {
        for (uint32_t i = 0; i < 256; i++)
        {
            push(static_cast<InType>(i));
        }
    }
    else if constexpr (std::is_same<OutType, cl_char>::value)
    {
        for (int32_t i = -127; i <= 128; i++)
        {
            push(static_cast<InType>(i));
        }
    }
    else if constexpr (std::is_same<OutType, cl_ushort>::value)
    {
        for (uint32_t i = 0; i < (64 * 1024); i++)
        {
            push(static_cast<InType>(i));
        }
    }
    else if constexpr (std::is_same<OutType, cl_short>::value)
    {
        for (int32_t i = -(64 * 1024 / 2 - 1); i <= (64 * 1024 / 2); i++)
        {
            push(static_cast<InType>(i));
        }
    }
    else if constexpr (std::is_same<OutType, cl_uint>::value)
    {
        for (const auto val : GetIntSpecialValues<uint32_t>(mutex))
        {
            push(static_cast<InType>(val));
        }
    }
    else if constexpr (std::is_same<OutType, cl_int>::value)
    {
        for (const auto val : GetIntSpecialValues<uint32_t>(mutex))
        {
            push(static_cast<InType>(bitcast<uint32_t, OutType>(val)));
        }
    }
    else if constexpr (std::is_same<OutType, cl_ulong>::value)
    {
        for (const auto val : GetIntSpecialValues<uint64_t>(mutex))
        {
            push(static_cast<InType>(val));
        }
    }
    else if constexpr (std::is_same<OutType, cl_long>::value)
    {
        for (const auto val : GetIntSpecialValues<uint64_t>(mutex))
        {
            push(static_cast<InType>(bitcast<uint64_t, OutType>(val)));
        }
    }
    else if constexpr (std::is_same<OutType, cl_half>::value)
    {
        for (uint32_t i = 0; i < (64 * 1024); i++)
        {
            push(static_cast<InType>(bitcast<uint16_t, OutType>(i)));
        }
    }
    else if constexpr (std::is_same<OutType, cl_float>::value
                       && !std::is_same<InType, cl_float>::value)
    {
        for (const auto &val : specialValuesFloat)
        {
            push(static_cast<InType>(val));
        }
    }
    else if constexpr (std::is_same<OutType, cl_double>::value
                       && !std::is_same<InType, cl_double>::value)
    {
        for (const auto &val : specialValuesDouble)
        {
            push(static_cast<InType>(val));
        }
    }
    return vec;
}

#endif // _conversions_h

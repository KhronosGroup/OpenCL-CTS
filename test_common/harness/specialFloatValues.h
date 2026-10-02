// Copyright (c) 2026 The Khronos Group Inc.
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

#ifndef SPECIAL_FLOAT_VALUES_H
#define SPECIAL_FLOAT_VALUES_H

#include <CL/cl.h>
#include <CL/cl_half.h>

#include <limits>
#include <type_traits>

namespace special_float_values {

// A type-independent representation of an edge-case value.
struct AbstractValue
{
    enum class Kind
    {
        PosZero,
        NegZero,
        PosInf,
        NegInf,
        NaN,
        Finite,
        Int,
        SmallestPosDenorm,
        SmallestNegDenorm,
    } kind;

    double d = 0.0;
    int i = 0;

    // Return whether the value is supported by the capabilities.
    bool isSupportedBy(cl_device_fp_config capabilities) const
    {
        switch (kind)
        {
            case Kind::PosInf:
            case Kind::NegInf:
            case Kind::NaN: return (capabilities & CL_FP_INF_NAN) != 0;
            case Kind::SmallestPosDenorm:
            case Kind::SmallestNegDenorm:
                return (capabilities & CL_FP_DENORM) != 0;
            default: return true;
        }
    }

    // Materialize the value as T.
    template <typename T> T toValue() const;
};

constexpr AbstractValue pos_zero() { return { AbstractValue::Kind::PosZero }; }
constexpr AbstractValue neg_zero() { return { AbstractValue::Kind::NegZero }; }
constexpr AbstractValue pos_inf() { return { AbstractValue::Kind::PosInf }; }
constexpr AbstractValue neg_inf() { return { AbstractValue::Kind::NegInf }; }
constexpr AbstractValue nan() { return { AbstractValue::Kind::NaN }; }
constexpr AbstractValue finite(double value)
{
    return { AbstractValue::Kind::Finite, value };
}
constexpr AbstractValue integer(int value)
{
    return { AbstractValue::Kind::Int, 0.0, value };
}
constexpr AbstractValue smallest_pos_denorm()
{
    return { AbstractValue::Kind::SmallestPosDenorm };
}
constexpr AbstractValue smallest_neg_denorm()
{
    return { AbstractValue::Kind::SmallestNegDenorm };
}

template <typename T> inline T AbstractValue::toValue() const
{
    // Encode half values directly to avoid depending on host floating-point
    // conversion behavior.
    if constexpr (std::is_same_v<T, cl_half>)
    {
        cl_half bits = 0;
        switch (kind)
        {
            case AbstractValue::Kind::PosZero: bits = 0x0000; break;
            case AbstractValue::Kind::NegZero: bits = 0x8000; break;
            case AbstractValue::Kind::PosInf: bits = 0x7c00; break;
            case AbstractValue::Kind::NegInf: bits = 0xfc00; break;
            case AbstractValue::Kind::NaN: bits = 0x7e00; break;
            case AbstractValue::Kind::Finite:
                bits = cl_half_from_float(static_cast<float>(d), CL_HALF_RTE);
                break;
            case AbstractValue::Kind::Int:
                bits = cl_half_from_float(static_cast<float>(i), CL_HALF_RTE);
                break;
            case AbstractValue::Kind::SmallestPosDenorm: bits = 0x0001; break;
            case AbstractValue::Kind::SmallestNegDenorm: bits = 0x8001; break;
        }
        return bits;
    }
    else
    {
        switch (kind)
        {
            case AbstractValue::Kind::PosZero: return T(0);
            case AbstractValue::Kind::NegZero: return -T(0);
            case AbstractValue::Kind::PosInf:
                return std::numeric_limits<T>::infinity();
            case AbstractValue::Kind::NegInf:
                return -std::numeric_limits<T>::infinity();
            case AbstractValue::Kind::NaN:
                return std::numeric_limits<T>::quiet_NaN();
            case AbstractValue::Kind::Finite: return static_cast<T>(d);
            case AbstractValue::Kind::Int: return static_cast<T>(i);
            case AbstractValue::Kind::SmallestPosDenorm:
                return std::numeric_limits<T>::denorm_min();
            case AbstractValue::Kind::SmallestNegDenorm:
                return -std::numeric_limits<T>::denorm_min();
        }
    }

    return T{};
}

} // namespace special_float_values

#endif // SPECIAL_FLOAT_VALUES_H

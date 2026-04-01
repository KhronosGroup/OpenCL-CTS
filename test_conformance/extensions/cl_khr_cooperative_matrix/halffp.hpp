// Copyright (c) 2024-2026 The Khronos Group Inc.
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

#ifndef HALFFP_HPP
#define HALFFP_HPP

#include "CL/cl_half.h"

#include <sstream>
#include <type_traits>

struct HalfFP final
{
    HalfFP() = default;
    HalfFP(float f)
    {
        // This class must be standard layout so that a pointer to it is
        // equivalent to a pointer to its first member variable (the uint16_t
        // data). If this assert trips, then the compiler being used is not
        // C++11 compliant.
        static_assert(std::is_standard_layout<HalfFP>::value,
                      "Bad C++11 implementation.");
        static_assert(sizeof(HalfFP) == sizeof(uint16_t),
                      "Bad C++11 implementation.");
        data = cl_half_from_float(f, CL_HALF_RTE);
    }

    operator float() const { return cl_half_to_float(data); }

    HalfFP operator-() const
    {
        HalfFP copy = *this;
        copy.data ^= 0x8000;
        return copy;
    }

    HalfFP operator+(const HalfFP &o) const
    {
        return HalfFP(static_cast<float>(*this) + o);
    }

    HalfFP operator-(const HalfFP &o) const
    {
        return HalfFP(static_cast<float>(*this) - o);
    }

    HalfFP operator/(const HalfFP &o) const
    {
        return HalfFP(static_cast<float>(*this) / static_cast<float>(o));
    }

    bool operator!=(const HalfFP &o) const
    {
        return static_cast<float>(data) != static_cast<float>(o.data);
    }

    bool operator==(const HalfFP &o) const { return !(*this != o); }

    HalfFP &operator+=(float f)
    {
        (*this) = HalfFP(static_cast<float>(*this) + f);
        return *this;
    }

    std::ostream &operator<<(std::ostream &out) const
    {
        return out << static_cast<float>(*this);
    }

    uint16_t data = 0;
};

#endif // HALFFP_HPP

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
#ifndef TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP
#define TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <cmath>

namespace detail 
{

template<class INT_TYPE>
inline int clz(INT_TYPE x)
{
    int count = 0;
    if(std::is_unsigned<INT_TYPE>::value)
    {
        cl_ulong value = x;
        value <<= 8 * sizeof(value) - (8 * sizeof(x));
        for(count = 0; 0 == (value & (CL_LONG_MIN)); count++)
        {
            value <<= 1;
        }
    }
    else
    {            
        cl_long value = x;
        value <<= 8 * sizeof(value) - (8 * sizeof(x));
        for(count = 0; 0 == (value & (CL_LONG_MIN)); count++)
        {
            value <<= 1;
        }
    }
    return count;
}

} // namespace detail 

inline cl_float half2float(cl_half us)
{
    uint32_t u = us;
    uint32_t sign = (u << 16) & 0x80000000;
    int32_t exponent = (u & 0x7c00) >> 10;
    uint32_t mantissa = (u & 0x03ff) << 13;
    union{ cl_uint u; cl_float f;}uu;

    if( exponent == 0 )
    {
        if( mantissa == 0 )
            return sign ? -0.0f : 0.0f;

        int shift = detail::clz( mantissa ) - 8;
        exponent -= shift-1;
        mantissa <<= shift;
        mantissa &= 0x007fffff;
    }
    else
        if( exponent == 31)
        {
            uu.u = mantissa | sign;
            if( mantissa )
                uu.u |= 0x7fc00000;
            else
                uu.u |= 0x7f800000;

            return uu.f;
        }

    exponent += 127 - 15;
    exponent <<= 23;

    exponent |= mantissa;
    uu.u = exponent | sign;

    return uu.f;
}

inline cl_ushort float2half_rte(cl_float f)
{
    union{ cl_float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    cl_float x = fabsf(f);

    //Nan
    if( x != x )
    {
        u.u >>= (24-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( x >= MAKE_HEX_FLOAT(0x1.ffep15f, 0x1ffeL, 3) )
        return 0x7c00 | sign;

    // underflow
    if( x <= MAKE_HEX_FLOAT(0x1.0p-25f, 0x1L, -25) )
        return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

    // very small
    if( x < MAKE_HEX_FLOAT(0x1.8p-24f, 0x18L, -28) )
        return sign | 1;

    // half denormal
    if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
    {
        u.f = x * MAKE_HEX_FLOAT(0x1.0p-125f, 0x1L, -125);
        return sign | u.u;
    }

    u.f *= MAKE_HEX_FLOAT(0x1.0p13f, 0x1L, 13);
    u.u &= 0x7f800000;
    x += u.f;
    u.f = x - u.f;
    u.f *= MAKE_HEX_FLOAT(0x1.0p-112f, 0x1L, -112);

    return (u.u >> (24-11)) | sign;
}

#endif // TEST_CONFORMANCE_CLCPP_HALF_UTILS_HPP

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
#include <stdint.h>
#include <math.h>
#include "fplib.h"

#define FLT_MANT_DIG    24
#define as_float(x)     (*((float *)(&x)))
#define as_long(x)      (*((int64_t *)(&x)))

static uint32_t clz(uint64_t value)
{
    uint32_t num_zeros;

    for( num_zeros = 0; num_zeros < (sizeof(uint64_t)*8); num_zeros++)
    {
        if(0x8000000000000000 & (value << num_zeros))
            break;
    }
    return num_zeros;
}

float qcom_s64_2_f32(int64_t data, bool sat, roundingMode rnd)
{
    switch (rnd) {
        case qcomRTZ: {
            int sign = 0;
            if (!data)
                return 0.0f;
            if (data < 0){
                data = - data;
                sign = 1;
            }
            uint32_t    exponent   = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int         mantShift  = 40 - clz(data);
            uint32_t    mantissa;
            if (mantShift >= 0)
                mantissa = (uint32_t)((uint64_t)data >> mantShift);
            else
                mantissa = (uint32_t)((uint64_t)data << -mantShift);
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            if (sign)
                result |= 0x80000000;
            return as_float(result);
            break;
        }
        case qcomRTE: return (float)(data); break;
        case qcomRTP: {
            int         sign    = 0;
            int         inExact = 0;
            uint32_t    f       = 0xdf000000;
            if (!data)
                return 0.0f;
            if (data == 0x8000000000000000)
                return as_float(f);
            if (data < 0){
                data = - data;
                sign = 1;
            }
            uint32_t    exponent    = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int         mantShift   = 40 - clz(data);
            uint32_t mantissa;
            if (mantShift >= 0){
                uint64_t temp = (uint64_t)data >> mantShift;
                uint64_t mask = (1 << mantShift) - 1;
                if ((temp << mantShift) != data)
                    inExact = 1;
                mantissa = (uint32_t)temp;
            }
            else
            {
                mantissa = (uint32_t)((uint64_t)data << -mantShift);
            }
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            if (sign)
                result |= 0x80000000;
            if (sign)
                return as_float(result); // for negative inputs return rtz results
            else
            {
                if(inExact)
                { // for positive inputs return higher next fp
                    uint32_t high_float = 0x7f7fffff;
                    return nextafterf(as_float(result), as_float(high_float)); // could be simplified with some inc and carry operation
                }
                else
                    return as_float(result);
            }
        }
        break;
        case qcomRTN: {
            int sign = 0;
            int inExact = 0;
            uint32_t f = 0xdf000000;
            if (!data)
                return 0.0f;
            if (data == 0x8000000000000000)
                return as_float(f);
            if (data < 0){
                data = - data;
                sign = 1;
            }
            uint32_t    exponent    = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int         mantShift   = 40 - clz(data);
            uint32_t    mantissa;
            if (mantShift >= 0){
                uint64_t temp = (uint64_t)data >> mantShift;
                uint64_t mask = (1 << mantShift) - 1;
                if (temp << mantShift != data)
                    inExact = 1;
                mantissa = (uint32_t)temp;
            }
            else
                mantissa = (uint32_t)((uint64_t)data << -mantShift);
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            if (sign)
                result |= 0x80000000;
            if (!sign)
                return as_float(result); // for positive inputs return RTZ result
            else{
                if(inExact){ // for negative inputs find the lower next fp number
                    uint32_t low_float = 0xff7fffff;
                    return nextafterf(as_float(result), as_float(low_float)); // could be simplified with some inc and carry operation
                }
                else
                    return as_float(result);
            }
        }
    }
    return 0.0f;
}

float qcom_u64_2_f32(uint64_t data, bool sat, roundingMode rnd)
{
    switch (rnd) {
        case qcomRTZ: {
            if (!data)
                return 0.0f;
            uint32_t    exponent    = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int         mantShift   = 40 - clz(data);
            uint32_t    mantissa;
            if (mantShift >= 0)
                mantissa = (uint32_t)(data >> mantShift);
            else
                mantissa = (uint32_t)(data << -mantShift);
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            return as_float(result);
            break;
        }
        case qcomRTE: return (float)(data); break;
        case qcomRTP: {
            int inExact = 0;
            if (!data)
                return 0.0f;
            uint32_t    exponent    = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int         mantShift   = 40 - clz(data);
            uint32_t    mantissa;
            if (mantShift >= 0){
                uint64_t temp = data >> mantShift;
                uint64_t mask = (1 << mantShift) - 1;
                if (temp << mantShift != data)
                    inExact = 1;
                mantissa = (uint32_t)temp;
            }
            else
                mantissa = (uint32_t)(data << -mantShift);
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            if(inExact){ // for positive inputs return higher next fp
                uint32_t high_float = 0x7f7fffff;
                return nextafterf(as_float(result), as_float(high_float)); // could be simplified with some inc and carry operation
            }
            else
                return as_float(result);
        }
        case qcomRTN: {
            int inExact = 0;
            if (!data)
                return 0.0f;
            uint32_t  exponent    = (127 + 64 - clz(data) - 1) << (FLT_MANT_DIG - 1); //add 1 for the implied 1.0 in normalized fp32 numbers
            int       mantShift   = 40 - clz(data);
            uint32_t  mantissa;
            if (mantShift >= 0){
                uint64_t temp = (uint64_t)data >> mantShift;
                uint64_t mask = (1 << mantShift) - 1;
                if (temp << mantShift != data)
                    inExact = 1;
                mantissa = (uint32_t)temp;
            }
            else
                mantissa = (uint32_t)((uint64_t)data << -mantShift);
            mantissa &= 0x7fffff;//mask off the leading 1

            uint32_t result = exponent | mantissa;
            return as_float(result); // for positive inputs return RTZ result
        }
    }
    return 0.0f;
}

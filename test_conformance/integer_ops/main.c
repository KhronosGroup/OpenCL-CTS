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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
    test_integer_clz,
    test_integer_ctz,
    test_integer_hadd,
    test_integer_rhadd,
    test_integer_mul_hi,
    test_integer_rotate,
    test_integer_clamp,
    test_integer_mad_sat,
    test_integer_mad_hi,
    test_integer_min,
    test_integer_max,
    test_integer_upsample,

    test_abs,
    test_absdiff,
    test_add_sat,
    test_sub_sat,

    test_integer_addAssign,
    test_integer_subtractAssign,
    test_integer_multiplyAssign,
    test_integer_divideAssign,
    test_integer_moduloAssign,
    test_integer_andAssign,
    test_integer_orAssign,
    test_integer_exclusiveOrAssign,

    test_unary_ops_increment,
    test_unary_ops_decrement,
    test_unary_ops_full,

    test_intmul24,
    test_intmad24,

    test_long_math,
    test_long_logic,
    test_long_shift,
    test_long_compare,

    test_ulong_math,
    test_ulong_logic,
    test_ulong_shift,
    test_ulong_compare,

    test_int_math,
    test_int_logic,
    test_int_shift,
    test_int_compare,

    test_uint_math,
    test_uint_logic,
    test_uint_shift,
    test_uint_compare,

    test_short_math,
    test_short_logic,
    test_short_shift,
    test_short_compare,

    test_ushort_math,
    test_ushort_logic,
    test_ushort_shift,
    test_ushort_compare,

    test_char_math,
    test_char_logic,
    test_char_shift,
    test_char_compare,

    test_uchar_math,
    test_uchar_logic,
    test_uchar_shift,
    test_uchar_compare,

    test_popcount,


    // Quick
    test_quick_long_math,
    test_quick_long_logic,
    test_quick_long_shift,
    test_quick_long_compare,

    test_quick_ulong_math,
    test_quick_ulong_logic,
    test_quick_ulong_shift,
    test_quick_ulong_compare,

    test_quick_int_math,
    test_quick_int_logic,
    test_quick_int_shift,
    test_quick_int_compare,

    test_quick_uint_math,
    test_quick_uint_logic,
    test_quick_uint_shift,
    test_quick_uint_compare,

    test_quick_short_math,
    test_quick_short_logic,
    test_quick_short_shift,
    test_quick_short_compare,

    test_quick_ushort_math,
    test_quick_ushort_logic,
    test_quick_ushort_shift,
    test_quick_ushort_compare,

    test_quick_char_math,
    test_quick_char_logic,
    test_quick_char_shift,
    test_quick_char_compare,

    test_quick_uchar_math,
    test_quick_uchar_logic,
    test_quick_uchar_shift,
    test_quick_uchar_compare,

    test_vector_scalar_ops,
};


const char    *basefn_names[] = {
    "integer_clz",
    "integer_ctz",
    "integer_hadd",
    "integer_rhadd",
    "integer_mul_hi",
    "integer_rotate",
    "integer_clamp",
    "integer_mad_sat",
    "integer_mad_hi",
    "integer_min",
    "integer_max",
    "integer_upsample",

    "integer_abs",
    "integer_abs_diff",
    "integer_add_sat",
    "integer_sub_sat",

    "integer_addAssign",
    "integer_subtractAssign",
    "integer_multiplyAssign",
    "integer_divideAssign",
    "integer_moduloAssign",
    "integer_andAssign",
    "integer_orAssign",
    "integer_exclusiveOrAssign",

    "unary_ops_increment",
    "unary_ops_decrement",
    "unary_ops_full",

    "integer_mul24",
    "integer_mad24",

    "long_math",
    "long_logic",
    "long_shift",
    "long_compare",

    "ulong_math",
    "ulong_logic",
    "ulong_shift",
    "ulong_compare",

    "int_math",
    "int_logic",
    "int_shift",
    "int_compare",

    "uint_math",
    "uint_logic",
    "uint_shift",
    "uint_compare",

    "short_math",
    "short_logic",
    "short_shift",
    "short_compare",

    "ushort_math",
    "ushort_logic",
    "ushort_shift",
    "ushort_compare",

    "char_math",
    "char_logic",
    "char_shift",
    "char_compare",

    "uchar_math",
    "uchar_logic",
    "uchar_shift",
    "uchar_compare",

    "popcount",

    // Quick
    "quick_long_math",
    "quick_long_logic",
    "quick_long_shift",
    "quick_long_compare",

    "quick_ulong_math",
    "quick_ulong_logic",
    "quick_ulong_shift",
    "quick_ulong_compare",

    "quick_int_math",
    "quick_int_logic",
    "quick_int_shift",
    "quick_int_compare",

    "quick_uint_math",
    "quick_uint_logic",
    "quick_uint_shift",
    "quick_uint_compare",

    "quick_short_math",
    "quick_short_logic",
    "quick_short_shift",
    "quick_short_compare",

    "quick_ushort_math",
    "quick_ushort_logic",
    "quick_ushort_shift",
    "quick_ushort_compare",

    "quick_char_math",
    "quick_char_logic",
    "quick_char_shift",
    "quick_char_compare",

    "quick_uchar_math",
    "quick_uchar_logic",
    "quick_uchar_shift",
    "quick_uchar_compare",

    "vector_scalar",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

void fill_test_values( cl_long *outBufferA, cl_long *outBufferB, size_t numElements, MTdata d )
{
    static const cl_long sUniqueValues[] = { 0x3333333333333333LL, 0x5555555555555555LL, 0x9999999999999999LL, 0xaaaaaaaaaaaaaaaaLL, 0xccccccccccccccccLL,
        0x3030303030303030LL, 0x5050505050505050LL, 0x9090909090909090LL,  0xa0a0a0a0a0a0a0a0LL, 0xc0c0c0c0c0c0c0c0LL, 0xf0f0f0f0f0f0f0f0LL,
        0x0303030303030303LL, 0x0505050505050505LL, 0x0909090909090909LL,  0x0a0a0a0a0a0a0a0aLL, 0x0c0c0c0c0c0c0c0cLL, 0x0f0f0f0f0f0f0f0fLL,
        0x3300330033003300LL, 0x5500550055005500LL, 0x9900990099009900LL,  0xaa00aa00aa00aa00LL, 0xcc00cc00cc00cc00LL, 0xff00ff00ff00ff00LL,
        0x0033003300330033LL, 0x0055005500550055LL, 0x0099009900990099LL,  0x00aa00aa00aa00aaLL, 0x00cc00cc00cc00ccLL, 0x00ff00ff00ff00ffLL,
        0x3333333300000000LL, 0x5555555500000000LL, 0x9999999900000000LL,  0xaaaaaaaa00000000LL, 0xcccccccc00000000LL, 0xffffffff00000000LL,
        0x0000000033333333LL, 0x0000000055555555LL, 0x0000000099999999LL,  0x00000000aaaaaaaaLL, 0x00000000ccccccccLL, 0x00000000ffffffffLL,
        0x3333000000003333LL, 0x5555000000005555LL, 0x9999000000009999LL,  0xaaaa00000000aaaaLL, 0xcccc00000000ccccLL, 0xffff00000000ffffLL};
    static cl_long sSpecialValues[ 128 + 128 + 128 + ( sizeof( sUniqueValues ) / sizeof( sUniqueValues[ 0 ] ) ) ] = { 0 };

    if( sSpecialValues[ 0 ] == 0 )
    {
        // Init the power-of-two special values
        for( size_t i = 0; i < 64; i++ )
        {
            sSpecialValues[ i ] = 1LL << i;
            sSpecialValues[ i + 64 ] = -1LL << i;
            sSpecialValues[ i + 128 ] = sSpecialValues[ i ] - 1;
            sSpecialValues[ i + 128 + 64 ] = sSpecialValues[ i ] - 1;
            sSpecialValues[ i + 256 ] = sSpecialValues[ i ] + 1;
            sSpecialValues[ i + 256 + 64 ] = sSpecialValues[ i ] + 1;
        }
        memcpy( &sSpecialValues[ 128 + 128 + 128 ], sUniqueValues, sizeof( sUniqueValues ) );
    }

    size_t i, aIdx = 0, bIdx = 0;
    size_t numSpecials = sizeof( sSpecialValues ) / sizeof( sSpecialValues[ 0 ] );

    for( i = 0; i < numElements; i++ )
    {
        outBufferA[ i ] = sSpecialValues[ aIdx ];
        outBufferB[ i ] = sSpecialValues[ bIdx ];
        bIdx++;
        if( bIdx == numSpecials )
        {
            bIdx = 0;
            aIdx++;
            if( aIdx == numSpecials )
                break;
        }
    }
    if( i < numElements )
    {
        // Fill remainder with random values
        for( ; i < numElements; i++ )
        {
            int a = (int)genrand_int32(d);
            int b = (int)genrand_int32(d);
            outBufferA[ i ] = ((cl_long)a <<33 | (cl_long)b) ^ ((cl_long)b << 16);

            a = (int)genrand_int32(d);
            b = (int)genrand_int32(d);
            outBufferB[ i ] = ((cl_long)a <<33 | (cl_long)b) ^ ((cl_long)b << 16);
        }
    }
    else if( aIdx < numSpecials )
    {
        log_info( "WARNING: Not enough space to fill all special values for long test! (need %d additional elements)\n", (int)( ( numSpecials - aIdx ) * numSpecials ) );
    }
}



int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false /* image support required */, false /* force no context creation */, 0 );
}





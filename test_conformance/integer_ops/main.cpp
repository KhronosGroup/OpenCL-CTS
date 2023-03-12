//
// Copyright (c) 2017-2022 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST(integer_clz),
    ADD_TEST_VERSION(integer_ctz, Version(2, 0)),
    ADD_TEST(integer_hadd),
    ADD_TEST(integer_rhadd),
    ADD_TEST(integer_mul_hi),
    ADD_TEST(integer_rotate),
    ADD_TEST(integer_clamp),
    ADD_TEST(integer_mad_sat),
    ADD_TEST(integer_mad_hi),
    ADD_TEST(integer_min),
    ADD_TEST(integer_max),
    ADD_TEST(integer_upsample),

    ADD_TEST(integer_abs),
    ADD_TEST(integer_abs_diff),
    ADD_TEST(integer_add_sat),
    ADD_TEST(integer_sub_sat),

    ADD_TEST(integer_addAssign),
    ADD_TEST(integer_subtractAssign),
    ADD_TEST(integer_multiplyAssign),
    ADD_TEST(integer_divideAssign),
    ADD_TEST(integer_moduloAssign),
    ADD_TEST(integer_andAssign),
    ADD_TEST(integer_orAssign),
    ADD_TEST(integer_exclusiveOrAssign),

    ADD_TEST(unary_ops_increment),
    ADD_TEST(unary_ops_decrement),
    ADD_TEST(unary_ops_full),

    ADD_TEST(integer_mul24),
    ADD_TEST(integer_mad24),

    ADD_TEST(extended_bit_ops_extract),
    ADD_TEST(extended_bit_ops_insert),
    ADD_TEST(extended_bit_ops_reverse),

    ADD_TEST(long_math),
    ADD_TEST(long_logic),
    ADD_TEST(long_shift),
    ADD_TEST(long_compare),

    ADD_TEST(ulong_math),
    ADD_TEST(ulong_logic),
    ADD_TEST(ulong_shift),
    ADD_TEST(ulong_compare),

    ADD_TEST(int_math),
    ADD_TEST(int_logic),
    ADD_TEST(int_shift),
    ADD_TEST(int_compare),

    ADD_TEST(uint_math),
    ADD_TEST(uint_logic),
    ADD_TEST(uint_shift),
    ADD_TEST(uint_compare),

    ADD_TEST(short_math),
    ADD_TEST(short_logic),
    ADD_TEST(short_shift),
    ADD_TEST(short_compare),

    ADD_TEST(ushort_math),
    ADD_TEST(ushort_logic),
    ADD_TEST(ushort_shift),
    ADD_TEST(ushort_compare),

    ADD_TEST(char_math),
    ADD_TEST(char_logic),
    ADD_TEST(char_shift),
    ADD_TEST(char_compare),

    ADD_TEST(uchar_math),
    ADD_TEST(uchar_logic),
    ADD_TEST(uchar_shift),
    ADD_TEST(uchar_compare),

    ADD_TEST(popcount),

    // Quick
    ADD_TEST(quick_long_math),
    ADD_TEST(quick_long_logic),
    ADD_TEST(quick_long_shift),
    ADD_TEST(quick_long_compare),

    ADD_TEST(quick_ulong_math),
    ADD_TEST(quick_ulong_logic),
    ADD_TEST(quick_ulong_shift),
    ADD_TEST(quick_ulong_compare),

    ADD_TEST(quick_int_math),
    ADD_TEST(quick_int_logic),
    ADD_TEST(quick_int_shift),
    ADD_TEST(quick_int_compare),

    ADD_TEST(quick_uint_math),
    ADD_TEST(quick_uint_logic),
    ADD_TEST(quick_uint_shift),
    ADD_TEST(quick_uint_compare),

    ADD_TEST(quick_short_math),
    ADD_TEST(quick_short_logic),
    ADD_TEST(quick_short_shift),
    ADD_TEST(quick_short_compare),

    ADD_TEST(quick_ushort_math),
    ADD_TEST(quick_ushort_logic),
    ADD_TEST(quick_ushort_shift),
    ADD_TEST(quick_ushort_compare),

    ADD_TEST(quick_char_math),
    ADD_TEST(quick_char_logic),
    ADD_TEST(quick_char_shift),
    ADD_TEST(quick_char_compare),

    ADD_TEST(quick_uchar_math),
    ADD_TEST(quick_uchar_logic),
    ADD_TEST(quick_uchar_shift),
    ADD_TEST(quick_uchar_compare),

    ADD_TEST(vector_scalar),

    ADD_TEST(integer_dot_product),
};

const int test_num = ARRAY_SIZE(test_list);

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
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}


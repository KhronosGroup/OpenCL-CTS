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
#include "harness/testHarness.h"
#include "harness/mt19937.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

void fill_test_values( cl_long *outBufferA, cl_long *outBufferB, size_t numElements, MTdata d )
{
    static const cl_long sUniqueValues[] = {
        (cl_long)0x3333333333333333LL, (cl_long)0x5555555555555555LL,
        (cl_long)0x9999999999999999LL, (cl_long)0xaaaaaaaaaaaaaaaaLL,
        (cl_long)0xccccccccccccccccLL, (cl_long)0x3030303030303030LL,
        (cl_long)0x5050505050505050LL, (cl_long)0x9090909090909090LL,
        (cl_long)0xa0a0a0a0a0a0a0a0LL, (cl_long)0xc0c0c0c0c0c0c0c0LL,
        (cl_long)0xf0f0f0f0f0f0f0f0LL, (cl_long)0x0303030303030303LL,
        (cl_long)0x0505050505050505LL, (cl_long)0x0909090909090909LL,
        (cl_long)0x0a0a0a0a0a0a0a0aLL, (cl_long)0x0c0c0c0c0c0c0c0cLL,
        (cl_long)0x0f0f0f0f0f0f0f0fLL, (cl_long)0x3300330033003300LL,
        (cl_long)0x5500550055005500LL, (cl_long)0x9900990099009900LL,
        (cl_long)0xaa00aa00aa00aa00LL, (cl_long)0xcc00cc00cc00cc00LL,
        (cl_long)0xff00ff00ff00ff00LL, (cl_long)0x0033003300330033LL,
        (cl_long)0x0055005500550055LL, (cl_long)0x0099009900990099LL,
        (cl_long)0x00aa00aa00aa00aaLL, (cl_long)0x00cc00cc00cc00ccLL,
        (cl_long)0x00ff00ff00ff00ffLL, (cl_long)0x3333333300000000LL,
        (cl_long)0x5555555500000000LL, (cl_long)0x9999999900000000LL,
        (cl_long)0xaaaaaaaa00000000LL, (cl_long)0xcccccccc00000000LL,
        (cl_long)0xffffffff00000000LL, (cl_long)0x0000000033333333LL,
        (cl_long)0x0000000055555555LL, (cl_long)0x0000000099999999LL,
        (cl_long)0x00000000aaaaaaaaLL, (cl_long)0x00000000ccccccccLL,
        (cl_long)0x00000000ffffffffLL, (cl_long)0x3333000000003333LL,
        (cl_long)0x5555000000005555LL, (cl_long)0x9999000000009999LL,
        (cl_long)0xaaaa00000000aaaaLL, (cl_long)0xcccc00000000ccccLL,
        (cl_long)0xffff00000000ffffLL
    };
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
    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}


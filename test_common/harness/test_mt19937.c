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
#include "mt19937.h"
#include <stdio.h>

int main( void )
{
    MTdata d = init_genrand(42);
    int i;
    const cl_uint reference[16] = { 0x5fe1dc66, 0x8b255210, 0x0380b0c8, 0xc87d2ce4,
                                    0x55c31f24, 0x8bcd21ab, 0x14d5fef5, 0x9416d2b6,
                                    0xdf875de9, 0x00517d76, 0xd861c944, 0xa7676404,
                                    0x5491aff4, 0x67616209, 0xc368b3fb, 0x929dfc92 };
    int errcount = 0;

    for( i = 0; i < 65536; i++ )
    {
        cl_uint u = genrand_int32( d );
        if( 0 == (i & 4095) )
        {
            if( u != reference[i>>12] )
            {
                printf("ERROR: expected *0x%8.8x at %d.  Got 0x%8.8x\n", reference[i>>12], i, u );
                errcount++;
            }
        }
    }

    free_mtdata(d);

    if( errcount )
        printf("mt19937 test failed.\n");
    else
        printf("mt19937 test passed.\n");


    return 0;
}
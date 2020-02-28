/*
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)

   Modifications for use in OpenCL by Ian Ollmann, Apple Inc.

*/

#include <stdio.h>
#include <stdlib.h>
#include "mt19937.h"
#include "mingw_compat.h"
#include "harness/alloc.h"

#ifdef __SSE2__
    #include <emmintrin.h>
#endif

/* Period parameters */
#define N 624   /* vector code requires multiple of 4 here */
#define M 397
#define MATRIX_A    (cl_uint) 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK  (cl_uint) 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK  (cl_uint) 0x7fffffffUL /* least significant r bits */

typedef struct _MTdata
{
    cl_uint mt[N];
#ifdef __SSE2__
    cl_uint cache[N];
#endif
    cl_int  mti;
}_MTdata;

/* initializes mt[N] with a seed */
MTdata init_genrand(cl_uint s)
{
    MTdata r = (MTdata) align_malloc( sizeof( _MTdata ), 16 );
    if( NULL != r )
    {
        cl_uint *mt = r->mt;
        int mti = 0;
        mt[0]= s; // & 0xffffffffUL;
        for (mti=1; mti<N; mti++) {
            mt[mti] = (cl_uint)
            (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
            /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
            /* In the previous versions, MSBs of the seed affect   */
            /* only MSBs of the array mt[].                        */
            /* 2002/01/09 modified by Makoto Matsumoto             */
    //        mt[mti] &= 0xffffffffUL;
            /* for >32 bit machines */
        }
        r->mti = mti;
    }

    return r;
}

void    free_mtdata( MTdata d )
{
    if(d)
        align_free(d);
}

/* generates a random number on [0,0xffffffff]-interval */
cl_uint genrand_int32( MTdata d)
{
    /* mag01[x] = x * MATRIX_A  for x=0,1 */
    static const cl_uint mag01[2]={0x0UL, MATRIX_A};
#ifdef __SSE2__
    static volatile int init = 0;
    static union{ __m128i v; cl_uint s[4]; } upper_mask, lower_mask, one, matrix_a, c0, c1;
#endif


    cl_uint *mt = d->mt;
    cl_uint y;

    if (d->mti == N)
    { /* generate N words at one time */
        int kk;

#ifdef __SSE2__
        if( 0 == init )
        {
            upper_mask.s[0] = upper_mask.s[1] = upper_mask.s[2] = upper_mask.s[3] = UPPER_MASK;
            lower_mask.s[0] = lower_mask.s[1] = lower_mask.s[2] = lower_mask.s[3] = LOWER_MASK;
            one.s[0] = one.s[1] = one.s[2] = one.s[3] = 1;
            matrix_a.s[0] = matrix_a.s[1] = matrix_a.s[2] = matrix_a.s[3] = MATRIX_A;
            c0.s[0] = c0.s[1] = c0.s[2] = c0.s[3] = (cl_uint) 0x9d2c5680UL;
            c1.s[0] = c1.s[1] = c1.s[2] = c1.s[3] = (cl_uint) 0xefc60000UL;
            init = 1;
        }
#endif

        kk = 0;
#ifdef __SSE2__
        // vector loop
        for( ; kk + 4 <= N-M; kk += 4 )
        {
            __m128i vy = _mm_or_si128(  _mm_and_si128( _mm_load_si128( (__m128i*)(mt + kk) ), upper_mask.v ),
                                        _mm_and_si128( _mm_loadu_si128( (__m128i*)(mt + kk + 1) ), lower_mask.v ));        //  ((mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK))

            __m128i mask = _mm_cmpeq_epi32( _mm_and_si128( vy, one.v), one.v );                                         // y & 1 ? -1 : 0
            __m128i vmag01 = _mm_and_si128( mask, matrix_a.v );                                                         // y & 1 ? MATRIX_A, 0    =  mag01[y & (cl_uint) 0x1UL]
            __m128i vr = _mm_xor_si128( _mm_loadu_si128( (__m128i*)(mt + kk + M)), (__m128i) _mm_srli_epi32( vy, 1 ) );    // mt[kk+M] ^ (y >> 1)
            vr = _mm_xor_si128( vr, vmag01 );                                                                           // mt[kk+M] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL]
            _mm_store_si128( (__m128i*) (mt + kk ), vr );
        }
#endif
        for ( ;kk<N-M;kk++) {
            y = (cl_uint) ((mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK));
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL];
        }

#ifdef __SSE2__
        // advance to next aligned location
        for (;kk<N-1 && (kk & 3);kk++) {
            y = (cl_uint) ((mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK));
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL];
        }

        // vector loop
        for( ; kk + 4 <= N-1; kk += 4 )
        {
            __m128i vy = _mm_or_si128(  _mm_and_si128( _mm_load_si128( (__m128i*)(mt + kk) ), upper_mask.v ),
                                        _mm_and_si128( _mm_loadu_si128( (__m128i*)(mt + kk + 1) ), lower_mask.v ));        //  ((mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK))

            __m128i mask = _mm_cmpeq_epi32( _mm_and_si128( vy, one.v), one.v );                                         // y & 1 ? -1 : 0
            __m128i vmag01 = _mm_and_si128( mask, matrix_a.v );                                                         // y & 1 ? MATRIX_A, 0    =  mag01[y & (cl_uint) 0x1UL]
            __m128i vr = _mm_xor_si128( _mm_loadu_si128( (__m128i*)(mt + kk + M - N)), _mm_srli_epi32( vy, 1 ) );          // mt[kk+M-N] ^ (y >> 1)
            vr = _mm_xor_si128( vr, vmag01 );                                                                           // mt[kk+M] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL]
            _mm_store_si128( (__m128i*) (mt + kk ), vr );
        }
#endif

        for (;kk<N-1;kk++) {
            y = (cl_uint) ((mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK));
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL];
        }
        y = (cl_uint)((mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK));
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & (cl_uint) 0x1UL];

#ifdef __SSE2__
        // Do the tempering ahead of time in vector code
        for( kk = 0; kk + 4 <= N; kk += 4 )
        {
            __m128i vy = _mm_load_si128( (__m128i*)(mt + kk ) );                            // y = mt[k];
            vy = _mm_xor_si128( vy, _mm_srli_epi32( vy, 11 ) );                             // y ^= (y >> 11);
            vy = _mm_xor_si128( vy, _mm_and_si128( _mm_slli_epi32( vy, 7 ), c0.v) );        // y ^= (y << 7) & (cl_uint) 0x9d2c5680UL;
            vy = _mm_xor_si128( vy, _mm_and_si128( _mm_slli_epi32( vy, 15 ), c1.v) );       // y ^= (y << 15) & (cl_uint) 0xefc60000UL;
            vy = _mm_xor_si128( vy, _mm_srli_epi32( vy, 18 ) );                             // y ^= (y >> 18);
            _mm_store_si128( (__m128i*)(d->cache+kk), vy );
        }
#endif

        d->mti = 0;
    }
#ifdef __SSE2__
    y = d->cache[d->mti++];
#else
    y = mt[d->mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & (cl_uint) 0x9d2c5680UL;
    y ^= (y << 15) & (cl_uint) 0xefc60000UL;
    y ^= (y >> 18);
#endif


    return y;
}

cl_ulong genrand_int64( MTdata d)
{
    return ((cl_ulong) genrand_int32(d) << 32) | (cl_uint) genrand_int32(d);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(MTdata d)
{
    return genrand_int32(d)*(1.0/4294967295.0);
    /* divided by 2^32-1 */
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(MTdata d)
{
    return genrand_int32(d)*(1.0/4294967296.0);
    /* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(MTdata d)
{
    return (((double)genrand_int32(d)) + 0.5)*(1.0/4294967296.0);
    /* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(MTdata d)
{
    unsigned long a=genrand_int32(d)>>5, b=genrand_int32(d)>>6;
    return(a*67108864.0+b)*(1.0/9007199254740992.0);
}

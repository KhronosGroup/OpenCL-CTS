
/*
 *  mt19937.h
 *
 *  Mersenne Twister.
 *
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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER
 OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
 */

#ifndef MT19937_H
#define MT19937_H 1

#if defined(__APPLE__)
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif

/*
 *      Interfaces here have been modified from original sources so that they
 *      are safe to call reentrantly, so long as a different MTdata is used
 *      on each thread.
 */

typedef struct _MTdata *MTdata;

/* Create the random number generator with seed */
MTdata init_genrand(cl_uint /*seed*/);

/* release memory used by a MTdata private data */
void free_mtdata(MTdata /*data*/);

/* generates a random number on [0,0xffffffff]-interval */
cl_uint genrand_int32(MTdata /*data*/);

/* generates a random number on [0,0xffffffffffffffffULL]-interval */
cl_ulong genrand_int64(MTdata /*data*/);

/* generates a random number on [0,1]-real-interval */
double genrand_real1(MTdata /*data*/);

/* generates a random number on [0,1)-real-interval */
double genrand_real2(MTdata /*data*/);

/* generates a random number on (0,1)-real-interval */
double genrand_real3(MTdata /*data*/);

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(MTdata /*data*/);


#ifdef __cplusplus

#include <cassert>

struct MTdataHolder
{
    MTdataHolder(cl_uint seed)
    {
        m_mtdata = init_genrand(seed);
        assert(m_mtdata != nullptr);
    }

    MTdataHolder(MTdata mtdata): m_mtdata(mtdata) {}

    ~MTdataHolder() { free_mtdata(m_mtdata); }

    operator MTdata() const { return m_mtdata; }

private:
    MTdata m_mtdata;
};

#endif // #ifdef __cplusplus

#endif /* MT19937_H */

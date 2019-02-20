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
#include <string.h>
#include <stdlib.h>
#include "cl_utils.h"
#include "tests.h"
#include <math.h>

extern const char *addressSpaceNames[];

cl_ushort float2half_rte( float f );
cl_ushort float2half_rtz( float f );
cl_ushort float2half_rtp( float f );
cl_ushort float2half_rtn( float f );
cl_ushort double2half_rte( double f );
cl_ushort double2half_rtz( double f );
cl_ushort double2half_rtp( double f );
cl_ushort double2half_rtn( double f );

cl_ushort float2half_rte( float f )
{
    union{ float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    float x = fabsf(f);

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

cl_ushort float2half_rtz( float f )
{
    union{ float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    float x = fabsf(f);

    //Nan
    if( x != x )
    {
        u.u >>= (24-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( x >= MAKE_HEX_FLOAT(0x1.0p16f, 0x1L, 16) )
    {
        if( x == INFINITY )
            return 0x7c00 | sign;

        return 0x7bff | sign;
    }

    // underflow
    if( x < MAKE_HEX_FLOAT(0x1.0p-24f, 0x1L, -24) )
        return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

    // half denormal
    if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
    {
        x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
        return (cl_ushort)((int) x | sign);
    }

    u.u &= 0xFFFFE000U;
    u.u -= 0x38000000U;

    return (u.u >> (24-11)) | sign;
}

cl_ushort float2half_rtp( float f )
{
    union{ float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    float x = fabsf(f);

    //Nan
    if( x != x )
    {
        u.u >>= (24-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( f > MAKE_HEX_FLOAT(0x1.ffcp15f, 0x1ffcL, 3) )
        return 0x7c00;

    if( f <= MAKE_HEX_FLOAT(-0x1.0p16f, -0x1L, 16) )
    {
        if( f == -INFINITY )
            return 0xfc00;

        return 0xfbff;
    }

    // underflow
    if( x < MAKE_HEX_FLOAT(0x1.0p-24f, 0x1L, -24) )
    {
        if( f > 0 )
            return 1;
        return sign;
    }

    // half denormal
    if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
    {
        x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
        int r = (int) x;
        r += (float) r != x && f > 0.0f;

        return (cl_ushort)( r | sign);
    }

    float g = u.f;
    u.u &= 0xFFFFE000U;
    if( g > u.f )
        u.u += 0x00002000U;
    u.u -= 0x38000000U;

    return (u.u >> (24-11)) | sign;
}


cl_ushort float2half_rtn( float f )
{
    union{ float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    float x = fabsf(f);

    //Nan
    if( x != x )
    {
        u.u >>= (24-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( f >= MAKE_HEX_FLOAT(0x1.0p16f, 0x1L, 16) )
    {
        if( f == INFINITY )
            return 0x7c00;

        return 0x7bff;
    }

    if( f < MAKE_HEX_FLOAT(-0x1.ffcp15f, -0x1ffcL, 3) )
        return 0xfc00;

    // underflow
    if( x < MAKE_HEX_FLOAT(0x1.0p-24f, 0x1L, -24) )
    {
        if( f < 0 )
            return 0x8001;
        return sign;
    }

    // half denormal
    if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
    {
        x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
        int r = (int) x;
        r += (float) r != x && f < 0.0f;

        return (cl_ushort)( r | sign);
    }

    u.u &= 0xFFFFE000U;
    if( u.f > f )
        u.u += 0x00002000U;
    u.u -= 0x38000000U;

    return (u.u >> (24-11)) | sign;
}

cl_ushort double2half_rte( double f )
{
    union{ double f; cl_ulong u; } u = {f};
    cl_ulong sign = (u.u >> 48) & 0x8000;
    double x = fabs(f);

    //Nan
    if( x != x )
    {
        u.u >>= (53-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( x >= MAKE_HEX_DOUBLE(0x1.ffep15, 0x1ffeLL, 3) )
        return 0x7c00 | sign;

    // underflow
    if( x <= MAKE_HEX_DOUBLE(0x1.0p-25, 0x1LL, -25) )
        return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

    // very small
    if( x < MAKE_HEX_DOUBLE(0x1.8p-24, 0x18LL, -28) )
        return sign | 1;

    // half denormal
    if( x < MAKE_HEX_DOUBLE(0x1.0p-14, 0x1LL, -14) )
    {
        u.f = x * MAKE_HEX_DOUBLE(0x1.0p-1050, 0x1LL, -1050);
        return sign | u.u;
    }

    u.f *= MAKE_HEX_DOUBLE(0x1.0p42, 0x1LL, 42);
    u.u &= 0x7ff0000000000000ULL;
    x += u.f;
    u.f = x - u.f;
    u.f *= MAKE_HEX_DOUBLE(0x1.0p-1008, 0x1LL, -1008);

    return (u.u >> (53-11)) | sign;
}

cl_ushort double2half_rtz( double f )
{
    union{ double f; cl_ulong u; } u = {f};
    cl_ulong sign = (u.u >> 48) & 0x8000;
    double x = fabs(f);

    //Nan
    if( x != x )
    {
        u.u >>= (53-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    if( x == INFINITY )
        return 0x7c00 | sign;

    // overflow
    if( x >= MAKE_HEX_DOUBLE(0x1.0p16, 0x1LL, 16) )
        return 0x7bff | sign;

    // underflow
    if( x < MAKE_HEX_DOUBLE(0x1.0p-24, 0x1LL, -24) )
        return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

    // half denormal
    if( x < MAKE_HEX_DOUBLE(0x1.0p-14, 0x1LL, -14) )
    {
        x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
        return (cl_ushort)((int) x | sign);
    }

    u.u &= 0xFFFFFC0000000000ULL;
    u.u -= 0x3F00000000000000ULL;

    return (u.u >> (53-11)) | sign;
}

cl_ushort double2half_rtp( double f )
{
    union{ double f; cl_ulong u; } u = {f};
    cl_ulong sign = (u.u >> 48) & 0x8000;
    double x = fabs(f);

    //Nan
    if( x != x )
    {
        u.u >>= (53-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( f > MAKE_HEX_DOUBLE(0x1.ffcp15, 0x1ffcLL, 3) )
        return 0x7c00;

    if( f <= MAKE_HEX_DOUBLE(-0x1.0p16, -0x1LL, 16) )
    {
        if( f == -INFINITY )
            return 0xfc00;

        return 0xfbff;
    }

    // underflow
    if( x < MAKE_HEX_DOUBLE(0x1.0p-24, 0x1LL, -24) )
    {
        if( f > 0 )
            return 1;
        return sign;
    }

    // half denormal
    if( x < MAKE_HEX_DOUBLE(0x1.0p-14, 0x1LL, -14) )
    {
        x *= MAKE_HEX_FLOAT(0x1.0p24f, 0x1L, 24);
        int r = (int) x;
        if( 0 == sign )
            r += (double) r != x;

        return (cl_ushort)( r | sign);
    }

    double g = u.f;
    u.u &= 0xFFFFFC0000000000ULL;
    if( g != u.f && 0 == sign)
        u.u += 0x0000040000000000ULL;
    u.u -= 0x3F00000000000000ULL;

    return (u.u >> (53-11)) | sign;
}


cl_ushort double2half_rtn( double f )
{
    union{ double f; cl_ulong u; } u = {f};
    cl_ulong sign = (u.u >> 48) & 0x8000;
    double x = fabs(f);

    //Nan
    if( x != x )
    {
        u.u >>= (53-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( f >= MAKE_HEX_DOUBLE(0x1.0p16, 0x1LL, 16) )
    {
        if( f == INFINITY )
            return 0x7c00;

        return 0x7bff;
    }

    if( f < MAKE_HEX_DOUBLE(-0x1.ffcp15, -0x1ffcLL, 3) )
        return 0xfc00;

    // underflow
    if( x < MAKE_HEX_DOUBLE(0x1.0p-24, 0x1LL, -24) )
    {
        if( f < 0 )
            return 0x8001;
        return sign;
    }

    // half denormal
    if( x < MAKE_HEX_DOUBLE(0x1.0p-14, 0x1LL, -14) )
    {
        x *= MAKE_HEX_DOUBLE(0x1.0p24, 0x1LL, 24);
        int r = (int) x;
        if( sign )
            r += (double) r != x;

        return (cl_ushort)( r | sign);
    }

    double g = u.f;
    u.u &= 0xFFFFFC0000000000ULL;
    if( g < u.f && sign)
        u.u += 0x0000040000000000ULL;
    u.u -= 0x3F00000000000000ULL;

    return (u.u >> (53-11)) | sign;
}

int Test_vstore_half( void )
{
    switch (get_default_rounding_mode(gDevice))
    {
        case CL_FP_ROUND_TO_ZERO:
            return Test_vStoreHalf_private(float2half_rtz, double2half_rte, "");
        case 0:
            return -1;
        default:
            return Test_vStoreHalf_private(float2half_rte, double2half_rte, "");
    }
}

int Test_vstore_half_rte( void )
{
    return Test_vStoreHalf_private(float2half_rte, double2half_rte, "_rte");
}

int Test_vstore_half_rtz( void )
{
    return Test_vStoreHalf_private(float2half_rtz, double2half_rtz, "_rtz");
}

int Test_vstore_half_rtp( void )
{
    return Test_vStoreHalf_private(float2half_rtp, double2half_rtp, "_rtp");
}

int Test_vstore_half_rtn( void )
{
    return Test_vStoreHalf_private(float2half_rtn, double2half_rtn, "_rtn");
}

int Test_vstorea_half( void )
{
    switch (get_default_rounding_mode(gDevice))
    {
        case CL_FP_ROUND_TO_ZERO:
            return Test_vStoreaHalf_private(float2half_rtz, double2half_rte, "");
        case 0:
            return -1;
        default:
            return Test_vStoreaHalf_private(float2half_rte, double2half_rte, "");
    }
}

int Test_vstorea_half_rte( void )
{
    return Test_vStoreaHalf_private(float2half_rte, double2half_rte, "_rte");
}

int Test_vstorea_half_rtz( void )
{
    return Test_vStoreaHalf_private(float2half_rtz, double2half_rtz, "_rtz");
}

int Test_vstorea_half_rtp( void )
{
    return Test_vStoreaHalf_private(float2half_rtp, double2half_rtp, "_rtp");
}

int Test_vstorea_half_rtn( void )
{
    return Test_vStoreaHalf_private(float2half_rtn, double2half_rtn, "_rtn");
}

#pragma mark -

int Test_vStoreHalf_private( f2h referenceFunc, d2h doubleReferenceFunc, const char *roundName )
{
    int vectorSize, error;
    cl_program  programs[kVectorSizeCount+kStrangeVectorSizeCount][3];
    cl_kernel   kernels[kVectorSizeCount+kStrangeVectorSizeCount][3];

    uint64_t time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    memset( min_time, -1, sizeof( min_time ) );
    cl_program  doublePrograms[kVectorSizeCount+kStrangeVectorSizeCount][3];
    cl_kernel   doubleKernels[kVectorSizeCount+kStrangeVectorSizeCount][3];
    uint64_t doubleTime[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_double_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    memset( min_double_time, -1, sizeof( min_double_time ) );

    vlog( "Testing vstore_half%s\n", roundName );
    fflush( stdout );

    bool aligned= false;

    for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        const char *source[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], i, f );\n"
            "}\n"
        };

        const char *source_v3[] = {
            "__kernel void test( __global float *p, __global half *f,\n"
            "                   uint extra_last_thread)\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   } "
            "   vstore_half3",roundName,"( vload3(i, p-adjust), i, f-adjust );\n"
            "}\n"
        };

        const char *source_private_store[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __private ushort data[16];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t offset = 0;\n"
            "   size_t vecsize = vec_step(p[i]);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], 0, (__private half *)(&data[0]) );\n"
            "   for(offset = 0; offset < vecsize; offset++)\n"
            "   {\n"
            "       vstore_half(vload_half(offset, (__private half *)data), 0, &f[vecsize*i+offset]);\n"
            "   }\n"
            "}\n"
        };


        const char *source_private_store_v3[] = {
            "__kernel void test( __global float *p, __global half *f,\n"
            "                   uint extra_last_thread )\n"
            "{\n"
            "   __private ushort data[4];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   size_t offset = 0;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   } "
            "   vstore_half3",roundName,"( vload3(i, p-adjust), 0, (__private half *)(&data[0]) );\n"
            "   for(offset = 0; offset < 3; offset++)\n"
            "   {\n"
            "       vstore_half(vload_half(offset, (__private half *) data), 0, &f[3*i+offset-adjust]);\n"
            "   }\n"
            "}\n"
        };

        char local_buf_size[10];
        sprintf(local_buf_size, "%lld", (uint64_t)gWorkGroupSize);


        const char *source_local_store[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local ushort data[16*", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   size_t lsize = get_local_size(0);\n"
            "   size_t vecsize = vec_step(p[0]);\n"
            "   event_t async_event;\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], lid, (__local half *)(&data[0]) );\n"
            "   barrier( CLK_LOCAL_MEM_FENCE ); \n"
            "   async_event = async_work_group_copy((__global ushort *)f+vecsize*(i-lid), (__local ushort *)(&data[0]), vecsize*lsize, 0);\n" // investigate later
            "   wait_group_events(1, &async_event);\n"
            "}\n"
        };

        const char *source_local_store_v3[] = {
            "__kernel void test( __global float *p, __global half *f,\n"
            "                   uint extra_last_thread )\n"
            "{\n"
            "   __local ushort data[3*(", local_buf_size, "+1)];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   size_t lsize = get_local_size(0);\n"
            "   event_t async_event;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   } "
            "   vstore_half3",roundName,"( vload3(i,p-adjust), lid, (__local half *)(&data[0]) );\n"
            "   barrier( CLK_LOCAL_MEM_FENCE ); \n"
            "   async_event = async_work_group_copy((__global ushort *)(f+3*(i-lid)), (__local ushort *)(&data[adjust]), lsize*3-adjust, 0);\n" // investigate later
            "   wait_group_events(1, &async_event);\n"
            "}\n"
        };

        const char *double_source[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], i, f );\n"
            "}\n"
        };

        const char *double_source_private_store[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __private ushort data[16];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t offset = 0;\n"
            "   size_t vecsize = vec_step(p[i]);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], 0, (__private half *)(&data[0]) );\n"
            "   for(offset = 0; offset < vecsize; offset++)\n"
            "   {\n"
            "       vstore_half(vload_half(offset, (__private half *)data), 0, &f[vecsize*i+offset]);\n"
            "   }\n"
            "}\n"
        };


        const char *double_source_local_store[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local ushort data[16*", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   size_t vecsize = vec_step(p[0]);\n"
            "   size_t lsize = get_local_size(0);\n"
            "   event_t async_event;\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], lid, (__local half *)(&data[0]) );\n"
            "   barrier( CLK_LOCAL_MEM_FENCE ); \n"
            "   async_event = async_work_group_copy((__global ushort *)(f+vecsize*(i-lid)), (__local ushort *)(&data[0]), vecsize*lsize, 0);\n" // investigate later
            "   wait_group_events(1, &async_event);\n"
            "}\n"
        };


        const char *double_source_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double *p, __global half *f ,\n"
            "                   uint extra_last_thread)\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   } "
            "   vstore_half3",roundName,"( vload3(i,p-adjust), i, f -adjust);\n"
            "}\n"
        };

        const char *double_source_private_store_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double *p, __global half *f,\n"
            "                   uint extra_last_thread )\n"
            "{\n"
            "   __private ushort data[4];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   size_t offset = 0;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   } "
            "   vstore_half3",roundName,"( vload3(i, p-adjust), 0, (__private half *)(&data[0]) );\n"
            "   for(offset = 0; offset < 3; offset++)\n"
            "   {\n"
            "       vstore_half(vload_half(offset, (__private half *)data), 0, &f[3*i+offset-adjust]);\n"
            "   }\n"
            "}\n"
        };

        const char *double_source_local_store_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double *p, __global half *f,\n"
            "                   uint extra_last_thread )\n"
            "{\n"
            "   __local ushort data[3*(", local_buf_size, "+1)];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   size_t lsize = get_local_size(0);\n"
            "   event_t async_event;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     adjust = 3-extra_last_thread;\n"
            "   }\n "
            "   vstore_half3",roundName,"( vload3(i,p-adjust), lid, (__local half *)(&data[0]) );\n"
            "   barrier( CLK_LOCAL_MEM_FENCE ); \n"
            "   async_event = async_work_group_copy((__global ushort *)(f+3*(i-lid)), (__local ushort *)(&data[adjust]), lsize*3-adjust, 0);\n" // investigate later
            "   wait_group_events(1, &async_event);\n"
            "}\n"
        };



        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][0] = MakeProgram( source_v3, sizeof(source_v3) / sizeof( source_v3[0]) );
        } else {
            programs[vectorSize][0] = MakeProgram( source, sizeof(source) / sizeof( source[0]) );
        }
        if( NULL == programs[ vectorSize ][0] )
        {
            gFailCount++;
            return -1;
        }

        kernels[ vectorSize ][0] = clCreateKernel( programs[ vectorSize ][0], "test", &error );
        if( NULL == kernels[vectorSize][0] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create kernel. (%d)\n", error );
            return error;
        }

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][1] = MakeProgram( source_private_store_v3, sizeof(source_private_store_v3) / sizeof( source_private_store_v3[0]) );
        } else {
            programs[vectorSize][1] = MakeProgram( source_private_store, sizeof(source_private_store) / sizeof( source_private_store[0]) );
        }
        if( NULL == programs[ vectorSize ][1] )
        {
            gFailCount++;
            return -1;
        }

        kernels[ vectorSize ][1] = clCreateKernel( programs[ vectorSize ][1], "test", &error );
        if( NULL == kernels[vectorSize][1] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create private kernel. (%d)\n", error );
            return error;
        }

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][2] = MakeProgram( source_local_store_v3, sizeof(source_local_store_v3) / sizeof( source_local_store_v3[0]) );
            if(  NULL == programs[ vectorSize ][2] )
            {
                unsigned q;
                for ( q= 0; q < sizeof( source_local_store_v3) / sizeof( source_local_store_v3[0]); q++)
                    vlog_error("%s", source_local_store_v3[q]);

                gFailCount++;
                return -1;

            }
        } else {
            programs[vectorSize][2] = MakeProgram( source_local_store, sizeof(source_local_store) / sizeof( source_local_store[0]) );
            if( NULL == programs[ vectorSize ][2] )
            {
                unsigned q;
                for ( q= 0; q < sizeof( source_local_store) / sizeof( source_local_store[0]); q++)
                    vlog_error("%s", source_local_store[q]);

                gFailCount++;
                return -1;

            }
        }

        kernels[ vectorSize ][2] = clCreateKernel( programs[ vectorSize ][2], "test", &error );
        if( NULL == kernels[vectorSize][2] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create local kernel. (%d)\n", error );
            return error;
        }

        if( gTestDouble )
        {
            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize][0] = MakeProgram( double_source_v3, sizeof(double_source_v3) / sizeof( double_source_v3[0]) );
            } else {
                doublePrograms[vectorSize][0] = MakeProgram( double_source, sizeof(double_source) / sizeof( double_source[0]) );
            }
            if( NULL == doublePrograms[ vectorSize ][0] )
            {
                gFailCount++;
                return -1;
            }

            doubleKernels[ vectorSize ][0] = clCreateKernel( doublePrograms[ vectorSize ][0], "test", &error );
            if( NULL == kernels[vectorSize][0] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double kernel. (%d)\n", error );
                return error;
            }

            if(g_arrVecSizes[vectorSize] == 3)
                doublePrograms[vectorSize][1] = MakeProgram( double_source_private_store_v3, sizeof(double_source_private_store_v3) / sizeof( double_source_private_store_v3[0]) );
            else
                doublePrograms[vectorSize][1] = MakeProgram( double_source_private_store, sizeof(double_source_private_store) / sizeof( double_source_private_store[0]) );

            if( NULL == doublePrograms[ vectorSize ][1] )
            {
                gFailCount++;
                return -1;
            }

            doubleKernels[ vectorSize ][1] = clCreateKernel( doublePrograms[ vectorSize ][1], "test", &error );
            if( NULL == kernels[vectorSize][1] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double private kernel. (%d)\n", error );
                return error;
            }

            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize][2] = MakeProgram( double_source_local_store_v3, sizeof(double_source_local_store_v3) / sizeof( double_source_local_store_v3[0]) );
            } else {
                doublePrograms[vectorSize][2] = MakeProgram( double_source_local_store, sizeof(double_source_local_store) / sizeof( double_source_local_store[0]) );
            }
            if( NULL == doublePrograms[ vectorSize ][2] )
            {
                gFailCount++;
                return -1;
            }

            doubleKernels[ vectorSize ][2] = clCreateKernel( doublePrograms[ vectorSize ][2], "test", &error );
            if( NULL == kernels[vectorSize][2] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double local kernel. (%d)\n", error );
                return error;
            }
        }
    } // end for vector size

    // Figure out how many elements are in a work block
    size_t elementSize = MAX( sizeof(cl_ushort), sizeof(float));
    size_t blockCount = getBufferSize(gDevice) / elementSize; // elementSize is power of 2
    uint64_t lastCase = 1ULL << (8*sizeof(float)); // number of floats.
    size_t stride = blockCount;

    if (gWimpyMode)
        stride = (uint64_t)blockCount * (uint64_t)gWimpyReductionFactor;

    // we handle 64-bit types a bit differently.
    if( lastCase == 0 )
        lastCase = 0x100000000ULL;

    uint64_t i, j;
    error = 0;
    uint64_t printMask = (lastCase >> 4) - 1;
    cl_uint count = 0;
    int addressSpace;
    size_t loopCount;

    for( i = 0; i < (uint64_t)lastCase; i += stride )
    {
        count = (cl_uint) MIN( blockCount, lastCase - i );

        //Init the input stream
        cl_uint *p = (cl_uint *)gIn_single;
        for( j = 0; j < count; j++ )
            p[j] = (cl_uint) (j + i);

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_TRUE, 0, count * sizeof( float ), gIn_single, 0, NULL, NULL)) )
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        //create the reference result
        const float *s = (float *)gIn_single;
        cl_ushort *d = (cl_ushort *)gOut_half_reference;
        for( j = 0; j < count; j++ )
            d[j] = referenceFunc( s[j] );

        if( gTestDouble )
        {
            //Init the input stream
            cl_double *q = (cl_double *)gIn_double;
            for( j = 0; j < count; j++ )
                q[j] = DoubleFromUInt32 ((uint32_t)(j + i));

            if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_TRUE, 0, count * sizeof( double ), gIn_double, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }

            //create the reference result
            const double *t = (const double *)gIn_double;
            cl_ushort *dd = (cl_ushort *)gOut_half_reference_double;
            for( j = 0; j < count; j++ )
                dd[j] = doubleReferenceFunc( t[j] );
        }


        //Check the vector lengths
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        { // here we loop through vector sizes
            for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
                cl_uint pattern = 0xdeaddead;
                memset_pattern4( gOut_half, &pattern, getBufferSize(gDevice)/2);
                if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( (error = RunKernel( kernels[vectorSize][addressSpace],
                                       gInBuffer_single, gOutBuffer_half,
                                       numVecs(count, vectorSize, aligned) ,
                                       runsOverBy(count, vectorSize, aligned) ) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( memcmp( gOut_half, gOut_half_reference, count * sizeof( cl_ushort )) )
                {
                    uint16_t *u1 = (uint16_t *)gOut_half;
                    uint16_t *u2 = (uint16_t *)gOut_half_reference;
                    for( j = 0; j < count; j++ )
                    {
                        if( u1[j] != u2[j] )
                        {
                            if( (u1[j] & 0x7fff) > 0x7c00 && (u2[j] & 0x7fff) > 0x7c00 )
                                continue;

                            // retry per section 6.5.3.3
                            if( IsFloatSubnormal( ((float *) gIn_single)[j] ) )
                            {
                                cl_ushort correct2 = referenceFunc(  0.0f );
                                cl_ushort correct3 = referenceFunc( -0.0f );
                                if( (u1[j] == correct2) || (u1[j] == correct3) )
                                    continue;
                            }

                            // if reference result is sub normal, test if the output is flushed to zero
                            if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                                continue;

                            vlog_error( "%lld) (of %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, (uint64_t)count, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );

                            --j;
                            vlog_error( "before %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );

                            j += 2;
                            vlog_error( "after %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );

                            j += 1;
                            vlog_error( "after %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );

                            j += 1;
                            vlog_error( "after %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );
                            j += 1;
                            vlog_error( "after %lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );

                            gFailCount++;
                            goto exit;
                        }
                    }
                }

                if( gTestDouble )
                {
                    memset_pattern4( gOut_half, &pattern, getBufferSize(gDevice)/2);
                    if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                    {
                        vlog_error( "Failure in clWriteArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = RunKernel( doubleKernels[vectorSize][addressSpace], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                           runsOverBy(count, vectorSize, aligned) ) ) )
                    {
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                    {
                        vlog_error( "Failure in clReadArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    if( memcmp( gOut_half, gOut_half_reference_double, count * sizeof( cl_ushort )) )
                    {
                        uint16_t *u1 = (uint16_t *)gOut_half;
                        uint16_t *u2 = (uint16_t *)gOut_half_reference_double;
                        for( j = 0; j < count; j++ )
                        {
                            if( u1[j] != u2[j] )
                            {
                                if( (u1[j] & 0x7fff) > 0x7c00 && (u2[j] & 0x7fff) > 0x7c00 )
                                    continue;

                                if( IsDoubleSubnormal( ((double *) gIn_double)[j] ) )
                                {
                                    cl_ushort correct2 = doubleReferenceFunc(  0.0 );
                                    cl_ushort correct3 = doubleReferenceFunc( -0.0 );
                                    if( (u1[j] == correct2) || (u1[j] == correct3) )
                                        continue;
                                }

                                // if reference result is sub normal, test if the output is flushed to zero
                                if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                                    continue;

                                vlog_error( "\n\t%lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address space = %s (double precision)\n",
                                           j, ((double*)gIn_double)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );
                                gFailCount++;
                                goto exit;
                            }
                        }
                    }
                }
            }
        }

        if( ((i+blockCount) & ~printMask) == (i+blockCount) )
        {
            vlog( "." );
            fflush( stdout );
        }
    }  // end last case

    loopCount = count == blockCount ? 1 : 100;
    if( gReportTimes )
    {
        //Init the input stream
        cl_float *p = (cl_float *)gIn_single;
        for( j = 0; j < count; j++ )
            p[j] = (float)((double) (rand() - RAND_MAX/2) / (RAND_MAX/2));

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_TRUE, 0, count * sizeof( float ), gIn_single, 0, NULL, NULL)) )
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        if( gTestDouble )
        {
            //Init the input stream
            cl_double *q = (cl_double *)gIn_double;
            for( j = 0; j < count; j++ )
                q[j] = ((double) (rand() - RAND_MAX/2) / (RAND_MAX/2));

            if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_TRUE, 0, count * sizeof( double ), gIn_double, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }
        }

        //Run again for timing
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        {
            uint64_t bestTime = -1ULL;
            for( j = 0; j < loopCount; j++ )
            {
                uint64_t startTime = ReadTime();


                if( (error = RunKernel( kernels[vectorSize][0], gInBuffer_single, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                       runsOverBy(count, vectorSize, aligned)) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clFinish(gQueue)) )
                {
                    vlog_error( "Failure in clFinish\n" );
                    gFailCount++;
                    goto exit;
                }
                uint64_t currentTime = ReadTime() - startTime;
                if( currentTime < bestTime )
                    bestTime = currentTime;
                time[ vectorSize ] += currentTime;
            }
            if( bestTime < min_time[ vectorSize ] )
                min_time[ vectorSize ] = bestTime ;

            if( gTestDouble )
            {
                bestTime = -1ULL;
                for( j = 0; j < loopCount; j++ )
                {
                    uint64_t startTime = ReadTime();
                    if( (error = RunKernel( doubleKernels[vectorSize][0], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                           runsOverBy(count, vectorSize, aligned)) ) )
                    {
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = clFinish(gQueue)) )
                    {
                        vlog_error( "Failure in clFinish\n" );
                        gFailCount++;
                        goto exit;
                    }
                    uint64_t currentTime = ReadTime() - startTime;
                    if( currentTime < bestTime )
                        bestTime = currentTime;
                    doubleTime[ vectorSize ] += currentTime;
                }
                if( bestTime < min_double_time[ vectorSize ] )
                    min_double_time[ vectorSize ] = bestTime;
            }
        }
    }

    if( 0 == gFailCount )
    {
        if( gWimpyMode )
        {
            vlog( "\tfloat: Wimp Passed\n" );
            if( gTestDouble )
                vlog( "\tdouble: Wimp Passed\n" );
        }
        else
        {
            vlog( "\tfloat Passed\n" );
            if( gTestDouble )
                vlog( "\tdouble Passed\n" );
        }
    }

    if( gReportTimes )
    {
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0,
                      "average us/elem", "vStoreHalf%s avg. (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize]) );
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( min_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0,
                      "best us/elem", "vStoreHalf%s best (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize])  );
        if( gTestDouble )
        {
            for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( doubleTime[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0,
                          "average us/elem (double)", "vStoreHalf%s avg. d (%s vector size: %d)", roundName, addressSpaceNames[0],  (g_arrVecSizes[vectorSize])  );
            for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( min_double_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0,
                          "best us/elem (double)", "vStoreHalf%s best d (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize]) );
        }
    }

exit:
    //clean up
    for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
            clReleaseKernel( kernels[ vectorSize ][ addressSpace ] );
            clReleaseProgram( programs[ vectorSize ][ addressSpace ] );
            if( gTestDouble )
            {
                clReleaseKernel( doubleKernels[ vectorSize ][addressSpace] );
                clReleaseProgram( doublePrograms[ vectorSize ][addressSpace] );
            }
        }
    }

    gTestCount++;
    return error;
}

int Test_vStoreaHalf_private( f2h referenceFunc, d2h doubleReferenceFunc, const char *roundName )
{
    int vectorSize, error;
    cl_program  programs[kVectorSizeCount+kStrangeVectorSizeCount][3];
    cl_kernel   kernels[kVectorSizeCount+kStrangeVectorSizeCount][3];

    uint64_t time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    memset( min_time, -1, sizeof( min_time ) );
    cl_program  doublePrograms[kVectorSizeCount+kStrangeVectorSizeCount][3];
    cl_kernel   doubleKernels[kVectorSizeCount+kStrangeVectorSizeCount][3];
    uint64_t doubleTime[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_double_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    memset( min_double_time, -1, sizeof( min_double_time ) );

    bool aligned = true;

    vlog( "Testing vstorea_half%s\n", roundName );
    fflush( stdout );

    int minVectorSize = kMinVectorSize;
    // There is no aligned scalar vstorea_half
    if( 0 == minVectorSize )
        minVectorSize = 1;

    //Loop over vector sizes
    for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        const char *source[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], i, f );\n"
            "}\n"
        };

        const char *source_v3[] = {
            "__kernel void test( __global float3 *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half3",roundName,"( p[i], i, f );\n"
            "   vstore_half",roundName,"( ((__global  float *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        const char *source_private[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __private float", vector_size_name_extensions[vectorSize], " data;\n"
            "   size_t i = get_global_id(0);\n"
            "   data = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data, i, f );\n"
            "}\n"
        };

        const char *source_private_v3[] = {
            "__kernel void test( __global float3 *p, __global half *f )\n"
            "{\n"
            "   __private float", vector_size_name_extensions[vectorSize], " data;\n"
            "   size_t i = get_global_id(0);\n"
            "   data = p[i];\n"
            "   vstorea_half3",roundName,"( data, i, f );\n"
            "   vstore_half",roundName,"( ((__global  float *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        char local_buf_size[10];
        sprintf(local_buf_size, "%lld", (uint64_t)gWorkGroupSize);
        const char *source_local[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local float", vector_size_name_extensions[vectorSize], " data[", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   data[lid] = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data[lid], i, f );\n"
            "}\n"
        };

        const char *source_local_v3[] = {
            "__kernel void test( __global float", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local float", vector_size_name_extensions[vectorSize], " data[", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   data[lid] = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data[lid], i, f );\n"
            "   vstore_half",roundName,"( ((__global float *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        const char *double_source[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], i, f );\n"
            "}\n"
        };

        const char *double_source_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( p[i], i, f );\n"
            "   vstore_half",roundName,"( ((__global double *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        const char *double_source_private[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __private double", vector_size_name_extensions[vectorSize], " data;\n"
            "   size_t i = get_global_id(0);\n"
            "   data = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data, i, f );\n"
            "}\n"
        };

        const char *double_source_private_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __private double", vector_size_name_extensions[vectorSize], " data;\n"
            "   size_t i = get_global_id(0);\n"
            "   data = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data, i, f );\n"
            "   vstore_half",roundName,"( ((__global  double *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        const char *double_source_local[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local double", vector_size_name_extensions[vectorSize], " data[", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   data[lid] = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data[lid], i, f );\n"
            "}\n"
        };

        const char *double_source_local_v3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( __global double", vector_size_name_extensions[vectorSize]," *p, __global half *f )\n"
            "{\n"
            "   __local double", vector_size_name_extensions[vectorSize], " data[", local_buf_size, "];\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   data[lid] = p[i];\n"
            "   vstorea_half",vector_size_name_extensions[vectorSize],roundName,"( data[lid], i, f );\n"
            "   vstore_half",roundName,"( ((__global double *)p)[4*i+3], 4*i+3, f);\n"
            "}\n"
        };

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][0] = MakeProgram( source_v3, sizeof(source_v3) / sizeof( source_v3[0]) );
            if( NULL == programs[ vectorSize ][0] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][0] = MakeProgram( source, sizeof(source) / sizeof( source[0]) );
            if( NULL == programs[ vectorSize ][0] )
            {
                gFailCount++;
                return -1;
            }
        }

        kernels[ vectorSize ][0] = clCreateKernel( programs[ vectorSize ][0], "test", &error );
        if( NULL == kernels[vectorSize][0] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create kernel. (%d)\n", error );
            return error;
        }

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][1] = MakeProgram( source_private_v3, sizeof(source_private_v3) / sizeof( source_private_v3[0]) );
            if( NULL == programs[ vectorSize ][1] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][1] = MakeProgram( source_private, sizeof(source_private) / sizeof( source_private[0]) );
            if( NULL == programs[ vectorSize ][1] )
            {
                gFailCount++;
                return -1;
            }
        }

        kernels[ vectorSize ][1] = clCreateKernel( programs[ vectorSize ][1], "test", &error );
        if( NULL == kernels[vectorSize][1] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create private kernel. (%d)\n", error );
            return error;
        }

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize][2] = MakeProgram( source_local_v3, sizeof(source_local_v3) / sizeof( source_local_v3[0]) );
            if( NULL == programs[ vectorSize ][2] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][2] = MakeProgram( source_local, sizeof(source_local) / sizeof( source_local[0]) );
            if( NULL == programs[ vectorSize ][2] )
            {
                gFailCount++;
                return -1;
            }
        }

        kernels[ vectorSize ][2] = clCreateKernel( programs[ vectorSize ][2], "test", &error );
        if( NULL == kernels[vectorSize][2] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create local kernel. (%d)\n", error );
            return error;
        }

        if( gTestDouble )
        {
            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize][0] = MakeProgram( double_source_v3, sizeof(double_source_v3) / sizeof( double_source_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][0] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][0] = MakeProgram( double_source, sizeof(double_source) / sizeof( double_source[0]) );
                if( NULL == doublePrograms[ vectorSize ][0] )
                {
                    gFailCount++;
                    return -1;
                }
            }

            doubleKernels[ vectorSize ][0] = clCreateKernel( doublePrograms[ vectorSize ][0], "test", &error );
            if( NULL == kernels[vectorSize][0] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double kernel. (%d)\n", error );
                return error;
            }

            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize][1] = MakeProgram( double_source_private_v3, sizeof(double_source_private_v3) / sizeof( double_source_private_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][1] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][1] = MakeProgram( double_source_private, sizeof(double_source_private) / sizeof( double_source_private[0]) );
                if( NULL == doublePrograms[ vectorSize ][1] )
                {
                    gFailCount++;
                    return -1;
                }
            }

            doubleKernels[ vectorSize ][1] = clCreateKernel( doublePrograms[ vectorSize ][1], "test", &error );
            if( NULL == kernels[vectorSize][1] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double private kernel. (%d)\n", error );
                return error;
            }

            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize][2] = MakeProgram( double_source_local_v3, sizeof(double_source_local_v3) / sizeof( double_source_local_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][2] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][2] = MakeProgram( double_source_local, sizeof(double_source_local) / sizeof( double_source_local[0]) );
                if( NULL == doublePrograms[ vectorSize ][2] )
                {
                    gFailCount++;
                    return -1;
                }
            }

            doubleKernels[ vectorSize ][2] = clCreateKernel( doublePrograms[ vectorSize ][2], "test", &error );
            if( NULL == kernels[vectorSize][2] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create double local kernel. (%d)\n", error );
                return error;
            }
        }
    }

    // Figure out how many elements are in a work block
    size_t elementSize = MAX( sizeof(cl_ushort), sizeof(float));
    size_t blockCount = getBufferSize(gDevice) / elementSize;
    uint64_t lastCase = 1ULL << (8*sizeof(float));
    size_t stride = blockCount;

    if (gWimpyMode)
        stride = (uint64_t)blockCount * (uint64_t)gWimpyReductionFactor;

    // we handle 64-bit types a bit differently.
    if( lastCase == 0 )
        lastCase = 0x100000000ULL;
    uint64_t i, j;
    error = 0;
    uint64_t printMask = (lastCase >> 4) - 1;
    cl_uint count = 0;
    int addressSpace;
    size_t loopCount;

    for( i = 0; i < (uint64_t)lastCase; i += stride )
    {
        count = (cl_uint) MIN( blockCount, lastCase - i );

        //Init the input stream
        cl_uint *p = (cl_uint *)gIn_single;
        for( j = 0; j < count; j++ )
            p[j] = (cl_uint) (j + i);

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_TRUE, 0, count * sizeof( float ), gIn_single, 0, NULL, NULL)) )
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        //create the reference result
        const float *s = (const float *)gIn_single;
        cl_ushort *d = (cl_ushort *)gOut_half_reference;
        for( j = 0; j < count; j++ )
            d[j] = referenceFunc( s[j] );

        if( gTestDouble )
        {
            //Init the input stream
            cl_double *q = (cl_double *)gIn_double;
            for( j = 0; j < count; j++ )
                q[j] = DoubleFromUInt32 ((uint32_t)(j + i));

            if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_TRUE, 0, count * sizeof( double ), gIn_double, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }

            //create the reference result
            const double *t = (const double *)gIn_double;
            cl_ushort *dd = (cl_ushort *)gOut_half_reference_double;
            for( j = 0; j < count; j++ )
                dd[j] = doubleReferenceFunc( t[j] );
        }

        //Check the vector lengths
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        {
            for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
                cl_uint pattern = 0xdeaddead;
                memset_pattern4( gOut_half, &pattern, getBufferSize(gDevice)/2);
                if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( (error = RunKernel( kernels[vectorSize][addressSpace], gInBuffer_single, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                       runsOverBy(count, vectorSize, aligned) ) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( memcmp( gOut_half, gOut_half_reference, count * sizeof( cl_ushort )) )
                {
                    uint16_t *u1 = (uint16_t *)gOut_half;
                    uint16_t *u2 = (uint16_t *)gOut_half_reference;
                    for( j = 0; j < count; j++ )
                    {
                        if( u1[j] != u2[j] )
                        {
                            if( (u1[j] & 0x7fff) > 0x7c00 && (u2[j] & 0x7fff) > 0x7c00 )
                                continue;

                            // retry per section 6.5.3.3
                            if( IsFloatSubnormal( ((float *) gIn_single)[j] ) )
                            {
                                cl_ushort correct2 = referenceFunc(  0.0f );
                                cl_ushort correct3 = referenceFunc( -0.0f );
                                if( (u1[j] == correct2) || (u1[j] == correct3) )
                                    continue;
                            }

                            // if reference result is sub normal, test if the output is flushed to zero
                            if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                                continue;

                            vlog_error( "%lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address_space = %s\n", j, ((float*)gIn_single)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]), addressSpaceNames[addressSpace] );
                            gFailCount++;
                            goto exit;
                        }
                    }
                }

                if( gTestDouble )
                {
                    memset_pattern4( gOut_half, &pattern, getBufferSize(gDevice)/2);
                    if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                    {
                        vlog_error( "Failure in clWriteArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = RunKernel( doubleKernels[vectorSize][addressSpace], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                           runsOverBy(count, vectorSize, aligned) ) ) )
                    {
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof( cl_ushort), gOut_half, 0, NULL, NULL)) )
                    {
                        vlog_error( "Failure in clReadArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    if( memcmp( gOut_half, gOut_half_reference_double, count * sizeof( cl_ushort )) )
                    {
                        uint16_t *u1 = (uint16_t *)gOut_half;
                        uint16_t *u2 = (uint16_t *)gOut_half_reference_double;
                        for( j = 0; j < count; j++ )
                        {
                            if( u1[j] != u2[j] )
                            {
                                if( (u1[j] & 0x7fff) > 0x7c00 && (u2[j] & 0x7fff) > 0x7c00 )
                                    continue;

                                if( IsDoubleSubnormal( ((double *) gIn_double)[j] ) )
                                {
                                    cl_ushort correct2 = doubleReferenceFunc(  0.0 );
                                    cl_ushort correct3 = doubleReferenceFunc( -0.0 );
                                    if( (u1[j] == correct2) || (u1[j] == correct3) )
                                        continue;
                                }

                                // if reference result is sub normal, test if the output is flushed to zero
                                if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                                    continue;

                                vlog_error( "\n\t%lld) Failure at %a: *0x%4.4x vs 0x%4.4x  vector_size = %d address space = %s (double precision)\n",
                                           j, ((double*)gIn_double)[j], u2[j], u1[j], (g_arrVecSizes[vectorSize]),  addressSpaceNames[addressSpace] );
                                gFailCount++;
                                goto exit;
                            }
                        }
                    }
                }
            }
        }  // end for vector size

        if( ((i+blockCount) & ~printMask) == (i+blockCount) )
        {
            vlog( "." );
            fflush( stdout );
        }
    }  // for end lastcase

    loopCount = count == blockCount ? 1 : 100;
    if( gReportTimes )
    {
        //Init the input stream
        cl_float *p = (cl_float *)gIn_single;
        for( j = 0; j < count; j++ )
            p[j] = (float)((double) (rand() - RAND_MAX/2) / (RAND_MAX/2));

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_TRUE, 0, count * sizeof( float ), gIn_single, 0, NULL, NULL)) )
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        if( gTestDouble )
        {
            //Init the input stream
            cl_double *q = (cl_double *)gIn_double;
            for( j = 0; j < count; j++ )
                q[j] = ((double) (rand() - RAND_MAX/2) / (RAND_MAX/2));

            if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_TRUE, 0, count * sizeof( double ), gIn_double, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }
        }

        //Run again for timing
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        {
            uint64_t bestTime = -1ULL;
            for( j = 0; j < loopCount; j++ )
            {
                uint64_t startTime = ReadTime();
                if( (error = RunKernel( kernels[vectorSize][0], gInBuffer_single, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                       runsOverBy(count, vectorSize, aligned)) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clFinish(gQueue)) )
                {
                    vlog_error( "Failure in clFinish\n" );
                    gFailCount++;
                    goto exit;
                }
                uint64_t currentTime = ReadTime() - startTime;
                if( currentTime < bestTime )
                    bestTime = currentTime;
                time[ vectorSize ] += currentTime;
            }
            if( bestTime < min_time[ vectorSize ] )
                min_time[ vectorSize ] = bestTime ;

            if( gTestDouble )
            {
                bestTime = -1ULL;
                for( j = 0; j < loopCount; j++ )
                {
                    uint64_t startTime = ReadTime();
                    if( (error = RunKernel( doubleKernels[vectorSize][0], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
                                           runsOverBy(count, vectorSize, aligned)) ) )
                    {
                        gFailCount++;
                        goto exit;
                    }

                    if( (error = clFinish(gQueue)) )
                    {
                        vlog_error( "Failure in clFinish\n" );
                        gFailCount++;
                        goto exit;
                    }
                    uint64_t currentTime = ReadTime() - startTime;
                    if( currentTime < bestTime )
                        bestTime = currentTime;
                    doubleTime[ vectorSize ] += currentTime;
                }
                if( bestTime < min_double_time[ vectorSize ] )
                    min_double_time[ vectorSize ] = bestTime;
            }
        }
    }

    if( gWimpyMode )
    {
        vlog( "\tfloat: Wimp Passed\n" );

        if( gTestDouble )
            vlog( "\tdouble: Wimp Passed\n" );
    }
    else
    {
        vlog( "\tfloat Passed\n" );
        if( gTestDouble )
            vlog( "\tdouble Passed\n" );
    }

    if( gReportTimes )
    {
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0,
                      "average us/elem", "vStoreaHalf%s avg. (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize]) );
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( min_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0,
                      "best us/elem", "vStoreaHalf%s best (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize])  );
        if( gTestDouble )
        {
            for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( doubleTime[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0,
                          "average us/elem (double)", "vStoreaHalf%s avg. d (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize])  );
            for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( min_double_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0,
                          "best us/elem (double)", "vStoreaHalf%s best d (%s vector size: %d)", roundName, addressSpaceNames[0], (g_arrVecSizes[vectorSize]) );
        }
    }

exit:
    //clean up
    for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
            clReleaseKernel( kernels[ vectorSize ][addressSpace] );
            clReleaseProgram( programs[ vectorSize ][addressSpace] );
            if( gTestDouble )
            {
                clReleaseKernel( doubleKernels[ vectorSize ][addressSpace] );
                clReleaseProgram( doublePrograms[ vectorSize ][addressSpace] );
            }
        }
    }

    gTestCount++;
    return error;
}


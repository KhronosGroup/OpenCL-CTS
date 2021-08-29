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
#include "harness/compat.h"
#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"

#include <string.h>

#include <algorithm>

#include "cl_utils.h"
#include "tests.h"

#include <CL/cl_half.h>

typedef struct ComputeReferenceInfoF_
{
    float *x;
    cl_ushort *r;
    f2h f;
    cl_ulong i;
    cl_uint lim;
    cl_uint count;
} ComputeReferenceInfoF;

typedef struct ComputeReferenceInfoD_
{
    double *x;
    cl_ushort *r;
    d2h f;
    cl_ulong i;
    cl_uint lim;
    cl_uint count;
} ComputeReferenceInfoD;

typedef struct CheckResultInfoF_
{
    const float *x;
    const cl_ushort *r;
    const cl_ushort *s;
    f2h f;
    const char *aspace;
    cl_uint lim;
    cl_uint count;
    int vsz;
} CheckResultInfoF;

typedef struct CheckResultInfoD_
{
    const double *x;
    const cl_ushort *r;
    const cl_ushort *s;
    d2h f;
    const char *aspace;
    cl_uint lim;
    cl_uint count;
    int vsz;
} CheckResultInfoD;

static cl_int
ReferenceF(cl_uint jid, cl_uint tid, void *userInfo)
{
    ComputeReferenceInfoF *cri = (ComputeReferenceInfoF *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    float *x = cri->x + off;
    cl_ushort *r = cri->r + off;
    f2h f = cri->f;
    cl_ulong i = cri->i + off;
    cl_uint j, rr;

    if (off + count > lim)
        count = lim - off;

    for (j = 0; j < count; ++j) {
        x[j] = as_float((cl_uint)(i + j));
        r[j] = f(x[j]);
    }

    return 0;
}

static cl_int
CheckF(cl_uint jid, cl_uint tid, void *userInfo)
{
    CheckResultInfoF *cri = (CheckResultInfoF *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    const float *x = cri->x + off;
    const cl_ushort *r = cri->r + off;
    const cl_ushort *s = cri->s + off;
    f2h f = cri->f;
    cl_uint j;
    cl_ushort correct2 = f( 0.0f);
    cl_ushort correct3 = f(-0.0f);
    cl_int ret = 0;

    if (off + count > lim)
        count = lim - off;

    if (!memcmp(r, s, count*sizeof(cl_ushort)))
        return 0;

    for (j = 0; j < count; j++) {
    if (s[j] == r[j])
        continue;

        // Pass any NaNs
        if ((s[j] & 0x7fff) > 0x7c00 && (r[j] & 0x7fff) > 0x7c00 )
            continue;

        // retry per section 6.5.3.3
        if (IsFloatSubnormal(x[j]) && (s[j] == correct2 || s[j] == correct3))
            continue;

        // if reference result is subnormal, pass any zero
        if (gIsEmbedded && IsHalfSubnormal(r[j]) && (s[j] == 0x0000 || s[j] == 0x8000))
            continue;

        vlog_error("\nFailure at [%u] with %.6a: *0x%04x vs 0x%04x,  vector_size = %d, address_space = %s\n",
                   j+off, x[j], r[j], s[j], cri->vsz, cri->aspace);

        ret = 1;
        break;
    }

    return ret;
}

static cl_int
ReferenceD(cl_uint jid, cl_uint tid, void *userInfo)
{
    ComputeReferenceInfoD *cri = (ComputeReferenceInfoD *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    double *x = cri->x + off;
    cl_ushort *r = cri->r + off;
    d2h f = cri->f;
    cl_uint j;
    cl_ulong i = cri->i + off;

    if (off + count > lim)
        count = lim - off;

    for (j = 0; j < count; ++j) {
        x[j] = as_double(DoubleFromUInt((cl_uint)(i + j)));
        r[j] = f(x[j]);
    }

    return 0;
}

static cl_int
CheckD(cl_uint jid, cl_uint tid, void *userInfo)
{
    CheckResultInfoD *cri = (CheckResultInfoD *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    const double *x = cri->x + off;
    const cl_ushort *r = cri->r + off;
    const cl_ushort *s = cri->s + off;
    d2h f = cri->f;
    cl_uint j;
    cl_ushort correct2 = f( 0.0);
    cl_ushort correct3 = f(-0.0);
    cl_int ret = 0;

    if (off + count > lim)
        count = lim - off;

    if (!memcmp(r, s, count*sizeof(cl_ushort)))
        return 0;

    for (j = 0; j < count; j++) {
    if (s[j] == r[j])
        continue;

        // Pass any NaNs
        if ((s[j] & 0x7fff) > 0x7c00 && (r[j] & 0x7fff) > 0x7c00)
            continue;

        if (IsDoubleSubnormal(x[j]) && (s[j] == correct2 || s[j] == correct3))
            continue;

        // if reference result is subnormal, pass any zero result
        if (gIsEmbedded && IsHalfSubnormal(r[j]) && (s[j] == 0x0000 || s[j] == 0x8000))
            continue;

        vlog_error("\nFailure at [%u] with %.13la: *0x%04x vs 0x%04x, vector_size = %d, address space = %s (double precision)\n",
                   j+off, x[j], r[j], s[j], cri->vsz, cri->aspace);

        ret = 1;
    break;
    }

    return ret;
}

static cl_half float2half_rte(float f)
{
    return cl_half_from_float(f, CL_HALF_RTE);
}

static cl_half float2half_rtz(float f)
{
    return cl_half_from_float(f, CL_HALF_RTZ);
}

static cl_half float2half_rtp(float f)
{
    return cl_half_from_float(f, CL_HALF_RTP);
}

static cl_half float2half_rtn(float f)
{
    return cl_half_from_float(f, CL_HALF_RTN);
}

static cl_half double2half_rte(double f)
{
    return cl_half_from_double(f, CL_HALF_RTE);
}

static cl_half double2half_rtz(double f)
{
    return cl_half_from_double(f, CL_HALF_RTZ);
}

static cl_half double2half_rtp(double f)
{
    return cl_half_from_double(f, CL_HALF_RTP);
}

static cl_half double2half_rtn(double f)
{
    return cl_half_from_double(f, CL_HALF_RTN);
}

int test_vstore_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    switch (get_default_rounding_mode(deviceID))
    {
        case CL_FP_ROUND_TO_ZERO:
            return Test_vStoreHalf_private(deviceID, float2half_rtz, double2half_rte, "");
        case 0:
            return -1;
        default:
            return Test_vStoreHalf_private(deviceID, float2half_rte, double2half_rte, "");
    }
}

int test_vstore_half_rte( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreHalf_private(deviceID, float2half_rte, double2half_rte, "_rte");
}

int test_vstore_half_rtz( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreHalf_private(deviceID, float2half_rtz, double2half_rtz, "_rtz");
}

int test_vstore_half_rtp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreHalf_private(deviceID, float2half_rtp, double2half_rtp, "_rtp");
}

int test_vstore_half_rtn( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreHalf_private(deviceID, float2half_rtn, double2half_rtn, "_rtn");
}

int test_vstorea_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    switch (get_default_rounding_mode(deviceID))
    {
        case CL_FP_ROUND_TO_ZERO:
            return Test_vStoreaHalf_private(deviceID,float2half_rtz, double2half_rte, "");
        case 0:
            return -1;
        default:
            return Test_vStoreaHalf_private(deviceID, float2half_rte, double2half_rte, "");
    }
}

int test_vstorea_half_rte( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreaHalf_private(deviceID, float2half_rte, double2half_rte, "_rte");
}

int test_vstorea_half_rtz( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreaHalf_private(deviceID, float2half_rtz, double2half_rtz, "_rtz");
}

int test_vstorea_half_rtp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreaHalf_private(deviceID, float2half_rtp, double2half_rtp, "_rtp");
}

int test_vstorea_half_rtn( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vStoreaHalf_private(deviceID, float2half_rtn, double2half_rtn, "_rtn");
}

#pragma mark -

int Test_vStoreHalf_private( cl_device_id device, f2h referenceFunc, d2h doubleReferenceFunc, const char *roundName )
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
            programs[vectorSize][0] = MakeProgram( device, source_v3, sizeof(source_v3) / sizeof( source_v3[0]) );
        } else {
            programs[vectorSize][0] = MakeProgram( device, source, sizeof(source) / sizeof( source[0]) );
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
            programs[vectorSize][1] = MakeProgram( device, source_private_store_v3, sizeof(source_private_store_v3) / sizeof( source_private_store_v3[0]) );
        } else {
            programs[vectorSize][1] = MakeProgram( device, source_private_store, sizeof(source_private_store) / sizeof( source_private_store[0]) );
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
            programs[vectorSize][2] = MakeProgram( device, source_local_store_v3, sizeof(source_local_store_v3) / sizeof( source_local_store_v3[0]) );
            if(  NULL == programs[ vectorSize ][2] )
            {
                unsigned q;
                for ( q= 0; q < sizeof( source_local_store_v3) / sizeof( source_local_store_v3[0]); q++)
                    vlog_error("%s", source_local_store_v3[q]);

                gFailCount++;
                return -1;

            }
        } else {
            programs[vectorSize][2] = MakeProgram( device, source_local_store, sizeof(source_local_store) / sizeof( source_local_store[0]) );
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
                doublePrograms[vectorSize][0] = MakeProgram( device, double_source_v3, sizeof(double_source_v3) / sizeof( double_source_v3[0]) );
            } else {
                doublePrograms[vectorSize][0] = MakeProgram( device, double_source, sizeof(double_source) / sizeof( double_source[0]) );
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
                doublePrograms[vectorSize][1] = MakeProgram( device, double_source_private_store_v3, sizeof(double_source_private_store_v3) / sizeof( double_source_private_store_v3[0]) );
            else
                doublePrograms[vectorSize][1] = MakeProgram( device, double_source_private_store, sizeof(double_source_private_store) / sizeof( double_source_private_store[0]) );

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
                doublePrograms[vectorSize][2] = MakeProgram( device, double_source_local_store_v3, sizeof(double_source_local_store_v3) / sizeof( double_source_local_store_v3[0]) );
            } else {
                doublePrograms[vectorSize][2] = MakeProgram( device, double_source_local_store, sizeof(double_source_local_store) / sizeof( double_source_local_store[0]) );
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
    size_t elementSize = std::max(sizeof(cl_ushort), sizeof(float));
    size_t blockCount = BUFFER_SIZE / elementSize; // elementSize is power of 2
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
    cl_uint threadCount = GetThreadCount();

    ComputeReferenceInfoF fref;
    fref.x = (float *)gIn_single;
    fref.r = (cl_half *)gOut_half_reference;
    fref.f = referenceFunc;
    fref.lim = blockCount;
    fref.count = (blockCount + threadCount - 1) / threadCount;

    CheckResultInfoF fchk;
    fchk.x = (const float *)gIn_single;
    fchk.r = (const cl_half *)gOut_half_reference;
    fchk.s = (const cl_half *)gOut_half;
    fchk.f = referenceFunc;
    fchk.lim = blockCount;
    fchk.count = (blockCount + threadCount - 1) / threadCount;

    ComputeReferenceInfoD dref;
    dref.x = (double *)gIn_double;
    dref.r = (cl_half *)gOut_half_reference_double;
    dref.f = doubleReferenceFunc;
    dref.lim = blockCount;
    dref.count = (blockCount + threadCount - 1) / threadCount;

    CheckResultInfoD dchk;
    dchk.x = (const double *)gIn_double;
    dchk.r = (const cl_half *)gOut_half_reference_double;
    dchk.s = (const cl_half *)gOut_half;
    dchk.f = doubleReferenceFunc;
    dchk.lim = blockCount;
    dchk.count = (blockCount + threadCount - 1) / threadCount;

    for( i = 0; i < lastCase; i += stride )
    {
        count = (cl_uint)std::min((uint64_t)blockCount, lastCase - i);
        fref.i = i;
        dref.i = i;

        // Compute the input and reference
        ThreadPool_Do(ReferenceF, threadCount, &fref);

        error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_FALSE, 0, count * sizeof(float ), gIn_single, 0, NULL, NULL);
        if (error) {
            vlog_error( "Failure in clWriteBuffer\n" );
            gFailCount++;
            goto exit;
        }

        if (gTestDouble) {
            ThreadPool_Do(ReferenceD, threadCount, &dref);

            error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_FALSE, 0, count * sizeof(double ), gIn_double, 0, NULL, NULL);
            if (error) {
                vlog_error( "Failure in clWriteBuffer\n" );
                gFailCount++;
                goto exit;
            }
        }

        for (vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++) {
            // Loop through vector sizes
            fchk.vsz = g_arrVecSizes[vectorSize];
            dchk.vsz = g_arrVecSizes[vectorSize];

            for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
                // Loop over address spaces
                fchk.aspace = addressSpaceNames[addressSpace];
                dchk.aspace = addressSpaceNames[addressSpace];

                cl_uint pattern = 0xdeaddead;
                memset_pattern4( gOut_half, &pattern, BUFFER_SIZE/2);

                error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_FALSE,
                                             0, count * sizeof(cl_half),
                                             gOut_half, 0, NULL, NULL);
                if (error) {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }

                error = RunKernel(device, kernels[vectorSize][addressSpace], gInBuffer_single, gOutBuffer_half,
                                       numVecs(count, vectorSize, aligned) ,
                                  runsOverBy(count, vectorSize, aligned));
                if (error) {
                    gFailCount++;
                    goto exit;
                }

                error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0,
                                            count * sizeof(cl_half), gOut_half,
                                            0, NULL, NULL);
                if (error) {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                error = ThreadPool_Do(CheckF, threadCount, &fchk);
                if (error) {
                            gFailCount++;
                            goto exit;
                        }

                if (gTestDouble) {
                    memset_pattern4( gOut_half, &pattern, BUFFER_SIZE/2);

                    error = clEnqueueWriteBuffer(
                        gQueue, gOutBuffer_half, CL_FALSE, 0,
                        count * sizeof(cl_half), gOut_half, 0, NULL, NULL);
                    if (error) {
                        vlog_error( "Failure in clWriteArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    error = RunKernel(device, doubleKernels[vectorSize][addressSpace], gInBuffer_double, gOutBuffer_half,
                                      numVecs(count, vectorSize, aligned),
                                      runsOverBy(count, vectorSize, aligned));
                    if (error) {
                        gFailCount++;
                        goto exit;
                    }

                    error = clEnqueueReadBuffer(
                        gQueue, gOutBuffer_half, CL_TRUE, 0,
                        count * sizeof(cl_half), gOut_half, 0, NULL, NULL);
                    if (error) {
                        vlog_error( "Failure in clReadArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    error = ThreadPool_Do(CheckD, threadCount, &dchk);
                    if (error) {
                                gFailCount++;
                                goto exit;
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


                if( (error = RunKernel(device, kernels[vectorSize][0], gInBuffer_single, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
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
                    if( (error = RunKernel(device, doubleKernels[vectorSize][0], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
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

    return error;
}

int Test_vStoreaHalf_private( cl_device_id device, f2h referenceFunc, d2h doubleReferenceFunc, const char *roundName )
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
            programs[vectorSize][0] = MakeProgram( device, source_v3, sizeof(source_v3) / sizeof( source_v3[0]) );
            if( NULL == programs[ vectorSize ][0] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][0] = MakeProgram( device, source, sizeof(source) / sizeof( source[0]) );
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
            programs[vectorSize][1] = MakeProgram( device, source_private_v3, sizeof(source_private_v3) / sizeof( source_private_v3[0]) );
            if( NULL == programs[ vectorSize ][1] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][1] = MakeProgram( device, source_private, sizeof(source_private) / sizeof( source_private[0]) );
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
            programs[vectorSize][2] = MakeProgram( device, source_local_v3, sizeof(source_local_v3) / sizeof( source_local_v3[0]) );
            if( NULL == programs[ vectorSize ][2] )
            {
                gFailCount++;
                return -1;
            }
        } else {
            programs[vectorSize][2] = MakeProgram( device, source_local, sizeof(source_local) / sizeof( source_local[0]) );
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
                doublePrograms[vectorSize][0] = MakeProgram( device, double_source_v3, sizeof(double_source_v3) / sizeof( double_source_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][0] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][0] = MakeProgram( device, double_source, sizeof(double_source) / sizeof( double_source[0]) );
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
                doublePrograms[vectorSize][1] = MakeProgram( device, double_source_private_v3, sizeof(double_source_private_v3) / sizeof( double_source_private_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][1] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][1] = MakeProgram( device, double_source_private, sizeof(double_source_private) / sizeof( double_source_private[0]) );
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
                doublePrograms[vectorSize][2] = MakeProgram( device, double_source_local_v3, sizeof(double_source_local_v3) / sizeof( double_source_local_v3[0]) );
                if( NULL == doublePrograms[ vectorSize ][2] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize][2] = MakeProgram( device, double_source_local, sizeof(double_source_local) / sizeof( double_source_local[0]) );
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
    size_t elementSize = std::max(sizeof(cl_ushort), sizeof(float));
    size_t blockCount = BUFFER_SIZE / elementSize;
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
    cl_uint threadCount = GetThreadCount();

    ComputeReferenceInfoF fref;
    fref.x = (float *)gIn_single;
    fref.r = (cl_half *)gOut_half_reference;
    fref.f = referenceFunc;
    fref.lim = blockCount;
    fref.count = (blockCount + threadCount - 1) / threadCount;

    CheckResultInfoF fchk;
    fchk.x = (const float *)gIn_single;
    fchk.r = (const cl_half *)gOut_half_reference;
    fchk.s = (const cl_half *)gOut_half;
    fchk.f = referenceFunc;
    fchk.lim = blockCount;
    fchk.count = (blockCount + threadCount - 1) / threadCount;

    ComputeReferenceInfoD dref;
    dref.x = (double *)gIn_double;
    dref.r = (cl_half *)gOut_half_reference_double;
    dref.f = doubleReferenceFunc;
    dref.lim = blockCount;
    dref.count = (blockCount + threadCount - 1) / threadCount;

    CheckResultInfoD dchk;
    dchk.x = (const double *)gIn_double;
    dchk.r = (const cl_half *)gOut_half_reference_double;
    dchk.s = (const cl_half *)gOut_half;
    dchk.f = doubleReferenceFunc;
    dchk.lim = blockCount;
    dchk.count = (blockCount + threadCount - 1) / threadCount;

    for( i = 0; i < (uint64_t)lastCase; i += stride )
    {
        count = (cl_uint)std::min((uint64_t)blockCount, lastCase - i);
        fref.i = i;
        dref.i = i;

        // Create the input and reference
        ThreadPool_Do(ReferenceF, threadCount, &fref);

        error = clEnqueueWriteBuffer(gQueue, gInBuffer_single, CL_FALSE, 0, count * sizeof(float ), gIn_single, 0, NULL, NULL);
        if (error) {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        if (gTestDouble) {
            ThreadPool_Do(ReferenceD, threadCount, &dref);

            error = clEnqueueWriteBuffer(gQueue, gInBuffer_double, CL_FALSE, 0, count * sizeof(double ), gIn_double, 0, NULL, NULL);
            if (error) {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }
        }

        for (vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++) {
            // Loop over vector legths
            fchk.vsz = g_arrVecSizes[vectorSize];
            dchk.vsz = g_arrVecSizes[vectorSize];

            for ( addressSpace = 0; addressSpace < 3; addressSpace++) {
                // Loop over address spaces
                fchk.aspace = addressSpaceNames[addressSpace];
                dchk.aspace = addressSpaceNames[addressSpace];

                cl_uint pattern = 0xdeaddead;
                memset_pattern4(gOut_half, &pattern, BUFFER_SIZE/2);

                error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_FALSE,
                                             0, count * sizeof(cl_half),
                                             gOut_half, 0, NULL, NULL);
                if (error) {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }

                error = RunKernel(device, kernels[vectorSize][addressSpace], gInBuffer_single, gOutBuffer_half,
                                  numVecs(count, vectorSize, aligned),
                                  runsOverBy(count, vectorSize, aligned));
                if (error) {
                    gFailCount++;
                    goto exit;
                }

                error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0,
                                            count * sizeof(cl_half), gOut_half,
                                            0, NULL, NULL);
                if (error) {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                error = ThreadPool_Do(CheckF, threadCount, &fchk);
                if (error) {
                            gFailCount++;
                            goto exit;
                        }

                if (gTestDouble) {
                    memset_pattern4(gOut_half, &pattern, BUFFER_SIZE/2);

                    error = clEnqueueWriteBuffer(
                        gQueue, gOutBuffer_half, CL_FALSE, 0,
                        count * sizeof(cl_half), gOut_half, 0, NULL, NULL);
                    if (error) {
                        vlog_error( "Failure in clWriteArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    error = RunKernel(device, doubleKernels[vectorSize][addressSpace], gInBuffer_double, gOutBuffer_half,
                                      numVecs(count, vectorSize, aligned),
                                      runsOverBy(count, vectorSize, aligned));
                    if (error) {
                        gFailCount++;
                        goto exit;
                    }

                    error = clEnqueueReadBuffer(
                        gQueue, gOutBuffer_half, CL_TRUE, 0,
                        count * sizeof(cl_half), gOut_half, 0, NULL, NULL);
                    if (error) {
                        vlog_error( "Failure in clReadArray\n" );
                        gFailCount++;
                        goto exit;
                    }

                    error = ThreadPool_Do(CheckD, threadCount, &dchk);
                    if (error) {
                                gFailCount++;
                                goto exit;
                            }
                        }
                    }
        }  // end for vector size

        if( ((i+blockCount) & ~printMask) == (i+blockCount) ) {
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
                if( (error = RunKernel(device, kernels[vectorSize][0], gInBuffer_single, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
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
                    if( (error = RunKernel(device, doubleKernels[vectorSize][0], gInBuffer_double, gOutBuffer_half, numVecs(count, vectorSize, aligned) ,
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

    return error;
}


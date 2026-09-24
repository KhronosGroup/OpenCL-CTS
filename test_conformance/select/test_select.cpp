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
#include "harness/typeWrappers.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <cinttypes>
#include <vector>

#if ! defined( _WIN32)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif
#include <limits.h>
#include "test_select.h"

#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/mt19937.h"
#include "harness/parseParameters.h"


//-----------------------------------------
// Static functions
//-----------------------------------------

// initialize src1 and src2 buffer with values based on stype
static void initSrcBuffer(void* src1, Type stype, MTdata);

// initialize the valued used to compare with in the select with
// vlaues [start, count)
static void initCmpBuffer(void *cmp, Type cmptype, uint32_t start,
                          const size_t count, const uint32_t vecsize,
                          MTdataHolder &d);

// make a program that uses select for the given stype (src/dest type),
// ctype (comparison type), veclen (vector length)
static cl_program makeSelectProgram(cl_kernel *kernel_ptr, cl_context context,
                                    Type stype, Type ctype,
                                    const size_t veclen);

// Creates and execute the select test for the given device, context,
// stype (source/dest type), cmptype (comparison type), using max_tg_size
// number of threads. It runs test for all the different vector lengths
// for the given stype and cmptype.
static int doTest(cl_command_queue queue, cl_context context,
                  Type stype, Type cmptype, cl_device_id device);

//-----------------------------------------
// Definitions and initializations
//-----------------------------------------

// Define the buffer size that we want to block our test with
#define BUFFER_SIZE (1024*1024)
#define KPAGESIZE 4096

#define test_error_count(errCode, msg)                                         \
    {                                                                          \
        auto errCodeResult = errCode;                                          \
        if (errCodeResult != CL_SUCCESS)                                       \
        {                                                                      \
            gFailCount++;                                                      \
            print_error(errCodeResult, msg);                                   \
            return errCode;                                                    \
        }                                                                      \
    }

//-----------------------------------------
// Static helper functions
//-----------------------------------------

// calculates log2 for a 32 bit number
int int_log2(size_t value) {
    if( 0 == value )
        return INT_MIN;

#if defined( __GNUC__ )
    return (unsigned) (8*sizeof(size_t) - 1UL - __builtin_clzl(value));
#else
    int result = -1;
    while(value)
    {
        result++;
        value >>= 1;
    }
    return result;
#endif
}


static void initSrcBuffer(void* src1, Type stype, MTdata d)
{
    unsigned int* s1 = (unsigned int *)src1;
    size_t i;

    for ( i=0 ; i < BUFFER_SIZE/sizeof(cl_int); i++)
        s1[i]   = genrand_int32(d);
}

static void initCmpBuffer(void *cmp, Type cmptype, uint32_t start,
                          const size_t count, const uint32_t vecsize,
                          MTdataHolder &d)

{
    assert(cmptype != kfloat);
    switch (type_size[cmptype]) {
        case 1: {
            uint8_t *ub = (uint8_t *)cmp;
            if (vecsize == 1)
                for (uint32_t i = 0; i < count; i++)
                    ub[i] = i % 2 ? 0 : (start + i) / 2;
            else
                for (uint32_t i = 0; i < count; i++)
                {
                    uint16_t vec_bitmask = (start + i) / vecsize;
                    uint16_t vec_idx = (start + i) % vecsize;
                    uint8_t gen = genrand_int32(d) & 0xff;
                    ub[i] = ((vec_bitmask >> vec_idx) & 0x1) ? gen | 0x80
                                                             : gen & 0x7f;
                }
            break;
        }
        case 2: {
            uint16_t* us = (uint16_t *)cmp;
            if (vecsize == 1)
            {
                for (uint32_t i = 0; i < count; i++)
                    us[i] = i % 2 ? 0 : genrand_int32(d) & 0xffff;
            }
            else
            {
                for (uint32_t i = 0; i < count; i++)
                {
                    uint16_t vec_bitmask = (start + i) / vecsize;
                    uint16_t vec_idx = (start + i) % vecsize;
                    uint16_t gen = genrand_int32(d) & 0xffff;
                    us[i] = ((vec_bitmask >> vec_idx) & 0x1) ? gen | 0x8000
                                                             : gen & 0x7fff;
                }
            }
            break;
        }
        case 4: {
            uint32_t *ui = (uint32_t *)cmp;
            if (vecsize == 1)
            {
                for (uint32_t i = 0; i < count; i++)
                    ui[i] = i % 2 ? 0 : genrand_int32(d);
            }
            else
            {
                for (uint32_t i = 0; i < count; i++)
                {
                    uint16_t vec_bitmask = (start + i) / vecsize;
                    uint16_t vec_idx = (start + i) % vecsize;
                    uint32_t gen = genrand_int32(d);
                    ui[i] = ((vec_bitmask >> vec_idx) & 0x1) ? gen | 0x80000000
                                                             : gen & 0x7fffffff;
                }
            }
            break;
        }
        case 8: {
            uint64_t *ul = (uint64_t *)cmp;
            if (vecsize == 1)
            {
                for (uint32_t i = 0; i < count; i++)
                    ul[i] = i % 2 ? 0 : genrand_int64(d);
            }
            else
            {
                for (uint32_t i = 0; i < count; i++)
                {
                    uint16_t vec_bitmask = (start + i) / vecsize;
                    uint16_t vec_idx = (start + i) % vecsize;
                    uint64_t gen = genrand_int64(d);
                    ul[i] = ((vec_bitmask >> vec_idx) & 0x1)
                        ? gen | 0x8000000000000000ULL
                        : gen & 0x7fffffffffffffffULL;
                }
            }
            break;
        }
        default:
            log_error("invalid cmptype %s\n",type_name[cmptype]);
    } // end switch
}

// Make the various incarnations of the program we want to run
//  stype: source and destination type for the select
//  ctype: compare type
static cl_program makeSelectProgram(cl_kernel *kernel_ptr,
                                    const cl_context context, Type srctype,
                                    Type cmptype, const size_t vec_len)
{
    char testname[256];
    char stypename[32];
    char ctypename[32];
    char extension[128] = "";
    int  err = 0;

    const char *source[] = {
        extension,
        "__kernel void ", testname,
        "(__global ", stypename, " *dest, __global ", stypename, " *src1,\n __global ",
        stypename, " *src2, __global ",  ctypename, " *cmp)\n",
        "{\n"
        "   size_t tid = get_global_id(0);\n"
        "   if( tid < get_global_size(0) )\n"
        "       dest[tid] = select(src1[tid], src2[tid], cmp[tid]);\n"
        "}\n"
    };


    const char *sourceV3[] = {
        extension,
        "__kernel void ", testname,
        "(__global ", stypename, " *dest, __global ", stypename, " *src1,\n __global ",
        stypename, " *src2, __global ",  ctypename, " *cmp)\n",
        "{\n"
        "   size_t tid = get_global_id(0);\n"
        "   size_t size = get_global_size(0);\n"
        "   if( tid + 1 < size ) // can't run off the end\n"
        "       vstore3( select( vload3(tid, src1), vload3(tid, src2), vload3(tid, cmp)), tid, dest );\n"
        "   else if(tid + 1 == size)\n"
        "   {\n"
        // If the size is odd, then we have odd * 3 elements, which is an odd number of scalars in the array
        // If the size is even, then we have even * 3 elements, which is an even number of scalars in the array
        // 3 will never divide evenly into a power of two sized buffer, so the last vec3 will overhang by 1 or 2.
        //  The only even number x in power_of_two < x <= power_of_two+2 is power_of_two+2.
        //  The only odd number x in power_of_two < x <= power_of_two+2 is power_of_two+1.
        // Therefore, odd sizes overhang the end of the array by 1, and even sizes overhang by 2.
        "       size_t leftovers = 1 + (size & 1);\n"
        "       ", stypename, "3 a, b; \n"
        "       ", ctypename, "3 c;\n"
        "       switch( leftovers )  \n"
        "       {\n"
        "           case 2:\n"
        "               a.y = src1[3*tid+1];\n"
        "               b.y = src2[3*tid+1];\n"
        "               c.y = cmp[3*tid+1];\n"
        "           // fall through \n"
        "           case 1:\n"
        "               a.x = src1[3*tid];\n"
        "               b.x = src2[3*tid];\n"
        "               c.x = cmp[3*tid];\n"
        "               break;\n"
        "       }\n"
        "       a = select( a, b, c );\n"
        "       switch( leftovers )  \n"
        "       {\n"
        "           case 2:\n"
        "               dest[3*tid+1] = a.y;\n"
        "           // fall through \n"
        "           case 1:\n"
        "               dest[3*tid] = a.x;\n"
        "               break;\n"
        "       }\n"
        "   }\n"
        "}\n"
    };

    if (srctype == kdouble)
        strcpy( extension, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" );

    if (srctype == khalf)
        strcpy(extension, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");

    // create type name and testname
    switch( vec_len )
    {
        case 1:
            strncpy(stypename, type_name[srctype], sizeof(stypename) - 1);
            stypename[sizeof(stypename) - 1] = '\0';
            strncpy(ctypename, type_name[cmptype], sizeof(ctypename) - 1);
            ctypename[sizeof(ctypename) - 1] = '\0';
            snprintf(testname, sizeof(testname), "select_%s_%s", stypename, ctypename );
            log_info("Building %s(%s, %s, %s)\n", testname, stypename, stypename, ctypename);
            break;
        case 3:
            strncpy(stypename, type_name[srctype], sizeof(stypename) - 1);
            stypename[sizeof(stypename) - 1] = '\0';
            strncpy(ctypename, type_name[cmptype], sizeof(ctypename) - 1);
            ctypename[sizeof(ctypename) - 1] = '\0';
            snprintf(testname, sizeof(testname), "select_%s3_%s3", stypename, ctypename );
            log_info("Building %s(%s3, %s3, %s3)\n", testname, stypename, stypename, ctypename);
            break;
        case 2:
        case 4:
        case 8:
        case 16:
            snprintf(stypename,sizeof(stypename), "%s%d", type_name[srctype],(int)vec_len);
            snprintf(ctypename,sizeof(ctypename), "%s%d", type_name[cmptype],(int)vec_len);
            snprintf(testname, sizeof(testname), "select_%s_%s", stypename, ctypename );
            log_info("Building %s(%s, %s, %s)\n", testname, stypename, stypename, ctypename);
            break;
        default:
            log_error( "Unkown vector type. Aborting...\n" );
            exit(-1);
            break;
    }

    /*
     int j;
     for( j = 0; j < sizeof( source ) / sizeof( source[0] ); j++ )
     log_info( "%s", source[j] );
     */

    // create program
    cl_program program;
    const char **psrc = vec_len == 3 ? sourceV3 : source;
    size_t src_size = vec_len == 3 ? ARRAY_SIZE(sourceV3) : ARRAY_SIZE(source);

    if (create_single_kernel_helper(context, &program, kernel_ptr, src_size,
                                    psrc, testname))
    {
        log_error("Failed to build program (%d)\n", err);
        return NULL;
    }

    return program;
}

#define VECTOR_SIZE_COUNT   6

static int doTest(cl_command_queue queue, cl_context context, Type stype, Type cmptype, cl_device_id device)
{
    int err = CL_SUCCESS;
    MTdataHolder d(gRandomSeed);
    const size_t element_count[VECTOR_SIZE_COUNT] = { 1, 2, 3, 4, 8, 16 };
    clMemWrapper src1, src2, cmp, dest;

    const size_t block_elements = BUFFER_SIZE / type_size[stype];

    // It is more efficient to create the tests all at once since we
    // use the same test data on each of the vector sizes
    clProgramWrapper programs[VECTOR_SIZE_COUNT];
    clKernelWrapper kernels[VECTOR_SIZE_COUNT];

    if (stype == kdouble && !is_extension_available(device, "cl_khr_fp64"))
    {
        log_info("Skipping double because cl_khr_fp64 extension is not supported.\n");
        return 0;
    }

    if (stype == khalf && !is_extension_available(device, "cl_khr_fp16"))
    {
        log_info(
            "Skipping half because cl_khr_fp16 extension is not supported.\n");
        return 0;
    }

    if (gIsEmbedded)
    {
       if (( stype == klong || stype == kulong ) && ! is_extension_available( device, "cles_khr_int64" ))
       {
         log_info("Long types unsupported, skipping.");
         return 0;
       }

       if (( cmptype == klong || cmptype == kulong ) && ! is_extension_available( device, "cles_khr_int64" ))
       {
         log_info("Long types unsupported, skipping.");
         return 0;
       }
    }

    src1 = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    test_error_count(err, "Error: could not allocate src1 buffer\n");
    src2 = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    test_error_count(err, "Error: could not allocate src2 buffer\n");
    cmp = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    test_error_count(err, "Error: could not allocate cmp buffer\n");
    dest = clCreateBuffer( context, CL_MEM_WRITE_ONLY, BUFFER_SIZE, NULL, &err );
    test_error_count(err, "Error: could not allocate dest buffer\n");

    for (size_t vecsize = 0; vecsize < VECTOR_SIZE_COUNT; ++vecsize)
    {
        programs[vecsize] = makeSelectProgram(&kernels[vecsize], context, stype,
                                              cmptype, element_count[vecsize]);
        if (!programs[vecsize] || !kernels[vecsize])
        {
            return -1;
        }

        err = clSetKernelArg(kernels[vecsize], 0, sizeof dest, &dest);
        test_error_count(err, "Error: Cannot set kernel arg dest!\n");
        err = clSetKernelArg(kernels[vecsize], 1, sizeof src1, &src1);
        test_error_count(err, "Error: Cannot set kernel arg dest!\n");
        err = clSetKernelArg(kernels[vecsize], 2, sizeof src2, &src2);
        test_error_count(err, "Error: Cannot set kernel arg dest!\n");
        err = clSetKernelArg(kernels[vecsize], 3, sizeof cmp, &cmp);
        test_error_count(err, "Error: Cannot set kernel arg dest!\n");
    }

    std::vector<char> ref(BUFFER_SIZE);
    std::vector<char> sref(BUFFER_SIZE);
    std::vector<char> src1_host(BUFFER_SIZE);
    std::vector<char> src2_host(BUFFER_SIZE);
    std::vector<char> cmp_host(BUFFER_SIZE);
    std::vector<char> dest_host(BUFFER_SIZE);

    log_info("Testing...");

    initSrcBuffer(src1_host.data(), stype, d);
    initSrcBuffer(src2_host.data(), stype, d);
    err = clEnqueueWriteBuffer(queue, src1, CL_FALSE, 0, BUFFER_SIZE,
                               src1_host.data(), 0, NULL, NULL);
    test_error_count(err, "Error: Could not write src1");

    err = clEnqueueWriteBuffer(queue, src2, CL_FALSE, 0, BUFFER_SIZE,
                               src2_host.data(), 0, NULL, NULL);
    test_error_count(err, "Error: Could not write src2");


    for (int vector_idx = 0; vector_idx < VECTOR_SIZE_COUNT; ++vector_idx)
    {
        const uint32_t vecsize = element_count[vector_idx];
        const size_t vector_size = vecsize * type_size[stype];
        const size_t vector_count =
            (BUFFER_SIZE + vector_size - 1) / vector_size;
        const uint32_t full_msb_mask_elements = vecsize * (1u << vecsize);
        const uint32_t min_cmp_elements = 64 * 1024;
        const uint32_t nb_elements =
            std::max(min_cmp_elements, full_msb_mask_elements);

        for (uint32_t i = 0; i < nb_elements; i += block_elements)
        {
            initCmpBuffer(cmp_host.data(), cmptype, i, block_elements, vecsize,
                          d);

            err = clEnqueueWriteBuffer(queue, cmp, CL_FALSE, 0, BUFFER_SIZE,
                                       cmp_host.data(), 0, NULL, NULL);
            test_error_count(err, "Error: Could not write cmp");

            Select sfunc = (cmptype == ctype[stype][0]) ? vrefSelects[stype][0]
                                                        : vrefSelects[stype][1];
            (*sfunc)(ref.data(), src1_host.data(), src2_host.data(),
                     cmp_host.data(), block_elements);

            sfunc = (cmptype == ctype[stype][0]) ? refSelects[stype][0]
                                                 : refSelects[stype][1];
            (*sfunc)(sref.data(), src1_host.data(), src2_host.data(),
                     cmp_host.data(), block_elements);

            const cl_int pattern = -1;
            err = clEnqueueFillBuffer(queue, dest, &pattern, sizeof(cl_int), 0,
                                      BUFFER_SIZE, 0, nullptr, nullptr);
            test_error_count(err, "clEnqueueFillBuffer failed");


            err = clEnqueueNDRangeKernel(queue, kernels[vector_idx], 1, NULL,
                                         &vector_count, NULL, 0, NULL, NULL);
            test_error_count(err, "clEnqueueNDRangeKernel failed errcode\n");

            err = clEnqueueReadBuffer(queue, dest, CL_TRUE, 0, BUFFER_SIZE,
                                      dest_host.data(), 0, NULL, NULL);
            test_error_count(
                err, "Error: Reading buffer from dest to dest_host failed\n");

            if ((*checkResults[stype])(
                    dest_host.data(),
                    vector_idx == 0 ? sref.data() : ref.data(), block_elements,
                    element_count[vector_idx])
                != 0)
            {
                log_error("vec_size:%d indx: 0x%8x\n",
                          (int)element_count[vector_idx], i);
                return TEST_FAIL;
            }
        }
    }

    return err;
}

REGISTER_TEST(select_uchar_uchar)
{
    return doTest(queue, context, kuchar, kuchar, device);
}
REGISTER_TEST(select_uchar_char)
{
    return doTest(queue, context, kuchar, kchar, device);
}
REGISTER_TEST(select_char_uchar)
{
    return doTest(queue, context, kchar, kuchar, device);
}
REGISTER_TEST(select_char_char)
{
    return doTest(queue, context, kchar, kchar, device);
}
REGISTER_TEST(select_ushort_ushort)
{
    return doTest(queue, context, kushort, kushort, device);
}
REGISTER_TEST(select_ushort_short)
{
    return doTest(queue, context, kushort, kshort, device);
}
REGISTER_TEST(select_short_ushort)
{
    return doTest(queue, context, kshort, kushort, device);
}
REGISTER_TEST(select_short_short)
{
    return doTest(queue, context, kshort, kshort, device);
}
REGISTER_TEST(select_half_ushort)
{
    return doTest(queue, context, khalf, kushort, device);
}
REGISTER_TEST(select_half_short)
{
    return doTest(queue, context, khalf, kshort, device);
}
REGISTER_TEST(select_uint_uint)
{
    return doTest(queue, context, kuint, kuint, device);
}
REGISTER_TEST(select_uint_int)
{
    return doTest(queue, context, kuint, kint, device);
}
REGISTER_TEST(select_int_uint)
{
    return doTest(queue, context, kint, kuint, device);
}
REGISTER_TEST(select_int_int)
{
    return doTest(queue, context, kint, kint, device);
}
REGISTER_TEST(select_float_uint)
{
    return doTest(queue, context, kfloat, kuint, device);
}
REGISTER_TEST(select_float_int)
{
    return doTest(queue, context, kfloat, kint, device);
}
REGISTER_TEST(select_ulong_ulong)
{
    return doTest(queue, context, kulong, kulong, device);
}
REGISTER_TEST(select_ulong_long)
{
    return doTest(queue, context, kulong, klong, device);
}
REGISTER_TEST(select_long_ulong)
{
    return doTest(queue, context, klong, kulong, device);
}
REGISTER_TEST(select_long_long)
{
    return doTest(queue, context, klong, klong, device);
}
REGISTER_TEST(select_double_ulong)
{
    return doTest(queue, context, kdouble, kulong, device);
}
REGISTER_TEST(select_double_long)
{
    return doTest(queue, context, kdouble, klong, device);
}

int main(int argc, const char *argv[])
{
    test_start();

    return runTestHarness(argc, argv, false, 0);
}

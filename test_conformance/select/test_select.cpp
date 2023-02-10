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

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
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
static void initCmpBuffer(void* cmp, Type cmptype, uint64_t start, size_t count);

// make a program that uses select for the given stype (src/dest type),
// ctype (comparison type), veclen (vector length)
static cl_program makeSelectProgram(cl_kernel *kernel_ptr, const cl_context context, Type stype, Type ctype, size_t veclen );

// Creates and execute the select test for the given device, context,
// stype (source/dest type), cmptype (comparison type), using max_tg_size
// number of threads. It runs test for all the different vector lengths
// for the given stype and cmptype.
static int doTest(cl_command_queue queue, cl_context context,
                  Type stype, Type cmptype, cl_device_id device);


static void printUsage( void );

//-----------------------------------------
// Definitions and initializations
//-----------------------------------------

// Define the buffer size that we want to block our test with
#define BUFFER_SIZE (1024*1024)
#define KPAGESIZE 4096


// When we indicate non wimpy mode, the types that are 32 bits value will
// test their entire range and 64 bits test will test the 32 bit
// range.  Otherwise, we test a subset of the range
// [-min_short, min_short]
static bool  s_wimpy_mode = false;
static int s_wimpy_reduction_factor = 256;

// Tests are broken into the major test which is based on the
// src and cmp type and their corresponding vector types and
// sub tests which is for each individual test.  The following
// tracks the subtests
int s_test_cnt = 0;

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

static void initCmpBuffer(void* cmp, Type cmptype, uint64_t start, size_t count) {
    int i;
    assert(cmptype != kfloat);
    switch (type_size[cmptype]) {
        case 1: {
            uint8_t* ub = (uint8_t *)cmp;
            for (i=0; i < count; ++i)
                ub[i] = (uint8_t)start++;
            break;
        }
        case 2: {
            uint16_t* us = (uint16_t *)cmp;
            for (i=0; i < count; ++i)
                us[i] = (uint16_t)start++;
            break;
        }
        case 4: {
            if (!s_wimpy_mode) {
                uint32_t* ui = (uint32_t *)cmp;
                for (i=0; i < count; ++i)
                    ui[i] = (uint32_t)start++;
            }
            else {
                // The short test doesn't iterate over the entire 32 bit space so
                // we alternate between positive and negative values
                int32_t* ui = (int32_t *)cmp;
                int32_t sign = 1;
                for (i=0; i < count; ++i, ++start) {
                    ui[i] = (int32_t)start*sign;
                    sign = sign * -1;
                }
            }
            break;
        }
        case 8: {
            // We don't iterate over the entire space of 64 bit so for the
            // selects, we want to test positive and negative values
            int64_t* ll = (int64_t *)cmp;
            int64_t sign = 1;
            for (i=0; i < count; ++i, ++start) {
                ll[i] = start*sign;
                sign = sign * -1;
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
static cl_program makeSelectProgram(cl_kernel *kernel_ptr, const cl_context context, Type srctype, Type cmptype, size_t vec_len)
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

    // create type name and testname
    switch( vec_len )
    {
        case 1:
            strncpy(stypename, type_name[srctype], sizeof(stypename));
            strncpy(ctypename, type_name[cmptype], sizeof(ctypename));
            snprintf(testname, sizeof(testname), "select_%s_%s", stypename, ctypename );
            log_info("Building %s(%s, %s, %s)\n", testname, stypename, stypename, ctypename);
            break;
        case 3:
            strncpy(stypename, type_name[srctype], sizeof(stypename));
            strncpy(ctypename, type_name[cmptype], sizeof(ctypename));
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
    int s_test_fail = 0;
    MTdataHolder d;
    const size_t element_count[VECTOR_SIZE_COUNT] = { 1, 2, 3, 4, 8, 16 };
    cl_mem src1 = NULL;
    cl_mem src2 = NULL;
    cl_mem cmp = NULL;
    cl_mem dest = NULL;
    void *ref = NULL;
    void *sref = NULL;

    cl_ulong blocks = type_size[stype] * 0x100000000ULL / BUFFER_SIZE;
    size_t block_elements = BUFFER_SIZE / type_size[stype];
    size_t step = s_wimpy_mode ? s_wimpy_reduction_factor : 1;
    cl_ulong cmp_stride = block_elements * step;

    // It is more efficient to create the tests all at once since we
    // use the same test data on each of the vector sizes
    int vecsize;
    cl_program programs[VECTOR_SIZE_COUNT];
    cl_kernel  kernels[VECTOR_SIZE_COUNT];

    if(stype == kdouble && ! is_extension_available( device, "cl_khr_fp64" ))
    {
        log_info("Skipping double because cl_khr_fp64 extension is not supported.\n");
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

    for (vecsize = 0; vecsize < VECTOR_SIZE_COUNT; ++vecsize)
    {
        programs[vecsize] = makeSelectProgram(&kernels[vecsize], context, stype, cmptype, element_count[vecsize] );
        if (!programs[vecsize] || !kernels[vecsize]) {
            ++s_test_fail;
            ++s_test_cnt;
            return -1;
        }
    }

    ref = malloc( BUFFER_SIZE );
    if( NULL == ref ){ log_error("Error: could not allocate ref buffer\n" ); goto exit; }
    sref = malloc( BUFFER_SIZE );
    if( NULL == sref ){ log_error("Error: could not allocate ref buffer\n" ); goto exit; }
    src1 = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    if( err ) { log_error( "Error: could not allocate src1 buffer\n" );  ++s_test_fail; goto exit; }
    src2 = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    if( err ) { log_error( "Error: could not allocate src2 buffer\n" );  ++s_test_fail; goto exit; }
    cmp = clCreateBuffer( context, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &err );
    if( err ) { log_error( "Error: could not allocate cmp buffer\n" );  ++s_test_fail; goto exit; }
    dest = clCreateBuffer( context, CL_MEM_WRITE_ONLY, BUFFER_SIZE, NULL, &err );
    if( err ) { log_error( "Error: could not allocate dest buffer\n" );  ++s_test_fail; goto exit; }


    // We block the test as we are running over the range of compare values
    // "block the test" means "break the test into blocks"
    if( type_size[stype] == 4 )
        cmp_stride = block_elements * step * (0x100000000ULL / 0x100000000ULL);
    if( type_size[stype] == 8 )
        cmp_stride = block_elements * step * (0xffffffffffffffffULL / 0x100000000ULL + 1);

    log_info("Testing...");
    d = MTdataHolder(gRandomSeed);
    uint64_t i;
    for (i=0; i < blocks; i+=step)
    {
        void *s1 = clEnqueueMapBuffer( queue, src1, CL_TRUE, CL_MAP_WRITE, 0, BUFFER_SIZE, 0, NULL, NULL, &err );
        if( err ){ log_error( "Error: Could not map src1" ); goto exit; }
        // Setup the input data to change for each block
        initSrcBuffer( s1, stype, d);

        void *s2 = clEnqueueMapBuffer( queue, src2, CL_TRUE, CL_MAP_WRITE, 0, BUFFER_SIZE, 0, NULL, NULL, &err );
        if( err ){ log_error( "Error: Could not map src2" ); goto exit; }
        // Setup the input data to change for each block
        initSrcBuffer( s2, stype, d);

        void *s3 = clEnqueueMapBuffer( queue, cmp, CL_TRUE, CL_MAP_WRITE, 0, BUFFER_SIZE, 0, NULL, NULL, &err );
        if( err ){ log_error( "Error: Could not map cmp" ); goto exit; }
        // Setup the input data to change for each block
        initCmpBuffer(s3, cmptype, i * cmp_stride, block_elements);

        // Create the reference result
        Select sfunc = (cmptype == ctype[stype][0]) ? vrefSelects[stype][0] : vrefSelects[stype][1];
        (*sfunc)(ref, s1, s2, s3, block_elements);

        sfunc = (cmptype == ctype[stype][0]) ? refSelects[stype][0] : refSelects[stype][1];
        (*sfunc)(sref, s1, s2, s3, block_elements);

        if( (err = clEnqueueUnmapMemObject( queue, src1, s1, 0, NULL, NULL )))
        { log_error( "Error: coult not unmap src1\n" );  ++s_test_fail; goto exit; }
        if( (err = clEnqueueUnmapMemObject( queue, src2, s2, 0, NULL, NULL )))
        { log_error( "Error: coult not unmap src2\n" );  ++s_test_fail; goto exit; }
        if( (err = clEnqueueUnmapMemObject( queue, cmp, s3, 0, NULL, NULL )))
        { log_error( "Error: coult not unmap cmp\n" );  ++s_test_fail; goto exit; }

        for (vecsize = 0; vecsize < VECTOR_SIZE_COUNT; ++vecsize)
        {
            size_t vector_size = element_count[vecsize] * type_size[stype];
            size_t vector_count =  (BUFFER_SIZE + vector_size - 1) / vector_size;

            if((err = clSetKernelArg(kernels[vecsize], 0,  sizeof dest, &dest) ))
            { log_error( "Error: Cannot set kernel arg dest! %d\n", err ); ++s_test_fail; goto exit; }
            if((err = clSetKernelArg(kernels[vecsize], 1,  sizeof src1, &src1) ))
            { log_error( "Error: Cannot set kernel arg dest! %d\n", err ); ++s_test_fail; goto exit; }
            if((err = clSetKernelArg(kernels[vecsize], 2,  sizeof src2, &src2) ))
            { log_error( "Error: Cannot set kernel arg dest! %d\n", err ); ++s_test_fail; goto exit; }
            if((err = clSetKernelArg(kernels[vecsize], 3,  sizeof cmp, &cmp) ))
            { log_error( "Error: Cannot set kernel arg dest! %d\n", err ); ++s_test_fail; goto exit; }


            // Wipe destination
            void *d = clEnqueueMapBuffer( queue, dest, CL_TRUE, CL_MAP_WRITE, 0, BUFFER_SIZE, 0, NULL, NULL, &err );
            if( err ){ log_error( "Error: Could not map dest" );  ++s_test_fail; goto exit; }
            memset( d, -1, BUFFER_SIZE );
            if( (err = clEnqueueUnmapMemObject( queue, dest, d, 0, NULL, NULL ) ) ){ log_error( "Error: Could not unmap dest" ); ++s_test_fail; goto exit; }

            err = clEnqueueNDRangeKernel(queue, kernels[vecsize], 1, NULL, &vector_count, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                log_error("clEnqueueNDRangeKernel failed errcode:%d\n", err);
                ++s_test_fail;
                goto exit;
            }

            d = clEnqueueMapBuffer( queue, dest, CL_TRUE, CL_MAP_READ, 0, BUFFER_SIZE, 0, NULL, NULL, &err );
            if( err ){ log_error( "Error: Could not map dest # 2" );  ++s_test_fail; goto exit; }

            if ((*checkResults[stype])(d, vecsize == 0 ? sref : ref, block_elements, element_count[vecsize])!=0){
                log_error("vec_size:%d indx: 0x%16.16llx\n", (int)element_count[vecsize], i);
                ++s_test_fail;
                goto exit;
            }

            if( (err = clEnqueueUnmapMemObject( queue, dest, d, 0, NULL, NULL ) ) )
            {
                log_error( "Error: Could not unmap dest" );
                ++s_test_fail;
                goto exit;
            }
        } // for vecsize
    } // for i

    if (!s_wimpy_mode)
        log_info(" Passed\n\n");
    else
        log_info(" Wimpy Passed\n\n");

exit:
    if( src1 )  clReleaseMemObject( src1 );
    if( src2 )  clReleaseMemObject( src2 );
    if( cmp )   clReleaseMemObject( cmp );
    if( dest)   clReleaseMemObject( dest );
    if( ref )   free(ref );
    if( sref )  free(sref );

    for (vecsize = 0; vecsize < VECTOR_SIZE_COUNT; vecsize++) {
        clReleaseKernel(kernels[vecsize]);
        clReleaseProgram(programs[vecsize]);
    }
    ++s_test_cnt;
    if (s_test_fail)
    {
        err = TEST_FAIL;
        gFailCount++;
    }
    return err;
}

int test_select_uchar_uchar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kuchar, kuchar, deviceID);
}
int test_select_uchar_char(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kuchar, kchar, deviceID);
}
int test_select_char_uchar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kchar, kuchar, deviceID);
}
int test_select_char_char(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kchar, kchar, deviceID);
}
int test_select_ushort_ushort(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kushort, kushort, deviceID);
}
int test_select_ushort_short(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kushort, kshort, deviceID);
}
int test_select_short_ushort(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kshort, kushort, deviceID);
}
int test_select_short_short(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kshort, kshort, deviceID);
}
int test_select_uint_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kuint, kuint, deviceID);
}
int test_select_uint_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kuint, kint, deviceID);
}
int test_select_int_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kint, kuint, deviceID);
}
int test_select_int_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kint, kint, deviceID);
}
int test_select_float_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kfloat, kuint, deviceID);
}
int test_select_float_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kfloat, kint, deviceID);
}
int test_select_ulong_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kulong, kulong, deviceID);
}
int test_select_ulong_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kulong, klong, deviceID);
}
int test_select_long_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, klong, kulong, deviceID);
}
int test_select_long_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, klong, klong, deviceID);
}
int test_select_double_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kdouble, kulong, deviceID);
}
int test_select_double_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(queue, context, kdouble, klong, deviceID);
}

test_definition test_list[] = {
    ADD_TEST( select_uchar_uchar ),
    ADD_TEST( select_uchar_char ),
    ADD_TEST( select_char_uchar ),
    ADD_TEST( select_char_char ),
    ADD_TEST( select_ushort_ushort ),
    ADD_TEST( select_ushort_short ),
    ADD_TEST( select_short_ushort ),
    ADD_TEST( select_short_short ),
    ADD_TEST( select_uint_uint ),
    ADD_TEST( select_uint_int ),
    ADD_TEST( select_int_uint ),
    ADD_TEST( select_int_int ),
    ADD_TEST( select_float_uint ),
    ADD_TEST( select_float_int ),
    ADD_TEST( select_ulong_ulong ),
    ADD_TEST( select_ulong_long ),
    ADD_TEST( select_long_ulong ),
    ADD_TEST( select_long_long ),
    ADD_TEST( select_double_ulong ),
    ADD_TEST( select_double_long ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char* argv[])
{
    test_start();

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return EXIT_FAILURE;
    }

    const char ** argList = (const char **)calloc( argc, sizeof( char*) );

    if( NULL == argList )
    {
        log_error( "Failed to allocate memory for argList array.\n" );
        return 1;
    }

    argList[0] = argv[0];
    size_t argCount = 1;

    for( int i = 1; i < argc; ++i )
    {
        const char *arg = argv[i];
        if (arg == NULL)
            break;

        if (arg[0] == '-')
        {
            arg++;
            while(*arg != '\0')
            {
                switch(*arg) {
                    case 'h':
                        printUsage();
                        return 0;
                    case 'w':
                        s_wimpy_mode = true;
                        break;
                    case '[':
                        parseWimpyReductionFactor(arg, s_wimpy_reduction_factor);
                        break;
                    default:
                        break;
                }
                arg++;
            }
        }
        else
        {
            argList[argCount] = arg;
            argCount++;
        }
    }

    if (getenv("CL_WIMPY_MODE")) {
        s_wimpy_mode = true;
    }

    if (s_wimpy_mode) {
        log_info("\n");
        log_info("*** WARNING: Testing in Wimpy mode!                     ***\n");
        log_info("*** Wimpy mode is not sufficient to verify correctness. ***\n");
        log_info("*** It gives warm fuzzy feelings and then nevers calls. ***\n\n");
        log_info("*** Wimpy Reduction Factor: %-27u ***\n\n", s_wimpy_reduction_factor);
    }

    int err = runTestHarness(argCount, argList, test_num, test_list, false, 0);

    free( argList );

    return err;
}

static void printUsage( void )
{
    log_info("test_select:  [-w] <optional: test_names> \n");
    log_info("\tdefault is to run the full test on the default device\n");
    log_info("\t-w run in wimpy mode (smoke test)\n");
    log_info("\t-[2^n] Set wimpy reduction factor, recommended range of n is 1-12, default factor(%u)\n", s_wimpy_reduction_factor);
    log_info("\n");
    log_info("Test names:\n");
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}

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

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#if ! defined( _WIN32)
#if ! defined( __ANDROID__ )
#include <sys/sysctl.h>
#endif
#endif
#include <limits.h>
#include "test_select.h"


#include "../../test_common/harness/testHarness.h"


#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/mt19937.h"
#include "../../test_common/harness/parseParameters.h"


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
int s_test_fail = 0;

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

    int i; // generic, re-usable loop variable

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

    if (create_single_kernel_helper(context, &program, kernel_ptr, (cl_uint)(vec_len == 3 ? sizeof(sourceV3) / sizeof(sourceV3[0]) : sizeof(source) / sizeof(source[0])), vec_len == 3 ? sourceV3 : source, testname))
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
    MTdata    d;
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
    d = init_genrand( gRandomSeed );
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

    free_mtdata(d);
    for (vecsize = 0; vecsize < VECTOR_SIZE_COUNT; vecsize++) {
        clReleaseKernel(kernels[vecsize]);
        clReleaseProgram(programs[vecsize]);
    }
    ++s_test_cnt;
    return err;
}

static void printUsage( void )
{
    log_info("test_select:  [-cghw] [test_name|start_test_num] \n");
    log_info("  default is to run the full test on the default device\n");
    log_info("  -w run in wimpy mode (smoke test)\n");
    log_info("  -[2^n] Set wimpy reduction factor, recommended range of n is 1-12, default factor(%u)\n", s_wimpy_reduction_factor);
    log_info("  test_name will run only one test of that name\n");
    log_info("  start_test_num will start running from that num\n");
}

static void printArch( void )
{
    log_info( "sizeof( void*) = %d\n", (int) sizeof( void *) );

#if defined( __APPLE__ )

#if defined( __ppc__ )
    log_info( "ARCH:\tppc\n" );
#elif defined( __ppc64__ )
    log_info( "ARCH:\tppc64\n" );
#elif defined( __i386__ )
    log_info( "ARCH:\ti386\n" );
#elif defined( __x86_64__ )
    log_info( "ARCH:\tx86_64\n" );
#elif defined( __arm__ )
    log_info( "ARCH:\tarm\n" );
#elif defined( __aarch64__ )
    log_info( "ARCH:\taarch64\n" );
#else
#error unknown arch
#endif

    int type = 0;
    size_t typeSize = sizeof( type );
    sysctlbyname( "hw.cputype", &type, &typeSize, NULL, 0 );
    log_info( "cpu type:\t%d\n", type );
    typeSize = sizeof( type );
    sysctlbyname( "hw.cpusubtype", &type, &typeSize, NULL, 0 );
    log_info( "cpu subtype:\t%d\n", type );

#endif
}




//-----------------------------------------
// main
//-----------------------------------------
int main(int argc, const char* argv[]) {
    int i;
    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    cl_platform_id platform_id;
    long           test_start_num = 0;   // start test number
    const char*    exec_testname = NULL;
    cl_device_id      device_id;
    uint32_t       device_frequency = 0;
    uint32_t       compute_devices = 0;


    test_start();

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        test_finish();
        return -1;
    }

    // Maybe we want turn off sleep

    // Check the environmental to see if there is device preference
    char *device_env = getenv("CL_DEVICE_TYPE");
    if (device_env != NULL) {
        if( strcmp( device_env, "gpu" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_GPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_GPU;
        else if( strcmp( device_env, "cpu" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_CPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_CPU;
        else if( strcmp( device_env, "accelerator" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( device_env, "default" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            device_type = CL_DEVICE_TYPE_DEFAULT;
        else
        {
            log_error( "Unknown CL_DEVICE_TYPE environment variable: %s.\nAborting...\n", device_env );
            abort();
        }
    }

    // Check for the wimpy mode environment variable
    if (getenv("CL_WIMPY_MODE")) {
      log_info("*** Detected CL_WIMPY_MODE env\n");
      s_wimpy_mode = 1;
    }

    // Determine if we want to run a particular test or if we want to
    // start running from a certain point and if we want to run on cpu/gpu
    // usage: test_selects [test_name] [start test num] [run_long]
    // default is to run all tests on the gpu and be short
    // test names are of the form select_[src/dest type]_[cmp_type]
    // In the long test, we run the full range for any type >= 32 bits
    // and 32 bits subset for the 64 bit value.
    for (i=1; i < argc; ++i) {
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
                    case 'w':  // Wimpy mode
                        s_wimpy_mode = true;
                        break;
                    case '[':
                        // wimpy reduction factor can be set with the option -[2^n]
                        // Default factor is 256, and n practically can be from 1 to 12
                        {
                            const char *arg_temp = strchr(&arg[1], ']');
                            if (arg_temp != 0)
                            {
                                int new_factor = atoi(&arg[1]);
                                arg = arg_temp; // Advance until ']'
                                if (new_factor && !(new_factor & (new_factor - 1)))
                                {
                                    vlog(" WimpyReduction factor changed from %d to %d \n", s_wimpy_reduction_factor, new_factor);
                                    s_wimpy_reduction_factor = new_factor;
                                }
                                else
                                {
                                vlog(" Error in WimpyReduction factor must be power of 2 \n");
                                }
                            }
                        }
                        break;
                    default:
                        log_error( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        printUsage();
                        return 0;
                }
                arg++;
            }
        }
        else {
            char* t = NULL;
            long num = strtol(argv[i], &t, 0);
            if (t != argv[i])
                test_start_num = num;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_CPU" ) )
                device_type = CL_DEVICE_TYPE_CPU;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_GPU" ) )
                device_type = CL_DEVICE_TYPE_GPU;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_ACCELERATOR" ) )
                device_type = CL_DEVICE_TYPE_ACCELERATOR;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_DEFAULT" ) )
                device_type = CL_DEVICE_TYPE_DEFAULT;
            else if( 0 == strcmp( argv[i], "randomize" ) ) {
                gRandomSeed = (cl_uint) time( NULL );
                log_info("\nRandom seed: %u.\n", gRandomSeed );
            } else {
                // assume it is a test name that we want to execute
                exec_testname = argv[i];
            }
        }
    }


    int err;

    // Get platform
    err = clGetPlatformIDs(1, &platform_id, NULL);
    checkErr(err,"clGetPlatformIDs failed");

    // Get Device information
    err = clGetDeviceIDs(platform_id, device_type, 1, &device_id, 0);
    checkErr(err,"clGetComputeDevices");

    err =  clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    checkErr(err,"clGetComputeConfigInfo 1");

    size_t config_size = sizeof( device_frequency );
#if MULTITHREAD
    if( (err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, config_size, &compute_devices, NULL )) )
#endif
        compute_devices = 1;

    config_size = sizeof(device_frequency);
    if((err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, config_size, &device_frequency, NULL )))
        device_frequency = 1;

    //detect whether profile of the device is embedded
    char profile[1024] = "";
    if( (err = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL ) ) ){}
    else if( strstr(profile, "EMBEDDED_PROFILE" ) )
    {
        gIsEmbedded = 1;
    }


    log_info( "\nCompute Device info:\n" );
    log_info( "\tProcessing with %d devices\n", compute_devices );
    log_info( "\tDevice Frequency: %d MHz\n", device_frequency );

    printDeviceHeader( device_id );
    printArch();

    log_info( "Test binary built %s %s\n", __DATE__, __TIME__ );
    if (s_wimpy_mode) {
        log_info("\n");
        log_info("*** WARNING: Testing in Wimpy mode!                     ***\n");
        log_info("*** Wimpy mode is not sufficient to verify correctness. ***\n");
        log_info("*** It gives warm fuzzy feelings and then nevers calls. ***\n\n");
        log_info("*** Wimpy Reduction Factor: %-27u ***\n\n", s_wimpy_reduction_factor);
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, notify_callback, NULL, NULL);
    checkNull(context, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);
    checkNull(queue, "clCreateCommandQueue");


    if (exec_testname) {
        // Parse name
        // Skip the first part of the name
        bool success = false;
        if (strncmp(exec_testname, "select_", 7) == 0) {
            int i;
            Type src_type = kTypeCount;
            Type cmp_type = kTypeCount;
            char* sptr = (char *)strchr(exec_testname, '_');
            if (sptr) {
                for (++sptr, i=0; i < kTypeCount; i++) {
                    if (strncmp(sptr, type_name[i], strlen(type_name[i])) == 0) {
                        src_type = (Type)i;
                        break;
                    }
                }
                sptr = strchr(sptr, '_');
                if (sptr) {
                    for (++sptr, i=0; i < kTypeCount; i++) {
                        if (strncmp(sptr, type_name[i], strlen(type_name[i])) == 0) {
                            cmp_type = (Type)i;
                            break;
                        }
                    }
                }
            }
            if (src_type != kTypeCount && cmp_type != kTypeCount) {
                success = true;
                log_info("Testing only select_%s_%s\n",
                         type_name[src_type], type_name[cmp_type]);
                if (doTest(queue, context, src_type, cmp_type, device_id) != 0)
                    log_error("*** select_%s_%s FAILED ***\n\n",
                              type_name[src_type], type_name[cmp_type]);
            }
        }
        if (!success) {
            log_error("can not find test:%s", exec_testname);
            return -1;
        }
    }
    else {
        int src_type, j;
        int test_num;
        test_num = 0;
        for (src_type = 0; src_type < kTypeCount; ++src_type) {
            for (j = 0; j < 2; ++j) {
                Type cmp_type = ctype[src_type][j];
                if (++test_num < test_start_num) {
                    log_info("%d) skipping select_%s_%s\n", test_num,
                             type_name[src_type], type_name[cmp_type]);
                }
                else {
                    log_info("%d) Testing select_%s_%s\n",
                             test_num, type_name[src_type], type_name[cmp_type]);
                    if (doTest(queue, context, (Type)src_type, cmp_type, device_id) != 0)
                        log_error("*** %d) select_%s_%s FAILED ***\n\n", test_num,
                                  type_name[src_type], type_name[cmp_type]);
                }
            }
        }
    }

    int error = clFinish(queue);
    if (error) {
        log_error("clFinish failed: %d\n", error);
    }

    clReleaseContext(context);
    clReleaseCommandQueue(queue);

    if (s_test_fail == 0) {
        if (s_test_cnt > 1)
            log_info("PASSED %d of %d tests.\n", s_test_cnt, s_test_cnt);
        else
            log_info("PASSED test.\n");
    } else if (s_test_fail > 0) {
        if (s_test_cnt > 1)
            log_error("FAILED %d of %d tests.\n", s_test_fail, s_test_cnt);
        else
            log_error("FAILED test.\n");
    }

    test_finish();
    return s_test_fail;
}

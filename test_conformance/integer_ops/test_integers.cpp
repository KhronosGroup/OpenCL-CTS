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
#include "testBase.h"
#include "harness/conversions.h"

#include <algorithm>
#include <cinttypes>

#define TEST_SIZE 512

const char *singleParamIntegerKernelSourcePattern =
"__kernel void sample_test(__global %s *sourceA, __global %s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    %s%s tmp = vload%s( tid, destValues );\n"
"    tmp %s= %s( vload%s( tid, sourceA ) );\n"
"    vstore%s( tmp, tid, destValues );\n"
"\n"
"}\n";

const char *singleParamSingleSizeIntegerKernelSourcePattern =
"__kernel void sample_test(__global %s *sourceA, __global %s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] %s= %s( sourceA[tid] );\n"
"}\n";

typedef bool (*singleParamIntegerVerifyFn)( void *source, void *destination, ExplicitType vecType );
static void patchup_divide_results( void *outData, const void *inDataA, const void *inDataB, size_t count, ExplicitType vecType );
bool verify_integer_divideAssign( void *source, void *destination, ExplicitType vecType );
bool verify_integer_moduloAssign( void *source, void *destination, ExplicitType vecType );

int test_single_param_integer_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                                  ExplicitType vecType, size_t vecSize, singleParamIntegerVerifyFn verifyFn,
                                     MTdata d, bool useOpKernel = false )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    cl_long inDataA[TEST_SIZE * 16], outData[TEST_SIZE * 16], inDataB[TEST_SIZE * 16], expected;
    int error, i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4];

    if (! gHasLong && strstr(get_explicit_type_name(vecType),"long"))
    {
       log_info( "WARNING: 64 bit integers are not supported on this device. Skipping %s\n", get_explicit_type_name(vecType) );
       return CL_SUCCESS;
    }

    /* Create the source */
    if( vecSize == 1 )
        sizeName[ 0 ] = 0;
    else
        sprintf( sizeName, "%d", (int)vecSize );

    if( vecSize == 1 )
        sprintf( kernelSource, singleParamSingleSizeIntegerKernelSourcePattern,
                get_explicit_type_name( vecType ), get_explicit_type_name( vecType ),
                useOpKernel ? fnName : "", useOpKernel ? "" : fnName );
    else
        sprintf( kernelSource, singleParamIntegerKernelSourcePattern,
                get_explicit_type_name( vecType ), get_explicit_type_name( vecType ),
                get_explicit_type_name( vecType ), sizeName, sizeName,
                useOpKernel ? fnName : "", useOpKernel ? "" : fnName, sizeName,
                sizeName );

    /* Create kernels */
    programPtr = kernelSource;
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
    {
        log_error("The program we attempted to compile was: \n%s\n", kernelSource);
        return -1;
    }

    /* Generate some streams */
    generate_random_data( vecType, vecSize * TEST_SIZE, d, inDataA );

    streams[0] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecType) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }

    if( useOpKernel )
    {
        // Op kernels use an r/w buffer for the second param, so we need to init it with data
        generate_random_data( vecType, vecSize * TEST_SIZE, d, inDataB );
    }
    streams[1] = clCreateBuffer(
        context, (CL_MEM_READ_WRITE | (useOpKernel ? CL_MEM_COPY_HOST_PTR : 0)),
        get_explicit_type_size(vecType) * vecSize * TEST_SIZE,
        (useOpKernel) ? &inDataB : NULL, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    memset(outData, 0xFF, get_explicit_type_size( vecType ) * TEST_SIZE * vecSize );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0,
                                 get_explicit_type_size( vecType ) * TEST_SIZE * vecSize,
                                 outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    // deal with division by 0 -- any answer is allowed here
    if( verifyFn == verify_integer_divideAssign || verifyFn == verify_integer_moduloAssign )
        patchup_divide_results( outData, inDataA, inDataB, TEST_SIZE * vecSize, vecType );

    /* And verify! */
    char *p = (char *)outData;
    char *in = (char *)inDataA;
    char *in2 = (char *)inDataB;
    for( i = 0; i < (int)TEST_SIZE; i++ )
    {
        for( size_t j = 0; j < vecSize; j++ )
        {
            if( useOpKernel )
                memcpy( &expected, in2, get_explicit_type_size( vecType ) );

            verifyFn( in, &expected, vecType );
            if( memcmp( &expected, p, get_explicit_type_size( vecType ) ) != 0 )
            {
                switch( get_explicit_type_size( vecType ))
                {
                    case 1:
                        if( useOpKernel )
                            log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%2.2x), got (0x%2.2x), sources (0x%2.2x, 0x%2.2x)\n",
                                      (int)i, (int)j,
                                      ((cl_uchar*)&expected)[0],
                                      *( (cl_uchar *)p ),
                                      *( (cl_uchar *)in ),
                                      *( (cl_uchar *)in2 ) );
                        else
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%2.2x), got (0x%2.2x), sources (0x%2.2x)\n",
                                  (int)i, (int)j,
                                   ((cl_uchar*)&expected)[0],
                                   *( (cl_uchar *)p ),
                                   *( (cl_uchar *)in ) );
                        break;

                    case 2:
                        if( useOpKernel )
                            log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%4.4x), got (0x%4.4x), sources (0x%4.4x, 0x%4.4x)\n",
                                      (int)i, (int)j, ((cl_ushort*)&expected)[0], *( (cl_ushort *)p ),
                                      *( (cl_ushort *)in ), *( (cl_ushort *)in2 ) );
                        else
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%4.4x), got (0x%4.4x), sources (0x%4.4x)\n",
                                  (int)i, (int)j, ((cl_ushort*)&expected)[0], *( (cl_ushort *)p ),
                                            *( (cl_ushort *)in ) );
                        break;

                    case 4:
                        if( useOpKernel )
                            log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%8.8x), got (0x%8.8x), sources (0x%8.8x, 0x%8.8x)\n",
                                      (int)i, (int)j, ((cl_uint*)&expected)[0], *( (cl_uint *)p ),
                                      *( (cl_uint *)in ), *( (cl_uint *)in2 ) );
                        else
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%8.8x), got (0x%8.8x), sources (0x%8.8x)\n",
                                  (int)i, (int)j, ((cl_uint*)&expected)[0], *( (cl_uint *)p ),
                                            *( (cl_uint *)in ) );
                        break;

                    case 8:
                        if( useOpKernel )
                            log_error("ERROR: Data sample %d:%d does not "
                                      "validate! Expected (0x%16.16" PRIx64
                                      "), got (0x%16.16" PRIx64
                                      "), sources (0x%16.16" PRIx64
                                      ", 0x%16.16" PRIx64 ")\n",
                                      (int)i, (int)j,
                                      ((cl_ulong *)&expected)[0],
                                      *((cl_ulong *)p), *((cl_ulong *)in),
                                      *((cl_ulong *)in2));
                        else
                            log_error("ERROR: Data sample %d:%d does not "
                                      "validate! Expected (0x%16.16" PRIx64
                                      "), got (0x%16.16" PRIx64
                                      "), sources (0x%16.16" PRIx64 ")\n",
                                      (int)i, (int)j,
                                      ((cl_ulong *)&expected)[0],
                                      *((cl_ulong *)p), *((cl_ulong *)in));
                        break;
                }
                return -1;
            }
            p += get_explicit_type_size( vecType );
            in += get_explicit_type_size( vecType );
            in2 += get_explicit_type_size( vecType );
        }
    }

    return 0;
}

int test_single_param_integer_fn( cl_command_queue queue, cl_context context, const char *fnName, singleParamIntegerVerifyFn verifyFn, bool useOpKernel = false )
{
    ExplicitType types[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 }; // TODO 3 not tested
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );

    for( typeIndex = 0; types[ typeIndex ] != kNumExplicitTypes; typeIndex++ )
    {
        if ((types[ typeIndex ] == kLong || types[ typeIndex ] == kULong) && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            if( test_single_param_integer_kernel(queue, context, fnName, types[ typeIndex ], vecSizes[ index ], verifyFn, seed, useOpKernel ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( types[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

bool verify_integer_clz( void *source, void *destination, ExplicitType vecType )
{
    cl_long testValue;
    int count;
    int typeBits;

    switch( vecType )
    {
        case kChar:
            testValue = *( (cl_char *)source );
            typeBits = 8 * sizeof( cl_char );
            break;
        case kUChar:
            testValue = *( (cl_uchar *)source );
            typeBits = 8 * sizeof( cl_uchar );
            break;
        case kShort:
            testValue = *( (cl_short *)source );
            typeBits = 8 * sizeof( cl_short );
            break;
        case kUShort:
            testValue = *( (cl_ushort *)source );
            typeBits = 8 * sizeof( cl_ushort );
            break;
        case kInt:
            testValue = *( (cl_int *)source );
            typeBits = 8 * sizeof( cl_int );
            break;
        case kUInt:
            testValue = *( (cl_uint *)source );
            typeBits = 8 * sizeof( cl_uint );
            break;
        case kLong:
            testValue = *( (cl_long *)source );
            typeBits = 8 * sizeof( cl_long );
            break;
        case kULong:
            // Hack for now: just treat it as a signed cl_long, since it won't matter for bitcounting
            testValue = *( (cl_ulong *)source );
            typeBits = 8 * sizeof( cl_ulong );
            break;
        default:
            // Should never happen
            return false;
    }

    count = typeBits;
    if( testValue )
    {
        testValue <<= 8 * sizeof( testValue ) - typeBits;
        for( count = 0; 0 == (testValue & CL_LONG_MIN); count++ )
            testValue <<= 1;
    }

    switch( vecType )
    {
        case kChar:
            *( (cl_char *)destination ) = count;
            break;
        case kUChar:
            *( (cl_uchar *)destination ) = count;
            break;
        case kShort:
            *( (cl_short *)destination ) = count;
            break;
        case kUShort:
            *( (cl_ushort *)destination ) = count;
            break;
        case kInt:
            *( (cl_int *)destination ) = count;
            break;
        case kUInt:
            *( (cl_uint *)destination ) = count;
            break;
        case kLong:
            *( (cl_long *)destination ) = count;
            break;
        case kULong:
            *( (cl_ulong *)destination ) = count;
            break;
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_clz(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_single_param_integer_fn( queue, context, "clz", verify_integer_clz );
}


bool verify_integer_ctz( void *source, void *destination, ExplicitType vecType )
{
  cl_long testValue;
  int count;
  int typeBits;

  switch( vecType )
  {
  case kChar:
    testValue = *( (cl_char *)source );
    typeBits = 8 * sizeof( cl_char );
    break;
  case kUChar:
    testValue = *( (cl_uchar *)source );
    typeBits = 8 * sizeof( cl_uchar );
    break;
  case kShort:
    testValue = *( (cl_short *)source );
    typeBits = 8 * sizeof( cl_short );
    break;
  case kUShort:
    testValue = *( (cl_ushort *)source );
    typeBits = 8 * sizeof( cl_ushort );
    break;
  case kInt:
    testValue = *( (cl_int *)source );
    typeBits = 8 * sizeof( cl_int );
    break;
  case kUInt:
    testValue = *( (cl_uint *)source );
    typeBits = 8 * sizeof( cl_uint );
    break;
  case kLong:
    testValue = *( (cl_long *)source );
    typeBits = 8 * sizeof( cl_long );
    break;
  case kULong:
    // Hack for now: just treat it as a signed cl_long, since it won't matter for bitcounting
    testValue = *( (cl_ulong *)source );
    typeBits = 8 * sizeof( cl_ulong );
    break;
  default:
    // Should never happen
    return false;
  }

  if ( testValue == 0 )
    count = typeBits;
  else
  {
    for( count = 0; (0 == (testValue & 0x1)); count++ )
      testValue >>= 1;
  }

  switch( vecType )
  {
  case kChar:
    *( (cl_char *)destination ) = count;
    break;
  case kUChar:
    *( (cl_uchar *)destination ) = count;
    break;
  case kShort:
    *( (cl_short *)destination ) = count;
    break;
  case kUShort:
    *( (cl_ushort *)destination ) = count;
    break;
  case kInt:
    *( (cl_int *)destination ) = count;
    break;
  case kUInt:
    *( (cl_uint *)destination ) = count;
    break;
  case kLong:
    *( (cl_long *)destination ) = count;
    break;
  case kULong:
    *( (cl_ulong *)destination ) = count;
    break;
  default:
    // Should never happen
    return false;
  }
  return true;
}


int test_integer_ctz(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  return test_single_param_integer_fn( queue, context, "ctz", verify_integer_ctz );
}

#define OP_CASE( op, sizeName, size ) \
    case sizeName: \
    {    \
        cl_##size *d = (cl_##size *)destination; \
        *d op##= *( (cl_##size *)source ); \
        break; \
    }

#define OP_CASES( op ) \
    switch( vecType ) \
    {                    \
        OP_CASE( op, kChar, char ) \
        OP_CASE( op, kUChar, uchar ) \
        OP_CASE( op, kShort, short ) \
        OP_CASE( op, kUShort, ushort ) \
        OP_CASE( op, kInt, int ) \
        OP_CASE( op, kUInt, uint ) \
        OP_CASE( op, kLong, long ) \
        OP_CASE( op, kULong, ulong ) \
        default: \
            break; \
    }

#define OP_TEST( op, opName ) \
    bool verify_integer_##opName##Assign( void *source, void *destination, ExplicitType vecType )    \
    {    \
        OP_CASES( op )    \
        return true; \
    }    \
    int test_integer_##opName##Assign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)    \
    {    \
        return test_single_param_integer_fn( queue, context, #op, verify_integer_##opName##Assign, true ); \
    }

OP_TEST( +, add )
OP_TEST( -, subtract )
OP_TEST( *, multiply )
OP_TEST( ^, exclusiveOr )
OP_TEST( |, or )
OP_TEST( &, and )

#define OP_CASE_GUARD( op, sizeName, size ) \
    case sizeName: \
    {    \
        cl_##size *d = (cl_##size *)destination; \
        cl_##size *s = (cl_##size *)source;     \
        if( *s == 0 )                           \
            *d = -1;                            \
        else                                    \
            *d op##= *s;                        \
        break; \
    }

#define OP_CASE_GUARD_SIGNED( op, sizeName, size, MIN_VAL ) \
    case sizeName: \
    {    \
        cl_##size *d = (cl_##size *)destination; \
        cl_##size *s = (cl_##size *)source;     \
        if( *s == 0 || (*d == MIN_VAL && *s == -1))  \
            *d = -1 - MIN_VAL;                  \
        else                                    \
            *d op##= *s;                        \
        break; \
    }

#define OP_CASES_GUARD( op ) \
    switch( vecType ) \
    {                    \
        OP_CASE_GUARD_SIGNED( op, kChar, char, CL_CHAR_MIN ) \
        OP_CASE_GUARD( op, kUChar, uchar ) \
        OP_CASE_GUARD_SIGNED( op, kShort, short, CL_SHRT_MIN ) \
        OP_CASE_GUARD( op, kUShort, ushort ) \
        OP_CASE_GUARD_SIGNED( op, kInt, int, CL_INT_MIN ) \
        OP_CASE_GUARD( op, kUInt, uint ) \
        OP_CASE_GUARD_SIGNED( op, kLong, long, CL_LONG_MIN ) \
        OP_CASE_GUARD( op, kULong, ulong ) \
        default: \
            break; \
    }

#define OP_TEST_GUARD( op, opName ) \
    bool verify_integer_##opName##Assign( void *source, void *destination, ExplicitType vecType )    \
    {    \
        OP_CASES_GUARD( op )    \
        return true;            \
    }    \
    int test_integer_##opName##Assign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)    \
    {    \
        return test_single_param_integer_fn( queue, context, #op, verify_integer_##opName##Assign, true ); \
    }

OP_TEST_GUARD( /, divide )
OP_TEST_GUARD( %, modulo )

#define PATCH_CASE( _out, _src, _dest, _count, _cl_type )           \
    {                                                               \
        const _cl_type *denom = (const _cl_type* ) _src;            \
        _cl_type *result = (_cl_type* ) _out;                       \
        for( size_t i = 0; i < _count; i++ )                        \
            if( denom[i] == 0 )                                     \
                result[i] = (_cl_type) -1;                          \
    }

#define PATCH_CASE_SIGNED( _out, _src, _dest, _count, _cl_type, _MIN_VAL )      \
    {                                                                           \
        const _cl_type *num = (const _cl_type* ) _dest;                         \
        const _cl_type *denom = (const _cl_type* ) _src;                        \
        _cl_type *result = (_cl_type* ) _out;                                   \
        for( size_t i = 0; i < _count; i++ )                                    \
            if( denom[i] == 0 || ( num[i] == _MIN_VAL && denom[i] == -1))       \
                result[i] = -1 - _MIN_VAL;                                      \
    }

static void patchup_divide_results( void *outData, const void *inDataA, const void *inDataB, size_t count, ExplicitType vecType )
{
    switch( vecType )
    {
        case kChar:
            PATCH_CASE_SIGNED( outData, inDataA, inDataB, count, cl_char, CL_CHAR_MIN )
            break;
        case kUChar:
            PATCH_CASE( outData, inDataA, inDataB, count, cl_uchar )
            break;
        case kShort:
            PATCH_CASE_SIGNED( outData, inDataA, inDataB, count, cl_short, CL_SHRT_MIN )
            break;
        case kUShort:
            PATCH_CASE( outData, inDataA, inDataB, count, cl_ushort )
            break;
        case kInt:
            PATCH_CASE_SIGNED( outData, inDataA, inDataB, count, cl_int, CL_INT_MIN )
            break;
        case kUInt:
            PATCH_CASE( outData, inDataA, inDataB, count, cl_uint )
            break;
        case kLong:
            PATCH_CASE_SIGNED( outData, inDataA, inDataB, count, cl_long, CL_LONG_MIN )
            break;
        case kULong:
            PATCH_CASE( outData, inDataA, inDataB, count, cl_ulong )
            break;
        default:
            log_error( "ERROR: internal test error -- unknown data type %d\n", vecType );
            break;
    }
}

const char *twoParamIntegerKernelSourcePattern =
"__kernel void sample_test(__global %s%s *sourceA, __global %s%s *sourceB, __global %s%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    %s%s sA = %s;\n"
"    %s%s sB = %s;\n"
"    %s%s dst = %s( sA, sB );\n"
"     %s;\n"
"\n"
"}\n";

typedef bool (*twoParamIntegerVerifyFn)( void *sourceA, void *sourceB, void *destination, ExplicitType vecType );

static char * build_load_statement( char *outString, size_t vecSize, const char *name )
{
    if( vecSize != 3 )
        sprintf( outString, "%s[ tid ]", name );
    else
        sprintf( outString, "vload3( tid, %s )", name );
    return outString;
}

static char * build_store_statement( char *outString, size_t vecSize, const char *name, const char *srcName )
{
    if( vecSize != 3 )
        sprintf( outString, "%s[ tid ] = %s", name, srcName );
    else
        sprintf( outString, "vstore3( %s, tid, %s )", srcName, name );
    return outString;
}

int test_two_param_integer_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                                     ExplicitType vecAType, ExplicitType vecBType, unsigned int vecSize, twoParamIntegerVerifyFn verifyFn, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    cl_long inDataA[TEST_SIZE * 16], inDataB[TEST_SIZE * 16], outData[TEST_SIZE * 16], expected;
    int error, i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4], paramSizeName[4];

    // embedded profiles don't support long/ulong datatypes
    if (! gHasLong && strstr(get_explicit_type_name(vecAType),"long"))
    {
       log_info( "WARNING: 64 bit integers are not supported on this device. Skipping %s\n", get_explicit_type_name(vecAType) );
       return CL_SUCCESS;
    }

    /* Create the source */
    if( vecSize == 1 )
        sizeName[ 0 ] = 0;
    else
        sprintf( sizeName, "%d", vecSize );
    if( ( vecSize == 1 ) || ( vecSize == 3 ) )
        paramSizeName[ 0 ] = 0;
        else
        sprintf( paramSizeName, "%d", vecSize );

    char sourceALoad[ 128 ], sourceBLoad[ 128 ], destStore[ 128 ];

    sprintf( kernelSource, twoParamIntegerKernelSourcePattern,
                get_explicit_type_name( vecAType ), paramSizeName,
                get_explicit_type_name( vecBType ), paramSizeName,
                get_explicit_type_name( vecAType ), paramSizeName,
                get_explicit_type_name( vecAType ), sizeName, build_load_statement( sourceALoad, (size_t)vecSize, "sourceA" ),
                get_explicit_type_name( vecBType ), sizeName, build_load_statement( sourceBLoad, (size_t)vecSize, "sourceB" ),
                get_explicit_type_name( vecAType ), sizeName,
                fnName,
                build_store_statement( destStore, (size_t)vecSize, "destValues", "dst" )
                );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        log_error("The program we attempted to compile was: \n%s\n", kernelSource);
        return -1;
    }

    /* Generate some streams */
    generate_random_data( vecAType, vecSize * TEST_SIZE, d, inDataA );
    generate_random_data( vecBType, vecSize * TEST_SIZE, d, inDataB );

    streams[0] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecAType) * vecSize * TEST_SIZE, &inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecBType) * vecSize * TEST_SIZE, &inDataB, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(
        context, CL_MEM_READ_WRITE,
        get_explicit_type_size(vecAType) * vecSize * TEST_SIZE, NULL, NULL);
    if( streams[2] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 2, sizeof( streams[2] ), &streams[2] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    memset(outData, 0xFF, get_explicit_type_size( vecAType ) * TEST_SIZE * vecSize);

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[2], CL_TRUE, 0,
                                 get_explicit_type_size( vecAType ) * TEST_SIZE * vecSize, outData, 0,
                                 NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    char *inA = (char *)inDataA;
    char *inB = (char *)inDataB;
    char *out = (char *)outData;
    for( i = 0; i < (int)TEST_SIZE; i++ )
    {
        for( size_t j = 0; j < vecSize; j++ )
        {
            bool test = verifyFn( inA, inB, &expected, vecAType );
            if( test && ( memcmp( &expected, out, get_explicit_type_size( vecAType ) ) != 0 ) )
            {
                switch( get_explicit_type_size( vecAType ))
                {
                    case 1:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%2.2x), got (0x%2.2x), sources (0x%2.2x, 0x%2.2x), TEST_SIZE %d\n",
                                   (int)i, (int)j, ((cl_uchar*)&expected)[ 0 ], *( (cl_uchar *)out ),
                                   *( (cl_uchar *)inA ),
                                   *( (cl_uchar *)inB ) ,
                                   TEST_SIZE);
                        break;

                    case 2:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%4.4x), got (0x%4.4x), sources (0x%4.4x, 0x%4.4x), TEST_SIZE %d\n",
                                   (int)i, (int)j, ((cl_ushort*)&expected)[ 0 ], *( (cl_ushort *)out ),
                                   *( (cl_ushort *)inA ),
                                   *( (cl_ushort *)inB ),
                                   TEST_SIZE);
                        break;

                    case 4:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%8.8x), got (0x%8.8x), sources (0x%8.8x, 0x%8.8x)\n",
                                  (int)i, (int)j, ((cl_uint*)&expected)[ 0 ], *( (cl_uint *)out ),
                                            *( (cl_uint *)inA ),
                                            *( (cl_uint *)inB ) );
                        break;

                    case 8:
                        log_error("ERROR: Data sample %d:%d does not validate! "
                                  "Expected (0x%16.16" PRIx64
                                  "), got (0x%16.16" PRIx64
                                  "), sources (0x%16.16" PRIx64
                                  ", 0x%16.16" PRIx64 ")\n",
                                  (int)i, (int)j, ((cl_ulong *)&expected)[0],
                                  *((cl_ulong *)out), *((cl_ulong *)inA),
                                  *((cl_ulong *)inB));
                        break;
                }
                return -1;
            }
            inA += get_explicit_type_size( vecAType );
            inB += get_explicit_type_size( vecBType );
            out += get_explicit_type_size( vecAType );
        }
    }

    return 0;
}

int test_two_param_integer_fn(cl_command_queue queue, cl_context context, const char *fnName, twoParamIntegerVerifyFn verifyFn)
{
    ExplicitType types[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 }; // TODO : 3 not tested
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );

    for( typeIndex = 0; types[ typeIndex ] != kNumExplicitTypes; typeIndex++ )
    {
        if (( types[ typeIndex ] == kLong || types[ typeIndex ] == kULong) && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            if( test_two_param_integer_kernel(queue, context, fnName, types[ typeIndex ], types[ typeIndex ], vecSizes[ index ], verifyFn, seed ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( types[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

int test_two_param_unmatched_integer_fn(cl_command_queue queue, cl_context context, const char *fnName, twoParamIntegerVerifyFn verifyFn)
{
    ExplicitType types[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 4, 8, 16, 0 };
    unsigned int index, typeAIndex, typeBIndex;
    int retVal = 0;
    RandomSeed seed( gRandomSeed );

    for( typeAIndex = 0; types[ typeAIndex ] != kNumExplicitTypes; typeAIndex++ )
    {
        if (( types[ typeAIndex ] == kLong || types[ typeAIndex ] == kULong) && !gHasLong)
            continue;

        for( typeBIndex = 0; types[ typeBIndex ] != kNumExplicitTypes; typeBIndex++ )
        {
            if (( types[ typeBIndex ] == kLong || types[ typeBIndex ] == kULong) && !gHasLong)
                continue;

            for( index = 0; vecSizes[ index ] != 0; index++ )
            {
                if( test_two_param_integer_kernel( queue, context, fnName, types[ typeAIndex ], types[ typeBIndex ], vecSizes[ index ], verifyFn, seed ) != 0 )
                {
                    log_error( "   Vector %s%d / %s%d FAILED\n", get_explicit_type_name( types[ typeAIndex ] ), vecSizes[ index ],  get_explicit_type_name( types[ typeBIndex ] ), vecSizes[ index ] );
                    retVal = -1;
                }
            }
        }
    }

    return retVal;
}

bool verify_integer_hadd( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    cl_long testValueA, testValueB, overflow;
    cl_ulong uValueA, uValueB, uOverflow;

    switch( vecType )
    {
        case kChar:
            testValueA = *( (cl_char *)sourceA );
            testValueB = *( (cl_char *)sourceB );
            *( (cl_char *)destination ) = (cl_char)( ( testValueA + testValueB ) >> 1 );
            break;
        case kUChar:
            testValueA = *( (cl_uchar *)sourceA );
            testValueB = *( (cl_uchar *)sourceB );
            *( (cl_uchar *)destination ) = (cl_uchar)( ( testValueA + testValueB ) >> 1 );
            break;
        case kShort:
            testValueA = *( (cl_short *)sourceA );
            testValueB = *( (cl_short *)sourceB );
            *( (cl_short *)destination ) = (cl_short)( ( testValueA + testValueB ) >> 1 );
            break;
        case kUShort:
            testValueA = *( (cl_ushort *)sourceA );
            testValueB = *( (cl_ushort *)sourceB );
            *( (cl_ushort *)destination ) = (cl_ushort)( ( testValueA + testValueB ) >> 1 );
            break;
        case kInt:
            testValueA = *( (cl_int *)sourceA );
            testValueB = *( (cl_int *)sourceB );
            *( (cl_int *)destination ) = (cl_int)( ( testValueA + testValueB ) >> 1 );
            break;
        case kUInt:
            testValueA = *( (cl_uint *)sourceA );
            testValueB = *( (cl_uint *)sourceB );
            *( (cl_uint *)destination ) = (cl_uint)( ( testValueA + testValueB ) >> 1 );
            break;
        case kLong:
            // The long way to avoid dropping bits
            testValueA = *( (cl_long *)sourceA );
            testValueB = *( (cl_long *)sourceB );
            overflow = ( testValueA & 0x1 ) + ( testValueB & 0x1 );
            *( (cl_long *)destination ) = ( ( testValueA >> 1 ) + ( testValueB >> 1 ) ) + ( overflow >> 1 );
            break;
        case kULong:
            // The long way to avoid dropping bits
            uValueA = *( (cl_ulong *)sourceA );
            uValueB = *( (cl_ulong *)sourceB );
            uOverflow = ( uValueA & 0x1 ) + ( uValueB & 0x1 );
            *( (cl_ulong *)destination ) = ( ( uValueA >> 1 ) + ( uValueB >> 1 ) ) + ( uOverflow >> 1 );
            break;
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_hadd(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "hadd", verify_integer_hadd );
}

bool verify_integer_rhadd( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    cl_long testValueA, testValueB, overflow;
    cl_ulong uValueA, uValueB, uOverflow;

    switch( vecType )
    {
        case kChar:
            testValueA = *( (cl_char *)sourceA );
            testValueB = *( (cl_char *)sourceB );
            *( (cl_char *)destination ) = (cl_char)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kUChar:
            testValueA = *( (cl_uchar *)sourceA );
            testValueB = *( (cl_uchar *)sourceB );
            *( (cl_uchar *)destination ) = (cl_uchar)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kShort:
            testValueA = *( (cl_short *)sourceA );
            testValueB = *( (cl_short *)sourceB );
            *( (cl_short *)destination ) = (cl_short)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kUShort:
            testValueA = *( (cl_ushort *)sourceA );
            testValueB = *( (cl_ushort *)sourceB );
            *( (cl_ushort *)destination ) = (cl_ushort)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kInt:
            testValueA = *( (cl_int *)sourceA );
            testValueB = *( (cl_int *)sourceB );
            *( (cl_int *)destination ) = (cl_int)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kUInt:
            testValueA = *( (cl_uint *)sourceA );
            testValueB = *( (cl_uint *)sourceB );
            *( (cl_uint *)destination ) = (cl_uint)( ( testValueA + testValueB + 1 ) >> 1 );
            break;
        case kLong:
            // The long way to avoid dropping bits
            testValueA = *( (cl_long *)sourceA );
            testValueB = *( (cl_long *)sourceB );
            overflow = ( testValueA | testValueB ) & 0x1;
            *( (cl_long *)destination ) = ( ( testValueA >> 1 ) + ( testValueB >> 1 ) ) + overflow;
            break;
        case kULong:
            // The long way to avoid dropping bits
            uValueA = *( (cl_ulong *)sourceA );
            uValueB = *( (cl_ulong *)sourceB );
            uOverflow = ( uValueA | uValueB ) & 0x1;
            *( (cl_ulong *)destination ) = ( ( uValueA >> 1 ) + ( uValueB >> 1 ) ) + uOverflow;
            break;
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_rhadd(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "rhadd", verify_integer_rhadd );
}

#define MIN_CASE( type, const ) \
    case const : \
    {            \
        cl_##type valueA = *( (cl_##type *)sourceA ); \
        cl_##type valueB = *( (cl_##type *)sourceB ); \
        *( (cl_##type *)destination ) = (cl_##type)( valueB < valueA ? valueB : valueA ); \
        break; \
    }

bool verify_integer_min( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    switch( vecType )
    {
        MIN_CASE( char, kChar )
        MIN_CASE( uchar, kUChar )
        MIN_CASE( short, kShort )
        MIN_CASE( ushort, kUShort )
        MIN_CASE( int, kInt )
        MIN_CASE( uint, kUInt )
        MIN_CASE( long, kLong )
        MIN_CASE( ulong, kULong )
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_min(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "min", verify_integer_min);
}

#define MAX_CASE( type, const ) \
    case const : \
    {            \
        cl_##type valueA = *( (cl_##type *)sourceA ); \
        cl_##type valueB = *( (cl_##type *)sourceB ); \
        *( (cl_##type *)destination ) = (cl_##type)( valueA < valueB ? valueB : valueA ); \
        break; \
    }

bool verify_integer_max( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    switch( vecType )
    {
            MAX_CASE( char, kChar )
            MAX_CASE( uchar, kUChar )
            MAX_CASE( short, kShort )
            MAX_CASE( ushort, kUShort )
            MAX_CASE( int, kInt )
            MAX_CASE( uint, kUInt )
            MAX_CASE( long, kLong )
            MAX_CASE( ulong, kULong )
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_max(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "max", verify_integer_max );
}


void multiply_unsigned_64_by_64( cl_ulong sourceA, cl_ulong sourceB, cl_ulong &destLow, cl_ulong &destHi )
{
    cl_ulong lowA, lowB;
    cl_ulong highA, highB;

    // Split up the values
    lowA = sourceA & 0xffffffff;
    highA = sourceA >> 32;
    lowB = sourceB & 0xffffffff;
    highB = sourceB >> 32;

    // Note that, with this split, our multiplication becomes:
    //     ( a * b )
    // = ( ( aHI << 32 + aLO ) * ( bHI << 32 + bLO ) ) >> 64
    // = ( ( aHI << 32 * bHI << 32 ) + ( aHI << 32 * bLO ) + ( aLO * bHI << 32 ) + ( aLO * bLO ) ) >> 64
    // = ( ( aHI * bHI << 64 ) + ( aHI * bLO << 32 ) + ( aLO * bHI << 32 ) + ( aLO * bLO ) ) >> 64
    // = ( aHI * bHI ) + ( aHI * bLO >> 32 ) + ( aLO * bHI >> 32 ) + ( aLO * bLO >> 64 )

    // Now, since each value is 32 bits, the max size of any multiplication is:
    // ( 2 ^ 32 - 1 ) * ( 2 ^ 32 - 1 ) = 2^64 - 4^32 + 1 = 2^64 - 2^33 + 1, which fits within 64 bits
    // Which means we can do each component within a 64-bit integer as necessary (each component above marked as AB1 - AB4)
    cl_ulong aHibHi = highA * highB;
    cl_ulong aHibLo = highA * lowB;
    cl_ulong aLobHi = lowA * highB;
    cl_ulong aLobLo = lowA * lowB;

    // Assemble terms.
    //  We note that in certain cases, sums of products cannot overflow:
    //
    //      The maximum product of two N-bit unsigned numbers is
    //
    //          (2**N-1)^2 = 2**2N - 2**(N+1) + 1
    //
    //      We note that we can add the maximum N-bit number to the 2N-bit product twice without overflow:
    //
    //          (2**N-1)^2 + 2*(2**N-1) = 2**2N - 2**(N+1) + 1 + 2**(N+1) - 2 = 2**2N - 1
    //
    //  If we breakdown the product of two numbers a,b into high and low halves of partial products as follows:
    //
    //                                              a.hi                a.lo
    // x                                            b.hi                b.lo
    //===============================================================================
    //  (b.hi*a.hi).hi      (b.hi*a.hi).lo
    //                      (b.lo*a.hi).hi      (b.lo*a.hi).lo
    //                      (b.hi*a.lo).hi      (b.hi*a.lo).lo
    // +                                        (b.lo*a.lo).hi      (b.lo*a.lo).lo
    //===============================================================================
    //
    // The (b.lo*a.lo).lo term cannot cause a carry, so we can ignore them for now.  We also know from above, that we can add (b.lo*a.lo).hi
    // and (b.hi*a.lo).lo to the 2N bit term [(b.lo*a.hi).hi + (b.lo*a.hi).lo] without overflow.  That takes care of all of the terms
    // on the right half that might carry.  Do that now.
    //
    cl_ulong aLobLoHi = aLobLo >> 32;
    cl_ulong aLobHiLo = aLobHi & 0xFFFFFFFFULL;
    aHibLo += aLobLoHi + aLobHiLo;

    // That leaves us with these terms:
    //
    //                                              a.hi                a.lo
    // x                                            b.hi                b.lo
    //===============================================================================
    //  (b.hi*a.hi).hi      (b.hi*a.hi).lo
    //                      (b.hi*a.lo).hi
    //                    [ (b.lo*a.hi).hi + (b.lo*a.hi).lo + other ]
    // +                                                                (b.lo*a.lo).lo
    //===============================================================================

    // All of the overflow potential from the right half has now been accumulated into the [ (b.lo*a.hi).hi + (b.lo*a.hi).lo ] 2N bit term.
    // We can safely separate into high and low parts. Per our rule above, we know we can accumulate the high part of that and (b.hi*a.lo).hi
    // into the 2N bit term (b.lo*a.hi) without carry.  The low part can be pieced together with (b.lo*a.lo).lo, to give the final low result

    destHi = aHibHi + (aHibLo >> 32 ) + (aLobHi >> 32);             // Cant overflow
    destLow = (aHibLo << 32) | ( aLobLo & 0xFFFFFFFFULL );
}

void multiply_signed_64_by_64( cl_long sourceA, cl_long sourceB, cl_ulong &destLow, cl_long &destHi )
{
    // Find sign of result
    cl_long aSign = sourceA >> 63;
    cl_long bSign = sourceB >> 63;
    cl_long resultSign = aSign ^ bSign;

    // take absolute values of the argument
    sourceA = (sourceA ^ aSign) - aSign;
    sourceB = (sourceB ^ bSign) - bSign;

    cl_ulong hi;
    multiply_unsigned_64_by_64( (cl_ulong) sourceA, (cl_ulong) sourceB, destLow, hi );

    // Fix the sign
    if( resultSign )
    {
        destLow ^= resultSign;
        hi  ^= resultSign;
        destLow -= resultSign;

        //carry if necessary
        if( 0 == destLow )
            hi -= resultSign;
    }

    destHi = (cl_long) hi;
}

bool verify_integer_mul_hi( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    cl_long testValueA, testValueB, highSigned;
    cl_ulong highUnsigned, lowHalf;

    switch( vecType )
    {
        case kChar:
            testValueA = *( (cl_char *)sourceA );
            testValueB = *( (cl_char *)sourceB );
            *( (cl_char *)destination ) = (cl_char)( ( testValueA * testValueB ) >> 8 );
            break;
        case kUChar:
            testValueA = *( (cl_uchar *)sourceA );
            testValueB = *( (cl_uchar *)sourceB );
            *( (cl_uchar *)destination ) = (cl_uchar)( ( testValueA * testValueB ) >> 8 );
            break;
        case kShort:
            testValueA = *( (cl_short *)sourceA );
            testValueB = *( (cl_short *)sourceB );
            *( (cl_short *)destination ) = (cl_short)( ( testValueA * testValueB ) >> 16 );
            break;
        case kUShort:
            testValueA = *( (cl_ushort *)sourceA );
            testValueB = *( (cl_ushort *)sourceB );
            *( (cl_ushort *)destination ) = (cl_ushort)( ( testValueA * testValueB ) >> 16 );
            break;
        case kInt:
            testValueA = *( (cl_int *)sourceA );
            testValueB = *( (cl_int *)sourceB );
            *( (cl_int *)destination ) = (cl_int)( ( testValueA * testValueB ) >> 32 );
            break;
        case kUInt:
            testValueA = *( (cl_uint *)sourceA );
            testValueB = *( (cl_uint *)sourceB );
            *( (cl_uint *)destination ) = (cl_uint)( ( testValueA * testValueB ) >> 32 );
            break;
        case kLong:
            testValueA = *( (cl_long *)sourceA );
            testValueB = *( (cl_long *)sourceB );

            multiply_signed_64_by_64( testValueA, testValueB, lowHalf, highSigned );
            *( (cl_long *)destination ) = highSigned;
            break;
        case kULong:
            testValueA = *( (cl_ulong *)sourceA );
            testValueB = *( (cl_ulong *)sourceB );

            multiply_unsigned_64_by_64( testValueA, testValueB, lowHalf, highUnsigned );
            *( (cl_ulong *)destination ) = highUnsigned;
            break;
        default:
            // Should never happen
            return false;
    }
    return true;
}

int test_integer_mul_hi(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "mul_hi", verify_integer_mul_hi );
}

bool verify_integer_rotate( void *sourceA, void *sourceB, void *destination, ExplicitType vecType )
{
    cl_ulong testValueA;
    char numBits;

    switch( vecType )
    {
        case kChar:
        case kUChar:
            testValueA = *( (cl_uchar *)sourceA );
            numBits = *( (cl_uchar *)sourceB );
            numBits &= 7;
            if ( numBits == 0 )
                *( (cl_uchar *)destination ) =  (cl_uchar)testValueA;
            else
                *( (cl_uchar *)destination ) = (cl_uchar)( ( testValueA << numBits ) | ( testValueA >> ( 8 - numBits ) ) );
            break;
        case kShort:
        case kUShort:
            testValueA = *( (cl_ushort *)sourceA );
            numBits = *( (cl_ushort *)sourceB );
            numBits &= 15;
            if ( numBits == 0 )
                *( (cl_ushort *)destination ) =  (cl_ushort)testValueA;
            else
                *( (cl_ushort *)destination ) = (cl_ushort)( ( testValueA << numBits ) | ( testValueA >> ( 16 - numBits ) ) );
            break;
        case kInt:
        case kUInt:
            testValueA = *( (cl_uint *)sourceA );
            numBits = *( (cl_uint *)sourceB );
            numBits &= 31;
            if ( numBits == 0 )
                *( (cl_uint *)destination ) =  (cl_uint) testValueA;
            else
                *( (cl_uint *)destination ) = (cl_uint)( ( testValueA << numBits ) | ( testValueA >> ( 32 - numBits ) ) );
            break;
        case kLong:
        case kULong:
            testValueA = *( (cl_ulong *)sourceA );
            numBits = *( (cl_ulong *)sourceB );
            numBits &= 63;
            if ( numBits == 0 )
                *( (cl_ulong *)destination ) =  (cl_ulong)testValueA;
            else
                *( (cl_ulong *)destination ) = (cl_ulong)( ( testValueA << numBits ) | ( testValueA >> ( 64 - numBits ) ) );
            break;
        default:
            // Should never happen
            log_error( "Unknown type encountered in verify_integer_rotate. Test failed. Aborting...\n" );
            abort();
            return false;
    }
    return true;
}

int test_integer_rotate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_two_param_integer_fn( queue, context, "rotate", verify_integer_rotate );
}

const char *threeParamIntegerKernelSourcePattern =
"__kernel void sample_test(__global %s%s *sourceA, __global %s%s *sourceB, __global %s%s *sourceC, __global %s%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    %s%s sA = %s;\n"
"    %s%s sB = %s;\n"
"    %s%s sC = %s;\n"
"    %s%s dst = %s( sA, sB, sC );\n"
"     %s;\n"
"\n"
"}\n";

typedef bool (*threeParamIntegerVerifyFn)( void *sourceA, void *sourceB, void *sourceC, void *destination,
                                            ExplicitType vecAType, ExplicitType vecBType, ExplicitType vecCType, ExplicitType destType );

int test_three_param_integer_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                                  ExplicitType vecAType, ExplicitType vecBType, ExplicitType vecCType, ExplicitType destType,
                                    unsigned int vecSize, threeParamIntegerVerifyFn verifyFn, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[4];
    cl_long inDataA[TEST_SIZE * 16], inDataB[TEST_SIZE * 16], inDataC[TEST_SIZE * 16], outData[TEST_SIZE * 16], expected;
    int error, i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4], paramSizeName[4];

    if (! gHasLong && strstr(get_explicit_type_name(vecAType),"long"))
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping %s\n", get_explicit_type_name(vecAType) );
        return CL_SUCCESS;
    }


    /* Create the source */
    if( vecSize == 1 )
        sizeName[ 0 ] = 0;
    else
        sprintf( sizeName, "%d", vecSize );
    if( ( vecSize == 1 ) || ( vecSize == 3 ) )
        paramSizeName[ 0 ] = 0;
        else
        sprintf( paramSizeName, "%d", vecSize );

    char sourceALoad[ 128 ], sourceBLoad[ 128 ], sourceCLoad[ 128 ], destStore[ 128 ];

    sprintf( kernelSource, threeParamIntegerKernelSourcePattern,
            get_explicit_type_name( vecAType ), paramSizeName,
            get_explicit_type_name( vecBType ), paramSizeName,
            get_explicit_type_name( vecCType ), paramSizeName,
            get_explicit_type_name( destType ), paramSizeName,
            get_explicit_type_name( vecAType ), sizeName, build_load_statement( sourceALoad, (size_t)vecSize, "sourceA" ),
            get_explicit_type_name( vecBType ), sizeName, build_load_statement( sourceBLoad, (size_t)vecSize, "sourceB" ),
            get_explicit_type_name( vecCType ), sizeName, build_load_statement( sourceCLoad, (size_t)vecSize, "sourceC" ),
            get_explicit_type_name( destType ), sizeName,
            fnName,
            build_store_statement( destStore, (size_t)vecSize, "destValues", "dst" )
            );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
    log_error("The program we attempted to compile was: \n%s\n", kernelSource);
        return -1;
    }

    /* Generate some streams */
    generate_random_data( vecAType, vecSize * TEST_SIZE, d, inDataA );
    generate_random_data( vecBType, vecSize * TEST_SIZE, d, inDataB );
    generate_random_data( vecCType, vecSize * TEST_SIZE, d, inDataC );

    streams[0] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecAType) * vecSize * TEST_SIZE, &inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecBType) * vecSize * TEST_SIZE, &inDataB, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecCType) * vecSize * TEST_SIZE, &inDataC, NULL);
    if( streams[2] == NULL )
    {
        log_error("ERROR: Creating input array C failed!\n");
        return -1;
    }
    streams[3] = clCreateBuffer(
        context, CL_MEM_READ_WRITE,
        get_explicit_type_size(destType) * vecSize * TEST_SIZE, NULL, NULL);
    if( streams[3] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 2, sizeof( streams[2] ), &streams[2] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 3, sizeof( streams[3] ), &streams[3] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    memset(outData, 0xFF, get_explicit_type_size( destType ) * TEST_SIZE * vecSize);

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[3], CL_TRUE, 0, get_explicit_type_size( destType ) * TEST_SIZE * vecSize, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    char *inA = (char *)inDataA;
    char *inB = (char *)inDataB;
    char *inC = (char *)inDataC;
    char *out = (char *)outData;
    for( i = 0; i < (int)TEST_SIZE; i++ )
    {
        for( size_t j = 0; j < vecSize; j++ )
        {
            bool test = verifyFn( inA, inB, inC, &expected, vecAType, vecBType, vecCType, destType );
            if( test && ( memcmp( &expected, out, get_explicit_type_size( destType ) ) != 0 ) )
            {
                switch( get_explicit_type_size( vecAType ))
                {
                    case 1:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%2.2x), got (0x%2.2x), sources (0x%2.2x, 0x%2.2x, 0x%2.2x)\n",
                                  (int)i, (int)j, ((cl_uchar*)&expected)[ 0 ], *( (cl_uchar *)out ),
                                            *( (cl_uchar *)inA ),
                                            *( (cl_uchar *)inB ),
                                            *( (cl_uchar *)inC ) );
                        break;

                    case 2:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%4.4x), got (0x%4.4x), sources (0x%4.4x, 0x%4.4x, 0x%4.4x)\n",
                                  (int)i, (int)j, ((cl_ushort*)&expected)[ 0 ], *( (cl_ushort *)out ),
                                            *( (cl_ushort *)inA ),
                                            *( (cl_ushort *)inB ),
                                            *( (cl_ushort *)inC ) );
                        break;

                    case 4:
                        log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%8.8x), got (0x%8.8x), sources (0x%8.8x, 0x%8.8x, 0x%8.8x)\n",
                                  (int)i, (int)j, ((cl_uint*)&expected)[ 0 ], *( (cl_uint *)out ),
                                            *( (cl_uint *)inA ),
                                            *( (cl_uint *)inB ),
                                            *( (cl_uint *)inC ) );
                        break;

                    case 8:
                        log_error("ERROR: Data sample %d:%d does not validate! "
                                  "Expected (0x%16.16" PRIx64
                                  "), got (0x%16.16" PRIx64
                                  "), sources (0x%16.16" PRIx64
                                  ", 0x%16.16" PRIx64 ", 0x%16.16" PRIx64 ")\n",
                                  (int)i, (int)j, ((cl_ulong *)&expected)[0],
                                  *((cl_ulong *)out), *((cl_ulong *)inA),
                                  *((cl_ulong *)inB), *((cl_ulong *)inC));
                        break;
                }
                return -1;
            }
            inA += get_explicit_type_size( vecAType );
            inB += get_explicit_type_size( vecBType );
            inC += get_explicit_type_size( vecCType );
            out += get_explicit_type_size( destType );
        }
    }

    return 0;
}

int test_three_param_integer_fn(cl_command_queue queue, cl_context context, const char *fnName, threeParamIntegerVerifyFn verifyFn)
{
    ExplicitType types[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index, typeAIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);

    for( typeAIndex = 0; types[ typeAIndex ] != kNumExplicitTypes; typeAIndex++ )
    {
        if ((types[ typeAIndex ] == kLong || types[ typeAIndex] == kULong) && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            if( test_three_param_integer_kernel(queue, context, fnName, types[ typeAIndex ], types[ typeAIndex ], types[ typeAIndex ], types[ typeAIndex ], vecSizes[ index ], verifyFn, seed ) != 0 )
            {
                log_error( "   Vector %s%d,%s%d,%s%d FAILED\n", get_explicit_type_name( types[ typeAIndex ] ), vecSizes[ index ],
                                                            get_explicit_type_name( types[ typeAIndex ] ), vecSizes[ index ] ,
                                                            get_explicit_type_name( types[ typeAIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

bool verify_integer_clamp( void *sourceA, void *sourceB, void *sourceC, void *destination,
                        ExplicitType vecAType, ExplicitType vecBType, ExplicitType vecCType, ExplicitType destType )
{
    if( vecAType == kULong || vecAType == kUInt || vecAType == kUShort || vecAType == kUChar )
    {
        cl_ulong valueA, valueB, valueC;

        switch( vecAType )
        {
            case kULong:
                valueA = ((cl_ulong*) sourceA)[0];
                valueB = ((cl_ulong*) sourceB)[0];
                valueC = ((cl_ulong*) sourceC)[0];
                break;
            case kUInt:
                valueA = ((cl_uint*) sourceA)[0];
                valueB = ((cl_uint*) sourceB)[0];
                valueC = ((cl_uint*) sourceC)[0];
                break;
            case kUShort:
                valueA = ((cl_ushort*) sourceA)[0];
                valueB = ((cl_ushort*) sourceB)[0];
                valueC = ((cl_ushort*) sourceC)[0];
                break;
            case kUChar:
                valueA = ((cl_uchar*) sourceA)[0];
                valueB = ((cl_uchar*) sourceB)[0];
                valueC = ((cl_uchar*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }


        if(valueB > valueC) {
            return false; // results are undefined : let expected alone.
        }

        switch( vecAType )
        {
            case kULong:
                ((cl_ulong *)destination)[0] =
                    std::max(std::min(valueA, valueC), valueB);
                break;
            case kUInt:
                ((cl_uint *)destination)[0] =
                    (cl_uint)(std::max(std::min(valueA, valueC), valueB));
                break;
            case kUShort:
                ((cl_ushort *)destination)[0] =
                    (cl_ushort)(std::max(std::min(valueA, valueC), valueB));
                break;
            case kUChar:
                ((cl_uchar *)destination)[0] =
                    (cl_uchar)(std::max(std::min(valueA, valueC), valueB));
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }




    }
    else
    {
        cl_long valueA, valueB, valueC;


        switch( vecAType )
        {
            case kLong:
                valueA = ((cl_long*) sourceA)[0];
                valueB = ((cl_long*) sourceB)[0];
                valueC = ((cl_long*) sourceC)[0];
                break;
            case kInt:
                valueA = ((cl_int*) sourceA)[0];
                valueB = ((cl_int*) sourceB)[0];
                valueC = ((cl_int*) sourceC)[0];
                break;
            case kShort:
                valueA = ((cl_short*) sourceA)[0];
                valueB = ((cl_short*) sourceB)[0];
                valueC = ((cl_short*) sourceC)[0];
                break;
            case kChar:
                valueA = ((cl_char*) sourceA)[0];
                valueB = ((cl_char*) sourceB)[0];
                valueC = ((cl_char*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

        if(valueB > valueC) {
            return false; // undefined behavior : leave "expected" alone
        }

        switch( vecAType )
        {
            case kLong:
                ((cl_long *)destination)[0] =
                    std::max(std::min(valueA, valueC), valueB);
                break;
            case kInt:
                ((cl_int *)destination)[0] =
                    (cl_int)(std::max(std::min(valueA, valueC), valueB));
                break;
            case kShort:
                ((cl_short *)destination)[0] =
                    (cl_short)(std::max(std::min(valueA, valueC), valueB));
                break;
            case kChar:
                ((cl_char *)destination)[0] =
                    (cl_char)(std::max(std::min(valueA, valueC), valueB));
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

    }
    return true;
}

int test_integer_clamp(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_three_param_integer_fn( queue, context, "clamp", verify_integer_clamp );
}

bool verify_integer_mad_sat( void *sourceA, void *sourceB, void *sourceC, void *destination,
                        ExplicitType vecAType, ExplicitType vecBType, ExplicitType vecCType, ExplicitType destType )
{
    if( vecAType == kULong || vecAType == kUInt || vecAType == kUShort || vecAType == kUChar )
    {
        cl_ulong valueA, valueB, valueC;

        switch( vecAType )
        {
            case kULong:
                valueA = ((cl_ulong*) sourceA)[0];
                valueB = ((cl_ulong*) sourceB)[0];
                valueC = ((cl_ulong*) sourceC)[0];
                break;
            case kUInt:
                valueA = ((cl_uint*) sourceA)[0];
                valueB = ((cl_uint*) sourceB)[0];
                valueC = ((cl_uint*) sourceC)[0];
                break;
            case kUShort:
                valueA = ((cl_ushort*) sourceA)[0];
                valueB = ((cl_ushort*) sourceB)[0];
                valueC = ((cl_ushort*) sourceC)[0];
                break;
            case kUChar:
                valueA = ((cl_uchar*) sourceA)[0];
                valueB = ((cl_uchar*) sourceB)[0];
                valueC = ((cl_uchar*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

        cl_ulong multHi, multLo;
        multiply_unsigned_64_by_64( valueA, valueB, multLo, multHi );

        multLo += valueC;
        multHi += multLo < valueC;  // carry if overflow
        if( multHi )
            multLo = 0xFFFFFFFFFFFFFFFFULL;

        switch( vecAType )
        {
            case kULong:
                ((cl_ulong*) destination)[0] = multLo;
                break;
            case kUInt:
                ((cl_uint *)destination)[0] =
                    (cl_uint)std::min(multLo, (cl_ulong)CL_UINT_MAX);
                break;
            case kUShort:
                ((cl_ushort *)destination)[0] =
                    (cl_ushort)std::min(multLo, (cl_ulong)CL_USHRT_MAX);
                break;
            case kUChar:
                ((cl_uchar *)destination)[0] =
                    (cl_uchar)std::min(multLo, (cl_ulong)CL_UCHAR_MAX);
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }
    }
    else
    {
        cl_long valueA, valueB, valueC;

        switch( vecAType )
        {
            case kLong:
                valueA = ((cl_long*) sourceA)[0];
                valueB = ((cl_long*) sourceB)[0];
                valueC = ((cl_long*) sourceC)[0];
                break;
            case kInt:
                valueA = ((cl_int*) sourceA)[0];
                valueB = ((cl_int*) sourceB)[0];
                valueC = ((cl_int*) sourceC)[0];
                break;
            case kShort:
                valueA = ((cl_short*) sourceA)[0];
                valueB = ((cl_short*) sourceB)[0];
                valueC = ((cl_short*) sourceC)[0];
                break;
            case kChar:
                valueA = ((cl_char*) sourceA)[0];
                valueB = ((cl_char*) sourceB)[0];
                valueC = ((cl_char*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

        cl_long multHi;
        cl_ulong multLo;
        multiply_signed_64_by_64( valueA, valueB, multLo, multHi );

        cl_ulong sum = multLo + valueC;
        // carry if overflow
        if( valueC >= 0 )
        {
            if( multLo > sum )
            {
                multHi++;
                if( CL_LONG_MIN == multHi )
                {
                    multHi = CL_LONG_MAX;
                    sum = CL_ULONG_MAX;
                }
            }
        }
        else
        {
            if( multLo < sum )
            {
                multHi--;
                if( CL_LONG_MAX == multHi )
                {
                    multHi = CL_LONG_MIN;
                    sum = 0;
                }
            }
        }

        // saturate
        if( multHi > 0 )
            sum = CL_LONG_MAX;
        else if( multHi < -1 )
            sum = CL_LONG_MIN;
        cl_long result = (cl_long) sum;

        switch( vecAType )
        {
            case kLong:
                ((cl_long*) destination)[0] = result;
                break;
            case kInt:
                result = std::min(result, (cl_long)CL_INT_MAX);
                result = std::max(result, (cl_long)CL_INT_MIN);
                ((cl_int*) destination)[0] = (cl_int) result;
                break;
            case kShort:
                result = std::min(result, (cl_long)CL_SHRT_MAX);
                result = std::max(result, (cl_long)CL_SHRT_MIN);
                ((cl_short*) destination)[0] = (cl_short) result;
                break;
            case kChar:
                result = std::min(result, (cl_long)CL_CHAR_MAX);
                result = std::max(result, (cl_long)CL_CHAR_MIN);
                ((cl_char*) destination)[0] = (cl_char) result;
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }
    }
    return true;
}

int test_integer_mad_sat(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_three_param_integer_fn( queue, context, "mad_sat", verify_integer_mad_sat );
}

bool verify_integer_mad_hi( void *sourceA, void *sourceB, void *sourceC, void *destination,
                            ExplicitType vecAType, ExplicitType vecBType, ExplicitType vecCType, ExplicitType destType )
{
    if( vecAType == kULong || vecAType == kUInt || vecAType == kUShort || vecAType == kUChar )
    {
        cl_ulong valueA, valueB, valueC;

        switch( vecAType )
        {
            case kULong:
                valueA = ((cl_ulong*) sourceA)[0];
                valueB = ((cl_ulong*) sourceB)[0];
                valueC = ((cl_ulong*) sourceC)[0];
                break;
            case kUInt:
                valueA = ((cl_uint*) sourceA)[0];
                valueB = ((cl_uint*) sourceB)[0];
                valueC = ((cl_uint*) sourceC)[0];
                break;
            case kUShort:
                valueA = ((cl_ushort*) sourceA)[0];
                valueB = ((cl_ushort*) sourceB)[0];
                valueC = ((cl_ushort*) sourceC)[0];
                break;
            case kUChar:
                valueA = ((cl_uchar*) sourceA)[0];
                valueB = ((cl_uchar*) sourceB)[0];
                valueC = ((cl_uchar*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

        cl_ulong multHi, multLo;
        multiply_unsigned_64_by_64( valueA, valueB, multLo, multHi );

        switch( vecAType )
        {
            case kULong:
                ((cl_ulong*) destination)[0] = multHi + valueC;
                break;
            case kUInt:
                ((cl_uint*) destination)[0] = (cl_uint) (( multLo >> 32) + valueC );
                break;
            case kUShort:
                ((cl_ushort*) destination)[0] = (cl_ushort) (( multLo >> 16) + valueC );
                break;
            case kUChar:
                ((cl_uchar*) destination)[0] = (cl_uchar) (( multLo >> 8) + valueC );
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }
    }
    else
    {
        cl_long valueA, valueB, valueC;

        switch( vecAType )
        {
            case kLong:
                valueA = ((cl_long*) sourceA)[0];
                valueB = ((cl_long*) sourceB)[0];
                valueC = ((cl_long*) sourceC)[0];
                break;
            case kInt:
                valueA = ((cl_int*) sourceA)[0];
                valueB = ((cl_int*) sourceB)[0];
                valueC = ((cl_int*) sourceC)[0];
                break;
            case kShort:
                valueA = ((cl_short*) sourceA)[0];
                valueB = ((cl_short*) sourceB)[0];
                valueC = ((cl_short*) sourceC)[0];
                break;
            case kChar:
                valueA = ((cl_char*) sourceA)[0];
                valueB = ((cl_char*) sourceB)[0];
                valueC = ((cl_char*) sourceC)[0];
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }

        cl_long multHi;
        cl_ulong multLo;
        multiply_signed_64_by_64( valueA, valueB, multLo, multHi );

        switch( vecAType )
        {
            case kLong:
                ((cl_long*) destination)[0] = multHi + valueC;
                break;
            case kInt:
                ((cl_int*) destination)[0] = (cl_int) ((multLo >> 32) + valueC);
                break;
            case kShort:
                ((cl_short*) destination)[0] = (cl_int) ((multLo >> 16) + valueC);
                break;
            case kChar:
                ((cl_char*) destination)[0] = (cl_char) (cl_int) ((multLo >> 8) + valueC);
                break;
            default:
                //error -- should never get here
                abort();
                break;
        }
    }
    return true;
}

int test_integer_mad_hi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_three_param_integer_fn( queue, context, "mad_hi", verify_integer_mad_hi );
}



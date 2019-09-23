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

#include "testBase.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include <float.h>

const char *crossKernelSource =
"__kernel void sample_test(__global float4 *sourceA, __global float4 *sourceB, __global float4 *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = cross( sourceA[tid], sourceB[tid] );\n"
"\n"
"}\n" ;

const char *crossKernelSourceV3 =
"__kernel void sample_test(__global float *sourceA, __global float *sourceB, __global float *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    vstore3( cross( vload3( tid, sourceA), vload3( tid,  sourceB) ), tid, destValues );\n"
"\n"
"}\n";

const char *twoToFloatKernelPattern =
"__kernel void sample_test(__global float%s *sourceA, __global float%s *sourceB, __global float *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"\n"
"}\n";

const char *twoToFloatKernelPatternV3 =
"__kernel void sample_test(__global float%s *sourceA, __global float%s *sourceB, __global float *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( vload3( tid, (__global float*) sourceA), vload3( tid, (__global float*) sourceB) );\n"
"\n"
"}\n";

const char *oneToFloatKernelPattern =
"__kernel void sample_test(__global float%s *sourceA, __global float *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid] );\n"
"\n"
"}\n";

const char *oneToFloatKernelPatternV3 =
"__kernel void sample_test(__global float%s *sourceA, __global float *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( vload3( tid, (__global float*) sourceA) );\n"
"\n"
"}\n";

const char *oneToOneKernelPattern =
"__kernel void sample_test(__global float%s *sourceA, __global float%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid] );\n"
"\n"
"}\n";

const char *oneToOneKernelPatternV3 =
"__kernel void sample_test(__global float%s *sourceA, __global float%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    vstore3( %s( vload3( tid, (__global float*) sourceA) ), tid, (__global float*) destValues );\n"
"\n"
"}\n";

#define TEST_SIZE (1 << 20)

double verifyFastDistance( float *srcA, float *srcB, size_t vecSize );
double verifyFastLength( float *srcA, size_t vecSize );



void vector2string( char *string, float *vector, size_t elements )
{
    *string++ = '{';
    *string++ = ' ';
    string += sprintf( string, "%a", vector[0] );
    size_t i;
    for( i = 1; i < elements; i++ )
        string += sprintf( string, ", %a", vector[i] );
    *string++ = ' ';
    *string++ = '}';
    *string = '\0';
}

void fillWithTrickyNumbers( float *aVectors, float *bVectors, size_t vecSize )
{
    static const cl_float trickyValues[] = { -FLT_EPSILON, FLT_EPSILON,
        MAKE_HEX_FLOAT(0x1.0p63f, 0x1L, 63), MAKE_HEX_FLOAT(0x1.8p63f, 0x18L, 59), MAKE_HEX_FLOAT(0x1.0p64f, 0x1L, 64), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.8p-63f, -0x18L, -67), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64),
        MAKE_HEX_FLOAT(0x1.0p-63f, 0x1L, -63), MAKE_HEX_FLOAT(0x1.8p-63f, 0x18L, -67), MAKE_HEX_FLOAT(0x1.0p-64f, 0x1L, -64), MAKE_HEX_FLOAT(-0x1.0p-63f, -0x1L, -63), MAKE_HEX_FLOAT(-0x1.8p-63f, -0x18L, -67), MAKE_HEX_FLOAT(-0x1.0p-64f, -0x1L, -64),
        FLT_MAX / 2.f, -FLT_MAX / 2.f, INFINITY,  -INFINITY, 0.f, -0.f };
    static const size_t trickyCount = sizeof( trickyValues ) / sizeof( trickyValues[0] );
    static const size_t stride[4] = {1, trickyCount, trickyCount*trickyCount, trickyCount*trickyCount*trickyCount };
    size_t i, j, k;

    for( j = 0; j < vecSize; j++ )
        for( k = 0; k < vecSize; k++ )
            for( i = 0; i < trickyCount; i++ )
                aVectors[ j + stride[j] * (i + k*trickyCount)*vecSize] = trickyValues[i];

    if( bVectors )
    {
        size_t copySize = vecSize * vecSize * trickyCount;
        memset( bVectors, 0, sizeof(float) * copySize );
        memset( aVectors + copySize, 0, sizeof(float) * copySize );
        memcpy( bVectors + copySize, aVectors, sizeof(float) * copySize );
    }
}


void cross_product( const float *vecA, const float *vecB, float *outVector, float *errorTolerances, float ulpTolerance )
{
    outVector[ 0 ] = ( vecA[ 1 ] * vecB[ 2 ] ) - ( vecA[ 2 ] * vecB[ 1 ] );
    outVector[ 1 ] = ( vecA[ 2 ] * vecB[ 0 ] ) - ( vecA[ 0 ] * vecB[ 2 ] );
    outVector[ 2 ] = ( vecA[ 0 ] * vecB[ 1 ] ) - ( vecA[ 1 ] * vecB[ 0 ] );
    outVector[ 3 ] = 0.0f;

    errorTolerances[ 0 ] = fmaxf( fabsf( vecA[ 1 ] ), fmaxf( fabsf( vecB[ 2 ] ), fmaxf( fabsf( vecA[ 2 ] ), fabsf( vecB[ 1 ] ) ) ) );
    errorTolerances[ 1 ] = fmaxf( fabsf( vecA[ 2 ] ), fmaxf( fabsf( vecB[ 0 ] ), fmaxf( fabsf( vecA[ 0 ] ), fabsf( vecB[ 2 ] ) ) ) );
    errorTolerances[ 2 ] = fmaxf( fabsf( vecA[ 0 ] ), fmaxf( fabsf( vecB[ 1 ] ), fmaxf( fabsf( vecA[ 1 ] ), fabsf( vecB[ 0 ] ) ) ) );

    errorTolerances[ 0 ] = errorTolerances[ 0 ] * errorTolerances[ 0 ] * ( ulpTolerance * FLT_EPSILON );    // This gives us max squared times ulp tolerance, i.e. the worst-case expected variance we could expect from this result
    errorTolerances[ 1 ] = errorTolerances[ 1 ] * errorTolerances[ 1 ] * ( ulpTolerance * FLT_EPSILON );
    errorTolerances[ 2 ] = errorTolerances[ 2 ] * errorTolerances[ 2 ] * ( ulpTolerance * FLT_EPSILON );
}




int test_geom_cross(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int vecsize;
    RandomSeed seed(gRandomSeed);

    /* Get the default rounding mode */
    cl_device_fp_config defaultRoundingMode = get_default_rounding_mode(deviceID);
    if( 0 == defaultRoundingMode )
        return -1;


    for(vecsize = 3; vecsize <= 4; ++vecsize)
    {
        clProgramWrapper program;
        clKernelWrapper kernel;
        clMemWrapper streams[3];
        BufferOwningPtr<cl_float> A(malloc(sizeof(cl_float) * TEST_SIZE * vecsize));
        BufferOwningPtr<cl_float> B(malloc(sizeof(cl_float) * TEST_SIZE * vecsize));
        BufferOwningPtr<cl_float> C(malloc(sizeof(cl_float) * TEST_SIZE * vecsize));
        cl_float testVector[4];
        int error, i;
        cl_float *inDataA = A;
        cl_float *inDataB = B;
        cl_float *outData = C;
        size_t threads[1], localThreads[1];

        /* Create kernels */
        if( create_single_kernel_helper( context, &program, &kernel, 1, vecsize == 3 ? &crossKernelSourceV3 : &crossKernelSource, "sample_test" ) )
            return -1;

        /* Generate some streams. Note: deliberately do some random data in w to verify that it gets ignored */
        for( i = 0; i < TEST_SIZE * vecsize; i++ )
        {
            inDataA[ i ] = get_random_float( -512.f, 512.f, seed );
            inDataB[ i ] = get_random_float( -512.f, 512.f, seed );
        }
        fillWithTrickyNumbers( inDataA, inDataB, vecsize );

        streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof(cl_float) * vecsize * TEST_SIZE, inDataA, NULL);
        if( streams[0] == NULL )
        {
            log_error("ERROR: Creating input array A failed!\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof(cl_float) * vecsize * TEST_SIZE, inDataB, NULL);
        if( streams[1] == NULL )
        {
            log_error("ERROR: Creating input array B failed!\n");
            return -1;
        }
        streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_float) * vecsize * TEST_SIZE, NULL, NULL);
        if( streams[2] == NULL )
        {
            log_error("ERROR: Creating output array failed!\n");
            return -1;
        }

        /* Assign streams and execute */
        for( i = 0; i < 3; i++ )
        {
            error = clSetKernelArg(kernel, i, sizeof( streams[i] ), &streams[i]);
            test_error( error, "Unable to set indexed kernel arguments" );
        }

        /* Run the kernel */
        threads[0] = TEST_SIZE;

        error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
        test_error( error, "Unable to get work group size to use" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
        test_error( error, "Unable to execute test kernel" );

        /* Now get the results */
        error = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof( cl_float ) * TEST_SIZE * vecsize, outData, 0, NULL, NULL );
        test_error( error, "Unable to read output array!" );

        /* And verify! */
        for( i = 0; i < TEST_SIZE; i++ )
        {
            float errorTolerances[ 4 ];
            // On an embedded device w/ round-to-zero, 3 ulps is the worst-case tolerance for cross product
            cross_product( inDataA + i * vecsize, inDataB + i * vecsize, testVector, errorTolerances, 3.f );

        // RTZ devices accrue approximately double the amount of error per operation.  Allow for that.
        if( defaultRoundingMode == CL_FP_ROUND_TO_ZERO )
        {
            errorTolerances[0] *= 2.0f;
            errorTolerances[1] *= 2.0f;
            errorTolerances[2] *= 2.0f;
            errorTolerances[3] *= 2.0f;
        }

            float errs[] = { fabsf( testVector[ 0 ] - outData[ i * vecsize + 0 ] ),
                             fabsf( testVector[ 1 ] - outData[ i * vecsize + 1 ] ),
                             fabsf( testVector[ 2 ] - outData[ i * vecsize + 2 ] ) };

            if( errs[ 0 ] > errorTolerances[ 0 ] || errs[ 1 ] > errorTolerances[ 1 ] || errs[ 2 ] > errorTolerances[ 2 ] )
            {
                log_error( "ERROR: Data sample %d does not validate! Expected (%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, testVector[0], testVector[1], testVector[2], testVector[3],
                          outData[i*vecsize], outData[i*vecsize+1], outData[i*vecsize+2], outData[i*vecsize+3] );
                log_error( "    Input: (%a %a %a) and (%a %a %a)\n",
                          inDataA[ i * vecsize + 0 ], inDataA[ i * vecsize + 1 ], inDataA[ i * vecsize + 2 ],
                          inDataB[ i * vecsize + 0 ], inDataB[ i * vecsize + 1 ], inDataB[ i * vecsize + 2 ] );
                log_error( "    Errors: (%a out of %a), (%a out of %a), (%a out of %a)\n",
                          errs[ 0 ], errorTolerances[ 0 ], errs[ 1 ], errorTolerances[ 1 ], errs[ 2 ], errorTolerances[ 2 ] );
                log_error("     ulp %f\n", Ulp_Error( outData[ i * vecsize + 1 ], testVector[ 1 ] ) );
                return -1;
            }
        }
    } // for(vecsize=...

    if(!is_extension_available(deviceID, "cl_khr_fp64")) {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    } else {
        log_info("Testing doubles...\n");
        return test_geom_cross_double( deviceID,  context,  queue,  num_elements, seed);
    }
}

float getMaxValue( float vecA[], float vecB[], size_t vecSize )
{
    float a = fmaxf( fabsf( vecA[ 0 ] ), fabsf( vecB[ 0 ] ) );
    for( size_t i = 1; i < vecSize; i++ )
        a = fmaxf( fabsf( vecA[ i ] ), fmaxf( fabsf( vecB[ i ] ), a ) );
    return a;
}

typedef double (*twoToFloatVerifyFn)( float *srcA, float *srcB, size_t vecSize );

int test_twoToFloat_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                           size_t vecSize, twoToFloatVerifyFn verifyFn, float ulpLimit, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    int error;
    size_t i, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    int hasInfNan = 1;
    cl_device_id device = NULL;

    error = clGetCommandQueueInfo( queue, CL_QUEUE_DEVICE, sizeof( device ), &device, NULL );
    test_error( error, "Unable to get command queue device" );

    /* Check for embedded devices doing nutty stuff */
    error = clGetDeviceInfo( device, CL_DEVICE_PROFILE, sizeof( kernelSource ), kernelSource, NULL );
    test_error( error, "Unable to get device profile" );
    if( 0 == strcmp( kernelSource, "EMBEDDED_PROFILE" ) )
    {
        cl_device_fp_config config = 0;
        error = clGetDeviceInfo( device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( config ), &config, NULL );
        test_error( error, "Unable to get CL_DEVICE_SINGLE_FP_CONFIG" );

        if( CL_FP_ROUND_TO_ZERO == (config & (CL_FP_ROUND_TO_NEAREST|CL_FP_ROUND_TO_ZERO)))
            ulpLimit *= 2.0f; // rtz operations average twice the accrued error of rte operations

        if( 0 == (config & CL_FP_INF_NAN) )
            hasInfNan = 0;
    }

    BufferOwningPtr<cl_float> A(malloc(sizeof(cl_float) * TEST_SIZE * 4));
    BufferOwningPtr<cl_float> B(malloc(sizeof(cl_float) * TEST_SIZE * 4));
    BufferOwningPtr<cl_float> C(malloc(sizeof(cl_float) * TEST_SIZE));

    cl_float *inDataA = A;
    cl_float *inDataB = B;
    cl_float *outData = C;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3 ? twoToFloatKernelPatternV3 : twoToFloatKernelPattern, sizeNames[vecSize-1], sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }
    /* Generate some streams */
    for( i = 0; i < TEST_SIZE * vecSize; i++ )
    {
        inDataA[ i ] = get_random_float( -512.f, 512.f, d );
        inDataB[ i ] = get_random_float( -512.f, 512.f, d );
    }
    fillWithTrickyNumbers( inDataA, inDataB, vecSize );

    /* Clamp values to be in range for fast_ functions */
    if( verifyFn == verifyFastDistance )
    {
        for( i = 0; i < TEST_SIZE * vecSize; i++ )
        {
            if( fabsf( inDataA[i] ) > MAKE_HEX_FLOAT(0x1.0p62f, 0x1L, 62) || fabsf( inDataA[i] ) < MAKE_HEX_FLOAT(0x1.0p-62f, 0x1L, -62) )
                inDataA[ i ] = get_random_float( -512.f, 512.f, d );
            if( fabsf( inDataB[i] ) > MAKE_HEX_FLOAT(0x1.0p62f, 0x1L, 62) || fabsf( inDataB[i] ) < MAKE_HEX_FLOAT(0x1.0p-62f, 0x1L, -62) )
                inDataB[ i ] = get_random_float( -512.f, 512.f, d );
        }
    }


    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof(cl_float) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof(cl_float) * vecSize * TEST_SIZE, inDataB, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_float) * TEST_SIZE, NULL, NULL);
    if( streams[2] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    for( i = 0; i < 3; i++ )
    {
        error = clSetKernelArg(kernel, (int)i, sizeof( streams[i] ), &streams[i]);
        test_error( error, "Unable to set indexed kernel arguments" );
    }

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof( cl_float ) * TEST_SIZE, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );


    /* And verify! */
    int skipCount = 0;
    for( i = 0; i < TEST_SIZE; i++ )
    {
        cl_float *src1 = inDataA + i * vecSize;
        cl_float *src2 = inDataB + i * vecSize;
        double expected = verifyFn( src1, src2, vecSize );
        if( (float) expected != outData[ i ] )
        {
            if( isnan(expected) && isnan( outData[i] ) )
                continue;

            if( ! hasInfNan )
            {
                size_t ii;
                for( ii = 0; ii < vecSize; ii++ )
                {
                    if( ! isfinite( src1[ii] ) || ! isfinite( src2[ii] ) )
                    {
                        skipCount++;
                        continue;
                    }
                }
                if( ! isfinite( (cl_float) expected ) )
                {
                    skipCount++;
                    continue;
                }
            }

            if( ulpLimit < 0 )
            {
                // Limit below zero means we need to test via a computed error (like cross product does)
                float maxValue =
                getMaxValue( inDataA + i * vecSize, inDataB + i * vecSize,vecSize );
                // In this case (dot is the only one that gets here), the ulp is 2*vecSize - 1 (n + n-1 max # of errors)
                float errorTolerance = maxValue * maxValue * ( 2.f * (float)vecSize - 1.f ) * FLT_EPSILON;

                // Limit below zero means test via epsilon instead
                double error =
                fabs( (double)expected - (double)outData[ i ] );
                if( error > errorTolerance )
                {

                    log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), sources (%a and %a) error of %g against tolerance %g\n",
                              (int)i, (int)vecSize, expected,
                              outData[ i ],
                              inDataA[i*vecSize],
                              inDataB[i*vecSize],
                              (float)error,
                              (float)errorTolerance );

                    char vecA[1000], vecB[1000];
                    vector2string( vecA, inDataA +i * vecSize, vecSize );
                    vector2string( vecB, inDataB + i * vecSize, vecSize );
                    log_error( "\tvector A: %s, vector B: %s\n", vecA, vecB );
                    return -1;
                }
            }
            else
            {
                float error = Ulp_Error( outData[ i ], expected );
                if( fabsf(error) > ulpLimit )
                {
                    log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), sources (%a and %a) ulp of %f\n",
                              (int)i, (int)vecSize, expected, outData[ i ], inDataA[i*vecSize], inDataB[i*vecSize], error );

                    char vecA[1000], vecB[1000];
                    vector2string( vecA, inDataA + i * vecSize, vecSize );
                    vector2string( vecB, inDataB + i * vecSize, vecSize );
                    log_error( "\tvector A: %s, vector B: %s\n", vecA, vecB );
                    return -1;
                }
            }
        }
    }

    if( skipCount )
        log_info( "Skipped %d tests out of %d because they contained Infs or NaNs\n\tEMBEDDED_PROFILE Device does not support CL_FP_INF_NAN\n", skipCount, TEST_SIZE );

    return 0;
}

double verifyDot( float *srcA, float *srcB, size_t vecSize )
{
    double total = 0.f;

    for( unsigned int i = 0; i < vecSize; i++ )
        total += (double)srcA[ i ] * (double)srcB[ i ];

    return total;
}

int test_geom_dot(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        if( test_twoToFloat_kernel( queue, context, "dot", sizes[size], verifyDot, -1.0f /*magic value*/, seed ) != 0 )
        {
            log_error( "   dot vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
    }

    if (retVal)
        return retVal;

    if(!is_extension_available(deviceID, "cl_khr_fp64"))
    {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    }

    log_info("Testing doubles...\n");
    return test_geom_dot_double( deviceID,  context,  queue,  num_elements, seed);
}

double verifyFastDistance( float *srcA, float *srcB, size_t vecSize )
{
    double total = 0, value;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
    {
        value = (double)srcA[i] - (double)srcB[i];
        total += value * value;
    }

    return sqrt( total );
}

int test_geom_fast_distance(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 8192.0f +                           // error in sqrt
        ( 1.5f * (float) sizes[size] +      // cumulative error for multiplications  (a-b+0.5ulp)**2 = (a-b)**2 + a*0.5ulp + b*0.5 ulp + 0.5 ulp for multiplication
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions

        if( test_twoToFloat_kernel( queue, context, "fast_distance",
                                   sizes[ size ], verifyFastDistance,
                                   maxUlps, seed ) != 0 )
        {
            log_error( "   fast_distance vector size %d FAILED\n",
                      (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   fast_distance vector size %d passed\n",
                     (int)sizes[ size ] );
        }
    }
    return retVal;
}


double verifyDistance( float *srcA, float *srcB, size_t vecSize )
{
    double total = 0, value;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
    {
        value = (double)srcA[i] - (double)srcB[i];
        total += value * value;
    }

    return sqrt( total );
}

int test_geom_distance(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 3.0f +                              // error in sqrt
        ( 1.5f * (float) sizes[size] +      // cumulative error for multiplications  (a-b+0.5ulp)**2 = (a-b)**2 + a*0.5ulp + b*0.5 ulp + 0.5 ulp for multiplication
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions

        if( test_twoToFloat_kernel( queue, context, "distance", sizes[ size ], verifyDistance, maxUlps, seed ) != 0 )
        {
            log_error( "   distance vector size %d FAILED\n",
                      (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   distance vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    if (retVal)
        return retVal;

    if(!is_extension_available(deviceID, "cl_khr_fp64"))
    {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    } else {
        log_info("Testing doubles...\n");
        return test_geom_distance_double( deviceID,  context,  queue,  num_elements, seed);
    }
}

typedef double (*oneToFloatVerifyFn)( float *srcA, size_t vecSize );

int test_oneToFloat_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                           size_t vecSize, oneToFloatVerifyFn verifyFn, float ulpLimit, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    BufferOwningPtr<cl_float> A(malloc(sizeof(cl_float) * TEST_SIZE * 4));
    BufferOwningPtr<cl_float> B(malloc(sizeof(cl_float) * TEST_SIZE));
    int error;
    size_t i, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    cl_float *inDataA = A;
    cl_float *outData = B;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3? oneToFloatKernelPatternV3 : oneToFloatKernelPattern, sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }

    /* Generate some streams */
    for( i = 0; i < TEST_SIZE * vecSize; i++ )
    {
        inDataA[ i ] = get_random_float( -512.f, 512.f, d );
    }
    fillWithTrickyNumbers( inDataA, NULL, vecSize );

    /* Clamp values to be in range for fast_ functions */
    if( verifyFn == verifyFastLength )
    {
        for( i = 0; i < TEST_SIZE * vecSize; i++ )
        {
            if( fabsf( inDataA[i] ) > MAKE_HEX_FLOAT(0x1.0p62f, 0x1L, 62) || fabsf( inDataA[i] ) < MAKE_HEX_FLOAT(0x1.0p-62f, 0x1L, -62) )
                inDataA[ i ] = get_random_float( -512.f, 512.f, d );
        }
    }

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR),
                                sizeof(cl_float) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),
                                sizeof(cl_float) * TEST_SIZE, NULL, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[ 1 ] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0],
                                           &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[1], true, 0,
                                sizeof( cl_float ) * TEST_SIZE, outData,
                                0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        double expected = verifyFn( inDataA + i * vecSize, vecSize );
        if( (float) expected != outData[ i ] )
        {
            float ulps = Ulp_Error( outData[i], expected );
            if( fabsf( ulps ) <= ulpLimit )
                continue;

            // We have to special case NAN
            if( isnan( outData[ i ] ) && isnan( expected ) )
                continue;

            if(! (fabsf(ulps) < ulpLimit) )
            {
                log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), source (%a), ulp %f\n",
                          (int)i, (int)vecSize, expected, outData[ i ],  inDataA[i*vecSize], ulps );
                char vecA[1000];
                vector2string( vecA, inDataA + i *vecSize, vecSize );
                log_error( "\tvector: %s", vecA );
                return -1;
            }
        }
    }

    return 0;
}

double verifyLength( float *srcA, size_t vecSize )
{
    double total = 0;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
    {
        total += (double)srcA[i] * (double)srcA[i];
    }

    return sqrt( total );
}

int test_geom_length(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed( gRandomSeed );

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 3.0f +                              // error in sqrt
        0.5f *                              // effect on e of taking sqrt( x + e )
        ( 0.5f * (float) sizes[size] +      // cumulative error for multiplications
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions

        if( test_oneToFloat_kernel( queue, context, "length", sizes[ size ], verifyLength, maxUlps, seed ) != 0 )
        {
            log_error( "   length vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   length vector vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    if (retVal)
        return retVal;

    if(!is_extension_available(deviceID, "cl_khr_fp64"))
    {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    }
    else
    {
        log_info("Testing doubles...\n");
        return test_geom_length_double( deviceID,  context,  queue,  num_elements, seed);
    }
}


double verifyFastLength( float *srcA, size_t vecSize )
{
    double total = 0;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
    {
        total += (double)srcA[i] * (double)srcA[i];
    }

    return sqrt( total );
}

int test_geom_fast_length(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 8192.0f +                           // error in half_sqrt
        ( 0.5f * (float) sizes[size] +      // cumulative error for multiplications
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions

        if( test_oneToFloat_kernel( queue, context, "fast_length", sizes[ size ], verifyFastLength, maxUlps, seed ) != 0 )
        {
            log_error( "   fast_length vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   fast_length vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}


typedef void (*oneToOneVerifyFn)( float *srcA, float *dstA, size_t vecSize );


int test_oneToOne_kernel(cl_command_queue queue, cl_context context, const char *fnName,
                         size_t vecSize, oneToOneVerifyFn verifyFn, float ulpLimit, int softball, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    BufferOwningPtr<cl_float> A(malloc(sizeof(cl_float) * TEST_SIZE
                                       * vecSize));
    BufferOwningPtr<cl_float> B(malloc(sizeof(cl_float) * TEST_SIZE
                                       * vecSize));
    int error;
    size_t i, j, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    cl_float *inDataA = A;
    cl_float *outData = B;
    float ulp_error = 0;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3 ? oneToOneKernelPatternV3: oneToOneKernelPattern, sizeNames[vecSize-1], sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr,  "sample_test" ) )
        return -1;

    /* Initialize data.  First element always 0. */
    memset( inDataA, 0, sizeof(cl_float) * vecSize );
    if( 0 == strcmp( fnName, "fast_normalize" ))
    { // keep problematic cases out of the fast function
        for( i = vecSize; i < TEST_SIZE * vecSize; i++ )
        {
            cl_float z = get_random_float( -MAKE_HEX_FLOAT( 0x1.0p60f, 1, 60), MAKE_HEX_FLOAT( 0x1.0p60f, 1, 60), d);
            if( fabsf(z) < MAKE_HEX_FLOAT( 0x1.0p-60f, 1, -60) )
                z = copysignf( 0.0f, z );
            inDataA[i] = z;
        }
    }
    else
    {
        for( i = vecSize; i < TEST_SIZE * vecSize; i++ )
            inDataA[i] = any_float(d);
    }

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof(cl_float) * vecSize* TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_float) * vecSize  * TEST_SIZE, NULL, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof( cl_float ) * TEST_SIZE  * vecSize, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        float expected[4];
        int fail = 0;
        verifyFn( inDataA + i * vecSize, expected, vecSize );
        for( j = 0; j < vecSize; j++ )
        {
            // We have to special case NAN
            if( isnan( outData[ i * vecSize + j ] )
               && isnan( expected[ j ] ) )
                continue;

            if( expected[j] != outData[ i * vecSize + j ] ) {
                ulp_error = Ulp_Error(  outData[i*vecSize+j], expected[ j ] );

                if( fabsf(ulp_error) > ulpLimit ) {
                    fail = 1;
                    break;
                }
            }

        }

        // try again with subnormals flushed to zero if the platform flushes
        if( fail && gFlushDenormsToZero )
        {
            float temp[4], expected2[4];
            for( j = 0; j < vecSize; j++ )
            {
                if( IsFloatSubnormal(inDataA[i*vecSize+j] ) )
                    temp[j] = copysignf( 0.0f, inDataA[i*vecSize+j] );
                else
                    temp[j] = inDataA[ i*vecSize +j];
            }

            verifyFn( temp, expected2, vecSize );
            fail = 0;

            for( j = 0; j < vecSize; j++ )
            {
                // We have to special case NAN
                if( isnan( outData[ i * vecSize + j ] ) && isnan( expected[ j ] ) )
                    continue;

                if( expected2[j] != outData[ i * vecSize + j ] )
                {
                    ulp_error = Ulp_Error(outData[i*vecSize + j ], expected[ j ]  );

                    if( fabsf(ulp_error) > ulpLimit )
                    {
                        if( IsFloatSubnormal(expected2[j]) )
                        {
                            expected2[j] = 0.0f;
                            if( expected2[j] !=  outData[i*vecSize + j ] )
                            {
                                ulp_error = Ulp_Error(  outData[ i * vecSize + j ], expected[ j ] );
                                if( fabsf(ulp_error) > ulpLimit ) {
                                    fail = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if( fail )
        {
            log_error( "ERROR: Data sample {%d,%d} at size %d does not validate! Expected %12.24f (%a), got %12.24f (%a), ulp %f\n",
                      (int)i, (int)j, (int)vecSize, expected[j], expected[j], outData[ i*vecSize+j], outData[ i*vecSize+j], ulp_error );
            log_error( "       Source: " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%g ", inDataA[ i * vecSize+q]);
            log_error( "\n             : " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%a ", inDataA[i*vecSize +q] );
            log_error( "\n" );
            log_error( "       Result: " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%g ", outData[ i *vecSize + q ] );
            log_error( "\n             : " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%a ", outData[ i * vecSize + q ] );
            log_error( "\n" );
            log_error( "       Expected: " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%g ", expected[ q ] );
            log_error( "\n             : " );
            for( size_t q = 0; q < vecSize; q++ )
                log_error( "%a ", expected[ q ] );
            log_error( "\n" );
            return -1;
        }
    }

    return 0;
}

void verifyNormalize( float *srcA, float *dst, size_t vecSize )
{
    double total = 0, value;
    unsigned int i;

    // We calculate everything as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
        total += (double)srcA[i] * (double)srcA[i];

    if( total == 0.f )
    {
        // Special edge case: copy vector over without change
        for( i = 0; i < vecSize; i++ )
            dst[i] = srcA[i];
        return;
    }

    // Deal with infinities
    if( total == INFINITY )
    {
        total = 0.0f;
        for( i = 0; i < vecSize; i++ )
        {
            if( fabsf( srcA[i]) == INFINITY )
                dst[i] = copysignf( 1.0f, srcA[i] );
            else
                dst[i] = copysignf( 0.0f, srcA[i] );
            total += (double)dst[i] * (double)dst[i];
        }

        srcA = dst;
    }

    value = sqrt( total );
    for( i = 0; i < vecSize; i++ )
        dst[i] = (float)( (double)srcA[i] / value );
}

int test_geom_normalize(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 2.5f +                              // error in rsqrt + error in multiply
        ( 0.5f * (float) sizes[size] +      // cumulative error for multiplications
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions
        if( test_oneToOne_kernel( queue, context, "normalize", sizes[ size ], verifyNormalize, maxUlps, 0, seed ) != 0 )
        {
            log_error( "   normalized vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   normalized vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    if (retVal)
        return retVal;

    if(!is_extension_available(deviceID, "cl_khr_fp64"))
    {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    } else {
        log_info("Testing doubles...\n");
        return test_geom_normalize_double( deviceID,  context,  queue,  num_elements, seed);
    }
}


int test_geom_fast_normalize(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;
    RandomSeed seed( gRandomSeed );

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        float maxUlps = 8192.5f +                           // error in rsqrt + error in multiply
        ( 0.5f * (float) sizes[size] +      // cumulative error for multiplications
         0.5f * (float) (sizes[size]-1));    // cumulative error for additions

        if( test_oneToOne_kernel( queue, context, "fast_normalize", sizes[ size ], verifyNormalize, maxUlps, 1, seed ) != 0 )
        {
            log_error( "   fast_normalize vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   fast_normalize vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}




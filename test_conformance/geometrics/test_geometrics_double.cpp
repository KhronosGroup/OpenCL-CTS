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
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"

const char *crossKernelSource_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double4 *sourceA, __global double4 *sourceB, __global double4 *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = cross( sourceA[tid], sourceB[tid] );\n"
"\n"
"}\n";

const char *crossKernelSource_doubleV3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double *sourceA, __global double *sourceB, __global double *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    vstore3( cross( vload3( tid, sourceA), vload3( tid, sourceB) ), tid, destValues);\n"
"\n"
"}\n";

const char *twoToFloatKernelPattern_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global double *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"\n"
"}\n";

const char *twoToFloatKernelPattern_doubleV3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global double *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( vload3( tid, (__global double*) sourceA), vload3( tid, (__global double*) sourceB ) );\n"
"\n"
"}\n";

const char *oneToFloatKernelPattern_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid] );\n"
"\n"
"}\n";

const char *oneToFloatKernelPattern_doubleV3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( vload3( tid, (__global double*) sourceA) );\n"
"\n"
"}\n";

const char *oneToOneKernelPattern_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid] );\n"
"\n"
"}\n";

const char *oneToOneKernelPattern_doubleV3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    vstore3( %s( vload3( tid, (__global double*) sourceA) ), tid, (__global double*) destValues );\n"
"\n"
"}\n";

#define TEST_SIZE (1 << 20)

double verifyLength_double( double *srcA, size_t vecSize );
double verifyDistance_double( double *srcA, double *srcB, size_t vecSize );



void vector2string_double( char *string, double *vector, size_t elements )
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

void fillWithTrickyNumbers_double( double *aVectors, double *bVectors, size_t vecSize )
{
    static const cl_double trickyValues[] = { -FLT_EPSILON, FLT_EPSILON,
        MAKE_HEX_DOUBLE(0x1.0p511, 0x1L, 511), MAKE_HEX_DOUBLE(0x1.8p511, 0x18L, 507), MAKE_HEX_DOUBLE(0x1.0p512, 0x1L, 512), MAKE_HEX_DOUBLE(-0x1.0p511, -0x1L, 511), MAKE_HEX_DOUBLE(-0x1.8p-511, -0x18L, -515), MAKE_HEX_DOUBLE(-0x1.0p512, -0x1L, 512),
        MAKE_HEX_DOUBLE(0x1.0p-511, 0x1L, -511), MAKE_HEX_DOUBLE(0x1.8p-511, 0x18L, -515), MAKE_HEX_DOUBLE(0x1.0p-512, 0x1L, -512), MAKE_HEX_DOUBLE(-0x1.0p-511, -0x1L, -511), MAKE_HEX_DOUBLE(-0x1.8p-511, -0x18L, -515), MAKE_HEX_DOUBLE(-0x1.0p-512, -0x1L, -512),
        DBL_MAX / 2., -DBL_MAX / 2., INFINITY,  -INFINITY, 0., -0. };
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
        memset( bVectors, 0, sizeof(double) * copySize );
        memset( aVectors + copySize, 0, sizeof(double) * copySize );
        memcpy( bVectors + copySize, aVectors, sizeof(double) * copySize );
    }
}


void cross_product_double( const double *vecA, const double *vecB, double *outVector, double *errorTolerances, double ulpTolerance )
{
    outVector[ 0 ] = ( vecA[ 1 ] * vecB[ 2 ] ) - ( vecA[ 2 ] * vecB[ 1 ] );
    outVector[ 1 ] = ( vecA[ 2 ] * vecB[ 0 ] ) - ( vecA[ 0 ] * vecB[ 2 ] );
    outVector[ 2 ] = ( vecA[ 0 ] * vecB[ 1 ] ) - ( vecA[ 1 ] * vecB[ 0 ] );
    outVector[ 3 ] = 0.0f;

    errorTolerances[ 0 ] = fmax( fabs( vecA[ 1 ] ), fmax( fabs( vecB[ 2 ] ), fmax( fabs( vecA[ 2 ] ), fabs( vecB[ 1 ] ) ) ) );
    errorTolerances[ 1 ] = fmax( fabs( vecA[ 2 ] ), fmax( fabs( vecB[ 0 ] ), fmax( fabs( vecA[ 0 ] ), fabs( vecB[ 2 ] ) ) ) );
    errorTolerances[ 2 ] = fmax( fabs( vecA[ 0 ] ), fmax( fabs( vecB[ 1 ] ), fmax( fabs( vecA[ 1 ] ), fabs( vecB[ 0 ] ) ) ) );

    errorTolerances[ 0 ] = errorTolerances[ 0 ] * errorTolerances[ 0 ] * ( ulpTolerance * FLT_EPSILON );    // This gives us max squared times ulp tolerance, i.e. the worst-case expected variance we could expect from this result
    errorTolerances[ 1 ] = errorTolerances[ 1 ] * errorTolerances[ 1 ] * ( ulpTolerance * FLT_EPSILON );
    errorTolerances[ 2 ] = errorTolerances[ 2 ] * errorTolerances[ 2 ] * ( ulpTolerance * FLT_EPSILON );
}

int test_geom_cross_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    cl_int error;
    cl_ulong maxAllocSize, maxGlobalMemSize;

    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( maxGlobalMemSize ), &maxGlobalMemSize, NULL );
    test_error( error, "Unable to get device config" );

    log_info("Device supports:\nCL_DEVICE_MAX_MEM_ALLOC_SIZE: %gMB\nCL_DEVICE_GLOBAL_MEM_SIZE: %gMB\n",
             maxGlobalMemSize/(1024.0*1024.0), maxAllocSize/(1024.0*1024.0));

    if (maxGlobalMemSize > (cl_ulong)SIZE_MAX) {
      maxGlobalMemSize = (cl_ulong)SIZE_MAX;
    }

    unsigned int size;
    unsigned int bufSize;
    unsigned int adjustment;
    int vecsize;

    adjustment = 32*1024*1024; /* Try to allocate a bit less than the limits */
    for(vecsize = 3; vecsize <= 4; ++vecsize)
    {
        /* Make sure we adhere to the maximum individual allocation size and global memory size limits. */
        size = TEST_SIZE;
        bufSize = sizeof(cl_double) * TEST_SIZE * vecsize;

        while ((bufSize > (maxAllocSize - adjustment)) || (3*bufSize > (maxGlobalMemSize - adjustment))) {
            size /= 2;
            bufSize = sizeof(cl_double) * size * vecsize;
        }

        /* Perform the test */
        clProgramWrapper program;
        clKernelWrapper kernel;
        clMemWrapper streams[3];
        cl_double testVector[4];
        int error;
        size_t threads[1], localThreads[1];
        BufferOwningPtr<cl_double> A(malloc(bufSize));
        BufferOwningPtr<cl_double> B(malloc(bufSize));
        BufferOwningPtr<cl_double> C(malloc(bufSize));
        cl_double *inDataA = A;
        cl_double *inDataB = B;
        cl_double *outData = C;

        /* Create kernels */
        if( create_single_kernel_helper( context, &program, &kernel, 1, vecsize == 3 ? &crossKernelSource_doubleV3 : &crossKernelSource_double, "sample_test" ) )
            return -1;

        /* Generate some streams. Note: deliberately do some random data in w to verify that it gets ignored */
        for (unsigned int i = 0; i < size * vecsize; i++)
        {
            inDataA[ i ] = get_random_double( -512.f, 512.f, d );
            inDataB[ i ] = get_random_double( -512.f, 512.f, d );
        }
        fillWithTrickyNumbers_double( inDataA, inDataB, vecsize );

        streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataA, NULL);
        if( streams[0] == NULL )
        {
            log_error("ERROR: Creating input array A failed!\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataB, NULL);
        if( streams[1] == NULL )
        {
            log_error("ERROR: Creating input array B failed!\n");
            return -1;
        }
        streams[2] =
            clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, NULL);
        if( streams[2] == NULL )
        {
            log_error("ERROR: Creating output array failed!\n");
            return -1;
        }

        /* Assign streams and execute */
        for (unsigned int i = 0; i < 3; i++)
        {
            error = clSetKernelArg(kernel, i, sizeof( streams[i] ), &streams[i]);
            test_error( error, "Unable to set indexed kernel arguments" );
        }

        /* Run the kernel */
        threads[0] = size;

        error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
        test_error( error, "Unable to get work group size to use" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
        test_error( error, "Unable to execute test kernel" );

        /* Now get the results */
        error = clEnqueueReadBuffer( queue, streams[2], true, 0, bufSize, outData, 0, NULL, NULL );
        test_error( error, "Unable to read output array!" );

        /* And verify! */
        for (unsigned int i = 0; i < size; i++)
        {
            double errorTolerances[ 4 ];
            // On an embedded device w/ round-to-zero, 3 ulps is the worst-case tolerance for cross product
            cross_product_double( inDataA + i * vecsize, inDataB + i * vecsize, testVector, errorTolerances, 3.f );

            double errs[] = {   fabs( testVector[ 0 ] - outData[ i * vecsize + 0 ] ),
                fabs( testVector[ 1 ] - outData[ i * vecsize + 1 ] ),
                fabs( testVector[ 2 ] - outData[ i * vecsize + 2 ] ) };

            if( errs[ 0 ] > errorTolerances[ 0 ] || errs[ 1 ] > errorTolerances[ 1 ] || errs[ 2 ] > errorTolerances[ 2 ] )
            {
                log_error("ERROR: Data sample %u does not validate! Expected "
                          "(%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, testVector[0], testVector[1], testVector[2],
                          testVector[3], outData[i * vecsize],
                          outData[i * vecsize + 1], outData[i * vecsize + 2],
                          outData[i * vecsize + 3]);
                log_error( "    Input: (%a %a %a) and (%a %a %a)\n",
                          inDataA[ i * vecsize + 0 ], inDataA[ i * vecsize + 1 ], inDataA[ i * vecsize + 2 ],
                          inDataB[ i * vecsize + 0 ], inDataB[ i * vecsize + 1 ], inDataB[ i * vecsize + 2 ] );
                log_error( "    Errors: (%a out of %a), (%a out of %a), (%a out of %a)\n",
                          errs[ 0 ], errorTolerances[ 0 ], errs[ 1 ], errorTolerances[ 1 ], errs[ 2 ], errorTolerances[ 2 ] );
                log_error("     ulp %g\n", Ulp_Error_Double( outData[ i * vecsize + 1 ], testVector[ 1 ] ) );
                return -1;
            }
        }
    }
    return 0;
}

double getMaxValue_double( double vecA[], double vecB[], size_t vecSize )
{
    double a = fmax( fabs( vecA[ 0 ] ), fabs( vecB[ 0 ] ) );
    for( size_t i = 1; i < vecSize; i++ )
        a = fmax( fabs( vecA[ i ] ), fmax( fabs( vecB[ i ] ), a ) );
    return a;
}

typedef double (*twoToFloatVerifyFn_double)( double *srcA, double *srcB, size_t vecSize );

int test_twoToFloat_kernel_double(cl_command_queue queue, cl_context context, const char *fnName,
                                  size_t vecSize, twoToFloatVerifyFn_double verifyFn, double ulpLimit, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    int error;
    size_t i, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    BufferOwningPtr<cl_double> A(malloc(sizeof(cl_double) * TEST_SIZE * vecSize));
    BufferOwningPtr<cl_double> B(malloc(sizeof(cl_double) * TEST_SIZE * vecSize));
    BufferOwningPtr<cl_double> C(malloc(sizeof(cl_double) * TEST_SIZE));

    cl_double *inDataA = A;
    cl_double *inDataB = B;
    cl_double *outData = C;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3 ? twoToFloatKernelPattern_doubleV3 : twoToFloatKernelPattern_double, sizeNames[vecSize-1], sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
        return -1;

    /* Generate some streams */
    for( i = 0; i < TEST_SIZE * vecSize; i++ )
    {
        inDataA[ i ] = any_double(d);
        inDataB[ i ] = any_double(d);
    }
    fillWithTrickyNumbers_double( inDataA, inDataB, vecSize );


    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_double) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_double) * vecSize * TEST_SIZE, inDataB, NULL);
    if( streams[1] == NULL )
    {
        log_error("ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_double) * TEST_SIZE, NULL, NULL);
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
    error = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof( cl_double ) * TEST_SIZE, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        double expected = verifyFn( inDataA + i * vecSize, inDataB + i * vecSize, vecSize );
        if( (double) expected != outData[ i ] )
        {
            if( isnan(expected) && isnan( outData[i] ) )
                continue;

            if( ulpLimit < 0 )
            {
                // Limit below zero means we need to test via a computed error (like cross product does)
                double maxValue =
                getMaxValue_double( inDataA + i * vecSize, inDataB + i * vecSize, vecSize );

                // In this case (dot is the only one that gets here), the ulp is 2*vecSize - 1 (n + n-1 max # of errors)
                double errorTolerance = maxValue * maxValue * ( 2.f * (double)vecSize - 1.f ) * FLT_EPSILON;

                // Limit below zero means test via epsilon instead
                double error = fabs( (double)expected - (double)outData[ i ] );
                if( error > errorTolerance )
                {

                    log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), sources (%a and %a) error of %g against tolerance %g\n",
                              (int)i, (int)vecSize, expected,
                              outData[ i ],
                              inDataA[i*vecSize],
                              inDataB[i*vecSize],
                              (double)error,
                              (double)errorTolerance );

                    char vecA[1000], vecB[1000];
                    vector2string_double( vecA, inDataA + i * vecSize, vecSize );
                    vector2string_double( vecB, inDataB + i * vecSize, vecSize );
                    log_error( "\tvector A: %s\n\tvector B: %s\n", vecA, vecB );
                    return -1;
                }
            }
            else
            {
                double error = Ulp_Error_Double( outData[ i ],
                                                expected );
                if( fabs(error) > ulpLimit )
                {
                    log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), sources (%a and %a) ulp of %f\n",
                              (int)i, (int)vecSize, expected,
                              outData[ i ],
                              inDataA[i*vecSize],
                              inDataB[i*vecSize],
                              error );

                    char vecA[1000], vecB[1000];
                    vector2string_double( vecA, inDataA + i * vecSize, vecSize );
                    vector2string_double( vecB, inDataB + i * vecSize, vecSize );
                    log_error( "\tvector A: %s\n\tvector B: %s\n", vecA, vecB );
                    return -1;
                }
            }
        }
    }
    return 0;
}

double verifyDot_double( double *srcA, double *srcB, size_t vecSize )
{
    double total = 0.f;

    for( unsigned int i = 0; i < vecSize; i++ )
        total += (double)srcA[ i ] * (double)srcB[ i ];

    return total;
}

int test_geom_dot_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;


    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        if( test_twoToFloat_kernel_double( queue, context, "dot", sizes[ size ], verifyDot_double, -1.0f /*magic value*/, d ) != 0 )
        {
            log_error( "   dot double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
    }
    return retVal;
}


int test_geom_fast_distance_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;

    abort();    //there is no double precision fast_distance

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        double maxUlps = 8192.0f +                           // error in sqrt
        0.5f *                              // effect on e of taking sqrt( x + e )
        ( 1.5f * (double) sizes[size] +      // cumulative error for multiplications  (a-b+0.5ulp)**2 = (a-b)**2 + a*0.5ulp + b*0.5 ulp + 0.5 ulp for multiplication
         0.5f * (double) (sizes[size]-1));    // cumulative error for additions

        if( test_twoToFloat_kernel_double( queue, context, "fast_distance", sizes[ size ], verifyDistance_double, maxUlps, d ) != 0 )
        {
            log_error( "   fast_distance double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   fast_distance double vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}


double verifyDistance_double( double *srcA, double *srcB, size_t vecSize )
{
    unsigned int i;
    double diff[4];

    for( i = 0; i < vecSize; i++ )
        diff[i] = srcA[i] - srcB[i];

    return verifyLength_double( diff, vecSize );
}

int test_geom_distance_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        double maxUlps = 3.0f +                              // error in sqrt
        0.5f *                              // effect on e of taking sqrt( x + e )
        ( 1.5f * (double) sizes[size] +      // cumulative error for multiplications  (a-b+0.5ulp)**2 = (a-b)**2 + a*0.5ulp + b*0.5 ulp + 0.5 ulp for multiplication
         0.5f * (double) (sizes[size]-1));    // cumulative error for additions

        maxUlps *= 2.0;         // our reference code may be in error too

        if( test_twoToFloat_kernel_double( queue, context, "distance", sizes[ size ], verifyDistance_double, maxUlps, d ) != 0 )
        {
            log_error( "   distance double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   distance double vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}

typedef double (*oneToFloatVerifyFn_double)( double *srcA, size_t vecSize );

int test_oneToFloat_kernel_double(cl_command_queue queue, cl_context context, const char *fnName,
                                  size_t vecSize, oneToFloatVerifyFn_double verifyFn, double ulpLimit, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    BufferOwningPtr<cl_double> A(malloc(sizeof(cl_double) * TEST_SIZE * vecSize));
    BufferOwningPtr<cl_double> B(malloc(sizeof(cl_double) * TEST_SIZE));
    int error;
    size_t i, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    cl_double *inDataA = A;
    cl_double *outData = B;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3 ? oneToFloatKernelPattern_doubleV3 : oneToFloatKernelPattern_double, sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
        return -1;

    /* Generate some streams */
    for( i = 0; i < TEST_SIZE * vecSize; i++ )
        inDataA[ i ] = any_double(d);

    fillWithTrickyNumbers_double( inDataA, NULL, vecSize );

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_double) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_double) * TEST_SIZE, NULL, NULL);
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

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof( cl_double ) * TEST_SIZE, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        double expected = verifyFn( inDataA + i * vecSize, vecSize );
        if( (double) expected != outData[ i ] )
        {
            double ulps = Ulp_Error_Double( outData[i], expected );
            if( fabs( ulps ) <= ulpLimit )
                continue;

            // We have to special case NAN
            if( isnan( outData[ i ] ) && isnan( expected ) )
                continue;

            if(! (fabs(ulps) < ulpLimit) )
            {
                log_error( "ERROR: Data sample %d at size %d does not validate! Expected (%a), got (%a), source (%a), ulp %f\n",
                          (int)i, (int)vecSize, expected, outData[ i ], inDataA[i*vecSize], ulps );
                char vecA[1000];
                vector2string_double( vecA, inDataA + i * vecSize, vecSize );
                log_error( "\tvector: %s", vecA );
                return -1;
            }
        }
    }

    return 0;
}

double verifyLength_double( double *srcA, size_t vecSize )
{
    double total = 0;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
        total += srcA[i] * srcA[i];

    // Deal with spurious overflow
    if( total == INFINITY )
    {
        total = 0.0;
        for( i = 0; i < vecSize; i++ )
        {
            double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p-600, 0x1LL, -600);
            total += f * f;
        }

        return sqrt( total ) * MAKE_HEX_DOUBLE(0x1.0p600, 0x1LL, 600);
    }

    // Deal with spurious underflow
    if( total < 4 /*max vector length*/ * DBL_MIN / DBL_EPSILON )
    {
        total = 0.0;
        for( i = 0; i < vecSize; i++ )
        {
            double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700);
            total += f * f;
        }

        return sqrt( total ) * MAKE_HEX_DOUBLE(0x1.0p-700, 0x1LL, -700);
    }

    return sqrt( total );
}

int test_geom_length_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        double maxUlps = 3.0f +                              // error in sqrt
        0.5f *                              // effect on e of taking sqrt( x + e )
        ( 0.5f * (double) sizes[size] +      // cumulative error for multiplications
         0.5f * (double) (sizes[size]-1));    // cumulative error for additions

        maxUlps *= 2.0;         // our reference code may be in error too
        if( test_oneToFloat_kernel_double( queue, context, "length", sizes[ size ], verifyLength_double, maxUlps, d ) != 0 )
        {
            log_error( "   length double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   length double vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}


double verifyFastLength_double( double *srcA, size_t vecSize )
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

int test_geom_fast_length_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;

    abort();    //there is no double precision fast_length

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        double maxUlps = 8192.0f +                           // error in half_sqrt
        0.5f *                              // effect on e of taking sqrt( x + e )
        ( 0.5f * (double) sizes[size] +      // cumulative error for multiplications
         0.5f * (double) (sizes[size]-1));    // cumulative error for additions

        if( test_oneToFloat_kernel_double( queue, context, "fast_length", sizes[ size ], verifyFastLength_double, maxUlps, d ) != 0 )
        {
            log_error( "   fast_length double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   fast_length double vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}


typedef void (*oneToOneVerifyFn_double)( double *srcA, double *dstA, size_t vecSize );

int test_oneToOne_kernel_double(cl_command_queue queue, cl_context context, const char *fnName,
                                size_t vecSize, oneToOneVerifyFn_double verifyFn, double ulpLimit, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    BufferOwningPtr<cl_double> A(malloc(sizeof(cl_double) * TEST_SIZE * vecSize));
    BufferOwningPtr<cl_double> B(malloc(sizeof(cl_double) * TEST_SIZE * vecSize));
    int error;
    size_t i, j, threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeNames[][4] = { "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    cl_double *inDataA = A;
    cl_double *outData = B;

    /* Create the source */
    sprintf( kernelSource, vecSize == 3 ? oneToOneKernelPattern_doubleV3 : oneToOneKernelPattern_double, sizeNames[vecSize-1], sizeNames[vecSize-1], fnName );

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
        return -1;

    /* initialize data */
    memset( inDataA, 0, vecSize * sizeof( cl_double ) );
    for( i = vecSize; i < TEST_SIZE * vecSize; i++ )
        inDataA[ i ] = any_double(d);


    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_double) * vecSize * TEST_SIZE, inDataA, NULL);
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_double) * vecSize * TEST_SIZE, NULL, NULL);
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
    error = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof( cl_double ) * TEST_SIZE * vecSize, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        double expected[4];
        verifyFn( inDataA + i * vecSize, expected, vecSize );
        for( j = 0; j < vecSize; j++ )
        {
            // We have to special case NAN
            if( isnan( outData[ i * vecSize + j ] ) && isnan( expected[ j ] ) )
                continue;

            if( expected[j] != outData[ i *vecSize+j ] )
            {
                double error =
                Ulp_Error_Double( outData[i*vecSize + j ], expected[ j ] );
                if( fabs(error) > ulpLimit )
                {
                    log_error( "ERROR: Data sample {%d,%d} at size %d does not validate! Expected %12.24f (%a), got %12.24f (%a), ulp %f\n",
                              (int)i, (int)j, (int)vecSize,
                              expected[j], expected[j],
                              outData[i*vecSize +j],
                              outData[i*vecSize +j], error );
                    log_error( "       Source: " );
                    for( size_t q = 0; q < vecSize; q++ )
                        log_error( "%g ", inDataA[ i * vecSize + q ] );
                    log_error( "\n             : " );
                    for( size_t q = 0; q < vecSize; q++ )
                        log_error( "%a ", inDataA[ i * vecSize + q ] );
                    log_error( "\n" );
                    log_error( "       Result: " );
                    for( size_t q = 0; q < vecSize; q++ )
                        log_error( "%g ", outData[i * vecSize + q ] );
                    log_error( "\n             : " );
                    for( size_t q = 0; q < vecSize; q++ )
                        log_error( "%a ", outData[i * vecSize + q ] );
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
        }
    }

    return 0;
}

void verifyNormalize_double( double *srcA, double *dst, size_t vecSize )
{
    double total = 0, value;
    unsigned int i;

    // We calculate everything as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op
    for( i = 0; i < vecSize; i++ )
        total += (double)srcA[i] * (double)srcA[i];

    if( total < vecSize * DBL_MIN / DBL_EPSILON )
    { //we may have incurred denormalization loss -- rescale
        total = 0;
        for( i = 0; i < vecSize; i++ )
        {
            dst[i] = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700);  //exact
            total += dst[i] * dst[i];
        }

        //If still zero
        if( total == 0.0 )
        {
            // Special edge case: copy vector over without change
            for( i = 0; i < vecSize; i++ )
                dst[i] = srcA[i];
            return;
        }

        srcA = dst;
    }
    else if( total == INFINITY )
    { //we may have incurred spurious overflow
        double scale = MAKE_HEX_DOUBLE(0x1.0p-512, 0x1LL, -512) / vecSize;
        total = 0;
        for( i = 0; i < vecSize; i++ )
        {
            dst[i] = srcA[i] * scale;  //exact
            total += dst[i] * dst[i];
        }

        // If there are infinities here, handle those
        if( total == INFINITY )
        {
            total = 0;
            for( i = 0; i < vecSize; i++ )
                {
                if( isinf(dst[i]) )
                {
                    dst[i] = copysign( 1.0, srcA[i] );
                    total += 1.0;
                }
                else
                    dst[i] = copysign( 0.0, srcA[i] );
        }
        }

        srcA = dst;
    }

    value = sqrt( total );

    for( i = 0; i < vecSize; i++ )
        dst[i] = srcA[i] / value;
}

int test_geom_normalize_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d)
{
    size_t sizes[] = { 1, 2, 3, 4, 0 };
    unsigned int size;
    int retVal = 0;

    for( size = 0; sizes[ size ] != 0 ; size++ )
    {
        double maxUlps = 2.5f +                              // error in rsqrt + error in multiply
        0.5f *                               // effect on e of taking sqrt( x + e )
        ( 0.5f * (double) sizes[size] +      // cumulative error for multiplications
         0.5f * (double) (sizes[size]-1));    // cumulative error for additions

        maxUlps *= 2.0; //our reference code is not infinitely precise and may have error of its own
        if( test_oneToOne_kernel_double( queue, context, "normalize", sizes[ size ], verifyNormalize_double, maxUlps, d ) != 0 )
        {
            log_error( "   normalize double vector size %d FAILED\n", (int)sizes[ size ] );
            retVal = -1;
        }
        else
        {
            log_info( "   normalize double vector size %d passed\n", (int)sizes[ size ] );
        }
    }
    return retVal;
}






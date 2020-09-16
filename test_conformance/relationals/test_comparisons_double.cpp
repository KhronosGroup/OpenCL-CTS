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
#include "harness/typeWrappers.h"

#define TEST_SIZE 512

const char *equivTestKernelPattern_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global long%s *destValues, __global long%s *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"    destValuesB[tid] = sourceA[tid] %s sourceB[tid];\n"
"\n"
"}\n";

const char *equivTestKernelPatternLessGreater_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global long%s *destValues, __global long%s *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"    destValuesB[tid] = (sourceA[tid] < sourceB[tid]) | (sourceA[tid] > sourceB[tid]);\n"
"\n"
"}\n";


const char *equivTestKernelPattern_double3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global long%s *destValues, __global long%s *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    double3 sampA = vload3(tid, (__global double *)sourceA);\n"
"    double3 sampB = vload3(tid, (__global double *)sourceB);\n"
"    vstore3(%s( sampA, sampB ), tid, (__global long *)destValues);\n"
"    vstore3(( sampA %s sampB ), tid, (__global long *)destValuesB);\n"
"\n"
"}\n";

const char *equivTestKernelPatternLessGreater_double3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void sample_test(__global double%s *sourceA, __global double%s *sourceB, __global long%s *destValues, __global long%s *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    double3 sampA = vload3(tid, (__global double *)sourceA);\n"
"    double3 sampB = vload3(tid, (__global double *)sourceB);\n"
"    vstore3(%s( sampA, sampB ), tid, (__global long *)destValues);\n"
"    vstore3(( sampA < sampB ) | (sampA > sampB), tid, (__global long *)destValuesB);\n"
"\n"
"}\n";


typedef bool (*equivVerifyFn)( double inDataA, double inDataB );

void verify_equiv_values_double( unsigned int vecSize, double *inDataA, double *inDataB, cl_long *outData, equivVerifyFn verifyFn )
{
    unsigned int i;
    cl_long trueResult;
    bool result;

    trueResult = ( vecSize == 1 ) ? 1 : -1;
    for( i = 0; i < vecSize; i++ )
    {
        result = verifyFn( inDataA[ i ], inDataB[ i ] );
        outData[ i ] = result ? trueResult : 0;
    }
}

void generate_equiv_test_data_double( double *outData, unsigned int vecSize, bool alpha, MTdata d )
{
    unsigned int i;

    generate_random_data( kDouble, vecSize * TEST_SIZE, d, outData );

    // Fill the first few vectors with NAN in each vector element (or the second set if we're alpha, so we can test either case)
    if( alpha )
        outData += vecSize * vecSize;
    for( i = 0; i < vecSize; i++ )
    {
        outData[ 0 ] = NAN;
        outData += vecSize + 1;
    }
    // Make sure the third set is filled regardless, to test the case where both have NANs
    if( !alpha )
        outData += vecSize * vecSize;
    for( i = 0; i < vecSize; i++ )
    {
        outData[ 0 ] = NAN;
        outData += vecSize + 1;
    }
}

int test_equiv_kernel_double(cl_context context, cl_command_queue queue, const char *fnName, const char *opName,
                             unsigned int vecSize, equivVerifyFn verifyFn, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[4];
    double inDataA[TEST_SIZE * 16], inDataB[ TEST_SIZE * 16 ];
    cl_long outData[TEST_SIZE * 16], expected[16];
    int error, i, j;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4];


    /* Create the source */
    if( vecSize == 1 )
        sizeName[ 0 ] = 0;
    else
        sprintf( sizeName, "%d", vecSize );

    if(DENSE_PACK_VECS && vecSize == 3) {
        if (strcmp(fnName, "islessgreater")) {
            sprintf( kernelSource, equivTestKernelPattern_double3, sizeName, sizeName, sizeName, sizeName, fnName, opName );
        } else {
            sprintf( kernelSource, equivTestKernelPatternLessGreater_double3, sizeName, sizeName, sizeName, sizeName, fnName );
        }
    } else {
        if (strcmp(fnName, "islessgreater")) {
            sprintf( kernelSource, equivTestKernelPattern_double, sizeName, sizeName, sizeName, sizeName, fnName, opName );
        } else {
            sprintf( kernelSource, equivTestKernelPatternLessGreater_double, sizeName, sizeName, sizeName, sizeName, fnName );
        }
    }

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }

    /* Generate some streams */
    generate_equiv_test_data_double( inDataA, vecSize, true, d );
    generate_equiv_test_data_double( inDataB, vecSize, false, d );

    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof( cl_double ) * vecSize * TEST_SIZE, &inDataA, &error);
    if( streams[0] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeof( cl_double ) * vecSize * TEST_SIZE, &inDataB, &error);
    if( streams[1] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer( context, CL_MEM_READ_WRITE, sizeof( cl_long ) * vecSize * TEST_SIZE, NULL, &error);
    if( streams[2] == NULL )
    {
        print_error( error, "Creating output array failed!\n");
        return -1;
    }
    streams[3] = clCreateBuffer( context, CL_MEM_READ_WRITE, sizeof( cl_long ) * vecSize * TEST_SIZE, NULL, &error);
    if( streams[3] == NULL )
    {
        print_error( error, "Creating output array failed!\n");
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

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof( cl_long ) * TEST_SIZE * vecSize, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        verify_equiv_values_double( vecSize, &inDataA[ i * vecSize ], &inDataB[ i * vecSize ], expected, verifyFn);

        for( j = 0; j < (int)vecSize; j++ )
        {
            if( expected[ j ] != outData[ i * vecSize + j ] )
            {
                log_error( "ERROR: Data sample %d:%d at size %d does not validate! Expected %lld, got %lld, source %f,%f\n",
                          i, j, vecSize, expected[ j ], outData[ i * vecSize + j ], inDataA[i*vecSize + j], inDataB[i*vecSize + j] );
                return -1;
            }
        }
    }

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[3], true, 0, sizeof( cl_long ) * TEST_SIZE * vecSize, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        verify_equiv_values_double( vecSize, &inDataA[ i * vecSize ], &inDataB[ i * vecSize ], expected, verifyFn);

        for( j = 0; j < (int)vecSize; j++ )
        {
            if( expected[ j ] != outData[ i * vecSize + j ] )
            {
                log_error( "ERROR: Data sample %d:%d at size %d does not validate! Expected %lld, got %lld, source %f,%f\n",
                          i, j, vecSize, expected[ j ], outData[ i * vecSize + j ], inDataA[i*vecSize + j], inDataB[i*vecSize + j] );
                return -1;
            }
        }
    }

    return 0;
}

int test_equiv_kernel_set_double(cl_device_id device, cl_context context, cl_command_queue queue, const char *fnName, const char *opName, equivVerifyFn verifyFn, MTdata d )
{
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index;
    int retVal = 0;

    if (!is_extension_available(device, "cl_khr_fp64")) {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
        return 0;
    }
    log_info("Testing doubles.\n");

    for( index = 0; vecSizes[ index ] != 0; index++ )
    {
        // Test!
        if( test_equiv_kernel_double(context, queue, fnName, opName, vecSizes[ index ], verifyFn, d ) != 0 )
        {
            log_error( "   Vector double%d FAILED\n", vecSizes[ index ] );
            retVal = -1;
        }
    }

    return retVal;
}

bool isequal_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return valueA == valueB;
}

int test_relational_isequal_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "isequal", "==", isequal_verify_fn_double, seed );
}

bool isnotequal_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return true;
    return valueA != valueB;
}

int test_relational_isnotequal_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "isnotequal", "!=", isnotequal_verify_fn_double, seed );
}

bool isgreater_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return valueA > valueB;
}

int test_relational_isgreater_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "isgreater", ">", isgreater_verify_fn_double, seed );
}

bool isgreaterequal_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return valueA >= valueB;
}

int test_relational_isgreaterequal_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "isgreaterequal", ">=", isgreaterequal_verify_fn_double, seed );
}

bool isless_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return valueA < valueB;
}

int test_relational_isless_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "isless", "<", isless_verify_fn_double, seed );
}

bool islessequal_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return valueA <= valueB;
}

int test_relational_islessequal_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "islessequal", "<=", islessequal_verify_fn_double, seed );
}

bool islessgreater_verify_fn_double( double valueA, double valueB )
{
    if( isnan( valueA ) || isnan( valueB ) )
        return false;
    return ( valueA < valueB ) || ( valueA > valueB );
}

int test_relational_islessgreater_double(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    RandomSeed seed(gRandomSeed);
    return test_equiv_kernel_set_double( device, context, queue, "islessgreater", "<>", islessgreater_verify_fn_double, seed );
}



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

const char *sample_single_test_kernel[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n" };

const char *sample_struct_array_test_kernel[] = {
"typedef struct {\n"
"int A;\n"
"int B;\n"
"} input_pair_t;\n"
"\n"
"__kernel void sample_test(__global input_pair_t *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src[tid].A + src[tid].B;\n"
"\n"
"}\n" };

const char *sample_const_test_kernel[] = {
"__kernel void sample_test(__constant int *src1, __constant int *src2, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src1[tid] + src2[tid];\n"
"\n"
"}\n" };

const char *sample_const_global_test_kernel[] = {
"__constant int addFactor = 1024;\n"
"__kernel void sample_test(__global int *src1, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src1[tid] + addFactor;\n"
"\n"
"}\n" };

const char *sample_two_kernel_program[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n",
"__kernel void sample_test2(__global int *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)src[tid];\n"
"\n"
"}\n" };


REGISTER_TEST(get_kernel_info)
{
    int error;
    cl_program program, testProgram;
    cl_context testContext;
    cl_kernel kernel;
    cl_char name[ 512 ];
    cl_uint numArgs, numInstances;
    size_t paramSize;


    /* Create reference */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_single_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &paramSize );
    test_error( error, "Unable to get kernel function name param size" );
    if( paramSize != strlen( "sample_test" ) + 1 )
    {
        log_error( "ERROR: Kernel function name param returns invalid size (expected %d, got %d)\n", (int)strlen( "sample_test" ) + 1, (int)paramSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_FUNCTION_NAME, sizeof( name ), name, NULL );
    test_error( error, "Unable to get kernel function name" );
    if( strcmp( (char *)name, "sample_test" ) != 0 )
    {
        log_error( "ERROR: Kernel function name returned invalid value (expected sample_test, got %s)\n", (char *)name );
        return -1;
    }


    error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, 0, NULL, &paramSize );
    test_error( error, "Unable to get kernel arg count param size" );
    if( paramSize != sizeof( numArgs ) )
    {
        log_error( "ERROR: Kernel arg count param returns invalid size (expected %d, got %d)\n", (int)sizeof( numArgs ), (int)paramSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, sizeof( numArgs ), &numArgs, NULL );
    test_error( error, "Unable to get kernel arg count" );
    if( numArgs != 2 )
    {
        log_error( "ERROR: Kernel arg count returned invalid value (expected %d, got %d)\n", 2, numArgs );
        return -1;
    }


    error = clGetKernelInfo( kernel, CL_KERNEL_REFERENCE_COUNT, 0, NULL, &paramSize );
    test_error( error, "Unable to get kernel reference count param size" );
    if( paramSize != sizeof( numInstances ) )
    {
        log_error( "ERROR: Kernel reference count param returns invalid size (expected %d, got %d)\n", (int)sizeof( numInstances ), (int)paramSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_REFERENCE_COUNT, sizeof( numInstances ), &numInstances, NULL );
    test_error( error, "Unable to get kernel reference count" );


    error = clGetKernelInfo( kernel, CL_KERNEL_PROGRAM, 0, NULL, &paramSize );
    test_error( error, "Unable to get kernel program param size" );
    if( paramSize != sizeof( testProgram ) )
    {
        log_error( "ERROR: Kernel program param returns invalid size (expected %d, got %d)\n", (int)sizeof( testProgram ), (int)paramSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_PROGRAM, sizeof( testProgram ), &testProgram, NULL );
    test_error( error, "Unable to get kernel program" );
    if( testProgram != program )
    {
        log_error( "ERROR: Kernel program returned invalid value (expected %p, got %p)\n", program, testProgram );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_CONTEXT, sizeof( testContext ), &testContext, NULL );
    test_error( error, "Unable to get kernel context" );
    if( testContext != context )
    {
        log_error( "ERROR: Kernel context returned invalid value (expected %p, got %p)\n", context, testContext );
        return -1;
    }

    /* Release memory */
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    return 0;
}

REGISTER_TEST(execute_kernel_local_sizes)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper            streams[2];
    size_t    threads[1], localThreads[1];
    RandomSeed seed( gRandomSeed );
    int i;

    num_elements = 100;
    std::vector<cl_float> inputData(num_elements);
    std::vector<cl_int> outputData(num_elements);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_single_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Write some test data */
    for (i = 0; i < num_elements; i++)
        inputData[i] = get_random_float(-(float) 0x7fffffff, (float) 0x7fffffff, seed);

    error = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                                 sizeof(cl_float) * num_elements,
                                 (void *)inputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to set testing kernel data" );

    /* Set the arguments */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set kernel arguments" );

    /* Test running the kernel and verifying it */
    threads[0] = (size_t)num_elements;
    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

    /* Try again */
    if( localThreads[0] > 1 )
        localThreads[0] /= 2;
    while( localThreads[0] > 1 && 0 != threads[0] % localThreads[0] )
        localThreads[0]--;
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

    /* And again */
    if( localThreads[0] > 1 )
        localThreads[0] /= 2;
    while( localThreads[0] > 1 && 0 != threads[0] % localThreads[0] )
        localThreads[0]--;
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

    /* One more time */
    localThreads[0] = (unsigned int)1;
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

    return 0;
}

REGISTER_TEST(set_kernel_arg_by_index)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper    streams[2];
    size_t    threads[1], localThreads[1];
    RandomSeed seed( gRandomSeed );
    int i;

    num_elements = 10;
    std::vector<cl_float> inputData(num_elements);
    std::vector<cl_int> outputData(num_elements);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_single_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Write some test data */
    for (i = 0; i < num_elements; i++)
        inputData[i] = get_random_float(-(float) 0x7fffffff, (float) 0x7fffffff, seed);

    error = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                                 sizeof(cl_float) * num_elements,
                                 (void *)inputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to set testing kernel data" );

    /* Test setting the arguments by index manually */
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = (size_t)num_elements;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

    return 0;
}

REGISTER_TEST(set_kernel_arg_constant)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper            streams[3];
    size_t    threads[1], localThreads[1];
    int i;
    cl_ulong maxSize;
    MTdata d;

    num_elements = 10;
    std::vector<cl_int> outputData(num_elements);
    std::vector<cl_int> randomTestDataA(num_elements);
    std::vector<cl_int> randomTestDataB(num_elements);

    /* Verify our test buffer won't be bigger than allowed */
    maxSize = get_device_info_max_constant_buffer_size(
        device, MAX_DEVICE_MEMORY_SIZE_DIVISOR);
    if (maxSize < sizeof(cl_int) * num_elements)
    {
        log_error( "ERROR: Unable to test constant argument to kernel: max size of constant buffer is reported as %d!\n", (int)maxSize );
        return -1;
    }

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_const_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    d = init_genrand( gRandomSeed );
    for (i = 0; i < num_elements; i++)
    {
        randomTestDataA[i] = (cl_int)genrand_int32(d) & 0xffffff;    /* Make sure values are positive, just so we don't have to */
        randomTestDataB[i] = (cl_int)genrand_int32(d) & 0xffffff;    /* deal with overflow on the verification */
    }
    free_mtdata(d); d = NULL;

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_int) * num_elements,
                                randomTestDataA.data(), &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_int) * num_elements,
                                randomTestDataB.data(), &error);
    test_error( error, "Creating test array failed" );
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 2, sizeof( streams[2] ), &streams[2]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = (size_t)num_elements;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != randomTestDataA[i] + randomTestDataB[i])
        {
            log_error( "ERROR: Data sample %d did not verify! %d does not match %d + %d (%d)\n", i, outputData[i], randomTestDataA[i], randomTestDataB[i], ( randomTestDataA[i] + randomTestDataB[i] ) );
            return -1;
        }
    }

    return 0;
}

REGISTER_TEST(set_kernel_arg_struct_array)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper            streams[2];
    size_t    threads[1], localThreads[1];
    int i;
    MTdata d;

    num_elements = 10;
    std::vector<cl_int> outputData(num_elements);

    typedef struct img_pair_type
    {
        int A;
        int B;
    } image_pair_t;

    std::vector<image_pair_t> image_pair(num_elements);


    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_struct_array_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    d = init_genrand( gRandomSeed );
    for (i = 0; i < num_elements; i++)
    {
        image_pair[i].A = (cl_int)genrand_int32(d);
        image_pair[i].B = (cl_int)genrand_int32(d);
    }
    free_mtdata(d); d = NULL;

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                sizeof(image_pair_t) * num_elements,
                                (void *)image_pair.data(), &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Test running the kernel and verifying it */
    threads[0] = (size_t)num_elements;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != image_pair[i].A + image_pair[i].B)
        {
            log_error( "ERROR: Data did not verify!\n" );
            return -1;
        }
    }

    return 0;
}

REGISTER_TEST(create_kernels_in_program)
{
    int error;
    cl_program program;
    cl_kernel  kernel[3];
    unsigned int kernelCount;

    error = create_single_kernel_helper(context, &program, NULL, 2, sample_two_kernel_program, NULL);
    test_error(error, "Unable to build test program");

    /* Try getting the kernel count */
    error = clCreateKernelsInProgram( program, 0, NULL, &kernelCount );
    test_error( error, "Unable to get kernel count for built program" );
    if( kernelCount != 2 )
    {
        log_error( "ERROR: Returned kernel count from clCreateKernelsInProgram is incorrect! (got %d, expected 2)\n", kernelCount );
        return -1;
    }

    /* Try actually getting the kernels */
    error = clCreateKernelsInProgram( program, 2, kernel, NULL );
    test_error( error, "Unable to get kernels for built program" );
    clReleaseKernel( kernel[0] );
    clReleaseKernel( kernel[1] );

    clReleaseProgram( program );
    return 0;
}

REGISTER_TEST(kernel_global_constant)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper            streams[2];
    size_t    threads[1], localThreads[1];
    int i;
    MTdata d;

    num_elements = 10;
    std::vector<cl_int> outputData(num_elements);
    std::vector<cl_int> randomTestDataA(num_elements);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_const_global_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    d = init_genrand( gRandomSeed );
    for (i = 0; i < num_elements; i++)
    {
        randomTestDataA[i] = (cl_int)genrand_int32(d) & 0xffff;    /* Make sure values are positive and small, just so we don't have to */
    }
    free_mtdata(d); d = NULL;

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_int) * num_elements,
                                randomTestDataA.data(), &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = (size_t)num_elements;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                sizeof(cl_int) * num_elements,
                                (void *)outputData.data(), 0, NULL, NULL);
    test_error( error, "Unable to get result data" );

    for (i = 0; i < num_elements; i++)
    {
        if (outputData[i] != randomTestDataA[i] + 1024)
        {
            log_error( "ERROR: Data sample %d did not verify! %d does not match %d + 1024 (%d)\n", i, outputData[i], randomTestDataA[i], ( randomTestDataA[i] + 1024 ) );
            return -1;
        }
    }

    return 0;
}

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
#include "Utility.h"

#include <string.h>
#include "FunctionList.h"

int TestMacro_Int_Float(const Func *f, MTdata);
int TestMacro_Int_Double(const Func *f, MTdata);

extern const vtbl _macro_unary = { "macro_unary", TestMacro_Int_Float,
                                   TestMacro_Int_Double };

static int BuildKernel( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p );

static int BuildKernel( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "__kernel void math_kernel", sizeNames[vectorSize], "( __global int", sizeNames[vectorSize], "* out, __global float", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i] );\n"
                            "}\n"
                        };
    const char *c3[] = {    "__kernel void math_kernel", sizeNames[vectorSize], "( __global int* out, __global float* in)\n"
                            "{\n"
                            "   size_t i = get_global_id(0);\n"
                            "   if( i + 1 < get_global_size(0) )\n"
                            "   {\n"
                            "       float3 f0 = vload3( 0, in + 3 * i );\n"
                            "       int3 i0 = ", name, "( f0 );\n"
                            "       vstore3( i0, 0, out + 3*i );\n"
                            "   }\n"
                            "   else\n"
                            "   {\n"
                            "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                            "       int3 i0;\n"
                            "       float3 f0;\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 1:\n"
                            "               f0 = (float3)( in[3*i], 0xdead, 0xdead ); \n"
                            "               break;\n"
                            "           case 0:\n"
                            "               f0 = (float3)( in[3*i], in[3*i+1], 0xdead ); \n"
                            "               break;\n"
                            "       }\n"
                            "       i0 = ", name, "( f0 );\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 0:\n"
                            "               out[3*i+1] = i0.y; \n"
                            "               // fall through\n"
                            "           case 1:\n"
                            "               out[3*i] = i0.x; \n"
                            "               break;\n"
                            "       }\n"
                            "   }\n"
                            "}\n"
                        };

    const char **kern = c;
    size_t kernSize = sizeof(c)/sizeof(c[0]);

    if( sizeValues[vectorSize] == 3 )
    {
        kern = c3;
        kernSize = sizeof(c3)/sizeof(c3[0]);
    }

    char testName[32];
    snprintf( testName, sizeof( testName ) -1, "math_kernel%s", sizeNames[vectorSize] );

    return MakeKernels(kern, (cl_uint) kernSize, testName, kernel_count, k, p);
}

static int BuildKernelDouble( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global long", sizeNames[vectorSize], "* out, __global double", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i] );\n"
                            "}\n"
                        };

    const char *c3[] = {    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global long* out, __global double* in)\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   if( i + 1 < get_global_size(0) )\n"
                        "   {\n"
                        "       double3 d0 = vload3( 0, in + 3 * i );\n"
                        "       long3 l0 = ", name, "( d0 );\n"
                        "       vstore3( l0, 0, out + 3*i );\n"
                        "   }\n"
                        "   else\n"
                        "   {\n"
                        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                        "       double3 d0;\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 1:\n"
                        "               d0 = (double3)( in[3*i], NAN, NAN ); \n"
                        "               break;\n"
                        "           case 0:\n"
                        "               d0 = (double3)( in[3*i], in[3*i+1], NAN ); \n"
                        "               break;\n"
                        "       }\n"
                        "       long3 l0 = ", name, "( d0 );\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 0:\n"
                        "               out[3*i+1] = l0.y; \n"
                        "               // fall through\n"
                        "           case 1:\n"
                        "               out[3*i] = l0.x; \n"
                        "               break;\n"
                        "       }\n"
                        "   }\n"
                        "}\n"
                    };

    const char **kern = c;
    size_t kernSize = sizeof(c)/sizeof(c[0]);

    if( sizeValues[vectorSize] == 3 )
    {
        kern = c3;
        kernSize = sizeof(c3)/sizeof(c3[0]);
    }


    char testName[32];
    snprintf( testName, sizeof( testName ) -1, "math_kernel%s", sizeNames[vectorSize] );

    return MakeKernels(kern, (cl_uint) kernSize, testName, kernel_count, k, p);
}

typedef struct BuildKernelInfo
{
    cl_uint     offset;            // the first vector size to build
    cl_uint     kernel_count;
    cl_kernel   **kernels;
    cl_program  *programs;
    const char  *nameInCode;
}BuildKernelInfo;

static cl_int BuildKernel_FloatFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p );
static cl_int BuildKernel_FloatFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p )
{
    BuildKernelInfo *info = (BuildKernelInfo*) p;
    cl_uint i = info->offset + job_id;
    return BuildKernel( info->nameInCode, i, info->kernel_count, info->kernels[i], info->programs + i );
}

static cl_int BuildKernel_DoubleFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p );
static cl_int BuildKernel_DoubleFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p )
{
    BuildKernelInfo *info = (BuildKernelInfo*) p;
    cl_uint i = info->offset + job_id;
    return BuildKernelDouble( info->nameInCode, i, info->kernel_count, info->kernels[i], info->programs + i );
}

//Thread specific data for a worker thread
typedef struct ThreadInfo
{
    cl_mem      inBuf;                              // input buffer for the thread
    cl_mem      outBuf[ VECTOR_SIZE_COUNT ];        // output buffers for the thread
    cl_command_queue tQueue;                        // per thread command queue to improve performance
}ThreadInfo;

typedef struct TestInfo
{
    size_t      subBufferSize;                      // Size of the sub-buffer in elements
    const Func  *f;                                 // A pointer to the function info
    cl_program  programs[ VECTOR_SIZE_COUNT ];      // programs for various vector sizes
    cl_kernel   *k[VECTOR_SIZE_COUNT ];             // arrays of thread-specific kernels for each worker thread:  k[vector_size][thread_id]
    ThreadInfo  *tinfo;                             // An array of thread specific information for each worker thread
    cl_uint     threadCount;                        // Number of worker threads
    cl_uint     jobCount;                           // Number of jobs
    cl_uint     step;                               // step between each chunk and the next.
    cl_uint     scale;                              // stride between individual test values
    int         ftz;                                // non-zero if running in flush to zero mode

}TestInfo;

static cl_int TestFloat( cl_uint job_id, cl_uint thread_id, void *p );

int TestMacro_Int_Float(const Func *f, MTdata d)
{
    TestInfo    test_info;
    cl_int      error;
    size_t      i, j;

    logFunctionInfo(f->name,sizeof(cl_float),gTestFastRelaxed);

    // Init test_info
    memset( &test_info, 0, sizeof( test_info ) );
    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE / (sizeof( cl_float) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.scale =  1;
    if (gWimpyMode )
    {
        test_info.subBufferSize = gWimpyBufferSize / (sizeof( cl_float) * RoundUpToNextPowerOfTwo(test_info.threadCount));
        test_info.scale =  (cl_uint) sizeof(cl_float) * 2 * gWimpyReductionFactor;
    }
    test_info.step = (cl_uint) test_info.subBufferSize * test_info.scale;
    if (test_info.step / test_info.subBufferSize != test_info.scale)
    {
        //there was overflow
        test_info.jobCount = 1;
    }
    else
    {
        test_info.jobCount = (cl_uint)((1ULL << 32) / test_info.step);
    }

    test_info.f = f;
    test_info.ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    // cl_kernels aren't thread safe, so we make one for each vector size for every thread
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        size_t array_size = test_info.threadCount * sizeof( cl_kernel );
        test_info.k[i] = (cl_kernel*)malloc( array_size );
        if( NULL == test_info.k[i] )
        {
            vlog_error( "Error: Unable to allocate storage for kernels!\n" );
            error = CL_OUT_OF_HOST_MEMORY;
            goto exit;
        }
        memset( test_info.k[i], 0, array_size );
    }
    test_info.tinfo = (ThreadInfo*)malloc( test_info.threadCount * sizeof(*test_info.tinfo) );
    if( NULL == test_info.tinfo )
    {
        vlog_error( "Error: Unable to allocate storage for thread specific data.\n" );
        error = CL_OUT_OF_HOST_MEMORY;
        goto exit;
    }
    memset( test_info.tinfo, 0, test_info.threadCount * sizeof(*test_info.tinfo) );
    for( i = 0; i < test_info.threadCount; i++ )
    {
        cl_buffer_region region = { i * test_info.subBufferSize * sizeof( cl_float), test_info.subBufferSize * sizeof( cl_float) };
        test_info.tinfo[i].inBuf = clCreateSubBuffer( gInBuffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if( error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error( "Error: Unable to create sub-buffer of gInBuffer for region {%zd, %zd}\n", region.origin, region.size );
            goto exit;
        }

        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer( gOutBuffer[j], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
            if( error || NULL == test_info.tinfo[i].outBuf[j] )
            {
                vlog_error( "Error: Unable to create sub-buffer of gOutBuffer for region {%zd, %zd}\n", region.origin, region.size );
                goto exit;
            }
        }
        test_info.tinfo[i].tQueue = clCreateCommandQueue(gContext, gDevice, 0, &error);
        if( NULL == test_info.tinfo[i].tQueue || error )
        {
            vlog_error( "clCreateCommandQueue failed. (%d)\n", error );
            goto exit;
        }
    }

    // Init the kernels
    {
        BuildKernelInfo build_info = { gMinVectorSizeIndex, test_info.threadCount, test_info.k, test_info.programs, f->nameInCode };
        if( (error = ThreadPool_Do( BuildKernel_FloatFn, gMaxVectorSizeIndex - gMinVectorSizeIndex, &build_info ) ))
            goto exit;
    }

    if( !gSkipCorrectnessTesting )
    {
        error = ThreadPool_Do( TestFloat, test_info.jobCount, &test_info );

        if( error )
            goto exit;

        if( gWimpyMode )
            vlog( "Wimp pass" );
        else
            vlog( "passed" );
    }

    if( gMeasureTimes )
    {
        //Init input array
        cl_uint *p = (cl_uint *)gIn;
        for( j = 0; j < BUFFER_SIZE / sizeof( float ); j++ )
            p[j] = genrand_int32(d);
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, BUFFER_SIZE, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeof( cl_float ) * sizeValues[j];
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;    // BUFFER_SIZE / vectorSize  rounded up
            if( ( error = clSetKernelArg( test_info.k[j][0], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(test_info.programs[j]); goto exit; }

            double sum = 0.0;
            double bestTime = INFINITY;
            for( i = 0; i < PERF_LOOP_COUNT; i++ )
            {
                uint64_t startTime = GetTime();
                if( (error = clEnqueueNDRangeKernel(gQueue, test_info.k[j][0], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
                {
                    vlog_error( "FAILED -- could not execute kernel\n" );
                    goto exit;
                }

                // Make sure OpenCL is done
                if( (error = clFinish(gQueue) ) )
                {
                    vlog_error( "Error %d at clFinish\n", error );
                    goto exit;
                }

                uint64_t endTime = GetTime();
                double time = SubtractTime( endTime, startTime );
                sum += time;
                if( time < bestTime )
                    bestTime = time;
            }

            if( gReportAverageTimes )
                bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double) gDeviceFrequency * gComputeDevices * gSimdSize * 1e6 / (BUFFER_SIZE / sizeof( float ) );
            vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sf%s", f->name, sizeNames[j] );
        }
    }

    vlog( "\n" );

exit:
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        clReleaseProgram(test_info.programs[i]);
        if( test_info.k[i] )
        {
            for( j = 0; j < test_info.threadCount; j++ )
                clReleaseKernel(test_info.k[i][j]);

            free( test_info.k[i] );
        }
    }
    if( test_info.tinfo )
    {
        for( i = 0; i < test_info.threadCount; i++ )
        {
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
                clReleaseMemObject(test_info.tinfo[i].outBuf[j]);
            clReleaseCommandQueue(test_info.tinfo[i].tQueue);
        }

        free( test_info.tinfo );
    }

    return error;
}

static cl_int TestFloat( cl_uint job_id, cl_uint thread_id, void *data )
{
    const TestInfo *job = (const TestInfo *) data;
    size_t  buffer_elements = job->subBufferSize;
    size_t  buffer_size = buffer_elements * sizeof( cl_float );
    cl_uint scale = job->scale;
    cl_uint base = job_id * (cl_uint) job->step;
    ThreadInfo *tinfo = job->tinfo + thread_id;
    fptr    func = job->f->func;
    int     ftz = job->ftz;
    cl_uint j, k;
    cl_int error = CL_SUCCESS;
    cl_int ret   = CL_SUCCESS;
    const char *name = job->f->name;

    int signbit_test = 0;
    if(!strcmp(name, "signbit"))
        signbit_test = 1;

    #define ref_func(s) ( signbit_test ? func.i_f_f( s ) : func.i_f( s ) )

    // start the map of the output arrays
    cl_event e[ VECTOR_SIZE_COUNT ];
    cl_int  *out[ VECTOR_SIZE_COUNT ];
    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_int*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_WRITE, 0, buffer_size, 0, NULL, e + j, &error);
        if( error || NULL == out[j])
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            return error;
        }
    }


    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush failed\n" );

    // Write the new values to the input array
    cl_uint *p = (cl_uint*) gIn + thread_id * buffer_elements;
    for( j = 0; j < buffer_elements; j++ )
        p[j] = base + j * scale;

    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0, buffer_size, p, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        return error;
    }

    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        //Wait for the map to finish
        if( (error = clWaitForEvents(1, e + j) ))
        {
            vlog_error( "Error: clWaitForEvents failed! err: %d\n", error );
            return error;
        }
        if( (error = clReleaseEvent( e[j] ) ))
        {
            vlog_error( "Error: clReleaseEvent failed! err: %d\n", error );
            return error;
        }

        // Fill the result buffer with garbage, so that old results don't carry over
        uint32_t pattern = 0xffffdead;
        memset_pattern4(out[j], &pattern, buffer_size);
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL) ))
        {
            vlog_error( "Error: clEnqueueMapBuffer failed! err: %d\n", error );
            return error;
        }

        // run the kernel
        size_t vectorCount = (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id];  //each worker thread has its own copy of the cl_kernel
        cl_program program = job->programs[j];

        if( ( error = clSetKernelArg( kernel, 0, sizeof( tinfo->outBuf[j] ), &tinfo->outBuf[j] ))){ LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 1, sizeof( tinfo->inBuf ), &tinfo->inBuf ) )) { LogBuildError(program); return error; }

        if( (error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL, &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error( "FAILED -- could not execute kernel\n" );
            return error;
        }
    }


    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush 2 failed\n" );

    if( gSkipCorrectnessTesting )
        return CL_SUCCESS;

    //Calculate the correctly rounded reference result
    cl_int *r = (cl_int *)gOut_Ref + thread_id * buffer_elements;
    float *s = (float *)p;
    for( j = 0; j < buffer_elements; j++ )
        r[j] = ref_func( s[j] );

    // Read the data back -- no need to wait for the first N-1 buffers. This is an in order queue.
    for( j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_int*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
        if( error || NULL == out[j] )
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            return error;
        }
    }
    // Wait for the last buffer
    out[j] = (cl_int*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_TRUE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
    if( error || NULL == out[j] )
    {
        vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
        return error;
    }

    //Verify data
    cl_int *t = (cl_int *)r;
    for( j = 0; j < buffer_elements; j++ )
    {
        for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
        {
            cl_int *q = out[0];

            // If we aren't getting the correctly rounded result
            if( gMinVectorSizeIndex == 0 && t[j] != q[j])
            {
                // If we aren't getting the correctly rounded result
                if( ftz )
                {
                    if( IsFloatSubnormal( s[j]) )
                    {
                        int correct = ref_func( +0.0f );
                        int correct2 = ref_func( -0.0f );
                        if( correct == q[j] || correct2 == q[j] )
                            continue;
                    }
                }

                uint32_t err = t[j] - q[j];
                if( q[j] > t[j] )
                    err = q[j] - t[j];
                vlog_error( "\nERROR: %s: %d ulp error at %a: *%d vs. %d\n", name,  err, ((float*) s)[j], t[j], q[j] );
                error = -1;
                goto exit;
            }


            for( k = MAX(1, gMinVectorSizeIndex); k < gMaxVectorSizeIndex; k++ )
            {
                q = out[k];
                // If we aren't getting the correctly rounded result
                if( -t[j] != q[j] )
                {
                    if( ftz )
                    {
                        if( IsFloatSubnormal( s[j]))
                        {
                            int correct = -ref_func( +0.0f );
                            int correct2 = -ref_func( -0.0f );
                            if( correct == q[j] || correct2 == q[j] )
                                continue;
                        }
                    }

                    uint32_t err = -t[j] - q[j];
                    if( q[j] > -t[j] )
                        err = q[j] + t[j];
                    vlog_error( "\nERROR: %s%s: %d ulp error at %a: *%d vs. %d\n", name, sizeNames[k], err, ((float*) s)[j], -t[j], q[j] );
                  error = -1;
                  goto exit;
                }
            }
        }
    }

exit:
    ret = error;
    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL)) )
        {
            vlog_error( "Error: clEnqueueUnmapMemObject %d failed 2! err: %d\n", j, error );
            return error;
        }
    }

    if( (error = clFlush(tinfo->tQueue) ))
    {
        vlog( "clFlush 3 failed\n" );
        return error;
    }


    if( 0 == ( base & 0x0fffffff) )
    {
       if (gVerboseBruteForce)
       {
           vlog("base:%14u step:%10u scale:%10u buf_elements:%10zd ThreadCount:%2u\n", base, job->step, job->scale, buffer_elements, job->threadCount);
       } else
       {
          vlog("." );
       }
       fflush(stdout);
    }

    return ret;
}

static cl_int TestDouble( cl_uint job_id, cl_uint thread_id, void *data );

int TestMacro_Int_Double(const Func *f, MTdata d)
{
    TestInfo    test_info;
    cl_int      error;
    size_t      i, j;

    logFunctionInfo(f->name,sizeof(cl_double),gTestFastRelaxed);
    // Init test_info
    memset( &test_info, 0, sizeof( test_info ) );
    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE / (sizeof( cl_double) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.scale =  1;
    if (gWimpyMode )
    {
        test_info.subBufferSize = gWimpyBufferSize / (sizeof( cl_double) * RoundUpToNextPowerOfTwo(test_info.threadCount));
        test_info.scale =  (cl_uint) sizeof(cl_double) * 2 * gWimpyReductionFactor;
    }

    test_info.step = (cl_uint) test_info.subBufferSize * test_info.scale;
    if (test_info.step / test_info.subBufferSize != test_info.scale)
    {
        //there was overflow
        test_info.jobCount = 1;
    }
    else
    {
        test_info.jobCount = (cl_uint)((1ULL << 32) / test_info.step);
    }

    test_info.f = f;
    test_info.ftz = f->ftz || gForceFTZ;

    // cl_kernels aren't thread safe, so we make one for each vector size for every thread
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        size_t array_size = test_info.threadCount * sizeof( cl_kernel );
        test_info.k[i] = (cl_kernel*)malloc( array_size );
        if( NULL == test_info.k[i] )
        {
            vlog_error( "Error: Unable to allocate storage for kernels!\n" );
            error = CL_OUT_OF_HOST_MEMORY;
            goto exit;
        }
        memset( test_info.k[i], 0, array_size );
    }
    test_info.tinfo = (ThreadInfo*)malloc( test_info.threadCount * sizeof(*test_info.tinfo) );
    if( NULL == test_info.tinfo )
    {
        vlog_error( "Error: Unable to allocate storage for thread specific data.\n" );
        error = CL_OUT_OF_HOST_MEMORY;
        goto exit;
    }
    memset( test_info.tinfo, 0, test_info.threadCount * sizeof(*test_info.tinfo) );
    for( i = 0; i < test_info.threadCount; i++ )
    {
        cl_buffer_region region = { i * test_info.subBufferSize * sizeof( cl_double), test_info.subBufferSize * sizeof( cl_double) };
        test_info.tinfo[i].inBuf = clCreateSubBuffer( gInBuffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if( error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error( "Error: Unable to create sub-buffer of gInBuffer for region {%zd, %zd}\n", region.origin, region.size );
            goto exit;
        }

        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            /* Qualcomm fix: 9461 read-write flags must be compatible with parent buffer */
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer( gOutBuffer[j], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
            /* Qualcomm fix: end */
            if( error || NULL == test_info.tinfo[i].outBuf[j] )
            {
                vlog_error( "Error: Unable to create sub-buffer of gInBuffer for region {%zd, %zd}\n", region.origin, region.size );
                goto exit;
            }
        }
        test_info.tinfo[i].tQueue = clCreateCommandQueue(gContext, gDevice, 0, &error);
        if( NULL == test_info.tinfo[i].tQueue || error )
        {
            vlog_error( "clCreateCommandQueue failed. (%d)\n", error );
            goto exit;
        }
    }

    // Init the kernels
    {
        BuildKernelInfo build_info = { gMinVectorSizeIndex, test_info.threadCount, test_info.k, test_info.programs, f->nameInCode };
        if( (error = ThreadPool_Do( BuildKernel_DoubleFn, gMaxVectorSizeIndex - gMinVectorSizeIndex, &build_info ) ))
            goto exit;
    }

    if( !gSkipCorrectnessTesting )
    {
        error = ThreadPool_Do( TestDouble, test_info.jobCount, &test_info );

        if( error )
            goto exit;

        if( gWimpyMode )
            vlog( "Wimp pass" );
        else
            vlog( "passed" );
    }

    if( gMeasureTimes )
    {
        //Init input array
        cl_ulong *p = (cl_ulong *)gIn;
        for( j = 0; j < BUFFER_SIZE / sizeof( cl_double ); j++ )
            p[j] = DoubleFromUInt32(genrand_int32(d));
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, BUFFER_SIZE, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg( test_info.k[j][0], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(test_info.programs[j]); goto exit; }

            double sum = 0.0;
            double bestTime = INFINITY;
            for( i = 0; i < PERF_LOOP_COUNT; i++ )
            {
                uint64_t startTime = GetTime();
                if( (error = clEnqueueNDRangeKernel(gQueue, test_info.k[j][0], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
                {
                    vlog_error( "FAILED -- could not execute kernel\n" );
                    goto exit;
                }

                // Make sure OpenCL is done
                if( (error = clFinish(gQueue) ) )
                {
                    vlog_error( "Error %d at clFinish\n", error );
                    goto exit;
                }

                uint64_t endTime = GetTime();
                double time = SubtractTime( endTime, startTime );
                sum += time;
                if( time < bestTime )
                    bestTime = time;
            }

            if( gReportAverageTimes )
                bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double) gDeviceFrequency * gComputeDevices * gSimdSize * 1e6 / (BUFFER_SIZE / sizeof( double ) );
            vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sD%s", f->name, sizeNames[j] );
        }
        for( ; j < gMaxVectorSizeIndex; j++ )
            vlog( "\t     -- " );
    }

    vlog( "\n" );

exit:
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        clReleaseProgram(test_info.programs[i]);
        if( test_info.k[i] )
        {
            for( j = 0; j < test_info.threadCount; j++ )
                clReleaseKernel(test_info.k[i][j]);

            free( test_info.k[i] );
        }
    }
    if( test_info.tinfo )
    {
        for( i = 0; i < test_info.threadCount; i++ )
        {
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
                clReleaseMemObject(test_info.tinfo[i].outBuf[j]);
            clReleaseCommandQueue(test_info.tinfo[i].tQueue);
        }

        free( test_info.tinfo );
    }

    return error;
}

static cl_int TestDouble( cl_uint job_id, cl_uint thread_id, void *data )
{
    const TestInfo *job = (const TestInfo *) data;
    size_t  buffer_elements = job->subBufferSize;
    size_t  buffer_size = buffer_elements * sizeof( cl_double );
    cl_uint scale = job->scale;
    cl_uint base = job_id * (cl_uint) job->step;
    ThreadInfo *tinfo = job->tinfo + thread_id;
    dptr    dfunc = job->f->dfunc;
    cl_uint j, k;
    cl_int error;
    int ftz = job->ftz;
    const char *name = job->f->name;

    Force64BitFPUPrecision();

    // start the map of the output arrays
    cl_event e[ VECTOR_SIZE_COUNT ];
    cl_long *out[ VECTOR_SIZE_COUNT ];
    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_long*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_WRITE, 0, buffer_size, 0, NULL, e + j, &error);
        if( error || NULL == out[j])
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            return error;
        }
    }

    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush failed\n" );

    // Write the new values to the input array
    cl_double *p = (cl_double*) gIn + thread_id * buffer_elements;
    for( j = 0; j < buffer_elements; j++ )
        p[j] = DoubleFromUInt32( base + j * scale);

    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0, buffer_size, p, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        return error;
    }

    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        //Wait for the map to finish
        if( (error = clWaitForEvents(1, e + j) ))
        {
            vlog_error( "Error: clWaitForEvents failed! err: %d\n", error );
            return error;
        }
        if( (error = clReleaseEvent( e[j] ) ))
        {
            vlog_error( "Error: clReleaseEvent failed! err: %d\n", error );
            return error;
        }

        // Fill the result buffer with garbage, so that old results don't carry over
        uint32_t pattern = 0xffffdead;
        memset_pattern4(out[j], &pattern, buffer_size);
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL) ))
        {
            vlog_error( "Error: clEnqueueMapBuffer failed! err: %d\n", error );
            return error;
        }

        // run the kernel
        size_t vectorCount = (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id];  //each worker thread has its own copy of the cl_kernel
        cl_program program = job->programs[j];

        if( ( error = clSetKernelArg( kernel, 0, sizeof( tinfo->outBuf[j] ), &tinfo->outBuf[j] ))){ LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 1, sizeof( tinfo->inBuf ), &tinfo->inBuf ) )) { LogBuildError(program); return error; }

        if( (error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL, &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error( "FAILED -- could not execute kernel\n" );
            return error;
        }
    }


    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush 2 failed\n" );

    if( gSkipCorrectnessTesting )
        return CL_SUCCESS;

    //Calculate the correctly rounded reference result
    cl_long *r = (cl_long *)gOut_Ref + thread_id * buffer_elements;
    cl_double *s = (cl_double *)p;
    for( j = 0; j < buffer_elements; j++ )
        r[j] = dfunc.i_f( s[j] );

    // Read the data back -- no need to wait for the first N-1 buffers. This is an in order queue.
    for( j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_long*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
        if( error || NULL == out[j] )
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            return error;
        }
    }
    // Wait for the last buffer
    out[j] = (cl_long*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_TRUE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
    if( error || NULL == out[j] )
    {
        vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
        return error;
    }


    //Verify data
    cl_long *t = (cl_long *)r;
    for( j = 0; j < buffer_elements; j++ )
    {
        cl_long *q = out[0];


        // If we aren't getting the correctly rounded result
        if( gMinVectorSizeIndex == 0 && t[j] != q[j])
        {
            // If we aren't getting the correctly rounded result
            if( ftz )
            {
                if( IsDoubleSubnormal( s[j]) )
                {
                    cl_long correct = dfunc.i_f( +0.0f );
                    cl_long correct2 = dfunc.i_f( -0.0f );
                    if( correct == q[j] || correct2 == q[j] )
                        continue;
                }
            }

            cl_ulong err = t[j] - q[j];
            if( q[j] > t[j] )
                err = q[j] - t[j];
            vlog_error( "\nERROR: %sD: %zd ulp error at %.13la: *%zd vs. %zd\n", name,  err, ((double*) gIn)[j], t[j], q[j] );
            return -1;
        }


        for( k = MAX(1, gMinVectorSizeIndex); k < gMaxVectorSizeIndex; k++ )
        {
            q = out[k];
            // If we aren't getting the correctly rounded result
            if( -t[j] != q[j] )
            {
                if( ftz )
                {
                    if( IsDoubleSubnormal( s[j]))
                    {
                        int64_t correct = -dfunc.i_f( +0.0f );
                        int64_t correct2 = -dfunc.i_f( -0.0f );
                        if( correct == q[j] || correct2 == q[j] )
                            continue;
                    }
                }

                cl_ulong err = -t[j] - q[j];
                if( q[j] > -t[j] )
                    err = q[j] + t[j];
                vlog_error( "\nERROR: %sD%s: %zd ulp error at %.13la: *%zd vs. %zd\n", name, sizeNames[k], err, ((double*) gIn)[j], -t[j], q[j] );
                return -1;
            }
        }

    }

    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL)) )
        {
            vlog_error( "Error: clEnqueueUnmapMemObject %d failed 2! err: %d\n", j, error );
            return error;
        }
    }

    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush 3 failed\n" );


    if( 0 == ( base & 0x0fffffff) )
    {
       if (gVerboseBruteForce)
       {
           vlog("base:%14u step:%10u scale:%10u buf_elements:%10zd ThreadCount:%2u\n", base, job->step, job->scale, buffer_elements, job->threadCount);
       } else
       {
          vlog("." );
       }
       fflush(stdout);
    }

    return CL_SUCCESS;
}





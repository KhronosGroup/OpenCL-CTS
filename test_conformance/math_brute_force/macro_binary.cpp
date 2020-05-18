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

int TestMacro_Int_Float_Float(const Func *f, MTdata);
int TestMacro_Int_Double_Double(const Func *f, MTdata);

extern const vtbl _macro_binary = { "macro_binary", TestMacro_Int_Float_Float,
                                    TestMacro_Int_Double_Double };

static int BuildKernel( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p );

static int BuildKernel( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "__kernel void math_kernel", sizeNames[vectorSize], "( __global int", sizeNames[vectorSize], "* out, __global float", sizeNames[vectorSize], "* in1, __global float", sizeNames[vectorSize], "* in2 )\n"
        "{\n"
        "   int i = get_global_id(0);\n"
        "   out[i] = ", name, "( in1[i], in2[i] );\n"
        "}\n"
    };

    const char *c3[] = {    "__kernel void math_kernel", sizeNames[vectorSize], "( __global int* out, __global float* in, __global float* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       int3 i0 = ", name, "( f0, f1 );\n"
        "       vstore3( i0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
        "       float3 f0, f1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
        "               f1 = (float3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
        "               f1 = (float3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       int3 i0 = ", name, "( f0, f1 );\n"
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

    return MakeKernels(kern, (cl_uint) kernSize, testName, kernel_count, k, p);   }


static int BuildKernelDouble( const char *name, int vectorSize, cl_uint kernel_count, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "__kernel void math_kernel", sizeNames[vectorSize], "( __global long", sizeNames[vectorSize], "* out, __global double", sizeNames[vectorSize], "* in1, __global double", sizeNames[vectorSize], "* in2 )\n"
        "{\n"
        "   int i = get_global_id(0);\n"
        "   out[i] = ", name, "( in1[i], in2[i] );\n"
        "}\n"
    };

    const char *c3[] = {    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "__kernel void math_kernel", sizeNames[vectorSize], "( __global long* out, __global double* in, __global double* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       double3 f0 = vload3( 0, in + 3 * i );\n"
        "       double3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       long3 l0 = ", name, "( f0, f1 );\n"
        "       vstore3( l0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
        "       double3 f0, f1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (double3)( in[3*i], NAN, NAN ); \n"
        "               f1 = (double3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (double3)( in[3*i], in[3*i+1], NAN ); \n"
        "               f1 = (double3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       long3 l0 = ", name, "( f0, f1 );\n"
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


// A table of more difficult cases to get right
static const float specialValuesFloat[] = {
    -NAN, -INFINITY, -FLT_MAX, MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64), MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),  MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8), MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32), MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7), MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7), MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31), MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6), -1000.f, -100.f,  -4.0f, -3.5f,
    -3.0f, MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23), -2.5f, MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23), -2.0f, MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24), -1.5f, MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24),MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24), -1.0f, MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25), -0.5f, MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26),  MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26), -0.25f, MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150), -FLT_MIN, MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150), MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150), MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150), MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150), MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150), MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150), MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150), MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150), MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150), -0.0f,

    +NAN, +INFINITY, +FLT_MAX, MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40), MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64), MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39), MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39), MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63), MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8), MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32), MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7), MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7), MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31), MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6), +1000.f, +100.f, +4.0f, +3.5f,
    +3.0f, MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23), 2.5f, MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),+2.0f, MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24), 1.5f, MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24), MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24), +1.0f, MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25), +0.5f, MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26), MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26), +0.25f, MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150), +FLT_MIN, MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150), MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150), MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150), MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150), MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150), MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150), MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150), MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150), MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150), +0.0f
};

static const size_t specialValuesFloatCount = sizeof(specialValuesFloat) / sizeof(specialValuesFloat[0]);

//Thread specific data for a worker thread
typedef struct ThreadInfo
{
    cl_mem      inBuf;                              // input buffer for the thread
    cl_mem      inBuf2;                             // input buffer for the thread
    cl_mem      outBuf[ VECTOR_SIZE_COUNT ];        // output buffers for the thread
    MTdata      d;
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

int TestMacro_Int_Float_Float(const Func *f, MTdata d)
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
    if (gWimpyMode)
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
        test_info.tinfo[i].inBuf2 = clCreateSubBuffer( gInBuffer2, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
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

        test_info.tinfo[i].d = init_genrand(genrand_int32(d));
    }

    // Init the kernels
    {
        BuildKernelInfo build_info = { gMinVectorSizeIndex, test_info.threadCount, test_info.k, test_info.programs, f->nameInCode };
        if( (error = ThreadPool_Do( BuildKernel_FloatFn, gMaxVectorSizeIndex - gMinVectorSizeIndex, &build_info ) ))
            goto exit;
    }


    // Run the kernels
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
        //Init input arrays
        uint32_t *p = (uint32_t *)gIn;
        uint32_t *p2 = (uint32_t *)gIn2;
        for( j = 0; j < BUFFER_SIZE / sizeof( float ); j++ )
        {
            p[j] = genrand_int32(d);
            p2[j] = genrand_int32(d);
        }

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, BUFFER_SIZE, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, BUFFER_SIZE, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg( test_info.k[j][0], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(test_info.programs[j]); goto exit; }

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
            free_mtdata(test_info.tinfo[i].d);
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            clReleaseMemObject(test_info.tinfo[i].inBuf2);
            for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
                clReleaseMemObject(test_info.tinfo[i].outBuf[j]);
            clReleaseCommandQueue(test_info.tinfo[i].tQueue);
        }

        free( test_info.tinfo );
    }

    return error;
}

static cl_int TestFloat( cl_uint job_id, cl_uint thread_id, void *data  )
{
    const TestInfo *job = (const TestInfo *) data;
    size_t      buffer_elements = job->subBufferSize;
    size_t      buffer_size = buffer_elements * sizeof( cl_float );
    cl_uint     base = job_id * (cl_uint) job->step;
    ThreadInfo  *tinfo = job->tinfo + thread_id;
    fptr        func = job->f->func;
    int         ftz = job->ftz;
    MTdata      d = tinfo->d;
    cl_uint     j, k;
    cl_int      error;
    const char  *name = job->f->name;
    cl_int      *t,*r;
    cl_float    *s,*s2;

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

    //Init input array
    cl_uint *p = (cl_uint *)gIn + thread_id * buffer_elements;
    cl_uint *p2 = (cl_uint *)gIn2 + thread_id * buffer_elements;
    j = 0;
    int totalSpecialValueCount = specialValuesFloatCount * specialValuesFloatCount;
    int indx = (totalSpecialValueCount - 1) / buffer_elements;

    if( job_id <= (cl_uint)indx )
    { // test edge cases
        float *fp = (float *)p;
        float *fp2 = (float *)p2;
        uint32_t x, y;

    x = (job_id * buffer_elements) % specialValuesFloatCount;
    y = (job_id * buffer_elements) / specialValuesFloatCount;

        for( ; j < buffer_elements; j++ )
        {
            fp[j] = specialValuesFloat[x];
            fp2[j] = specialValuesFloat[y];
            if( ++x >= specialValuesFloatCount )
            {
                x = 0;
                y++;
                if( y >= specialValuesFloatCount )
                    break;
            }
        }
    }

    //Init any remaining values.
    for( ; j < buffer_elements; j++ )
    {
        p[j] = genrand_int32(d);
        p2[j] = genrand_int32(d);
    }


    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0, buffer_size, p, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        goto exit;
    }

    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf2, CL_FALSE, 0, buffer_size, p2, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        goto exit;
    }

    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        //Wait for the map to finish
        if( (error = clWaitForEvents(1, e + j) ))
        {
            vlog_error( "Error: clWaitForEvents failed! err: %d\n", error );
            goto exit;
        }
        if( (error = clReleaseEvent( e[j] ) ))
        {
            vlog_error( "Error: clReleaseEvent failed! err: %d\n", error );
            goto exit;
        }

        // Fill the result buffer with garbage, so that old results don't carry over
        uint32_t pattern = 0xffffdead;
        memset_pattern4(out[j], &pattern, buffer_size);
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL) ))
        {
            vlog_error( "Error: clEnqueueMapBuffer failed! err: %d\n", error );
            goto exit;
        }

        // run the kernel
        size_t vectorCount = (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id];  //each worker thread has its own copy of the cl_kernel
        cl_program program = job->programs[j];

        if( ( error = clSetKernelArg( kernel, 0, sizeof( tinfo->outBuf[j] ), &tinfo->outBuf[j] ))){ LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 1, sizeof( tinfo->inBuf ), &tinfo->inBuf ) )) { LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 2, sizeof( tinfo->inBuf2 ), &tinfo->inBuf2 ) )) { LogBuildError(program); return error; }

        if( (error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL, &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error( "FAILED -- could not execute kernel\n" );
            goto exit;
        }
    }

    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush 2 failed\n" );

    if( gSkipCorrectnessTesting )
        return CL_SUCCESS;

    //Calculate the correctly rounded reference result
    r = (cl_int *)gOut_Ref  + thread_id * buffer_elements;
    s = (float *)gIn  + thread_id * buffer_elements;
    s2 = (float *)gIn2  + thread_id * buffer_elements;
    for( j = 0; j < buffer_elements; j++ )
        r[j] = func.i_ff( s[j], s2[j] );


    // Read the data back -- no need to wait for the first N-1 buffers. This is an in order queue.
    for( j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_int*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
        if( error || NULL == out[j] )
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            goto exit;
        }
    }

    // Wait for the last buffer
    out[j] = (cl_int*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_TRUE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
    if( error || NULL == out[j] )
    {
        vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
        goto exit;
    }

    //Verify data
    t = (cl_int *)r;
    for( j = 0; j < buffer_elements; j++ )
    {
        cl_int *q = out[0];

        if( gMinVectorSizeIndex == 0 && t[j] != q[j] )
        {
            if( ftz )
            {
                if( IsFloatSubnormal( s[j])  )
                {
                    if( IsFloatSubnormal( s2[j] )  )
                    {
                        int correct = func.i_ff( 0.0f, 0.0f );
                        int correct2 = func.i_ff( 0.0f, -0.0f );
                        int correct3 = func.i_ff( -0.0f, 0.0f );
                        int correct4 = func.i_ff( -0.0f, -0.0f );

                        if( correct == q[j] || correct2 == q[j] || correct3 == q[j] || correct4 == q[j] )
                            continue;
                    }
                    else
                    {
                        int correct = func.i_ff( 0.0f, s2[j] );
                        int correct2 = func.i_ff( -0.0f, s2[j] );
                        if( correct == q[j] || correct2 == q[j]  )
                            continue;
                    }
                }
                else if( IsFloatSubnormal( s2[j] ) )
                {
                    int correct = func.i_ff( s[j], 0.0f );
                    int correct2 = func.i_ff( s[j], -0.0f );
                    if( correct == q[j] || correct2 == q[j]  )
                        continue;
                }

            }

            uint32_t err = t[j] - q[j];
            if( q[j] > t[j] )
                err = q[j] - t[j];
            vlog_error( "\nERROR: %s: %d ulp error at {%a, %a}: *0x%8.8x vs. 0x%8.8x (index: %d)\n", name, err, ((float*) s)[j], ((float*) s2)[j], t[j], q[j], j );
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
                    if( IsFloatSubnormal( s[j])  )
                    {
                        if( IsFloatSubnormal( s2[j] )  )
                        {
                            int correct = -func.i_ff( 0.0f, 0.0f );
                            int correct2 = -func.i_ff( 0.0f, -0.0f );
                            int correct3 = -func.i_ff( -0.0f, 0.0f );
                            int correct4 = -func.i_ff( -0.0f, -0.0f );

                            if( correct == q[j] || correct2 == q[j] || correct3 == q[j] || correct4 == q[j] )
                                continue;
                        }
                        else
                        {
                            int correct = -func.i_ff( 0.0f, s2[j] );
                            int correct2 = -func.i_ff( -0.0f, s2[j] );
                            if( correct == q[j] || correct2 == q[j]  )
                                continue;
                        }
                    }
                    else if( IsFloatSubnormal( s2[j] ) )
                    {
                        int correct = -func.i_ff( s[j], 0.0f );
                        int correct2 = -func.i_ff( s[j], -0.0f );
                        if( correct == q[j] || correct2 == q[j]  )
                            continue;
                    }

                }
                cl_uint err = -t[j] - q[j];
                if( q[j] > -t[j] )
                    err = q[j] + t[j];
                vlog_error( "\nERROR: %s%s: %d ulp error at {%a, %a}: *0x%8.8x vs. 0x%8.8x (index: %d)\n", name, sizeNames[k], err, ((float*) s)[j], ((float*) s2)[j], -t[j], q[j], j );
                error = -1;
                goto exit;
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

exit:
    return error;
}


// A table of more difficult cases to get right
static const double specialValuesDouble[] = {
    -NAN, -INFINITY, -DBL_MAX, MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12), MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11),  MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11), MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8), MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21), MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21), MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22), -1000., -100.,  -4.0, -3.5,
    -3.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51), -2.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51), -2.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52), -1.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), -1.0, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53), -0.5, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54),  MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54), -0.25, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074), -DBL_MIN, MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074), -0.0,

    +NAN, +INFINITY, +DBL_MAX, MAKE_HEX_DOUBLE(+0x1.0000000000001p64, +0x10000000000001LL, 12), MAKE_HEX_DOUBLE(+0x1.0p64, +0x1LL, 64), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),  MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11), MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8), MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21), MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21), MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22), +1000., +100.,  +4.0, +3.5,
    +3.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51), +2.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51), +2.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52), +1.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), +1.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53), +0.5, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54),  MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54), +0.25, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074), +DBL_MIN, MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074), +0.0,
};

static size_t specialValuesDoubleCount = sizeof( specialValuesDouble ) / sizeof( specialValuesDouble[0] );


static cl_int TestDouble( cl_uint job_id, cl_uint thread_id, void *p );

int TestMacro_Int_Double_Double(const Func *f, MTdata d)
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
    if (gWimpyMode)
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
        test_info.tinfo[i].inBuf2 = clCreateSubBuffer( gInBuffer2, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
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

        test_info.tinfo[i].d = init_genrand(genrand_int32(d));
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
        //Init input arrays
        uint64_t *p = (uint64_t *)gIn;
        uint64_t *p2 = (uint64_t *)gIn2;
        for( j = 0; j < BUFFER_SIZE / sizeof( double ); j++ )
        {
            p[j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
            p2[j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
        }

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, BUFFER_SIZE, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, BUFFER_SIZE, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg( test_info.k[j][0], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(test_info.programs[j]); goto exit; }
            if( ( error = clSetKernelArg( test_info.k[j][0], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(test_info.programs[j]); goto exit; }

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
    // Release
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
            free_mtdata(test_info.tinfo[i].d);
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            clReleaseMemObject(test_info.tinfo[i].inBuf2);
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
    size_t      buffer_elements = job->subBufferSize;
    size_t      buffer_size = buffer_elements * sizeof( cl_double );
    cl_uint     base = job_id * (cl_uint) job->step;
    ThreadInfo  *tinfo = job->tinfo + thread_id;
    dptr        dfunc = job->f->dfunc;
    int         ftz = job->ftz;
    MTdata      d = tinfo->d;
    cl_uint     j, k;
    cl_int      error;
    const char  *name = job->f->name;
    cl_long     *t,*r;
    cl_double   *s,*s2;

    Force64BitFPUPrecision();

    // start the map of the output arrays
    cl_event e[ VECTOR_SIZE_COUNT ];
    cl_long  *out[ VECTOR_SIZE_COUNT ];
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

    //Init input array
    double *p = (double *)gIn + thread_id * buffer_elements;
    double *p2 = (double *)gIn2 + thread_id * buffer_elements;
    j = 0;
    int totalSpecialValueCount = specialValuesDoubleCount * specialValuesDoubleCount;
    int indx = (totalSpecialValueCount - 1) / buffer_elements;

    if( job_id <= (cl_uint)indx )
    { // test edge cases
        uint32_t x, y;

    x = (job_id * buffer_elements) % specialValuesDoubleCount;
    y = (job_id * buffer_elements) / specialValuesDoubleCount;

        for( ; j < buffer_elements; j++ )
        {
            p[j] = specialValuesDouble[x];
            p2[j] = specialValuesDouble[y];
            if( ++x >= specialValuesDoubleCount )
            {
                x = 0;
                y++;
                if( y >= specialValuesDoubleCount )
                    break;
            }
        }
    }

    //Init any remaining values.
    for( ; j < buffer_elements; j++ )
    {
        ((cl_ulong*)p)[j] = genrand_int64(d);
        ((cl_ulong*)p2)[j] = genrand_int64(d);
    }


    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0, buffer_size, p, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        goto exit;
    }

    if( (error = clEnqueueWriteBuffer( tinfo->tQueue, tinfo->inBuf2, CL_FALSE, 0, buffer_size, p2, 0, NULL, NULL) ))
    {
        vlog_error( "Error: clEnqueueWriteBuffer failed! err: %d\n", error );
        goto exit;
    }

    for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
    {
        //Wait for the map to finish
        if( (error = clWaitForEvents(1, e + j) ))
        {
            vlog_error( "Error: clWaitForEvents failed! err: %d\n", error );
            goto exit;
        }
        if( (error = clReleaseEvent( e[j] ) ))
        {
            vlog_error( "Error: clReleaseEvent failed! err: %d\n", error );
            goto exit;
        }

        // Fill the result buffer with garbage, so that old results don't carry over
        uint32_t pattern = 0xffffdead;
        memset_pattern4(out[j], &pattern, buffer_size);
        if( (error = clEnqueueUnmapMemObject( tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL) ))
        {
            vlog_error( "Error: clEnqueueMapBuffer failed! err: %d\n", error );
            goto exit;
        }

        // run the kernel
        size_t vectorCount = (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id];  //each worker thread has its own copy of the cl_kernel
        cl_program program = job->programs[j];

        if( ( error = clSetKernelArg( kernel, 0, sizeof( tinfo->outBuf[j] ), &tinfo->outBuf[j] ))){ LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 1, sizeof( tinfo->inBuf ), &tinfo->inBuf ) )) { LogBuildError(program); return error; }
        if( ( error = clSetKernelArg( kernel, 2, sizeof( tinfo->inBuf2 ), &tinfo->inBuf2 ) )) { LogBuildError(program); return error; }

        if( (error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL, &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error( "FAILED -- could not execute kernel\n" );
            goto exit;
        }
    }

    // Get that moving
    if( (error = clFlush(tinfo->tQueue) ))
        vlog( "clFlush 2 failed\n" );

    if( gSkipCorrectnessTesting )
        return CL_SUCCESS;

    //Calculate the correctly rounded reference result
    r = (cl_long *)gOut_Ref  + thread_id * buffer_elements;
    s = (cl_double *)gIn  + thread_id * buffer_elements;
    s2 = (cl_double *)gIn2  + thread_id * buffer_elements;
    for( j = 0; j < buffer_elements; j++ )
        r[j] = dfunc.i_ff( s[j], s2[j] );


    // Read the data back -- no need to wait for the first N-1 buffers. This is an in order queue.
    for( j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++ )
    {
        out[j] = (cl_long*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
        if( error || NULL == out[j] )
        {
            vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
            goto exit;
        }
    }

    // Wait for the last buffer
    out[j] = (cl_long*) clEnqueueMapBuffer( tinfo->tQueue, tinfo->outBuf[j], CL_TRUE, CL_MAP_READ, 0, buffer_size, 0, NULL, NULL, &error);
    if( error || NULL == out[j] )
    {
        vlog_error( "Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error );
        goto exit;
    }

    //Verify data
    t = (cl_long *)r;
    for( j = 0; j < buffer_elements; j++ )
    {
        cl_long *q = (cl_long *) out[0];

        // If we aren't getting the correctly rounded result
        if( gMinVectorSizeIndex == 0 && t[j] != q[j] )
        {
            if( ftz )
            {
                if( IsDoubleSubnormal( s[j])  )
                {
                    if( IsDoubleSubnormal( s2[j] )  )
                    {
                        int64_t correct = dfunc.i_ff( 0.0f, 0.0f );
                        int64_t correct2 = dfunc.i_ff( 0.0f, -0.0f );
                        int64_t correct3 = dfunc.i_ff( -0.0f, 0.0f );
                        int64_t correct4 = dfunc.i_ff( -0.0f, -0.0f );

                        if( correct == q[j] || correct2 == q[j] || correct3 == q[j] || correct4 == q[j] )
                            continue;
                    }
                    else
                    {
                        int64_t correct = dfunc.i_ff( 0.0f, s2[j] );
                        int64_t correct2 = dfunc.i_ff( -0.0f, s2[j] );
                        if( correct == q[j] || correct2 == q[j]  )
                            continue;
                    }
                }
                else if( IsDoubleSubnormal( s2[j] ) )
                {
                    int64_t correct = dfunc.i_ff( s[j], 0.0f );
                    int64_t correct2 = dfunc.i_ff( s[j], -0.0f );
                    if( correct == q[j] || correct2 == q[j]  )
                        continue;
                }

            }

            uint64_t err = t[j] - q[j];
            if( q[j] > t[j] )
                err = q[j] - t[j];
            vlog_error( "\nERROR: %s: %lld ulp error at {%.13la, %.13la}: *%lld vs. %lld  (index: %d)\n", name, err, ((double*) s)[j], ((double*) s2)[j], t[j], q[j], j );
            error = -1;
            goto exit;
        }


        for( k = MAX(1, gMinVectorSizeIndex); k < gMaxVectorSizeIndex; k++ )
        {
            q = (cl_long*) out[k];
            // If we aren't getting the correctly rounded result
            if( -t[j] != q[j] )
            {
                if( ftz )
                {
                    if( IsDoubleSubnormal( s[j])  )
                    {
                        if( IsDoubleSubnormal( s2[j] )  )
                        {
                            int64_t correct = -dfunc.i_ff( 0.0f, 0.0f );
                            int64_t correct2 = -dfunc.i_ff( 0.0f, -0.0f );
                            int64_t correct3 = -dfunc.i_ff( -0.0f, 0.0f );
                            int64_t correct4 = -dfunc.i_ff( -0.0f, -0.0f );

                            if( correct == q[j] || correct2 == q[j] || correct3 == q[j] || correct4 == q[j] )
                                continue;
                        }
                        else
                        {
                            int64_t correct = -dfunc.i_ff( 0.0f, s2[j] );
                            int64_t correct2 = -dfunc.i_ff( -0.0f, s2[j] );
                            if( correct == q[j] || correct2 == q[j]  )
                                continue;
                        }
                    }
                    else if( IsDoubleSubnormal( s2[j] ) )
                    {
                        int64_t correct = -dfunc.i_ff( s[j], 0.0f );
                        int64_t correct2 = -dfunc.i_ff( s[j], -0.0f );
                        if( correct == q[j] || correct2 == q[j]  )
                            continue;
                    }

                }

                uint64_t err = -t[j] - q[j];
                if( q[j] > -t[j] )
                    err = q[j] + t[j];
                vlog_error( "\nERROR: %sD%s: %lld ulp error at {%.13la, %.13la}: *%lld vs. %lld  (index: %d)\n", name, sizeNames[k], err, ((double*) s)[j], ((double*) s2)[j], -t[j], q[j], j );
                error = -1;
                goto exit;
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

exit:
    return error;
}


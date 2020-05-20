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

#include <limits.h>
#include <string.h>
#include "FunctionList.h"

int TestFunc_FloatI_Float(const Func *f, MTdata);
int TestFunc_DoubleI_Double(const Func *f, MTdata);

extern const vtbl _unary_two_results_i = { "unary_two_results_i",
                                           TestFunc_FloatI_Float,
                                           TestFunc_DoubleI_Double };

static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p );

static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "__kernel void math_kernel", sizeNames[vectorSize], "( __global float", sizeNames[vectorSize], "* out, __global int", sizeNames[vectorSize], "* out2, __global float", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i], out2 + i );\n"
                            "}\n"
                        };
    const char *c3[] = {    "__kernel void math_kernel", sizeNames[vectorSize], "( __global float* out, __global int* out2, __global float* in)\n"
                            "{\n"
                            "   size_t i = get_global_id(0);\n"
                            "   if( i + 1 < get_global_size(0) )\n"
                            "   {\n"
                            "       float3 f0 = vload3( 0, in + 3 * i );\n"
                            "       int3 iout = INT_MIN;\n"
                            "       f0 = ", name, "( f0, &iout );\n"
                            "       vstore3( f0, 0, out + 3*i );\n"
                            "       vstore3( iout, 0, out2 + 3*i );\n"
                            "   }\n"
                            "   else\n"
                            "   {\n"
                            "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                            "       int3 iout = INT_MIN;\n"
                            "       float3 f0;\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 1:\n"
                            "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
                            "               break;\n"
                            "           case 0:\n"
                            "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
                            "               break;\n"
                            "       }\n"
                            "       f0 = ", name, "( f0, &iout );\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 0:\n"
                            "               out[3*i+1] = f0.y; \n"
                            "               out2[3*i+1] = iout.y; \n"
                            "               // fall through\n"
                            "           case 1:\n"
                            "               out[3*i] = f0.x; \n"
                            "               out2[3*i] = iout.x; \n"
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

    return MakeKernel(kern, (cl_uint) kernSize, testName, k, p);

}

static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global double", sizeNames[vectorSize], "* out, __global int", sizeNames[vectorSize], "* out2, __global double", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i], out2 + i );\n"
                            "}\n"
                        };
    const char *c3[] = {    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global double* out, __global int* out2, __global double* in)\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   if( i + 1 < get_global_size(0) )\n"
                        "   {\n"
                        "       double3 f0 = vload3( 0, in + 3 * i );\n"
                        "       int3 iout = INT_MIN;\n"
                        "       f0 = ", name, "( f0, &iout );\n"
                        "       vstore3( f0, 0, out + 3*i );\n"
                        "       vstore3( iout, 0, out2 + 3*i );\n"
                        "   }\n"
                        "   else\n"
                        "   {\n"
                        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                        "       int3 iout = INT_MIN;\n"
                        "       double3 f0;\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 1:\n"
                        "               f0 = (double3)( in[3*i], NAN, NAN ); \n"
                        "               break;\n"
                        "           case 0:\n"
                        "               f0 = (double3)( in[3*i], in[3*i+1], NAN ); \n"
                        "               break;\n"
                        "       }\n"
                        "       f0 = ", name, "( f0, &iout );\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 0:\n"
                        "               out[3*i+1] = f0.y; \n"
                        "               out2[3*i+1] = iout.y; \n"
                        "               // fall through\n"
                        "           case 1:\n"
                        "               out[3*i] = f0.x; \n"
                        "               out2[3*i] = iout.x; \n"
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

    return MakeKernel(kern, (cl_uint) kernSize, testName, k, p);

}

typedef struct BuildKernelInfo
{
    cl_uint     offset;            // the first vector size to build
    cl_kernel   *kernels;
    cl_program  *programs;
    const char  *nameInCode;
}BuildKernelInfo;

static cl_int BuildKernel_FloatFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p );
static cl_int BuildKernel_FloatFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p )
{
    BuildKernelInfo *info = (BuildKernelInfo*) p;
    cl_uint i = info->offset + job_id;
    return BuildKernel( info->nameInCode, i, info->kernels + i, info->programs + i );
}

static cl_int BuildKernel_DoubleFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p );
static cl_int BuildKernel_DoubleFn( cl_uint job_id, cl_uint thread_id UNUSED, void *p )
{
    BuildKernelInfo *info = (BuildKernelInfo*) p;
    cl_uint i = info->offset + job_id;
    return BuildKernelDouble( info->nameInCode, i, info->kernels + i, info->programs + i );
}

cl_ulong  abs_cl_long( cl_long i );
cl_ulong  abs_cl_long( cl_long i )
{
    cl_long mask = i >> 63;
    return (i ^ mask) - mask;
}

int TestFunc_FloatI_Float(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int64_t maxError2 = 0;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;
    float float_ulps;
     uint64_t step = bufferSize / sizeof( float );
    int scale = (int)((1ULL<<32) / (16 * bufferSize / sizeof( float )) + 1);
    cl_ulong  maxiError;

    logFunctionInfo(f->name,sizeof(cl_float),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }
    if( gIsEmbedded )
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    maxiError = float_ulps == INFINITY ? CL_ULONG_MAX : 0;

    // Init the kernels
    BuildKernelInfo build_info = { gMinVectorSizeIndex, kernels, programs, f->nameInCode };
    if( (error = ThreadPool_Do( BuildKernel_FloatFn, gMaxVectorSizeIndex - gMinVectorSizeIndex, &build_info ) ))
        return error;
/*
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
        if( (error =  BuildKernel( f->nameInCode, (int) i, kernels + i, programs + i) ) )
            return error;
*/

    for( i = 0; i < (1ULL<<32); i += step )
    {
        //Init input array
        uint32_t *p = (uint32_t *)gIn;
        if( gWimpyMode )
        {
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
                p[j] = (uint32_t) i + j * scale;
        }
        else
        {
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
                p[j] = (uint32_t) i + j;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }

        // write garbage into output arrays
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            uint32_t pattern = 0xffffdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0, bufferSize, gOut[j], 0, NULL, NULL) ))
            {
                vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n", error, j );
                goto exit;
            }

            memset_pattern4(gOut2[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j], CL_FALSE, 0, bufferSize, gOut2[j], 0, NULL, NULL) ))
            {
                vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2b(%d) ***\n", error, j );
                goto exit;
            }
        }

        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gOutBuffer2[j] ), &gOutBuffer2[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
            {
                vlog_error( "FAILED -- could not execute kernel\n" );
                goto exit;
            }
        }

        // Get that moving
        if( (error = clFlush(gQueue) ))
            vlog( "clFlush failed\n" );

        //Calculate the correctly rounded reference result
        float *r = (float *)gOut_Ref;
        int *r2 = (int *)gOut_Ref2;
        float *s = (float *)gIn;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
            r[j] = (float) f->func.f_fpI( s[j], r2+j );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer2[j], CL_TRUE, 0, bufferSize, gOut2[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray2 failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;

        //Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] || t2[j] != q2[j] )
                {
                    float test = ((float*) q)[j];
                    int correct2 = INT_MIN;
                    double correct = f->func.f_fpI( s[j], &correct2 );
                    float err = Ulp_Error( test, correct );
                    cl_long iErr = (int64_t) q2[j] - (int64_t) correct2;
                    int fail = ! (fabsf(err) <= float_ulps && abs_cl_long( iErr ) <= maxiError );
                    if( ftz )
                    {
                        // retry per section 6.5.3.2
                        if( IsFloatResultSubnormal(correct, float_ulps ) )
                        {
                            fail = fail && ! ( test == 0.0f && iErr == 0 );
                            if( ! fail )
                                err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if( IsFloatSubnormal( s[j] ) )
                        {
                            int correct5, correct6;
                            double correct3 = f->func.f_fpI( 0.0, &correct5 );
                            double correct4 = f->func.f_fpI( -0.0, &correct6 );
                            float err2 = Ulp_Error( test, correct3  );
                            float err3 = Ulp_Error( test, correct4  );
                            cl_long iErr2 = (long long) q2[j] - (long long) correct5;
                            cl_long iErr3 = (long long) q2[j] - (long long) correct6;

                            // Did +0 work?
                            if( fabsf(err2) <= float_ulps && abs_cl_long( iErr2 ) <= maxiError )
                            {
                                err = err2;
                                iErr = iErr2;
                                fail = 0;
                            }
                            // Did -0 work?
                            else if(fabsf(err3) <= float_ulps && abs_cl_long( iErr3 ) <= maxiError)
                            {
                                err = err3;
                                iErr = iErr3;
                                fail = 0;
                            }

                            // retry per section 6.5.3.4
                            if( fail && (IsFloatResultSubnormal(correct2, float_ulps ) || IsFloatResultSubnormal(correct3, float_ulps )) )
                            {
                                fail = fail && ! ( test == 0.0f && (abs_cl_long( iErr2 ) <= maxiError || abs_cl_long( iErr3 ) <= maxiError) );
                                if( ! fail )
                                {
                                    err = 0.0f;
                                    iErr = 0;
                                }
                            }
                        }
                    }
                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                    }
                    if( llabs(iErr) > maxError2 )
                    {
                        maxError2 = llabs(iErr );
                        maxErrorVal2 = s[j];
                    }

                    if( fail )
                    {
                        vlog_error( "\nERROR: %s%s: {%f, %d} ulp error at %a: *{%a, %d} vs. {%a, %d}\n", f->name, sizeNames[k], err, (int) iErr, ((float*) gIn)[j], ((float*) gOut_Ref)[j], ((int*) gOut_Ref2)[j], test, q2[j] );
                        error = -1;
                        goto exit;
                    }
                }
            }
        }

        if( 0 == (i & 0x0fffffff) )
        {
           if (gVerboseBruteForce)
           {
               vlog("base:%14u step:%10zu  bufferSize:%10zd \n", i, step, bufferSize);
           } else
           {
              vlog("." );
           }
           fflush(stdout);
        }
    }

    if( ! gSkipCorrectnessTesting )
    {
        if( gWimpyMode )
            vlog( "Wimp pass" );
        else
            vlog( "passed" );
    }

    if( gMeasureTimes )
    {
        //Init input array
        uint32_t *p = (uint32_t *)gIn;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
            p[j] = genrand_int32(d);
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gOutBuffer2[j] ), &gOutBuffer2[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            double sum = 0.0;
            double bestTime = INFINITY;
            for( k = 0; k < PERF_LOOP_COUNT; k++ )
            {
                uint64_t startTime = GetTime();
                if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
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
            double clocksPerOp = bestTime * (double) gDeviceFrequency * gComputeDevices * gSimdSize * 1e6 / (bufferSize / sizeof( float ) );
            vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sf%s", f->name, sizeNames[j] );
        }
    }

    if( ! gSkipCorrectnessTesting )
        vlog( "\t{%8.2f, %lld} @ %a", maxError, maxError2, maxErrorVal );
    vlog( "\n" );

exit:
    // Release
    for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}

int TestFunc_DoubleI_Double(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int64_t maxError2 = 0;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    double maxErrorVal2 = 0.0f;
    cl_ulong  maxiError = f->double_ulps == INFINITY ? CL_ULONG_MAX : 0;
    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;

    uint64_t step = bufferSize / sizeof( double );
    int scale = (int)((1ULL<<32) / (16 * bufferSize / sizeof( double )) + 1);

    logFunctionInfo(f->name,sizeof(cl_double),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }

    Force64BitFPUPrecision();

    // Init the kernels
    BuildKernelInfo build_info = { gMinVectorSizeIndex, kernels, programs, f->nameInCode };
    if( (error = ThreadPool_Do( BuildKernel_DoubleFn,
                                gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                &build_info ) ))
    {
        return error;
    }
/*
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
        if( (error =  BuildKernelDouble( f->nameInCode, (int) i, kernels + i, programs + i) ) )
            return error;
*/

    for( i = 0; i < (1ULL<<32); i += step )
    {
        //Init input array
        double *p = (double *)gIn;
        if( gWimpyMode )
        {
            for( j = 0; j < bufferSize / sizeof( double ); j++ )
                p[j] = DoubleFromUInt32((uint32_t) i + j * scale);
        }
        else
        {
            for( j = 0; j < bufferSize / sizeof( double ); j++ )
                p[j] = DoubleFromUInt32((uint32_t) i + j);
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }

        // write garbage into output arrays
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            uint32_t pattern = 0xffffdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0, bufferSize, gOut[j], 0, NULL, NULL) ))
            {
                vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n", error, j );
                goto exit;
            }

            memset_pattern4(gOut2[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j], CL_FALSE, 0, bufferSize, gOut2[j], 0, NULL, NULL) ))
            {
                vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2b(%d) ***\n", error, j );
                goto exit;
            }
        }

        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gOutBuffer2[j] ), &gOutBuffer2[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
            {
                vlog_error( "FAILED -- could not execute kernel\n" );
                goto exit;
            }
        }

        // Get that moving
        if( (error = clFlush(gQueue) ))
            vlog( "clFlush failed\n" );

        //Calculate the correctly rounded reference result
        double *r = (double *)gOut_Ref;
        int *r2 = (int *)gOut_Ref2;
        double *s = (double *)gIn;
        for( j = 0; j < bufferSize / sizeof( double ); j++ )
            r[j] = (double) f->dfunc.f_fpI( s[j], r2+j );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer2[j], CL_TRUE, 0, bufferSize, gOut2[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray2 failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;

        //Verify data
        uint64_t *t = (uint64_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for( j = 0; j < bufferSize / sizeof( double ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint64_t *q = (uint64_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] || t2[j] != q2[j] )
                {
                    double test = ((double*) q)[j];
                    int correct2 = INT_MIN;
                    long double correct = f->dfunc.f_fpI( s[j], &correct2 );
                    float err = Bruteforce_Ulp_Error_Double( test, correct );
                    cl_long iErr = (long long) q2[j] - (long long) correct2;
                    int fail = ! (fabsf(err) <= f->double_ulps && abs_cl_long( iErr ) <= maxiError );
                    if( ftz )
                    {
                        // retry per section 6.5.3.2
                        if( IsDoubleResultSubnormal(correct, f->double_ulps ) )
                        {
                            fail = fail && ! ( test == 0.0f && iErr == 0 );
                            if( ! fail )
                                err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if( IsDoubleSubnormal( s[j] ) )
                        {
                            int correct5, correct6;
                            long double correct3 = f->dfunc.f_fpI( 0.0, &correct5 );
                            long double correct4 = f->dfunc.f_fpI( -0.0, &correct6 );
                            float err2 = Bruteforce_Ulp_Error_Double( test, correct3  );
                            float err3 = Bruteforce_Ulp_Error_Double( test, correct4  );
                            cl_long iErr2 = (long long) q2[j] - (long long) correct5;
                            cl_long iErr3 = (long long) q2[j] - (long long) correct6;

                            // Did +0 work?
                            if( fabsf(err2) <= f->double_ulps && abs_cl_long( iErr2 ) <= maxiError )
                            {
                                err = err2;
                                iErr = iErr2;
                                fail = 0;
                            }
                            // Did -0 work?
                            else if(fabsf(err3) <= f->double_ulps && abs_cl_long( iErr3 ) <= maxiError)
                            {
                                err = err3;
                                iErr = iErr3;
                                fail = 0;
                            }

                            // retry per section 6.5.3.4
                            if( fail && (IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps )) )
                            {
                                fail = fail && ! ( test == 0.0f && (abs_cl_long( iErr2 ) <= maxiError || abs_cl_long( iErr3 ) <= maxiError) );
                                if( ! fail )
                                {
                                    err = 0.0f;
                                    iErr = 0;
                                }
                            }
                        }
                    }
                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                    }
                    if( llabs(iErr) > maxError2 )
                    {
                        maxError2 = llabs(iErr );
                        maxErrorVal2 = s[j];
                    }

                    if( fail )
                    {
                        vlog_error( "\nERROR: %sD%s: {%f, %d} ulp error at %.13la: *{%.13la, %d} vs. {%.13la, %d}\n", f->name, sizeNames[k], err, (int) iErr, ((double*) gIn)[j], ((double*) gOut_Ref)[j], ((int*) gOut_Ref2)[j], test, q2[j] );
                        error = -1;
                        goto exit;
                    }
                }
            }
        }

        if( 0 == (i & 0x0fffffff) )
        {
           if (gVerboseBruteForce)
           {
               vlog("base:%14u step:%10zu  bufferSize:%10zd \n", i, step, bufferSize);
           } else
           {
              vlog("." );
           }
           fflush(stdout);
        }
    }

    if( ! gSkipCorrectnessTesting )
    {
        if( gWimpyMode )
            vlog( "Wimp pass" );
        else
            vlog( "passed" );
    }

    if( gMeasureTimes )
    {
        //Init input array
        double *p = (double *)gIn;

        for( j = 0; j < bufferSize / sizeof( double ); j++ )
            p[j] = DoubleFromUInt32(genrand_int32(d));
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gOutBuffer2[j] ), &gOutBuffer2[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            double sum = 0.0;
            double bestTime = INFINITY;
            for( k = 0; k < PERF_LOOP_COUNT; k++ )
            {
                uint64_t startTime = GetTime();
                if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)) )
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
            double clocksPerOp = bestTime * (double) gDeviceFrequency * gComputeDevices * gSimdSize * 1e6 / (bufferSize / sizeof( double ) );
            vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sd%s", f->name, sizeNames[j] );
        }
        for( ; j < gMaxVectorSizeIndex; j++ )
            vlog( "\t     -- " );
    }

    if( ! gSkipCorrectnessTesting )
        vlog( "\t{%8.2f, %lld} @ %a", maxError, maxError2, maxErrorVal );
    vlog( "\n" );

exit:
    // Release
    for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}




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

int TestFunc_Int_Float(const Func *f, MTdata);
int TestFunc_Int_Double(const Func *f, MTdata);

extern const vtbl _i_unary = { "i_unary", TestFunc_Int_Float,
                               TestFunc_Int_Double };


static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p );

static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
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
                            "       int3 i0 = ", name, "( f0 );\n"
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

    return MakeKernel(kern, (cl_uint) kernSize, testName, k, p);
}

static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global int", sizeNames[vectorSize], "* out, __global double", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i] );\n"
                            "}\n"
                        };

    const char *c3[] = {"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global int* out, __global double* in)\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   if( i + 1 < get_global_size(0) )\n"
                        "   {\n"
                        "       double3 f0 = vload3( 0, in + 3 * i );\n"
                        "       int3 i0 = ", name, "( f0 );\n"
                        "       vstore3( i0, 0, out + 3*i );\n"
                        "   }\n"
                        "   else\n"
                        "   {\n"
                        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
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
                        "       int3 i0 = ", name, "( f0 );\n"
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

int TestFunc_Int_Float(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    int ftz = f->ftz || 0 == (gFloatCapabilities & CL_FP_DENORM) || gForceFTZ;
    size_t bufferSize = (gWimpyMode)?gWimpyBufferSize:BUFFER_SIZE;
    uint64_t step = bufferSize / sizeof( float );
    int scale = (int)((1ULL<<32) / (16 * bufferSize / sizeof( float )) + 1);

    logFunctionInfo(f->name,sizeof(cl_float),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }

    // This test is not using ThreadPool so we need to disable FTZ here
    // for reference computations
    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);

    Force64BitFPUPrecision();

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
        }

        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

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
        int *r = (int *)gOut_Ref;
        float *s = (float *)gIn;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
            r[j] = f->func.i_f( s[j] );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;

        //Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] )
                {
                    if( ftz && IsFloatSubnormal(s[j]))
                    {
                        unsigned int correct0 = f->func.i_f( 0.0 );
                        unsigned int correct1 = f->func.i_f( -0.0 );
                        if( q[j] == correct0 || q[j] == correct1 )
                            continue;
                    }

                    uint32_t err = t[j] - q[j];
                    if( q[j] > t[j] )
                        err = q[j] - t[j];
                    vlog_error( "\nERROR: %s%s: %d ulp error at %a (0x%8.8x): *%d vs. %d\n", f->name, sizeNames[k], err, ((float*) gIn)[j], ((cl_uint*) gIn)[j], t[j], q[j] );
                  error = -1;
                  goto exit;
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
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

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

    vlog( "\n" );
exit:
    RestoreFPState(&oldMode);
    // Release
    for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}

int TestFunc_Int_Double(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    int ftz = f->ftz || gForceFTZ;
    size_t bufferSize = (gWimpyMode)?gWimpyBufferSize:BUFFER_SIZE;
    uint64_t step = bufferSize / sizeof( cl_double );
    int scale = (int)((1ULL<<32) / (16 * bufferSize / sizeof( cl_double )) + 1);

    logFunctionInfo(f->name,sizeof(cl_double),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }
    // This test is not using ThreadPool so we need to disable FTZ here
    // for reference computations
    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);

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
            for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
                p[j] = DoubleFromUInt32( (uint32_t) i + j * scale );
        }
        else
        {
            for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
                p[j] = DoubleFromUInt32( (uint32_t) i + j );
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
        }

        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

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
        int *r = (int *)gOut_Ref;
        double *s = (double *)gIn;
        for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
            r[j] = f->dfunc.i_f( s[j] );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)) )
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;

        //Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] )
                {
                    if( ftz && IsDoubleSubnormal(s[j]))
                    {
                        unsigned int correct0 = f->dfunc.i_f( 0.0 );
                        unsigned int correct1 = f->dfunc.i_f( -0.0 );
                        if( q[j] == correct0 || q[j] == correct1 )
                            continue;
                    }

                    uint32_t err = t[j] - q[j];
                    if( q[j] > t[j] )
                        err = q[j] - t[j];
                    vlog_error( "\nERROR: %sD%s: %d ulp error at %.13la: *%d vs. %d\n", f->name, sizeNames[k], err, ((double*) gIn)[j], t[j], q[j] );
                  error = -1;
                  goto exit;
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
        for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
            p[j] = DoubleFromUInt32( genrand_int32(d) );
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
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

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
            vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sD%s", f->name, sizeNames[j] );
        }
        for( ; j < gMaxVectorSizeIndex; j++ )
            vlog( "\t     -- " );
    }

    vlog( "\n" );


exit:
    RestoreFPState(&oldMode);
    // Release
    for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}




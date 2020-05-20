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

int TestFunc_Float_UInt(const Func *f, MTdata);
int TestFunc_Double_ULong(const Func *f, MTdata);

extern const vtbl _unary_u = { "unary_u", TestFunc_Float_UInt,
                               TestFunc_Double_ULong };


static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p );

static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
{
    const char *c[] = {
                            "__kernel void math_kernel", sizeNames[vectorSize], "( __global float", sizeNames[vectorSize], "* out, __global uint", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i] );\n"
                            "}\n"
                        };
    const char *c3[] = {    "__kernel void math_kernel", sizeNames[vectorSize], "( __global float* out, __global uint* in)\n"
                            "{\n"
                            "   size_t i = get_global_id(0);\n"
                            "   if( i + 1 < get_global_size(0) )\n"
                            "   {\n"
                            "       uint3 u0 = vload3( 0, in + 3 * i );\n"
                            "       float3 f0 = ", name, "( u0 );\n"
                            "       vstore3( f0, 0, out + 3*i );\n"
                            "   }\n"
                            "   else\n"
                            "   {\n"
                            "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                            "       uint3 u0;\n"
                            "       float3 f0;\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 1:\n"
                            "               u0 = (uint3)( in[3*i], 0xdead, 0xdead ); \n"
                            "               break;\n"
                            "           case 0:\n"
                            "               u0 = (uint3)( in[3*i], in[3*i+1], 0xdead ); \n"
                            "               break;\n"
                            "       }\n"
                            "       f0 = ", name, "( u0 );\n"
                            "       switch( parity )\n"
                            "       {\n"
                            "           case 0:\n"
                            "               out[3*i+1] = f0.y; \n"
                            "               // fall through\n"
                            "           case 1:\n"
                            "               out[3*i] = f0.x; \n"
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
    const char *c[] = {
                            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                "__kernel void math_kernel", sizeNames[vectorSize], "( __global double", sizeNames[vectorSize], "* out, __global ulong", sizeNames[vectorSize], "* in)\n"
                            "{\n"
                            "   int i = get_global_id(0);\n"
                            "   out[i] = ", name, "( in[i] );\n"
                            "}\n"
                        };

    const char *c3[] = {    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel", sizeNames[vectorSize], "( __global double* out, __global ulong* in)\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   if( i + 1 < get_global_size(0) )\n"
                        "   {\n"
                        "       ulong3 u0 = vload3( 0, in + 3 * i );\n"
                        "       double3 f0 = ", name, "( u0 );\n"
                        "       vstore3( f0, 0, out + 3*i );\n"
                        "   }\n"
                        "   else\n"
                        "   {\n"
                        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
                        "       ulong3 u0;\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 1:\n"
                        "               u0 = (ulong3)( in[3*i], 0xdeaddeaddeaddeadUL, 0xdeaddeaddeaddeadUL ); \n"
                        "               break;\n"
                        "           case 0:\n"
                        "               u0 = (ulong3)( in[3*i], in[3*i+1], 0xdeaddeaddeaddeadUL ); \n"
                        "               break;\n"
                        "       }\n"
                        "       double3 f0 = ", name, "( u0 );\n"
                        "       switch( parity )\n"
                        "       {\n"
                        "           case 0:\n"
                        "               out[3*i+1] = f0.y; \n"
                        "               // fall through\n"
                        "           case 1:\n"
                        "               out[3*i] = f0.x; \n"
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

int TestFunc_Float_UInt(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal = 0.0f;
    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;

    uint64_t step = bufferSize / sizeof( float );
    int scale = (int)((1ULL<<32) / (16 * bufferSize / sizeof( double )) + 1);
    int isRangeLimited = 0;
    float float_ulps;
    float half_sin_cos_tan_limit = 0;

    logFunctionInfo(f->name,sizeof(cl_float),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }
    if( gIsEmbedded)
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    // Init the kernels
    BuildKernelInfo build_info = { gMinVectorSizeIndex, kernels, programs, f->nameInCode };
    if( (error = ThreadPool_Do( BuildKernel_FloatFn, gMaxVectorSizeIndex - gMinVectorSizeIndex, &build_info ) ))
        return error;
/*
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
        if( (error =  BuildKernel( f->nameInCode, (int) i, kernels + i, programs + i) ) )
            return error;
*/

    if( 0 == strcmp( f->name, "half_sin") || 0 == strcmp( f->name, "half_cos") )
    {
        isRangeLimited = 1;
        half_sin_cos_tan_limit = 1.0f + float_ulps * (FLT_EPSILON/2.0f);             // out of range results from finite inputs must be in [-1,1]
    }
    else if( 0 == strcmp( f->name, "half_tan"))
    {
        isRangeLimited = 1;
        half_sin_cos_tan_limit = INFINITY;             // out of range resut from finite inputs must be numeric
    }


    for( i = 0; i < (1ULL<<32); i += step  )
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
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }

        // write garbage into output arrays
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            uint32_t pattern = 0xffffdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0, bufferSize, gOut[j], 0, NULL, NULL)))
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
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ))){ LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)))
            {
                vlog_error( "FAILURE -- could not execute kernel\n" );
                goto exit;
            }
        }

        // Get that moving
        if( (error = clFlush(gQueue) ))
            vlog( "clFlush failed\n" );

        //Calculate the correctly rounded reference result
        float *r = (float*) gOut_Ref;
        cl_uint *s = (cl_uint*) gIn;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
            r[j] = (float) f->func.f_u( s[j] );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;


        //Verify data
        uint32_t *t = (uint32_t*) gOut_Ref;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint32_t *q = (uint32_t*)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] )
                {
                    float test = ((float*) q)[j];
                    double correct = f->func.f_u( s[j] );
                    float err = Ulp_Error( test, correct );
                    int fail = ! (fabsf(err) <= float_ulps);

                    // half_sin/cos/tan are only valid between +-2**16, Inf, NaN
                    if( isRangeLimited && fabsf(s[j]) > MAKE_HEX_FLOAT(0x1.0p16f, 0x1L, 16) && fabsf(s[j]) < INFINITY )
                    {
                        if( fabsf( test ) <= half_sin_cos_tan_limit )
                        {
                            err = 0;
                            fail = 0;
                        }
                    }

                     if( fail )
                    {
                        if( ftz )
                        {
                            // retry per section 6.5.3.2
                            if( IsFloatResultSubnormal(correct, float_ulps) )
                            {
                                fail = fail && ( test != 0.0f );
                                if( ! fail )
                                    err = 0.0f;
                            }
                        }
                    }
                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                    }
                    if( fail )
                    {
                        vlog_error( "\n%s%s: %f ulp error at 0x%8.8x: *%a vs. %a\n", f->name, sizeNames[k], err, ((uint32_t*) gIn)[j], ((float*) gOut_Ref)[j], test );
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
        uint32_t *p = (uint32_t*)gIn;
        if( strstr( f->name, "exp" ) || strstr( f->name, "sin" ) || strstr( f->name, "cos" ) || strstr( f->name, "tan" ) )
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
                ((float*)p)[j] = (float) genrand_real1(d);
        else if( strstr( f->name, "log" ) )
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
                p[j] = genrand_int32(d) & 0x7fffffff;
        else
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
                    vlog_error( "FAILURE -- could not execute kernel\n" );
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
        vlog( "\t%8.2f @ %a", maxError, maxErrorVal );
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

static cl_ulong random64( MTdata d )
{
    return (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
}

int TestFunc_Double_ULong(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;
    uint64_t step = bufferSize / sizeof( cl_double );

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

    for( i = 0; i < (1ULL<<32); i += step  )
    {
        //Init input array
        cl_ulong *p = (cl_ulong *)gIn;
        for( j = 0; j < bufferSize / sizeof( cl_ulong ); j++ )
            p[j] = random64(d);

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }

        // write garbage into output arrays
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            uint32_t pattern = 0xffffdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0, bufferSize, gOut[j], 0, NULL, NULL)))
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
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ))){ LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }

            if( (error = clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL, &localCount, NULL, 0, NULL, NULL)))
            {
                vlog_error( "FAILURE -- could not execute kernel\n" );
                goto exit;
            }
        }

        // Get that moving
        if( (error = clFlush(gQueue) ))
            vlog( "clFlush failed\n" );

        //Calculate the correctly rounded reference result
        double *r = (double*) gOut_Ref;
        cl_ulong *s = (cl_ulong*) gIn;
        for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
            r[j] = (double) f->dfunc.f_u( s[j] );

        // Read the data back
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0, bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error( "ReadArray failed %d\n", error );
                goto exit;
            }
        }

        if( gSkipCorrectnessTesting )
            break;


        //Verify data
        uint64_t *t = (uint64_t*) gOut_Ref;
        for( j = 0; j < bufferSize / sizeof( cl_double ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint64_t *q = (uint64_t*)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] )
                {
                    double test = ((double*) q)[j];
                    long double correct = f->dfunc.f_u( s[j] );
                    float err = Bruteforce_Ulp_Error_Double(test, correct);
                    int fail = ! (fabsf(err) <= f->double_ulps);

                    // half_sin/cos/tan are only valid between +-2**16, Inf, NaN
                    if( fail )
                    {
                        if( ftz )
                        {
                            // retry per section 6.5.3.2
                            if( IsDoubleResultSubnormal(correct, f->double_ulps) )
                            {
                                fail = fail && ( test != 0.0 );
                                if( ! fail )
                                    err = 0.0f;
                            }
                        }
                    }
                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                    }
                    if( fail )
                    {
                        vlog_error( "\n%s%sD: %f ulp error at 0x%16.16llx: *%.13la vs. %.13la\n", f->name, sizeNames[k], err, ((uint64_t*) gIn)[j], ((double*) gOut_Ref)[j], test );
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
        double *p = (double*) gIn;

        for( j = 0; j < bufferSize / sizeof( double ); j++ )
            p[j] = random64(d);
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
                    vlog_error( "FAILURE -- could not execute kernel\n" );
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

    if( ! gSkipCorrectnessTesting )
        vlog( "\t%8.2f @ %a", maxError, maxErrorVal );
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



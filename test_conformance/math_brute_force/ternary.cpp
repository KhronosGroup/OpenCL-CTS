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

#define CORRECTLY_ROUNDED 0
#define FLUSHED 1

int TestFunc_Float_Float_Float_Float(const Func *f, MTdata);
int TestFunc_Double_Double_Double_Double(const Func *f, MTdata);

extern const vtbl _ternary = { "ternary", TestFunc_Float_Float_Float_Float,
                               TestFunc_Double_Double_Double_Double };

static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p );
static int BuildKernelDouble( const char *name, int vectorSize, cl_kernel *k, cl_program *p );
static int BuildKernel( const char *name, int vectorSize, cl_kernel *k, cl_program *p )
{
    const char *c[] = {
        "__kernel void math_kernel", sizeNames[vectorSize], "( __global float", sizeNames[vectorSize], "* out, __global float", sizeNames[vectorSize], "* in1, __global float", sizeNames[vectorSize], "* in2,  __global float", sizeNames[vectorSize], "* in3 )\n"
        "{\n"
        "   int i = get_global_id(0);\n"
        "   out[i] = ", name, "( in1[i], in2[i], in3[i] );\n"
        "}\n"
    };

    const char *c3[] = {    "__kernel void math_kernel", sizeNames[vectorSize], "( __global float* out, __global float* in, __global float* in2 , __global float* in3)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       float3 f2 = vload3( 0, in3 + 3 * i );\n"
        "       f0 = ", name, "( f0, f1, f2 );\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
        "       float3 f0, f1, f2;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
        "               f1 = (float3)( in2[3*i], NAN, NAN ); \n"
        "               f2 = (float3)( in3[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
        "               f1 = (float3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               f2 = (float3)( in3[3*i], in3[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       f0 = ", name, "( f0, f1, f2 );\n"
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
        "__kernel void math_kernel", sizeNames[vectorSize], "( __global double", sizeNames[vectorSize], "* out, __global double", sizeNames[vectorSize], "* in1, __global double", sizeNames[vectorSize], "* in2,  __global double", sizeNames[vectorSize], "* in3 )\n"
        "{\n"
        "   int i = get_global_id(0);\n"
        "   out[i] = ", name, "( in1[i], in2[i], in3[i] );\n"
        "}\n"
    };

    const char *c3[] = {    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "__kernel void math_kernel", sizeNames[vectorSize], "( __global double* out, __global double* in, __global double* in2 , __global double* in3)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       double3 d0 = vload3( 0, in + 3 * i );\n"
        "       double3 d1 = vload3( 0, in2 + 3 * i );\n"
        "       double3 d2 = vload3( 0, in3 + 3 * i );\n"
        "       d0 = ", name, "( d0, d1, d2 );\n"
        "       vstore3( d0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two buffer size \n"
        "       double3 d0, d1, d2;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               d0 = (double3)( in[3*i], NAN, NAN ); \n"
        "               d1 = (double3)( in2[3*i], NAN, NAN ); \n"
        "               d2 = (double3)( in3[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               d0 = (double3)( in[3*i], in[3*i+1], NAN ); \n"
        "               d1 = (double3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               d2 = (double3)( in3[3*i], in3[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       d0 = ", name, "( d0, d1, d2 );\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = d0.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = d0.x; \n"
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


// A table of more difficult cases to get right
static const float specialValuesFloat[] = {
    -NAN, -INFINITY, -FLT_MAX, MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64), MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),  MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    -3.0f, MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23), -2.5f, MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23), -2.0f, MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24), -1.75f, -1.5f, -1.25f, MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24), MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24), MAKE_HEX_FLOAT(-0x1.003p0f, -0x1003000L, -24), -MAKE_HEX_FLOAT(0x1.001p0f, 0x1001000L, -24), -1.0f, MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150), -FLT_MIN, MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150), MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150), MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150), MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150), MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150), MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150), MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150), MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150), MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150), -0.0f,

    +NAN, +INFINITY, +FLT_MAX, MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40), MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64), MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39), MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39), MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63), MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    +3.0f, MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23), 2.5f, MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),+2.0f, MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24), 1.75f, 1.5f, 1.25f, MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24), MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24), MAKE_HEX_FLOAT(0x1.003p0f, 0x1003000L, -24), +MAKE_HEX_FLOAT(0x1.001p0f, 0x1001000L, -24), +1.0f, MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150), +FLT_MIN, MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150), MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150), MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150), MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150), MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150), MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150), MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150), MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150), MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150), +0.0f
};

static size_t specialValuesFloatCount = sizeof( specialValuesFloat ) / sizeof( specialValuesFloat[0] );


int TestFunc_Float_Float_Float_Float(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    float maxErrorVal3 = 0.0f;
    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;

    uint64_t step = bufferSize / sizeof( float );
    int skipNanInf = (0 == strcmp( "fma", f->nameInCode )) && ! gInfNanSupport;
    cl_uchar overflow[BUFFER_SIZE / sizeof( float )];
    float float_ulps;

    logFunctionInfo(f->name,sizeof(cl_float),gTestFastRelaxed);
    if( gWimpyMode )
    {
        step = (1ULL<<32) * gWimpyReductionFactor / (512);
    }

    if( gIsEmbedded )
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

    for( i = 0; i < (1ULL<<32); i += step )
    {
        //Init input array
        uint32_t *p = (uint32_t *)gIn;
        uint32_t *p2 = (uint32_t *)gIn2;
        uint32_t *p3 = (uint32_t *)gIn3;
        j = 0;
        if( i == 0 )
        { // test edge cases
            float *fp = (float *)gIn;
            float *fp2 = (float *)gIn2;
            float *fp3 = (float *)gIn3;
            uint32_t x, y, z;  x = y = z = 0;
            for( ; j < bufferSize / sizeof( float ); j++ )
            {
                fp[j] = specialValuesFloat[x];
                fp2[j] = specialValuesFloat[y];
                fp3[j] = specialValuesFloat[z];

                if( ++x >= specialValuesFloatCount )
                {
                    x = 0;
                    if( ++y >= specialValuesFloatCount )
                    {
                        y = 0;
                        if( ++z >= specialValuesFloatCount )
                            break;
                    }
                }
            }
            if( j == bufferSize / sizeof( float ) )
                vlog_error( "Test Error: not all special cases tested!\n" );
        }

        for( ; j < bufferSize / sizeof( float ); j++ )
        {
            p[j] = genrand_int32(d);
            p2[j] = genrand_int32(d);
            p3[j] = genrand_int32(d);
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, bufferSize, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0, bufferSize, gIn3, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error );
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
            size_t vectorSize = sizeof( cl_float ) * sizeValues[j];
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;    // bufferSize / vectorSize  rounded up
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 3, sizeof( gInBuffer3 ), &gInBuffer3 ) )) { LogBuildError(programs[j]); goto exit; }

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
        float *s = (float *)gIn;
        float *s2 = (float *)gIn2;
        float *s3 = (float *)gIn3;
        if( skipNanInf )
        {
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
            {
                feclearexcept(FE_OVERFLOW);
                r[j] = (float) f->func.f_fma( s[j], s2[j], s3[j], CORRECTLY_ROUNDED );
                overflow[j] = FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for( j = 0; j < bufferSize / sizeof( float ); j++ )
                r[j] = (float) f->func.f_fma( s[j], s2[j], s3[j], CORRECTLY_ROUNDED );
        }


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
                    float err;
                    int fail;
                    float test = ((float*) q)[j];
                    float correct = f->func.f_fma( s[j], s2[j], s3[j], CORRECTLY_ROUNDED );

                    // Per section 10 paragraph 6, accept any result if an input or output is a infinity or NaN or overflow
                    if( skipNanInf )
                    {
                        if( overflow[j]                                         ||
                           IsFloatInfinity(correct) || IsFloatNaN(correct)     ||
                           IsFloatInfinity(s[j])    || IsFloatNaN(s[j])        ||
                           IsFloatInfinity(s2[j])   || IsFloatNaN(s2[j])       ||
                           IsFloatInfinity(s3[j])   || IsFloatNaN(s3[j])       )
                            continue;
                    }


                    err = Ulp_Error( test, correct );
                    fail = ! (fabsf(err) <= float_ulps);

                    if( fail && ftz )
                    {
                        float correct2, err2;

                        // retry per section 6.5.3.2  with flushing on
                        if( 0.0f == test && 0.0f == f->func.f_fma( s[j], s2[j], s3[j], FLUSHED ) )
                        {
                            fail = 0;
                            err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if( fail && IsFloatSubnormal( s[j] ) )
                        { // look at me,
                            float err3, correct3;

                            if( skipNanInf )
                                feclearexcept( FE_OVERFLOW );

                            correct2 = f->func.f_fma( 0.0f, s2[j], s3[j], CORRECTLY_ROUNDED );
                            correct3 = f->func.f_fma( -0.0f, s2[j], s3[j], CORRECTLY_ROUNDED );

                            if( skipNanInf )
                            {
                                if( fetestexcept( FE_OVERFLOW ) )
                                    continue;

                                // Note: no double rounding here.  Reference functions calculate in single precision.
                                if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                   IsFloatInfinity(correct3) || IsFloatNaN(correct3)   )
                                    continue;
                            }

                            err2 = Ulp_Error( test, correct2  );
                            err3 = Ulp_Error( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( 0.0f == test &&
                                ( 0.0f == f->func.f_fma(  0.0f, s2[j], s3[j], FLUSHED )  ||
                                  0.0f == f->func.f_fma( -0.0f, s2[j], s3[j], FLUSHED ) )
                              )
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            //try with first two args as zero
                            if( IsFloatSubnormal( s2[j] ) )
                            { // its fun to have fun,
                                double correct4, correct5;
                                float err4, err5;

                                if( skipNanInf )
                                    feclearexcept( FE_OVERFLOW );

                                correct2 = f->func.f_fma( 0.0f, 0.0f, s3[j], CORRECTLY_ROUNDED );
                                correct3 = f->func.f_fma( -0.0f, 0.0f, s3[j], CORRECTLY_ROUNDED );
                                correct4 = f->func.f_fma( 0.0f, -0.0f, s3[j], CORRECTLY_ROUNDED );
                                correct5 = f->func.f_fma( -0.0f, -0.0f, s3[j], CORRECTLY_ROUNDED );

                                // Per section 10 paragraph 6, accept any result if an input or output is a infinity or NaN or overflow
                                if( !gInfNanSupport )
                                {
                                    if( fetestexcept(FE_OVERFLOW) )
                                        continue;

                                    // Note: no double rounding here.  Reference functions calculate in single precision.
                                    if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                       IsFloatInfinity(correct3) || IsFloatNaN(correct3) ||
                                       IsFloatInfinity(correct4) || IsFloatNaN(correct4) ||
                                       IsFloatInfinity(correct5) || IsFloatNaN(correct5) )
                                        continue;
                                }

                                err2 = Ulp_Error( test, correct2  );
                                err3 = Ulp_Error( test, correct3  );
                                err4 = Ulp_Error( test, correct4  );
                                err5 = Ulp_Error( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)) &&
                                                 (!(fabsf(err4) <= float_ulps)) && (!(fabsf(err5) <= float_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( 0.0f == test &&
                                    ( 0.0f == f->func.f_fma(  0.0f,  0.0f, s3[j], FLUSHED )  ||
                                      0.0f == f->func.f_fma( -0.0f,  0.0f, s3[j], FLUSHED )  ||
                                      0.0f == f->func.f_fma(  0.0f, -0.0f, s3[j], FLUSHED )  ||
                                      0.0f == f->func.f_fma( -0.0f, -0.0f, s3[j], FLUSHED )  )
                                )
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }

                                if( IsFloatSubnormal( s3[j] )  )
                                {
                                    if( test == 0.0f )  // 0*0+0 is 0
                                    {
                                        fail = 0;
                                        err = 0.0f;
                                    }
                                }
                            }
                            else if( IsFloatSubnormal( s3[j] ) )
                            {
                                double correct4, correct5;
                                float err4, err5;

                                if( skipNanInf )
                                    feclearexcept( FE_OVERFLOW );

                                correct2 = f->func.f_fma( 0.0f, s2[j], 0.0f, CORRECTLY_ROUNDED );
                                correct3 = f->func.f_fma( -0.0f, s2[j], 0.0f, CORRECTLY_ROUNDED );
                                correct4 = f->func.f_fma( 0.0f,  s2[j], -0.0f, CORRECTLY_ROUNDED );
                                correct5 = f->func.f_fma( -0.0f, s2[j], -0.0f, CORRECTLY_ROUNDED );

                                // Per section 10 paragraph 6, accept any result if an input or output is a infinity or NaN or overflow
                                if( !gInfNanSupport )
                                {
                                    if( fetestexcept(FE_OVERFLOW) )
                                        continue;

                                    // Note: no double rounding here.  Reference functions calculate in single precision.
                                    if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                       IsFloatInfinity(correct3) || IsFloatNaN(correct3) ||
                                       IsFloatInfinity(correct4) || IsFloatNaN(correct4) ||
                                       IsFloatInfinity(correct5) || IsFloatNaN(correct5) )
                                        continue;
                                }

                                err2 = Ulp_Error( test, correct2  );
                                err3 = Ulp_Error( test, correct3  );
                                err4 = Ulp_Error( test, correct4  );
                                err5 = Ulp_Error( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)) &&
                                                 (!(fabsf(err4) <= float_ulps)) && (!(fabsf(err5) <= float_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( 0.0f == test &&
                                    (   0.0f == f->func.f_fma( 0.0f, s2[j], 0.0f, FLUSHED )   ||
                                        0.0f == f->func.f_fma(-0.0f, s2[j], 0.0f, FLUSHED )   ||
                                        0.0f == f->func.f_fma( 0.0f, s2[j],-0.0f, FLUSHED )   ||
                                        0.0f == f->func.f_fma(-0.0f, s2[j],-0.0f, FLUSHED )   )
                                )
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if( fail && IsFloatSubnormal( s2[j] ) )
                        {
                            double correct2, correct3;
                            float err2, err3;

                            if( skipNanInf )
                                feclearexcept( FE_OVERFLOW );

                            correct2 = f->func.f_fma( s[j], 0.0f, s3[j], CORRECTLY_ROUNDED );
                            correct3 = f->func.f_fma( s[j], -0.0f, s3[j], CORRECTLY_ROUNDED );

                            if( skipNanInf )
                            {
                                if( fetestexcept( FE_OVERFLOW ) )
                                    continue;

                                // Note: no double rounding here.  Reference functions calculate in single precision.
                                if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                   IsFloatInfinity(correct3) || IsFloatNaN(correct3)   )
                                    continue;
                            }

                            err2 = Ulp_Error( test, correct2  );
                            err3 = Ulp_Error( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( 0.0f == test &&
                                (   0.0f == f->func.f_fma( s[j], 0.0f, s3[j], FLUSHED )  ||
                                    0.0f == f->func.f_fma( s[j], -0.0f, s3[j], FLUSHED ) )
                            )
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            //try with second two args as zero
                            if( IsFloatSubnormal( s3[j] ) )
                            {
                                double correct4, correct5;
                                float err4, err5;

                                if( skipNanInf )
                                    feclearexcept( FE_OVERFLOW );

                                correct2 = f->func.f_fma( s[j], 0.0f, 0.0f, CORRECTLY_ROUNDED );
                                correct3 = f->func.f_fma( s[j], -0.0f, 0.0f, CORRECTLY_ROUNDED );
                                correct4 = f->func.f_fma( s[j], 0.0f, -0.0f, CORRECTLY_ROUNDED );
                                correct5 = f->func.f_fma( s[j], -0.0f, -0.0f, CORRECTLY_ROUNDED );

                                // Per section 10 paragraph 6, accept any result if an input or output is a infinity or NaN or overflow
                                if( !gInfNanSupport )
                                {
                                    if( fetestexcept(FE_OVERFLOW) )
                                        continue;

                                    // Note: no double rounding here.  Reference functions calculate in single precision.
                                    if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                       IsFloatInfinity(correct3) || IsFloatNaN(correct3) ||
                                       IsFloatInfinity(correct4) || IsFloatNaN(correct4) ||
                                       IsFloatInfinity(correct5) || IsFloatNaN(correct5) )
                                        continue;
                                }

                                err2 = Ulp_Error( test, correct2  );
                                err3 = Ulp_Error( test, correct3  );
                                err4 = Ulp_Error( test, correct4  );
                                err5 = Ulp_Error( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)) &&
                                                 (!(fabsf(err4) <= float_ulps)) && (!(fabsf(err5) <= float_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( 0.0f == test &&
                                    (   0.0f == f->func.f_fma( s[j], 0.0f, 0.0f, FLUSHED )    ||
                                        0.0f == f->func.f_fma( s[j],-0.0f, 0.0f, FLUSHED )    ||
                                        0.0f == f->func.f_fma( s[j], 0.0f,-0.0f, FLUSHED )    ||
                                        0.0f == f->func.f_fma( s[j],-0.0f,-0.0f, FLUSHED )    )
                                )
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if( fail && IsFloatSubnormal(s3[j]) )
                        {
                            double correct2, correct3;
                            float err2, err3;

                            if( skipNanInf )
                                feclearexcept( FE_OVERFLOW );

                            correct2 = f->func.f_fma( s[j], s2[j], 0.0f, CORRECTLY_ROUNDED );
                            correct3 = f->func.f_fma( s[j], s2[j], -0.0f, CORRECTLY_ROUNDED );

                            if( skipNanInf )
                            {
                                if( fetestexcept( FE_OVERFLOW ) )
                                    continue;

                                // Note: no double rounding here.  Reference functions calculate in single precision.
                                if( IsFloatInfinity(correct2) || IsFloatNaN(correct2) ||
                                   IsFloatInfinity(correct3) || IsFloatNaN(correct3)   )
                                    continue;
                            }

                            err2 = Ulp_Error( test, correct2  );
                            err3 = Ulp_Error( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= float_ulps)) && (!(fabsf(err3) <= float_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( 0.0f == test &&
                                (   0.0f == f->func.f_fma( s[j], s2[j], 0.0f, FLUSHED ) ||
                                    0.0f == f->func.f_fma( s[j], s2[j],-0.0f, FLUSHED )  )
                            )
                            {
                                fail = 0;
                                err = 0.0f;
                            }
                        }
                    }

                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                        maxErrorVal2 = s2[j];
                        maxErrorVal3 = s3[j];
                    }

                    if( fail )
                    {
                        vlog_error( "\nERROR: %s%s: %f ulp error at {%a, %a, %a} ({0x%8.8x, 0x%8.8x, 0x%8.8x}): *%a vs. %a\n", f->name, sizeNames[k], err, s[j], s2[j], s3[j], ((cl_uint*)s)[j], ((cl_uint*)s2)[j], ((cl_uint*)s3)[j],  ((float*) gOut_Ref)[j], test );
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
               vlog("base:%14u step:%10u bufferSize:%10zd \n", i, step,  bufferSize);
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
        uint32_t *p2 = (uint32_t *)gIn2;
        uint32_t *p3 = (uint32_t *)gIn3;
        for( j = 0; j < bufferSize / sizeof( float ); j++ )
        {
            p[j] = genrand_int32(d);
            p2[j] = genrand_int32(d);
            p3[j] = genrand_int32(d);
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, bufferSize, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0, bufferSize, gIn3, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeof( cl_float ) * sizeValues[j];
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;    // bufferSize / vectorSize  rounded up
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 3, sizeof( gInBuffer3 ), &gInBuffer3 ) )) { LogBuildError(programs[j]); goto exit; }

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
        vlog( "\t%8.2f @ {%a, %a, %a}", maxError, maxErrorVal, maxErrorVal2, maxErrorVal3 );
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

// A table of more difficult cases to get right
static const double specialValuesDouble[] = {
    -NAN, -INFINITY, -DBL_MAX, MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12), MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11),  MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11), MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    -3.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51), -2.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51), -2.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52), -1.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), -1.0, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074), -DBL_MIN, MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074), -0.0,

    +NAN, +INFINITY, +DBL_MAX, MAKE_HEX_DOUBLE(+0x1.0000000000001p64, +0x10000000000001LL, 12), MAKE_HEX_DOUBLE(+0x1.0p64, +0x1LL, 64), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),  MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11), MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    +3.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51), +2.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51), +2.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52), +1.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), +1.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074), +DBL_MIN, MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074), +0.0,
};

static const size_t specialValuesDoubleCount = sizeof( specialValuesDouble ) / sizeof( specialValuesDouble[0] );


int TestFunc_Double_Double_Double_Double(const Func *f, MTdata d)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[ VECTOR_SIZE_COUNT ];
    cl_kernel kernels[ VECTOR_SIZE_COUNT ];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    double maxErrorVal2 = 0.0f;
    double maxErrorVal3 = 0.0f;
    logFunctionInfo(f->name,sizeof(cl_double),gTestFastRelaxed);

    size_t bufferSize = (gWimpyMode)? gWimpyBufferSize: BUFFER_SIZE;
    uint64_t step = bufferSize / sizeof( double );
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
        double *p2 = (double *)gIn2;
        double *p3 = (double *)gIn3;
        j = 0;
        if( i == 0 )
        { // test edge cases
            uint32_t x, y, z;  x = y = z = 0;
            for( ; j < bufferSize / sizeof( double ); j++ )
            {
                p[j] = specialValuesDouble[x];
                p2[j] = specialValuesDouble[y];
                p3[j] = specialValuesDouble[z];
                if( ++x >= specialValuesDoubleCount )
                {
                    x = 0;
                    if( ++y >= specialValuesDoubleCount )
                    {
                        y = 0;
                        if( ++z >= specialValuesDoubleCount )
                            break;
                    }
                }
            }
            if( j == bufferSize / sizeof( double ) )
                vlog_error( "Test Error: not all special cases tested!\n" );
        }

        for( ; j < bufferSize / sizeof( double ); j++ )
        {
            p[j] = DoubleFromUInt32(genrand_int32(d));
            p2[j] = DoubleFromUInt32(genrand_int32(d));
            p3[j] = DoubleFromUInt32(genrand_int32(d));
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, bufferSize, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0, bufferSize, gIn3, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error );
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
            size_t vectorSize = sizeof( cl_double ) * sizeValues[j];
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;    // bufferSize / vectorSize  rounded up
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 3, sizeof( gInBuffer3 ), &gInBuffer3 ) )) { LogBuildError(programs[j]); goto exit; }

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
        double *s = (double *)gIn;
        double *s2 = (double *)gIn2;
        double *s3 = (double *)gIn3;
        for( j = 0; j < bufferSize / sizeof( double ); j++ )
            r[j] = (double) f->dfunc.f_fff( s[j], s2[j], s3[j] );

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
        uint64_t *t = (uint64_t *)gOut_Ref;
        for( j = 0; j < bufferSize / sizeof( double ); j++ )
        {
            for( k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++ )
            {
                uint64_t *q = (uint64_t *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if( t[j] != q[j] )
                {
                    double test = ((double*) q)[j];
                    long double correct = f->dfunc.f_fff( s[j], s2[j], s3[j] );
                    float err = Bruteforce_Ulp_Error_Double( test, correct );
                    int fail = ! (fabsf(err) <= f->double_ulps);

                    if( fail && ftz )
                    {
                        // retry per section 6.5.3.2
                        if( IsDoubleSubnormal(correct) )
                        { // look at me,
                            fail = fail && ( test != 0.0f );
                            if( ! fail )
                                err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if( fail && IsDoubleSubnormal( s[j] ) )
                        { // look at me,
                            long double correct2 = f->dfunc.f_fff( 0.0, s2[j], s3[j] );
                            long double correct3 = f->dfunc.f_fff( -0.0, s2[j], s3[j] );
                            float err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                            float err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps ) )
                            { // look at me now,
                                fail = fail && ( test != 0.0f);
                                if( ! fail )
                                    err = 0.0f;
                            }

                            //try with first two args as zero
                            if( IsDoubleSubnormal( s2[j] ) )
                            { // its fun to have fun,
                                correct2 = f->dfunc.f_fff( 0.0, 0.0, s3[j] );
                                correct3 = f->dfunc.f_fff( -0.0, 0.0, s3[j] );
                                long double correct4 = f->dfunc.f_fff( 0.0, -0.0, s3[j] );
                                long double correct5 = f->dfunc.f_fff( -0.0, -0.0, s3[j] );
                                err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                                err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                                float err4 = Bruteforce_Ulp_Error_Double( test, correct4  );
                                float err5 = Bruteforce_Ulp_Error_Double( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)) &&
                                                 (!(fabsf(err4) <= f->double_ulps)) && (!(fabsf(err5) <= f->double_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps ) ||
                                   IsDoubleResultSubnormal( correct4, f->double_ulps ) || IsDoubleResultSubnormal( correct5, f->double_ulps ) )
                                {
                                    fail = fail && ( test != 0.0f);
                                    if( ! fail )
                                        err = 0.0f;
                                }

                                if( IsDoubleSubnormal( s3[j] )  )
                                { // but you have to know how!
                                    correct2 = f->dfunc.f_fff( 0.0, 0.0, 0.0f );
                                    correct3 = f->dfunc.f_fff( -0.0, 0.0, 0.0f );
                                    correct4 = f->dfunc.f_fff( 0.0, -0.0, 0.0f );
                                    correct5 = f->dfunc.f_fff( -0.0, -0.0, 0.0f );
                                    long double correct6 = f->dfunc.f_fff( 0.0, 0.0, -0.0f );
                                    long double correct7 = f->dfunc.f_fff( -0.0, 0.0, -0.0f );
                                    long double correct8 = f->dfunc.f_fff( 0.0, -0.0, -0.0f );
                                    long double correct9 = f->dfunc.f_fff( -0.0, -0.0, -0.0f );
                                    err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                                    err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                                    err4 = Bruteforce_Ulp_Error_Double( test, correct4  );
                                    err5 = Bruteforce_Ulp_Error_Double( test, correct5  );
                                    float err6 = Bruteforce_Ulp_Error_Double( test, correct6  );
                                    float err7 = Bruteforce_Ulp_Error_Double( test, correct7  );
                                    float err8 = Bruteforce_Ulp_Error_Double( test, correct8  );
                                    float err9 = Bruteforce_Ulp_Error_Double( test, correct9  );
                                    fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)) &&
                                                     (!(fabsf(err4) <= f->double_ulps)) && (!(fabsf(err5) <= f->double_ulps)) &&
                                                     (!(fabsf(err5) <= f->double_ulps)) && (!(fabsf(err6) <= f->double_ulps)) &&
                                                     (!(fabsf(err7) <= f->double_ulps)) && (!(fabsf(err8) <= f->double_ulps)));
                                    if( fabsf( err2 ) < fabsf(err ) )
                                        err = err2;
                                    if( fabsf( err3 ) < fabsf(err ) )
                                        err = err3;
                                    if( fabsf( err4 ) < fabsf(err ) )
                                        err = err4;
                                    if( fabsf( err5 ) < fabsf(err ) )
                                        err = err5;
                                    if( fabsf( err6 ) < fabsf(err ) )
                                        err = err6;
                                    if( fabsf( err7 ) < fabsf(err ) )
                                        err = err7;
                                    if( fabsf( err8 ) < fabsf(err ) )
                                        err = err8;
                                    if( fabsf( err9 ) < fabsf(err ) )
                                        err = err9;

                                    // retry per section 6.5.3.4
                                    if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps )  ||
                                       IsDoubleResultSubnormal( correct4, f->double_ulps ) || IsDoubleResultSubnormal( correct5, f->double_ulps )  ||
                                       IsDoubleResultSubnormal( correct6, f->double_ulps ) || IsDoubleResultSubnormal( correct7, f->double_ulps )  ||
                                       IsDoubleResultSubnormal( correct8, f->double_ulps ) || IsDoubleResultSubnormal( correct9, f->double_ulps )  )
                                    {
                                        fail = fail && ( test != 0.0f);
                                        if( ! fail )
                                            err = 0.0f;
                                    }
                                }
                            }
                            else if( IsDoubleSubnormal( s3[j] ) )
                            {
                                correct2 = f->dfunc.f_fff( 0.0, s2[j], 0.0 );
                                correct3 = f->dfunc.f_fff( -0.0, s2[j], 0.0 );
                                long double correct4 = f->dfunc.f_fff( 0.0,  s2[j], -0.0 );
                                long double correct5 = f->dfunc.f_fff( -0.0, s2[j], -0.0 );
                                err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                                err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                                float err4 = Bruteforce_Ulp_Error_Double( test, correct4  );
                                float err5 = Bruteforce_Ulp_Error_Double( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)) &&
                                                 (!(fabsf(err4) <= f->double_ulps)) && (!(fabsf(err5) <= f->double_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps )  ||
                                   IsDoubleResultSubnormal( correct4, f->double_ulps ) || IsDoubleResultSubnormal( correct5, f->double_ulps ) )
                                {
                                    fail = fail && ( test != 0.0f);
                                    if( ! fail )
                                        err = 0.0f;
                                }
                            }
                        }
                        else if( fail && IsDoubleSubnormal( s2[j] ) )
                        {
                            long double correct2 = f->dfunc.f_fff( s[j], 0.0, s3[j] );
                            long double correct3 = f->dfunc.f_fff( s[j], -0.0, s3[j] );
                            float err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                            float err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( IsDoubleResultSubnormal( correct2, f->double_ulps )  || IsDoubleResultSubnormal( correct3, f->double_ulps ) )
                            {
                                fail = fail && ( test != 0.0f);
                                if( ! fail )
                                    err = 0.0f;
                            }

                            //try with second two args as zero
                            if( IsDoubleSubnormal( s3[j] ) )
                            {
                                correct2 = f->dfunc.f_fff( s[j], 0.0, 0.0 );
                                correct3 = f->dfunc.f_fff( s[j], -0.0, 0.0 );
                                long double correct4 = f->dfunc.f_fff( s[j], 0.0, -0.0 );
                                long double correct5 = f->dfunc.f_fff( s[j], -0.0, -0.0 );
                                err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                                err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                                float err4 = Bruteforce_Ulp_Error_Double( test, correct4  );
                                float err5 = Bruteforce_Ulp_Error_Double( test, correct5  );
                                fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)) &&
                                                 (!(fabsf(err4) <= f->double_ulps)) && (!(fabsf(err5) <= f->double_ulps)));
                                if( fabsf( err2 ) < fabsf(err ) )
                                    err = err2;
                                if( fabsf( err3 ) < fabsf(err ) )
                                    err = err3;
                                if( fabsf( err4 ) < fabsf(err ) )
                                    err = err4;
                                if( fabsf( err5 ) < fabsf(err ) )
                                    err = err5;

                                // retry per section 6.5.3.4
                                if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps ) ||
                                   IsDoubleResultSubnormal( correct4, f->double_ulps ) || IsDoubleResultSubnormal( correct5, f->double_ulps ) )
                                {
                                    fail = fail && ( test != 0.0f);
                                    if( ! fail )
                                        err = 0.0f;
                                }
                            }
                        }
                        else if( fail && IsDoubleSubnormal(s3[j]) )
                        {
                            long double correct2 = f->dfunc.f_fff( s[j], s2[j], 0.0 );
                            long double correct3 = f->dfunc.f_fff( s[j], s2[j], -0.0 );
                            float err2 = Bruteforce_Ulp_Error_Double( test, correct2  );
                            float err3 = Bruteforce_Ulp_Error_Double( test, correct3  );
                            fail =  fail && ((!(fabsf(err2) <= f->double_ulps)) && (!(fabsf(err3) <= f->double_ulps)));
                            if( fabsf( err2 ) < fabsf(err ) )
                                err = err2;
                            if( fabsf( err3 ) < fabsf(err ) )
                                err = err3;

                            // retry per section 6.5.3.4
                            if( IsDoubleResultSubnormal( correct2, f->double_ulps ) || IsDoubleResultSubnormal( correct3, f->double_ulps ) )
                            {
                                fail = fail && ( test != 0.0f);
                                if( ! fail )
                                    err = 0.0f;
                            }
                        }
                    }

                    if( fabsf(err ) > maxError )
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                        maxErrorVal2 = s2[j];
                        maxErrorVal3 = s3[j];
                    }

                    if( fail )
                    {
                        vlog_error( "\nERROR: %sD%s: %f ulp error at {%.13la, %.13la, %.13la}: *%.13la vs. %.13la\n", f->name, sizeNames[k], err, s[j], s2[j], s3[j], ((double*) gOut_Ref)[j], test );
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
        double *p2 = (double *)gIn2;
        double *p3 = (double *)gIn3;
        for( j = 0; j < bufferSize / sizeof( double ); j++ )
        {
            p[j] = DoubleFromUInt32(genrand_int32(d));
            p2[j] = DoubleFromUInt32(genrand_int32(d));
            p3[j] = DoubleFromUInt32(genrand_int32(d));
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, bufferSize, gIn, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0, bufferSize, gIn2, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error );
            return error;
        }
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0, bufferSize, gIn3, 0, NULL, NULL) ))
        {
            vlog_error( "\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error );
            return error;
        }


        // Run the kernels
        for( j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++ )
        {
            size_t vectorSize = sizeof( cl_double ) * sizeValues[j];
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;    // bufferSize / vectorSize  rounded up
            if( ( error = clSetKernelArg(kernels[j], 0, sizeof( gOutBuffer[j] ), &gOutBuffer[j] ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 1, sizeof( gInBuffer ), &gInBuffer ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 2, sizeof( gInBuffer2 ), &gInBuffer2 ) )) { LogBuildError(programs[j]); goto exit; }
            if( ( error = clSetKernelArg( kernels[j], 3, sizeof( gInBuffer3 ), &gInBuffer3 ) )) { LogBuildError(programs[j]); goto exit; }

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

    if( ! gSkipCorrectnessTesting )
        vlog( "\t%8.2f @ {%a, %a, %a}", maxError, maxErrorVal, maxErrorVal2, maxErrorVal3 );
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



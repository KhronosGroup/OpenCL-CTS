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
#include "harness/testHarness.h"

#include <string.h>

#include <algorithm>

#include "cl_utils.h"
#include "tests.h"

#include <CL/cl_half.h>

int Test_vLoadHalf_private( cl_device_id device, bool aligned )
{
    cl_int error;
    int vectorSize;
    cl_program  programs[kVectorSizeCount+kStrangeVectorSizeCount][AS_NumAddressSpaces] = {{0}};
    cl_kernel   kernels[kVectorSizeCount+kStrangeVectorSizeCount][AS_NumAddressSpaces] = {{0}};
    uint64_t time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    size_t q;

    memset( min_time, -1, sizeof( min_time ) );

    const char *vector_size_names[]   = {"1", "2", "4", "8", "16", "3"};

    int minVectorSize = kMinVectorSize;

    // There is no aligned scalar vloada_half
    if (aligned && minVectorSize == 0) minVectorSize = 1;

    for (vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest;
         vectorSize++)
    {

        int effectiveVectorSize = g_arrVecSizes[vectorSize];
        if(effectiveVectorSize == 3 && aligned) {
            effectiveVectorSize = 4;
        }
        const char *source[] = {
            "__kernel void test( const __global half *p, __global float", vector_size_name_extensions[vectorSize], " *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   f[i] = vload", aligned ? "a" : "", "_half",vector_size_name_extensions[vectorSize],"( i, p );\n"
            "}\n"
        };

        const char *sourceV3[] = {
            "__kernel void test( const __global half *p, __global float *f,\n"
            "                   uint extra_last_thread)\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     if(extra_last_thread ==2) {\n"
            "       f[3*i+1] = vload_half(3*i+1, p);\n"
            "     }\n"
            "     f[3*i] = vload_half(3*i, p);\n"
            "   } else {\n"
            "     vstore3(vload_half3( i, p ),i,f);\n"
            "   }\n"
            "}\n"
        };

        const char *sourceV3aligned[] = {
            "__kernel void test( const __global half *p, __global float3 *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   f[i] = vloada_half3( i, p );\n"
            "   ((__global float *)f)[4*i+3] = vload_half(4*i+3,p);\n"
            "}\n"
        };

        const char *source_private1[] = {
            "__kernel void test( const __global half *p, __global float *f )\n"
            "{\n"
            "   __private ushort data[1];\n"
            "   __private half* hdata_p = (__private half*) data;\n"
            "   size_t i = get_global_id(0);\n"
            "   data[0] = ((__global ushort*)p)[i];\n"
            "   f[i] = vload", (aligned ? "a" : ""), "_half( 0, hdata_p );\n"
            "}\n"
        };

        const char *source_private2[] = {
            "__kernel void test( const __global half *p, __global float", vector_size_name_extensions[vectorSize], " *f )\n"
            "{\n"
            "   __private ", align_types[vectorSize], " data[", vector_size_names[vectorSize], "/", align_divisors[vectorSize], "];\n"
            "   __private half* hdata_p = (__private half*) data;\n"
            "   __global  ", align_types[vectorSize], "* i_p = (__global ", align_types[vectorSize], "*)p;\n"
            "   size_t i = get_global_id(0);\n"
            "   int k;\n"
            "   for (k=0; k<",vector_size_names[vectorSize],"/",align_divisors[vectorSize],"; k++)\n"
            "     data[k] = i_p[i+k];\n"
            "   f[i] = vload", aligned ? "a" : "", "_half",vector_size_name_extensions[vectorSize],"( 0, hdata_p );\n"
            "}\n"
        };

        const char *source_privateV3[] = {
            "__kernel void test( const __global half *p, __global float *f,"
            "                    uint extra_last_thread )\n"
            "{\n"
            "   __private ushort data[3];\n"
            "   __private half* hdata_p = (__private half*) data;\n"
            "   __global  ushort* i_p = (__global  ushort*)p;\n"
            "   size_t i = get_global_id(0);\n"
            "   int k;\n"
            //        "   data = vload3(i, i_p);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     if(extra_last_thread ==2) {\n"
            "       f[3*i+1] = vload_half(3*i+1, p);\n"
            "     }\n"
            "     f[3*i] = vload_half(3*i, p);\n"
            "   } else {\n"
            "     for (k=0; k<3; k++)\n"
            "       data[k] = i_p[i*3+k];\n"
            "     vstore3(vload_half3( 0, hdata_p ), i, f);\n"
            "   }\n"
            "}\n"
        };

        const char *source_privateV3aligned[] = {
            "__kernel void test( const __global half *p, __global float3 *f )\n"
            "{\n"
            "   ushort4 data[4];\n"  // declare as vector for alignment. Make four to check to see vloada_half3 index is working.
            "   half* hdata_p = (half*) &data;\n"
            "   size_t i = get_global_id(0);\n"
            "   global  ushort* i_p = (global  ushort*)p + i * 4;\n"
            "   int offset = i & 3;\n"
            "   data[offset] = (ushort4)( i_p[0], i_p[1], i_p[2], USHRT_MAX ); \n"
            "   data[offset^1] = USHRT_MAX; \n"
            "   data[offset^2] = USHRT_MAX; \n"
            "   data[offset^3] = USHRT_MAX; \n"
            //  test vloada_half3
            "   f[i] = vloada_half3( offset, hdata_p );\n"
            //  Fill in the 4th value so we don't have to special case this code elsewhere in the test.
            "   mem_fence(CLK_GLOBAL_MEM_FENCE );\n"
            "   ((__global float *)f)[4*i+3] = vload_half(4*i+3, p);\n"
            "}\n"
        };

        char local_buf_size[10];

        sprintf(local_buf_size, "%lld", (uint64_t)((effectiveVectorSize))*gWorkGroupSize);
        const char *source_local1[] = {
            "__kernel void test( const __global half *p, __global float *f )\n"
            "{\n"
            "   __local ushort data[",local_buf_size,"];\n"
            "   __local half* hdata_p = (__local half*) data;\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   data[lid] = ((__global ushort*)p)[i];\n"
            "   f[i] = vload", aligned ? "a" : "", "_half( lid, hdata_p );\n"
            "}\n"
        };

        const char *source_local2[] = {
            "#define VECTOR_LEN (",
            vector_size_names[vectorSize],
            "/",
            align_divisors[vectorSize],
            ")\n"
            "#define ALIGN_TYPE ",
            align_types[vectorSize],
            "\n"
            "__kernel void test( const __global half *p, __global float",
            vector_size_name_extensions[vectorSize],
            " *f )\n"
            "{\n"
            "   __local uchar data[",
            local_buf_size,
            "/",
            align_divisors[vectorSize],
            "*sizeof(ALIGN_TYPE)] ",
            "__attribute__((aligned(sizeof(ALIGN_TYPE))));\n"
            "   __local half* hdata_p = (__local half*) data;\n"
            "   __global ALIGN_TYPE* i_p = (__global ALIGN_TYPE*)p;\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   int k;\n"
            "   for (k=0; k<VECTOR_LEN; k++)\n"
            "     *(__local ",
            "ALIGN_TYPE*)&(data[(lid*VECTOR_LEN+k)*sizeof(ALIGN_TYPE)]) = ",
            "i_p[i*VECTOR_LEN+k];\n"
            "   f[i] = vload",
            aligned ? "a" : "",
            "_half",
            vector_size_name_extensions[vectorSize],
            "( lid, hdata_p );\n"
            "}\n"
        };

        const char *source_localV3[] = {
            "__kernel void test( const __global half *p, __global float *f,\n"
            "                    uint extra_last_thread)\n"
            "{\n"
            "   __local ushort data[", local_buf_size,"];\n"
            "   __local half* hdata_p = (__local half*) data;\n"
            "   __global  ushort* i_p = (__global  ushort*)p;\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t lid = get_local_id(0);\n"
            "   int k;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     if(extra_last_thread ==2) {\n"
            "       f[3*i+1] = vload_half(3*i+1, p);\n"
            "     }\n"
            "     f[3*i] = vload_half(3*i, p);\n"
            "   } else {\n"
            "     for (k=0; k<3; k++)\n"
            "       data[lid*3+k] = i_p[i*3+k];\n"
            "     vstore3( vload_half3( lid, hdata_p ),i,f);\n"
            "   };\n"
            "}\n"
        };

        const char *source_localV3aligned[] = {
            "__kernel void test( const __global half *p, __global float3 *f )\n"
            "{\n"
            "   __local ushort data[", local_buf_size,"];\n"
            "   __local half* hdata_p = (__local half*) data;\n"
            "   __global  ushort* i_p = (__global  ushort*)p;\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t lid = get_local_id(0);\n"
            "   int k;\n"
            "   for (k=0; k<4; k++)\n"
            "     data[lid*4+k] = i_p[i*4+k];\n"
            "   f[i] = vloada_half3( lid, hdata_p );\n"
            "   ((__global float *)f)[4*i+3] = vload_half(lid*4+3, hdata_p);\n"
            "}\n"
        };

        const char *source_constant[] = {
            "__kernel void test( __constant half *p, __global float", vector_size_name_extensions[vectorSize], " *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   f[i] = vload", aligned ? "a" : "", "_half",vector_size_name_extensions[vectorSize],"( i, p );\n"
            "}\n"
        };

        const char *source_constantV3[] = {
            "__kernel void test( __constant half *p, __global float *f,\n"
            "                    uint extra_last_thread)\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   if(last_i == i && extra_last_thread != 0) {\n"
            "     if(extra_last_thread ==2) {\n"
            "       f[3*i+1] = vload_half(3*i+1, p);\n"
            "     }\n"
            "     f[3*i] = vload_half(3*i, p);\n"
            "   } else {\n"
            "     vstore3(vload_half",vector_size_name_extensions[vectorSize],"( i, p ), i, f);\n"
            "   }\n"
            "}\n"
        };

        const char *source_constantV3aligned[] = {
            "__kernel void test( __constant half *p, __global float3 *f )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   f[i] = vloada_half3( i, p );\n"
            "   ((__global float *)f)[4*i+3] = vload_half(4*i+3,p);\n"
            "}\n"
        };


        if(g_arrVecSizes[vectorSize] != 3) {
            programs[vectorSize][AS_Global] = MakeProgram( device, source, sizeof( source) / sizeof( source[0])  );
            if( NULL == programs[ vectorSize ][AS_Global] ) {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create program.\n" );
                for ( q= 0; q < sizeof( source) / sizeof( source[0]); q++)
                    vlog_error("%s", source[q]);
                return -1;
            } else {
            }
        } else if(aligned) {
            programs[vectorSize][AS_Global] = MakeProgram( device, sourceV3aligned, sizeof( sourceV3aligned) / sizeof( sourceV3aligned[0])  );
            if( NULL == programs[ vectorSize ][AS_Global] ) {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create program.\n" );
                for ( q= 0; q < sizeof( sourceV3aligned) / sizeof( sourceV3aligned[0]); q++)
                    vlog_error("%s", sourceV3aligned[q]);
                return -1;
            } else {
            }
        } else {
            programs[vectorSize][AS_Global] = MakeProgram( device, sourceV3, sizeof( sourceV3) / sizeof( sourceV3[0])  );
            if( NULL == programs[ vectorSize ][AS_Global] ) {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create program.\n" );
                for ( q= 0; q < sizeof( sourceV3) / sizeof( sourceV3[0]); q++)
                    vlog_error("%s", sourceV3[q]);
                return -1;
            }
        }

        kernels[ vectorSize ][AS_Global] = clCreateKernel( programs[ vectorSize ][AS_Global], "test", &error );
        if( NULL == kernels[vectorSize][AS_Global] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create kernel. (%d)\n", error );
            return -2;
        }

        const char** source_ptr;
        uint32_t source_size;
        if (vectorSize == 0) {
            source_ptr = source_private1;
            source_size = sizeof( source_private1) / sizeof( source_private1[0]);
        } else if(g_arrVecSizes[vectorSize] == 3) {
            if(aligned) {
                source_ptr = source_privateV3aligned;
                source_size = sizeof( source_privateV3aligned) / sizeof( source_privateV3aligned[0]);
            } else {
                source_ptr = source_privateV3;
                source_size = sizeof( source_privateV3) / sizeof( source_privateV3[0]);
            }
        } else {
            source_ptr = source_private2;
            source_size = sizeof( source_private2) / sizeof( source_private2[0]);
        }
        programs[vectorSize][AS_Private] = MakeProgram( device, source_ptr, source_size );
        if( NULL == programs[ vectorSize ][AS_Private] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create private program.\n" );
            for ( q= 0; q < source_size; q++)
                vlog_error("%s", source_ptr[q]);
            return -1;
        }

        kernels[ vectorSize ][AS_Private] = clCreateKernel( programs[ vectorSize ][AS_Private], "test", &error );
        if( NULL == kernels[vectorSize][AS_Private] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create private kernel. (%d)\n", error );
            return -2;
        }

        if (vectorSize == 0) {
            source_ptr = source_local1;
            source_size = sizeof( source_local1) / sizeof( source_local1[0]);
        } else if(g_arrVecSizes[vectorSize] == 3) {
            if(aligned) {
                source_ptr = source_localV3aligned;
                source_size = sizeof(source_localV3aligned)/sizeof(source_localV3aligned[0]);
            } else  {
                source_ptr = source_localV3;
                source_size = sizeof(source_localV3)/sizeof(source_localV3[0]);
            }
        } else {
            source_ptr = source_local2;
            source_size = sizeof( source_local2) / sizeof( source_local2[0]);
        }
        programs[vectorSize][AS_Local] = MakeProgram( device, source_ptr, source_size );
        if( NULL == programs[ vectorSize ][AS_Local] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create local program.\n" );
            for ( q= 0; q < source_size; q++)
                vlog_error("%s", source_ptr[q]);
            return -1;
        }

        kernels[ vectorSize ][AS_Local] = clCreateKernel( programs[ vectorSize ][AS_Local], "test", &error );
        if( NULL == kernels[vectorSize][AS_Local] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create local kernel. (%d)\n", error );
            return -2;
        }

        if(g_arrVecSizes[vectorSize] == 3) {
            if(aligned) {
                programs[vectorSize][AS_Constant] = MakeProgram( device, source_constantV3aligned, sizeof(source_constantV3aligned) / sizeof( source_constantV3aligned[0])  );
                if( NULL == programs[ vectorSize ][AS_Constant] )
                {
                    gFailCount++;
                    vlog_error( "\t\tFAILED -- Failed to create constant program.\n" );
                    for ( q= 0; q < sizeof( source_constantV3aligned) / sizeof( source_constantV3aligned[0]); q++)
                        vlog_error("%s", source_constantV3aligned[q]);
                    return -1;
                }
            } else {
                programs[vectorSize][AS_Constant] = MakeProgram( device, source_constantV3, sizeof(source_constantV3) / sizeof( source_constantV3[0])  );
                if( NULL == programs[ vectorSize ][AS_Constant] )
                {
                    gFailCount++;
                    vlog_error( "\t\tFAILED -- Failed to create constant program.\n" );
                    for ( q= 0; q < sizeof( source_constantV3) / sizeof( source_constantV3[0]); q++)
                        vlog_error("%s", source_constantV3[q]);
                    return -1;
                }
            }
        } else {
            programs[vectorSize][AS_Constant] = MakeProgram( device, source_constant, sizeof(source_constant) / sizeof( source_constant[0])  );
            if( NULL == programs[ vectorSize ][AS_Constant] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create constant program.\n" );
                for ( q= 0; q < sizeof( source_constant) / sizeof( source_constant[0]); q++)
                    vlog_error("%s", source_constant[q]);
                return -1;
            }
        }

        kernels[ vectorSize ][AS_Constant] = clCreateKernel( programs[ vectorSize ][AS_Constant], "test", &error );
        if( NULL == kernels[vectorSize][AS_Constant] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create constant kernel. (%d)\n", error );
            return -2;
        }
    }

    // Figure out how many elements are in a work block
    size_t elementSize = std::max(sizeof(cl_half), sizeof(cl_float));
    size_t blockCount = getBufferSize(device) / elementSize; // elementSize is power of 2
    uint64_t lastCase = 1ULL << (8*sizeof(cl_half)); // number of things of size cl_half

    // we handle 64-bit types a bit differently.
    if( lastCase == 0 )
        lastCase = 0x100000000ULL;


    uint64_t i, j;
    uint64_t printMask = (lastCase >> 4) - 1;
    uint32_t count = 0;
    error = 0;
    int addressSpace;
    //    int reported_vector_skip = 0;

    for( i = 0; i < (uint64_t)lastCase; i += blockCount )
    {
        count = (uint32_t)std::min((uint64_t)blockCount, lastCase - i);

        //Init the input stream
        uint16_t *p = (uint16_t *)gIn_half;
        for( j = 0; j < count; j++ )
            p[j] = j + i;

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_half, CL_TRUE, 0, count * sizeof( cl_half ), gIn_half, 0, NULL, NULL)))
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        //create the reference result
        const unsigned short *s = (const unsigned short *)gIn_half;
        float *d = (float *)gOut_single_reference;
        for (j = 0; j < count; j++) d[j] = cl_half_to_float(s[j]);

        //Check the vector lengths
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        { // here we loop through vector sizes, 3 is last

            for ( addressSpace = 0; addressSpace < AS_NumAddressSpaces; addressSpace++) {
                uint32_t pattern = 0x7fffdead;

                /*
                 if (addressSpace == 3) {
                 vlog("Note: skipping address space %s due to small buffer size.\n", addressSpaceNames[addressSpace]);
                 continue;
                 }
                 */
                memset_pattern4( gOut_single, &pattern, getBufferSize(device));
                if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_single, CL_TRUE, 0, count * sizeof( float ), gOut_single, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if(g_arrVecSizes[vectorSize] == 3 && !aligned) {
                    // now we need to add the extra const argument for how
                    // many elements the last thread should take care of.
                }

                // okay, here is where we have to be careful
                if( (error = RunKernel(device, kernels[vectorSize][addressSpace], gInBuffer_half, gOutBuffer_single, numVecs(count, vectorSize, aligned) ,
                                       runsOverBy(count, vectorSize, aligned) ) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_single, CL_TRUE, 0, count * sizeof( float ), gOut_single, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( memcmp( gOut_single, gOut_single_reference, count * sizeof( float )) )
                {
                    uint32_t *u1 = (uint32_t *)gOut_single;
                    uint32_t *u2 = (uint32_t *)gOut_single_reference;
                    float *f1 = (float *)gOut_single;
                    float *f2 = (float *)gOut_single_reference;
                    for( j = 0; j < count; j++ )
                    {
                        if(isnan(f1[j]) && isnan(f2[j])) // both are nan dont compare them
                            continue;
                        if( u1[j] != u2[j])
                        {
                            vlog_error( " %lld)  (of %lld) Failure at 0x%4.4x:  %a vs *%a  (0x%8.8x vs *0x%8.8x)  vector_size = %d (%s) address space = %s, load is %s\n",
                                       j, (uint64_t)count, ((unsigned short*)gIn_half)[j], f1[j], f2[j], u1[j], u2[j], (g_arrVecSizes[vectorSize]),
                                       vector_size_names[vectorSize], addressSpaceNames[addressSpace],
                                       (aligned?"aligned":"unaligned"));
                            gFailCount++;
                            error = -1;
                            goto exit;
                        }
                    }
                }

                if( gReportTimes && addressSpace == 0)
                {
                    //Run again for timing
                    for( j = 0; j < 100; j++ )
                    {
                        uint64_t startTime = ReadTime();
                        error =
                        RunKernel(device, kernels[vectorSize][addressSpace], gInBuffer_half, gOutBuffer_single, numVecs(count, vectorSize, aligned) ,
                                  runsOverBy(count, vectorSize, aligned));
                        if(error)
                        {
                            gFailCount++;
                            goto exit;
                        }

                        if( (error = clFinish(gQueue)) )
                        {
                            vlog_error( "Failure in clFinish\n" );
                            gFailCount++;
                            goto exit;
                        }
                        uint64_t currentTime = ReadTime() - startTime;
                        time[ vectorSize ] += currentTime;
                        if( currentTime < min_time[ vectorSize ] )
                            min_time[ vectorSize ] = currentTime ;
                    }
                }
            }
        }

        if( ((i+blockCount) & ~printMask) == (i+blockCount) )
        {
            vlog( "." );
            fflush( stdout );
        }
    }

    vlog( "\n" );

    if( gReportTimes )
    {
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * 100), 0,
                      "average us/elem", "vLoad%sHalf avg. (%s, vector size: %d)", ( (aligned) ? "a" : ""), addressSpaceNames[0], (g_arrVecSizes[vectorSize])  );
        for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( min_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0,
                      "best us/elem", "vLoad%sHalf best (%s vector size: %d)", ( (aligned) ? "a" : ""), addressSpaceNames[0], (g_arrVecSizes[vectorSize]) );
    }

exit:
    //clean up
    for( vectorSize = minVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        for ( addressSpace = 0; addressSpace < AS_NumAddressSpaces; addressSpace++) {
            clReleaseKernel( kernels[ vectorSize ][addressSpace] );
            clReleaseProgram( programs[ vectorSize ][addressSpace] );
        }
    }

    return error;
}

int test_vload_half( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vLoadHalf_private( device, false );
}

int test_vloada_half( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    return Test_vLoadHalf_private( device, true );
}


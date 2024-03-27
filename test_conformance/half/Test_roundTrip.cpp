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
#include <string.h>

#include <algorithm>
#include <cinttypes>

#include "cl_utils.h"
#include "tests.h"
#include "harness/testHarness.h"

int test_roundTrip( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int vectorSize, error;
    uint64_t i, j;
    cl_program  programs[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    cl_kernel   kernels[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    cl_program  doublePrograms[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    cl_kernel   doubleKernels[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t doubleTime[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    uint64_t min_double_time[kVectorSizeCount+kStrangeVectorSizeCount] = {0};
    memset( min_time, -1, sizeof( min_time ) );
    memset( min_double_time, -1, sizeof( min_double_time ) );

    for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        const char *source[] = {
            "__kernel void test( const __global half *in, __global half *out )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],"( vload_half",vector_size_name_extensions[vectorSize],"(i, in),  i, out);\n"
            "}\n"
        };

        const char *doubleSource[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( const __global half *in, __global half *out )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstore_half",vector_size_name_extensions[vectorSize],"( convert_double", vector_size_name_extensions[vectorSize], "( vload_half",vector_size_name_extensions[vectorSize],"(i, in)),  i, out);\n"
            "}\n"
        };

        const char *sourceV3[] = {
            "__kernel void test( const __global half *in, __global half *out,"
            "                    uint extra_last_thread  )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   if(i == last_i && extra_last_thread != 0) { \n"
            "     adjust = 3-extra_last_thread;\n"
            "   }\n"
            "   vstore_half3( vload_half3(i, in-adjust),  i, out-adjust);\n"
            "}\n"
        };

        const char *doubleSourceV3[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( const __global half *in, __global half *out,"
            "                    uint extra_last_thread  )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   size_t last_i = get_global_size(0)-1;\n"
            "   size_t adjust = 0;\n"
            "   if(i == last_i && extra_last_thread != 0) { \n"
            "     adjust = 3-extra_last_thread;\n"
            "   }\n"
            "   vstore_half3( vload_half3(i, in-adjust),  i, out-adjust);\n"
            "}\n"
        };

/*
        const char *sourceV3aligned[] = {
            "__kernel void test( const __global half *in, __global half *out )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half3( vloada_half3(i, in),  i, out);\n"
            "   vstore_half(vload_half(4*i+3, in), 4*i+3, out);\n"
            "}\n"
        };

        const char *doubleSourceV3aligned[] = {
            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
            "__kernel void test( const __global half *in, __global half *out )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   vstorea_half3( vloada_half3(i, in),  i, out);\n"
            "   vstore_half(vload_half(4*i+3, in), 4*i+3, out);\n"
            "}\n"
        };
*/

        if(g_arrVecSizes[vectorSize] == 3) {
            programs[vectorSize] = MakeProgram( device, sourceV3, sizeof( sourceV3) / sizeof( sourceV3[0])  );
            if( NULL == programs[ vectorSize ] )
            {
                gFailCount++;

                return -1;
            }
        } else {
            programs[vectorSize] = MakeProgram( device, source, sizeof( source) / sizeof( source[0])  );
            if( NULL == programs[ vectorSize ] )
            {
                gFailCount++;
                return -1;
            }
        }

        kernels[ vectorSize ] = clCreateKernel( programs[ vectorSize ], "test", &error );
        if( NULL == kernels[vectorSize] )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create kernel. (%d)\n", error );
            return error;
        }

        if( gTestDouble )
        {
            if(g_arrVecSizes[vectorSize] == 3) {
                doublePrograms[vectorSize] = MakeProgram( device, doubleSourceV3, sizeof( doubleSourceV3) / sizeof( doubleSourceV3[0])  );
                if( NULL == doublePrograms[ vectorSize ] )
                {
                    gFailCount++;
                    return -1;
                }
            } else {
                doublePrograms[vectorSize] = MakeProgram( device, doubleSource, sizeof( doubleSource) / sizeof( doubleSource[0])  );
                if( NULL == doublePrograms[ vectorSize ] )
                {
                    gFailCount++;
                    return -1;
                }
            }

            doubleKernels[ vectorSize ] = clCreateKernel( doublePrograms[ vectorSize ], "test", &error );
            if( NULL == doubleKernels[vectorSize] )
            {
                gFailCount++;
                vlog_error( "\t\tFAILED -- Failed to create kernel. (%d)\n", error );
                return error;
            }
        }
    }

    // Figure out how many elements are in a work block
    size_t elementSize = std::max(sizeof(cl_half), sizeof(cl_float));
    size_t blockCount = (size_t)getBufferSize(device) / elementSize; //elementSize is a power of two
    uint64_t lastCase = 1ULL << (8*sizeof(cl_half)); // number of cl_half
    size_t stride = blockCount;

    error = 0;
    uint64_t printMask = (lastCase >> 4) - 1;
    uint32_t count;
    size_t loopCount;

    for( i = 0; i < (uint64_t)lastCase; i += stride )
    {
        count = (uint32_t)std::min((uint64_t)blockCount, lastCase - i);

        //Init the input stream
        uint16_t *p = (uint16_t *)gIn_half;
        for( j = 0; j < count; j++ )
            p[j] = j + i;

        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer_half, CL_TRUE, 0, count * sizeof( cl_half ), gIn_half, 0, NULL, NULL)) )
        {
            vlog_error( "Failure in clWriteArray\n" );
            gFailCount++;
            goto exit;
        }

        //Check the vector lengths
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        { // here we loop through vector sizes -- 3 is last.
            uint32_t pattern = 0xdeaddead;
            memset_pattern4( gOut_half, &pattern, (size_t)getBufferSize(device)/2);

            if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof(cl_half), gOut_half, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clWriteArray\n" );
                gFailCount++;
                goto exit;
            }


            // here is where "3" starts to cause problems.
            error = RunKernel(device, kernels[vectorSize], gInBuffer_half, gOutBuffer_half, numVecs(count, vectorSize, false) ,
                              runsOverBy(count, vectorSize, false) );
            if(error)
            {
                gFailCount++;
                goto exit;
            }

            if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof(cl_half), gOut_half, 0, NULL, NULL)) )
            {
                vlog_error( "Failure in clReadArray\n" );
                gFailCount++;
                goto exit;
            }

            if( (memcmp( gOut_half, gIn_half, count * sizeof(cl_half))) )
            {
                uint16_t *u1 = (uint16_t *)gOut_half;
                uint16_t *u2 = (uint16_t *)gIn_half;
                for( j = 0; j < count; j++ )
                {
                    if( u1[j] != u2[j] )
                    {
                        uint16_t abs1 = u1[j] & 0x7fff;
                        uint16_t abs2 = u2[j] & 0x7fff;
                        if( abs1 > 0x7c00 && abs2 > 0x7c00 )
                            continue; //any NaN is okay if NaN is input

                        // if reference result is sub normal, test if the output is flushed to zero
                        if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                            continue;

                        vlog_error("%" PRId64 ") (of %u)  Failure at 0x%4.4x:  "
                                   "0x%4.4x   vector_size = %d \n",
                                   j, count, u2[j], u1[j],
                                   (g_arrVecSizes[vectorSize]));
                        gFailCount++;
                        error = -1;
                        goto exit;
                    }
                }
            }

            if( gTestDouble )
            {
                memset_pattern4( gOut_half, &pattern, (size_t)getBufferSize(device)/2);
                if( (error = clEnqueueWriteBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof(cl_half), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clWriteArray\n" );
                    gFailCount++;
                    goto exit;
                }


                if( (error = RunKernel(device, doubleKernels[vectorSize], gInBuffer_half, gOutBuffer_half, numVecs(count, vectorSize, false) ,
                                       runsOverBy(count, vectorSize, false) ) ) )
                {
                    gFailCount++;
                    goto exit;
                }

                if( (error = clEnqueueReadBuffer(gQueue, gOutBuffer_half, CL_TRUE, 0, count * sizeof(cl_half), gOut_half, 0, NULL, NULL)) )
                {
                    vlog_error( "Failure in clReadArray\n" );
                    gFailCount++;
                    goto exit;
                }

                if( (memcmp( gOut_half, gIn_half, count * sizeof(cl_half))) )
                {
                    uint16_t *u1 = (uint16_t *)gOut_half;
                    uint16_t *u2 = (uint16_t *)gIn_half;
                    for( j = 0; j < count; j++ )
                    {
                        if( u1[j] != u2[j] )
                        {
                            uint16_t abs1 = u1[j] & 0x7fff;
                            uint16_t abs2 = u2[j] & 0x7fff;
                            if( abs1 > 0x7c00 && abs2 > 0x7c00 )
                                continue; //any NaN is okay if NaN is input

                            // if reference result is sub normal, test if the output is flushed to zero
                            if( IsHalfSubnormal(u2[j]) && ( (u1[j] == 0) || (u1[j] == 0x8000) ) )
                                continue;

                            vlog_error(
                                "%" PRId64 ") Failure at 0x%4.4x:  0x%4.4x   "
                                "vector_size = %d (double precision)\n",
                                j, u2[j], u1[j], (g_arrVecSizes[vectorSize]));
                            gFailCount++;
                            error = -1;
                            goto exit;
                        }
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

    loopCount = 100;
    if( gReportTimes )
    {
        //Run again for timing
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
        {
            uint64_t bestTime = -1ULL;

            for( j = 0; j < loopCount; j++ )
            {
                uint64_t startTime = ReadTime();
                if( (error = RunKernel(device, kernels[vectorSize], gInBuffer_half, gOutBuffer_half,numVecs(count, vectorSize, false) ,
                                       runsOverBy(count, vectorSize, false)) ) )
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
                if( currentTime < bestTime )
                    bestTime = currentTime;
                time[ vectorSize ] += currentTime;
            }
            if( bestTime < min_time[ vectorSize ] )
                min_time[ vectorSize ] = bestTime;

            if( gTestDouble )
            {
                bestTime = -1ULL;
                for( j = 0; j < loopCount; j++ )
                {
                    uint64_t startTime = ReadTime();
                    if( (error = RunKernel(device, doubleKernels[vectorSize], gInBuffer_half, gOutBuffer_half, numVecs(count, vectorSize, false) ,
                                           runsOverBy(count, vectorSize, false)) ) )
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
                    if( currentTime < bestTime )
                        bestTime = currentTime;
                    doubleTime[ vectorSize ] += currentTime;
                }
                if( bestTime < min_double_time[ vectorSize ] )
                    min_double_time[ vectorSize ] = bestTime;
            }
        }
    }

    if( gReportTimes )
    {
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0, "average us/elem", "roundTrip avg. (vector size: %d)", (g_arrVecSizes[vectorSize]) );
        for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
            vlog_perf( SubtractTime( min_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0, "best us/elem", "roundTrip best (vector size: %d)", (g_arrVecSizes[vectorSize])  );
        if( gTestDouble )
        {
            for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( doubleTime[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) (count * loopCount), 0, "average us/elem (double)", "roundTrip avg. d (vector size: %d)", (g_arrVecSizes[vectorSize])  );
            for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
                vlog_perf( SubtractTime( min_double_time[ vectorSize ], 0 ) * 1e6 * gDeviceFrequency * gComputeDevices / (double) count, 0, "best us/elem (double)", "roundTrip best d (vector size: %d)", (g_arrVecSizes[vectorSize]) );
        }
    }

exit:
    //clean up
    for( vectorSize = kMinVectorSize; vectorSize < kLastVectorSizeToTest; vectorSize++)
    {
        clReleaseKernel( kernels[ vectorSize ] );
        clReleaseProgram( programs[ vectorSize ] );
        if( gTestDouble )
        {
            clReleaseKernel( doubleKernels[ vectorSize ] );
            clReleaseProgram( doublePrograms[ vectorSize ] );
        }
    }

    return error;
}



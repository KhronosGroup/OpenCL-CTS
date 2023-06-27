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
#include "harness/conversions.h"
#include "harness/stringHelpers.h"
#include "harness/typeWrappers.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "procs.h"

// clang-format off

static char extension[128] = { 0 };
static char strLoad[128] = { 0 };
static char strStore[128] = { 0 };
static const char *regLoad = "as_%s%s(src[tid]);\n";
static const char *v3Load = "as_%s%s(vload3(tid,(__global %s*)src));\n";
static const char *regStore = "dst[tid] = tmp;\n";
static const char *v3Store = "vstore3(tmp, tid, (__global %s*)dst);\n";

static const char* astype_kernel_pattern[] = {
extension,
"__kernel void test_fn( __global %s%s *src, __global %s%s *dst )\n"
"{\n"
"    int tid = get_global_id( 0 );\n",
"    %s%s tmp = ", strLoad,
"    ", strStore,
"}\n"};

// clang-format on

int test_astype_set( cl_device_id device, cl_context context, cl_command_queue queue, ExplicitType inVecType, ExplicitType outVecType,
                    unsigned int vecSize, unsigned int outVecSize,
                    int numElements )
{
    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 2 ];

    size_t threads[ 1 ], localThreads[ 1 ];
    size_t typeSize = get_explicit_type_size( inVecType );
    size_t outTypeSize = get_explicit_type_size(outVecType);
    char sizeNames[][ 3 ] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    MTdataHolder d(gRandomSeed);

    std::ostringstream sstr;
    if (outVecType == kDouble || inVecType == kDouble)
        sstr << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (outVecType == kHalf || inVecType == kHalf)
        sstr << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

    strcpy(extension, sstr.str().c_str());

    if (vecSize == 3)
        std::snprintf(strLoad, sizeof(strLoad), v3Load,
                      get_explicit_type_name(outVecType), sizeNames[outVecSize],
                      get_explicit_type_name(inVecType));
    else
        std::snprintf(strLoad, sizeof(strLoad), regLoad,
                      get_explicit_type_name(outVecType),
                      sizeNames[outVecSize]);

    if (outVecSize == 3)
        std::snprintf(strStore, sizeof(strStore), v3Store,
                      get_explicit_type_name(outVecType));
    else
        std::snprintf(strStore, sizeof(strStore), "%s", regStore);

    auto str =
        concat_kernel(astype_kernel_pattern,
                      sizeof(astype_kernel_pattern) / sizeof(const char *));
    std::string kernelSource =
        str_sprintf(str, get_explicit_type_name(inVecType), sizeNames[vecSize],
                    get_explicit_type_name(outVecType), sizeNames[outVecSize],
                    get_explicit_type_name(outVecType), sizeNames[outVecSize]);

    const char *ptr = kernelSource.c_str();
    error = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "test_fn" );
    test_error( error, "Unable to create testing kernel" );

    // Create some input values
    size_t inBufferSize = sizeof(char)* numElements * get_explicit_type_size( inVecType ) * vecSize;
    std::vector<char> inBuffer(inBufferSize);
    size_t outBufferSize = sizeof(char)* numElements * get_explicit_type_size( outVecType ) *outVecSize;
    std::vector<char> outBuffer(outBufferSize);

    generate_random_data(inVecType, numElements * vecSize, d,
                         &inBuffer.front());

    // Create I/O streams and set arguments
    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, inBufferSize,
                                &inBuffer.front(), &error);
    test_error( error, "Unable to create I/O stream" );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE, outBufferSize, NULL, &error );
    test_error( error, "Unable to create I/O stream" );

    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel argument" );


    // Run the kernel
    threads[ 0 ] = numElements;
    error = get_max_common_work_group_size( context, kernel, threads[ 0 ], &localThreads[ 0 ] );
    test_error( error, "Unable to get group size to run with" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to run kernel" );

    // Get the results and compare
    // The beauty is that astype is supposed to return the bit pattern as a different type, which means
    // the output should have the exact same bit pattern as the input. No interpretation necessary!
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, outBufferSize,
                                &outBuffer.front(), 0, NULL, NULL);
    test_error( error, "Unable to read results" );

    char *expected = &inBuffer.front();
    char *actual = &outBuffer.front();
    size_t compSize = typeSize*vecSize;
    if(outTypeSize*outVecSize < compSize) {
        compSize = outTypeSize*outVecSize;
    }

    if(outVecSize == 4 && vecSize == 3)
    {
        // as_type4(vec3) should compile but produce undefined results??
        return 0;
    }

    if(outVecSize != 3 && vecSize != 3 && outVecSize != vecSize)
    {
        // as_typen(vecm) should compile and run but produce
        // implementation-defined results for m != n
        // and n*sizeof(type) = sizeof(vecm)
        return 0;
    }

    for( int i = 0; i < numElements; i++ )
    {
        if( memcmp( expected, actual, compSize ) != 0 )
        {
            char expectedString[ 1024 ], actualString[ 1024 ];
            log_error( "ERROR: Data sample %d of %d for as_%s%d( %s%d ) did not validate (expected {%s}, got {%s})\n",
                      (int)i, (int)numElements, get_explicit_type_name( outVecType ), vecSize, get_explicit_type_name( inVecType ), vecSize,
                      GetDataVectorString( expected, typeSize, vecSize, expectedString ),
                      GetDataVectorString( actual, typeSize, vecSize, actualString ) );
            log_error("Src is :\n%s\n----\n%d threads %d localthreads\n",
                      kernelSource.c_str(), (int)threads[0],
                      (int)localThreads[0]);
            return 1;
        }
        expected += typeSize * vecSize;
        actual += outTypeSize * outVecSize;
    }

    return 0;
}

int test_astype(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Note: although casting to different vector element sizes that match the same size (i.e. short2 -> char4) is
    // legal in OpenCL 1.0, the result is dependent on the device it runs on, which means there's no actual way
    // for us to verify what is "valid". So the only thing we can test are types that match in size independent
    // of the element count (char -> uchar, etc)
    const std::vector<ExplicitType> vecTypes = { kChar,   kUChar, kShort,
                                                 kUShort, kInt,   kUInt,
                                                 kLong,   kULong, kFloat,
                                                 kHalf,   kDouble };
    const unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int inTypeIdx, outTypeIdx, sizeIdx, outSizeIdx;
    size_t inTypeSize, outTypeSize;
    int error = 0;

    bool fp16Support = is_extension_available(device, "cl_khr_fp16");
    bool fp64Support = is_extension_available(device, "cl_khr_fp64");

    auto skip_type = [&](ExplicitType et) {
        if ((et == kLong || et == kULong) && !gHasLong)
            return true;
        else if (et == kDouble && !fp64Support)
            return true;
        else if (et == kHalf && !fp16Support)
            return true;
        return false;
    };

    for (inTypeIdx = 0; inTypeIdx < vecTypes.size(); inTypeIdx++)
    {
        inTypeSize = get_explicit_type_size(vecTypes[inTypeIdx]);

        if (skip_type(vecTypes[inTypeIdx])) continue;

        for (outTypeIdx = 0; outTypeIdx < vecTypes.size(); outTypeIdx++)
        {
            outTypeSize = get_explicit_type_size(vecTypes[outTypeIdx]);

            if (skip_type(vecTypes[outTypeIdx])) continue;

            // change this check
            if( inTypeIdx == outTypeIdx ) {
                continue;
            }

            log_info( " (%s->%s)\n", get_explicit_type_name( vecTypes[ inTypeIdx ] ), get_explicit_type_name( vecTypes[ outTypeIdx ] ) );
            fflush( stdout );

            for( sizeIdx = 0; vecSizes[ sizeIdx ] != 0; sizeIdx++ )
            {
                for(outSizeIdx = 0; vecSizes[outSizeIdx] != 0; outSizeIdx++)
                {
                    if(vecSizes[sizeIdx]*inTypeSize !=
                       vecSizes[outSizeIdx]*outTypeSize )
                    {
                        continue;
                    }
                    error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], vecSizes[ sizeIdx ], vecSizes[outSizeIdx], n_elems );
                }
            }
            if(get_explicit_type_size(vecTypes[inTypeIdx]) ==
               get_explicit_type_size(vecTypes[outTypeIdx])) {
                // as_type3(vec4) allowed, as_type4(vec3) not allowed
                error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], 3, 4, n_elems );
                error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], 4, 3, n_elems );
            }

        }
    }
    return error;
}



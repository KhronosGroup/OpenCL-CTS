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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

struct work_item_data
{
    cl_uint workDim;
    cl_uint globalSize[ 3 ];
    cl_uint globalID[ 3 ];
    cl_uint localSize[ 3 ];
    cl_uint localID[ 3 ];
    cl_uint numGroups[ 3 ];
    cl_uint groupID[ 3 ];
};

static const char *workItemKernelCode =
"typedef struct {\n"
"    uint workDim;\n"
"    uint globalSize[ 3 ];\n"
"    uint globalID[ 3 ];\n"
"    uint localSize[ 3 ];\n"
"    uint localID[ 3 ];\n"
"    uint numGroups[ 3 ];\n"
"    uint groupID[ 3 ];\n"
" } work_item_data;\n"
"\n"
"__kernel void sample_kernel( __global work_item_data *outData )\n"
"{\n"
"    int id = get_global_id(0);\n"
"   outData[ id ].workDim = (uint)get_work_dim();\n"
"    for( uint i = 0; i < get_work_dim(); i++ )\n"
"   {\n"
"       outData[ id ].globalSize[ i ] = (uint)get_global_size( i );\n"
"       outData[ id ].globalID[ i ] = (uint)get_global_id( i );\n"
"       outData[ id ].localSize[ i ] = (uint)get_local_size( i );\n"
"       outData[ id ].localID[ i ] = (uint)get_local_id( i );\n"
"       outData[ id ].numGroups[ i ] = (uint)get_num_groups( i );\n"
"       outData[ id ].groupID[ i ] = (uint)get_group_id( i );\n"
"   }\n"
"}";

#define NUM_TESTS 1

int test_work_item_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outData;
    std::vector<work_item_data> testData(10240);
    size_t threads[3], localThreads[3];
    MTdata d;


    error = create_single_kernel_helper( context, &program, &kernel, 1, &workItemKernelCode, "sample_kernel" );
    test_error( error, "Unable to create testing kernel" );

    outData =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(work_item_data) * testData.size(), NULL, &error);
    test_error( error, "Unable to create output buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( outData ), &outData );
    test_error( error, "Unable to set kernel arg" );

    d = init_genrand( gRandomSeed );
    for( size_t dim = 1; dim <= 3; dim++ )
    {
        for( int i = 0; i < NUM_TESTS; i++  )
        {
            for( size_t j = 0; j < dim; j++ )
            {
                // All of our thread sizes should be within the max local sizes, since they're all <= 20
                threads[ j ] = (size_t)random_in_range( 1, 20, d );
                localThreads[ j ] = threads[ j ] / (size_t)random_in_range( 1, (int)threads[ j ], d );
                while( localThreads[ j ] > 1 && ( threads[ j ] % localThreads[ j ] != 0 ) )
                    localThreads[ j ]--;

                // Hack for now: localThreads > 1 are iffy
                localThreads[ j ] = 1;
            }
            error = clEnqueueNDRangeKernel( queue, kernel, (cl_uint)dim, NULL, threads, localThreads, 0, NULL, NULL );
            test_error( error, "Unable to run kernel" );

            error =
                clEnqueueReadBuffer(queue, outData, CL_TRUE, 0,
                                    sizeof(work_item_data) * testData.size(),
                                    testData.data(), 0, NULL, NULL);
            test_error( error, "Unable to read results" );

            // Validate
            for( size_t q = 0; q < threads[0]; q++ )
            {
                // We can't really validate the actual value of each one, but we can validate that they're within a sane range
                if( testData[ q ].workDim != (cl_uint)dim )
                {
                    log_error( "ERROR: get_work_dim() did not return proper value for %d dimensions (expected %d, got %d)\n", (int)dim, (int)dim, (int)testData[ q ].workDim );
                    free_mtdata(d);
                    return -1;
                }
                for( size_t j = 0; j < dim; j++ )
                {
                    if( testData[ q ].globalSize[ j ] != (cl_uint)threads[ j ] )
                    {
                        log_error( "ERROR: get_global_size(%d) did not return proper value for %d dimensions (expected %d, got %d)\n",
                                    (int)j, (int)dim, (int)threads[ j ], (int)testData[ q ].globalSize[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                    if (testData[q].globalID[j] >= (cl_uint)threads[j])
                    {
                        log_error( "ERROR: get_global_id(%d) did not return proper value for %d dimensions (max %d, got %d)\n",
                                  (int)j, (int)dim, (int)threads[ j ], (int)testData[ q ].globalID[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                    if( testData[ q ].localSize[ j ] != (cl_uint)localThreads[ j ] )
                    {
                        log_error( "ERROR: get_local_size(%d) did not return proper value for %d dimensions (expected %d, got %d)\n",
                                  (int)j, (int)dim, (int)localThreads[ j ], (int)testData[ q ].localSize[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                    if (testData[q].localID[j] >= (cl_uint)localThreads[j])
                    {
                        log_error( "ERROR: get_local_id(%d) did not return proper value for %d dimensions (max %d, got %d)\n",
                                  (int)j, (int)dim, (int)localThreads[ j ], (int)testData[ q ].localID[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                    size_t groupCount = ( threads[ j ] + localThreads[ j ] - 1 ) / localThreads[ j ];
                    if( testData[ q ].numGroups[ j ] != (cl_uint)groupCount )
                    {
                        log_error( "ERROR: get_num_groups(%d) did not return proper value for %d dimensions (expected %d with global dim %d and local dim %d, got %d)\n",
                                  (int)j, (int)dim, (int)groupCount, (int)threads[ j ], (int)localThreads[ j ], (int)testData[ q ].numGroups[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                    if (testData[q].groupID[j] >= (cl_uint)groupCount)
                    {
                        log_error( "ERROR: get_group_id(%d) did not return proper value for %d dimensions (max %d, got %d)\n",
                                  (int)j, (int)dim, (int)groupCount, (int)testData[ q ].groupID[ j ] );
                        free_mtdata(d);
                        return -1;
                    }
                }
            }
        }
    }

    free_mtdata(d);
    return 0;
}



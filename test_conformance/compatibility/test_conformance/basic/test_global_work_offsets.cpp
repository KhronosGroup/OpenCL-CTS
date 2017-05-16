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
#include "procs.h"
#include <ctype.h>


const char *work_offset_test[] = {
    "__kernel void test( __global int * outputID_A, \n"
    "                        __global int * outputID_B, __global int * outputID_C )\n"
    "{\n"
    "    size_t id0 = get_local_id( 0 ) + get_group_id( 0 ) * get_local_size( 0 );\n"
    "    size_t id1 = get_local_id( 1 ) + get_group_id( 1 ) * get_local_size( 1 );\n"
    "    size_t id2 = get_local_id( 2 ) + get_group_id( 2 ) * get_local_size( 2 );\n"
    "    size_t id = ( id2 * get_global_size( 0 ) * get_global_size( 1 ) ) + ( id1 * get_global_size( 0 ) ) + id0;\n"
    "\n"
    "    outputID_A[ id ] = get_global_id( 0 );\n"
    "    outputID_B[ id ] = get_global_id( 1 );\n"
    "    outputID_C[ id ] = get_global_id( 2 );\n"
    "}\n"
    };

#define MAX_TEST_ITEMS 16 * 16 * 16
#define NUM_TESTS 16
#define MAX_OFFSET 256

#define CHECK_RANGE( v, m, c ) \
    if( ( v >= (cl_int)m ) || ( v < 0 ) ) \
    {    \
        log_error( "ERROR: ouputID_%c[%lu]: %d is < 0 or >= %lu\n", c, i, v, m ); \
        return -1;    \
    }

int check_results( size_t threads[], size_t offsets[], cl_int outputA[], cl_int outputB[], cl_int outputC[] )
{
    size_t offsettedSizes[ 3 ] = { threads[ 0 ] + offsets[ 0 ], threads[ 1 ] + offsets[ 1 ], threads[ 2 ] + offsets[ 2 ] };
    size_t limit = threads[ 0 ] * threads[ 1 ] * threads[ 2 ];

    static char counts[ MAX_OFFSET + 32 ][ MAX_OFFSET + 16 ][ MAX_OFFSET + 16 ];
    memset( counts, 0, sizeof( counts ) );

    for( size_t i = 0; i < limit; i++ )
    {
        // Check ranges first
        CHECK_RANGE( outputA[ i ], offsettedSizes[ 0 ], 'A' )
        CHECK_RANGE( outputB[ i ], offsettedSizes[ 1 ], 'B' )
        CHECK_RANGE( outputC[ i ], offsettedSizes[ 2 ], 'C' )

        // Now set the value in the map
        counts[ outputA[ i ] ][ outputB[ i ] ][ outputC[ i ] ]++;
    }

    // Now check the map
    int missed = 0, multiple = 0, errored = 0, corrected = 0;
    for( size_t x = 0; x < offsettedSizes[ 0 ]; x++ )
    {
        for( size_t y = 0; y < offsettedSizes[ 1 ]; y++ )
        {
            for( size_t z = 0; z < offsettedSizes[ 2 ]; z++ )
            {
                const char * limitMsg = " (further errors of this type suppressed)";
                if( ( x >= offsets[ 0 ] ) && ( y >= offsets[ 1 ] ) && ( z >= offsets[ 2 ] ) )
                {
                    if( counts[ x ][ y ][ z ] < 1 )
                    {
                        if( missed < 3 )
                            log_error( "ERROR: Map value (%ld,%ld,%ld) was missed%s\n", x, y, z, ( missed == 2 ) ? limitMsg : "" );
                        missed++;
                    }
                    else if( counts[ x ][ y ][ z ] > 1 )
                    {
                        if( multiple < 3 )
                            log_error( "ERROR: Map value (%ld,%ld,%ld) was returned multiple times%s\n", x, y, z, ( multiple == 2 ) ? limitMsg : "" );
                        multiple++;
                    }
                }
                else
                {
                    if( counts[ x ][ y ][ z ] > 0 )
                    {
                        if( errored < 3 )
                            log_error( "ERROR: Map value (%ld,%ld,%ld) was erroneously returned%s\n", x, y, z, ( errored == 2 ) ? limitMsg : "" );
                        errored++;
                    }
                }
                    }
                }
                    }

    if( missed || multiple || errored )
    {
        size_t diffs[3] = { ( offsets[ 0 ] > threads[ 0 ] ? 0 : threads[ 0 ] - offsets[ 0 ] ),
                        ( offsets[ 1 ] > threads[ 1 ] ? 0 : threads[ 1 ] - offsets[ 1 ] ),
                        ( offsets[ 2 ] > threads[ 2 ] ? 0 : threads[ 2 ] - offsets[ 2 ] ) };
            int diff = (int)( ( threads[ 0 ] - diffs[ 0 ] ) * ( threads[ 1 ] - diffs[ 1 ] ) * ( threads[ 2 ] - diffs[ 2 ] ) );

        if( ( multiple == 0 ) && ( missed == diff ) && ( errored == diff ) )
            log_error( "ERROR: Global work offset values are not being respected by get_global_id()\n" );
        else
            log_error( "ERROR: Global work offset values did not function as expected (%d missed, %d reported multiple times, %d erroneously hit)\n",
                            missed, multiple, errored );
    }
    return ( missed | multiple | errored | corrected );
}

int test_global_work_offsets(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 7 ];

    int error;
    size_t    threads[] = {1,1,1}, localThreads[] = {1,1,1}, offsets[] = {0,0,0};
    cl_int outputA[ MAX_TEST_ITEMS ], outputB[ MAX_TEST_ITEMS ], outputC[ MAX_TEST_ITEMS ];


    // Create the kernel
    if( create_single_kernel_helper( context, &program, &kernel, 1, work_offset_test, "test" ) != 0 )
    {
        return -1;
    }

    //// Create some output streams

    // Use just one output array to init them all (no need to init every single stack storage here)
    memset( outputA, 0xff, sizeof( outputA ) );
    for( int i = 0; i < 3; i++ )
    {
        streams[ i ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sizeof(outputA), outputA, &error );
        test_error( error, "Unable to create output array" );
    }

    // Run a few different times
    MTdata seed = init_genrand( gRandomSeed );
    for( int test = 0; test < NUM_TESTS; test++ )
    {
        // Choose a random combination of thread size, but in total less than MAX_TEST_ITEMS
        threads[ 0 ] = random_in_range( 1, 32, seed );
        threads[ 1 ] = random_in_range( 1, 16, seed );
        threads[ 2 ] = random_in_range( 1, MAX_TEST_ITEMS / (int)( threads[ 0 ] * threads[ 1 ] ), seed );

        // Make sure we get the local thread count right
        error = get_max_common_3D_work_group_size( context, kernel, threads, localThreads );
        test_error( error, "Unable to determine local work group sizes" );

        // Randomize some offsets
        for( int j = 0; j < 3; j++ )
            offsets[ j ] = random_in_range( 0, MAX_OFFSET, seed );

        log_info( "\tTesting %ld,%ld,%ld (%ld,%ld,%ld) with offsets (%ld,%ld,%ld)...\n",
                 threads[ 0 ], threads[ 1 ], threads[ 2 ], localThreads[ 0 ], localThreads[ 1 ], localThreads[ 2 ],
                 offsets[ 0 ], offsets[ 1 ], offsets[ 2 ] );

        // Now set up and run
        for( int i = 0; i < 3; i++ )
        {
            error = clSetKernelArg( kernel, i, sizeof( streams[i] ), &streams[i] );
            test_error( error, "Unable to set indexed kernel arguments" );
        }

        error = clEnqueueNDRangeKernel( queue, kernel, 3, offsets, threads, localThreads, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        // Read our results back now
        cl_int * resultBuffers[] = { outputA, outputB, outputC };
        for( int i = 0; i < 3; i++ )
        {
            error = clEnqueueReadBuffer( queue, streams[ i ], CL_TRUE, 0, sizeof( outputA ), resultBuffers[ i ], 0, NULL, NULL );
            test_error( error, "Unable to get result data" );
        }

        // Now we need to check the results. The outputs should have one entry for each possible ID,
        // but they won't be in order, so we need to construct a count map to determine what we got
        if( check_results( threads, offsets, outputA, outputB, outputC ) )
        {
            log_error( "\t(Test failed for global dim %ld,%ld,%ld, local dim %ld,%ld,%ld, offsets %ld,%ld,%ld)\n",
                      threads[ 0 ], threads[ 1 ], threads[ 2 ], localThreads[ 0 ], localThreads[ 1 ], localThreads[ 2 ],
                      offsets[ 0 ], offsets[ 1 ], offsets[ 2 ] );
            return -1;
        }
    }

    free_mtdata(seed);

    // All done!
    return 0;
}

const char *get_offset_test[] = {
    "__kernel void test( __global int * outOffsets )\n"
    "{\n"
    "    // We use local ID here so we don't have to worry about offsets\n"
    "   // Also note that these should be the same for ALL threads, so we won't worry about contention\n"
    "    outOffsets[ 0 ] = (int)get_global_offset( 0 );\n"
    "    outOffsets[ 1 ] = (int)get_global_offset( 1 );\n"
    "    outOffsets[ 2 ] = (int)get_global_offset( 2 );\n"
    "}\n"
};

int test_get_global_offset(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 1 ];

    int error;
    size_t    threads[] = {1,1,1}, localThreads[] = {1,1,1}, offsets[] = {0,0,0};
    cl_int outOffsets[ 3 ];


    // Create the kernel
    if( create_single_kernel_helper( context, &program, &kernel, 1, get_offset_test, "test" ) != 0 )
    {
        return -1;
    }

    // Create some output streams, and storage for a single control ID
    memset( outOffsets, 0xff, sizeof( outOffsets ) );
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sizeof( outOffsets ), outOffsets, &error );
    test_error( error, "Unable to create control ID buffer" );

    // Run a few different times
    MTdata seed = init_genrand( gRandomSeed );
    for( int test = 0; test < NUM_TESTS; test++ )
    {
        // Choose a random combination of thread size, but in total less than MAX_TEST_ITEMS
        threads[ 0 ] = random_in_range( 1, 32, seed );
        threads[ 1 ] = random_in_range( 1, 16, seed );
        threads[ 2 ] = random_in_range( 1, MAX_TEST_ITEMS / (int)( threads[ 0 ] * threads[ 1 ] ), seed );

        // Make sure we get the local thread count right
        error = get_max_common_3D_work_group_size( context, kernel, threads, localThreads );
        test_error( error, "Unable to determine local work group sizes" );

        // Randomize some offsets
        for( int j = 0; j < 3; j++ )
            offsets[ j ] = random_in_range( 0, MAX_OFFSET, seed );

        log_info( "\tTesting %ld,%ld,%ld (%ld,%ld,%ld) with offsets (%ld,%ld,%ld)...\n",
                 threads[ 0 ], threads[ 1 ], threads[ 2 ], localThreads[ 0 ], localThreads[ 1 ], localThreads[ 2 ],
                 offsets[ 0 ], offsets[ 1 ], offsets[ 2 ] );

        // Now set up and run
        error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
        test_error( error, "Unable to set indexed kernel arguments" );

        error = clEnqueueNDRangeKernel( queue, kernel, 3, offsets, threads, localThreads, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        // Read our results back now
        error = clEnqueueReadBuffer( queue, streams[ 0 ], CL_TRUE, 0, sizeof( outOffsets ), outOffsets, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );

        // And check!
        int errors = 0;
        for( int j = 0; j < 3; j++ )
        {
            if( outOffsets[ j ] != (cl_int)offsets[ j ] )
            {
                log_error( "ERROR: get_global_offset( %d ) did not return expected value (expected %ld, got %d)\n", j, offsets[ j ], outOffsets[ j ] );
                errors++;
            }
        }
        if( errors > 0 )
            return errors;
    }
    free_mtdata(seed);

    // All done!
    return 0;
}


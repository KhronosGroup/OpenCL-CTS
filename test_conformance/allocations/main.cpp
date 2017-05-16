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
#include "testBase.h"

#include "allocation_functions.h"
#include "allocation_fill.h"
#include "allocation_execute.h"
#include "../../test_common/harness/testHarness.h"
#include "../../test_common/harness/parseParameters.h"
#include <time.h>

typedef long long unsigned llu;

cl_device_id g_device_id;
cl_device_type g_device_type = CL_DEVICE_TYPE_DEFAULT;
clContextWrapper g_context;
clCommandQueueWrapper g_queue;
int g_repetition_count = 1;
int g_tests_to_run = 0;
int g_reduction_percentage = 100;
int g_write_allocations = 1;
int g_multiple_allocations = 0;
int g_execute_kernel = 1;

cl_uint checksum;

void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [single|multiple] [numReps] [reduction%%] allocType\n", execName );
    log_info( "Where:\n" );
    log_info( "\tsingle - Tests using a single allocation as large as possible\n" );
    log_info( "\tmultiple - Tests using as many allocations as possible\n" );
    log_info( "\n" );
    log_info( "\tnumReps - Optional integer specifying the number of repetitions to run and average the result (defaults to 1)\n" );
    log_info( "\treduction%% - Optional integer, followed by a %% sign, that acts as a multiplier for the target amount of memory.\n" );
    log_info( "\t              Example: target amount of 512MB and a reduction of 75%% will result in a target of 384MB.\n" );
    log_info( "\n" );
    log_info( "\tallocType - Allocation type to test with. Can be one of the following:\n" );
    log_info( "\t\tbuffer\n");
    log_info( "\t\timage2d_read\n");
    log_info( "\t\timage2d_write\n");
    log_info( "\t\tbuffer_non_blocking\n");
    log_info( "\t\timage2d_read_non_blocking\n");
    log_info( "\t\timage2d_write_non_blocking\n");
    log_info( "\t\tall (runs all of the above in sequence)\n" );
    log_info( "\tdo_not_force_fill - Disable explicitly write data to all memory objects after creating them.\n" );
    log_info( "\t Without this, the kernel execution can not verify its checksum.\n" );
    log_info( "\tdo_not_execute - Disable executing a kernel that accesses all of the memory objects.\n" );
}


int init_cl() {
    cl_platform_id platform;
    int error;

    error = clGetPlatformIDs(1, &platform, NULL);
    test_error(error, "clGetPlatformIDs failed");

    error = clGetDeviceIDs(platform, g_device_type, 1, &g_device_id, NULL);
    test_error(error, "clGetDeviceIDs failed");

    /* Create a context */
    g_context = clCreateContext( NULL, 1, &g_device_id, notify_callback, NULL, &error );
    test_error(error, "clCreateContext failed");

    /* Create command queue */
    g_queue = clCreateCommandQueueWithProperties( g_context, g_device_id, 0, &error );
    test_error(error, "clCreateCommandQueue failed");

    return error;
}


int main(int argc, const char *argv[])
{
    int error;
    int count;
    cl_mem mems[MAX_NUMBER_TO_ALLOCATE];
    cl_ulong max_individual_allocation_size, global_mem_size;
    char            str[ 128 ],  *endPtr;
    int r;
    int number_of_mems_used;
    int failure_counts = 0;
    int test, test_to_run = 0;
    int randomize = 0;
    size_t final_size, max_size, current_test_size;

    test_start();

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        test_finish();
        return -1;
    }

    // Parse arguments
    checkDeviceTypeOverride( &g_device_type );
    for( int i = 1; i < argc; i++ )
    {
        strncpy( str, argv[ i ], sizeof( str ) - 1 );

        if( strcmp( str, "cpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_CPU" ) == 0 )
            g_device_type = CL_DEVICE_TYPE_CPU;
        else if( strcmp( str, "gpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_GPU" ) == 0 )
            g_device_type = CL_DEVICE_TYPE_GPU;
        else if( strcmp( str, "accelerator" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            g_device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( str, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            g_device_type = CL_DEVICE_TYPE_DEFAULT;

        else if( strcmp( str, "multiple" ) == 0 )
            g_multiple_allocations = 1;
        else if( strcmp( str, "randomize" ) == 0 )
            randomize = 1;
        else if( strcmp( str, "single" ) == 0 )
            g_multiple_allocations = 0;

        else if( ( r = (int)strtol( str, &endPtr, 10 ) ) && ( endPtr != str ) && ( *endPtr == 0 ) )
        {
            // By spec, that means the entire string was an integer, so take it as a repetition count
            g_repetition_count = r;
        }

        else if( strcmp( str, "all" ) == 0 )
        {
            g_tests_to_run = BUFFER | IMAGE_READ | IMAGE_WRITE | BUFFER_NON_BLOCKING | IMAGE_READ_NON_BLOCKING | IMAGE_WRITE_NON_BLOCKING;
        }

        else if( strchr( str, '%' ) != NULL )
        {
            // Reduction percentage (let strtol ignore the percentage)
            g_reduction_percentage = (int)strtol( str, NULL, 10 );
        }

        else if( g_tests_to_run == 0 )
        {
            if( strcmp( str, "buffer" ) == 0 )
            {
                g_tests_to_run |= BUFFER;
            }
            else if( strcmp( str, "image2d_read" ) == 0 )
            {
                g_tests_to_run |= IMAGE_READ;
            }
            else if( strcmp( str, "image2d_write" ) == 0 )
            {
                g_tests_to_run |= IMAGE_WRITE;
            }
            else if( strcmp( str, "buffer_non_blocking" ) == 0 )
            {
                g_tests_to_run |= BUFFER_NON_BLOCKING;
            }
            else if( strcmp( str, "image2d_read_non_blocking" ) == 0 )
            {
                g_tests_to_run |= IMAGE_READ_NON_BLOCKING;
            }
            else if( strcmp( str, "image2d_write_non_blocking" ) == 0 )
            {
                g_tests_to_run |= IMAGE_WRITE_NON_BLOCKING;
            }
            if( g_tests_to_run == 0 )
                break;    // Argument is invalid; break to print usage
        }

        else if( strcmp( str, "do_not_force_fill" ) == 0 )
        {
            g_write_allocations = 0;
        }

        else if( strcmp( str, "do_not_execute" ) == 0 )
        {
            g_execute_kernel = 0;
        }

    }

    if( randomize )
    {
        gRandomSeed = (cl_uint) time( NULL );
        log_info( "Random seed: %u.\n", gRandomSeed );
        gReSeed = 1;
    }

    if( g_tests_to_run == 0 )
    {
        // Allocation type was never specified, or one of the arguments was invalid. Print usage and bail
        printUsage( argv[ 0 ] );
        return -1;
    }

    // All ready to go, so set up an environment
    error = init_cl();
    if (error) {
        test_finish();
        return -1;
    }

    if( printDeviceHeader( g_device_id ) != CL_SUCCESS )
    {
        test_finish();
        return -1;
    }


    error = clGetDeviceInfo(g_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_individual_allocation_size), &max_individual_allocation_size, NULL);
    if ( error ) {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_MAX_MEM_ALLOC_SIZE");
        test_finish();
        return -1;
    }
    error = clGetDeviceInfo(g_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    if ( error ) {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_GLOBAL_MEM_SIZE");
        test_finish();
        return -1;
    }

    log_info("Device reports CL_DEVICE_MAX_MEM_ALLOC_SIZE=%llu bytes (%gMB), CL_DEVICE_GLOBAL_MEM_SIZE=%llu bytes (%gMB).\n",
             llu( max_individual_allocation_size ), toMB(max_individual_allocation_size),
             llu( global_mem_size ), toMB(global_mem_size));

    if (global_mem_size > (cl_ulong)SIZE_MAX) {
      global_mem_size = (cl_ulong)SIZE_MAX;
    }

    if( max_individual_allocation_size > global_mem_size )
    {
        log_error( "FAILURE:  CL_DEVICE_MAX_MEM_ALLOC_SIZE (%llu) is greater than the CL_DEVICE_GLOBAL_MEM_SIZE (%llu)\n", llu( max_individual_allocation_size ), llu( global_mem_size ) );
        test_finish();
        return -1;
    }

    // We may need to back off the global_mem_size on unified memory devices to leave room for application and operating system code
    // and associated data in the working set, so we dont start pathologically paging.
    // Check to see if we are a unified memory device
    cl_bool hasUnifiedMemory = CL_FALSE;
    if( ( error = clGetDeviceInfo( g_device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof( hasUnifiedMemory ), &hasUnifiedMemory, NULL )))
    {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_HOST_UNIFIED_MEMORY");
        test_finish();
        return -1;
    }
    // we share unified memory so back off to 1/2 the global memory size.
    if( CL_TRUE == hasUnifiedMemory )
    {
        global_mem_size -= global_mem_size /2;
        log_info( "Device shares memory with the host, so backing off the maximum combined allocation size to be %gMB to avoid rampant paging.\n", toMB( global_mem_size ) );
    }
    else
    {
        // Lets just use 60% of total available memory as framework/driver may not allow using all of it
        // e.g. vram on GPU is used by window server and even for this test, we need some space for context,
        // queue, kernel code on GPU.
        global_mem_size *= 0.60;
    }

    // Pick the baseline size based on whether we are doing a single large or multiple allocations
    if (!g_multiple_allocations) {
        max_size = (size_t)max_individual_allocation_size;
    } else {
        max_size = (size_t)global_mem_size;
    }


    // Adjust based on the percentage
    if (g_reduction_percentage != 100) {
        log_info("NOTE: reducing max allocations to %d%%.\n", g_reduction_percentage);
        max_size = (size_t)((double)max_size * (double)g_reduction_percentage/100.0);
    }

    // Round to nearest MB.
    max_size &= (size_t)(0xFFFFFFFFFF00000ULL);

    log_info("** Target allocation size (rounded to nearest MB) is: %lu bytes (%gMB).\n", max_size, toMB(max_size));

    // Run all the requested tests
    RandomSeed seed( gRandomSeed );
    for (test=0; test<6; test++) {
        if (test == 0) test_to_run = BUFFER;
        if (test == 1) test_to_run = IMAGE_READ;
        if (test == 2) test_to_run = IMAGE_WRITE;
        if (test == 3) test_to_run = BUFFER_NON_BLOCKING;
        if (test == 4) test_to_run = IMAGE_READ_NON_BLOCKING;
        if (test == 5) test_to_run = IMAGE_WRITE_NON_BLOCKING;
        if (!(g_tests_to_run & test_to_run))
            continue;

        // Skip image tests if we don't support images on the device
        if (test > 0 && checkForImageSupport(g_device_id)) {
            log_info("Can not test image allocation because device does not support images.\n");
            continue;
        }

        // This section was added in order to fix a bug in the test
        // If CL_DEVICE_MAX_MEM_ALLOC_SIZE is much grater than CL_DEVICE_IMAGE2D_MAX_WIDTH * CL_DEVICE_IMAGE2D_MAX_HEIGHT
        // The test will fail in image allocations as the size requested for the allocation will be much grater than the maximum size allowed for image
        if ( (test_to_run != BUFFER) && (test_to_run != BUFFER_NON_BLOCKING) ) {
          size_t max_width, max_height;
          cl_ulong max_image2d_size;
          error = clGetDeviceInfo(g_device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( max_width ), &max_width, NULL );
          test_error_abort( error, "clGetDeviceInfo failed for CL_DEVICE_IMAGE2D_MAX_WIDTH");
          error = clGetDeviceInfo(g_device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( max_height ), &max_height, NULL );
          test_error_abort( error, "clGetDeviceInfo failed for CL_DEVICE_IMAGE2D_MAX_HEIGHT");
          max_image2d_size = (cl_ulong)max_height*max_width*4*sizeof(cl_uint);

          if (max_individual_allocation_size > max_image2d_size)
          {
            max_individual_allocation_size = max_image2d_size;
          }
        }

        // Pick the baseline size based on whether we are doing a single large or multiple allocations
    if (!g_multiple_allocations) {
      max_size = (size_t)max_individual_allocation_size;
    } else {
      max_size = (size_t)global_mem_size;
    }

        // Adjust based on the percentage
        if (g_reduction_percentage != 100) {
            log_info("NOTE: reducing max allocations to %d%%.\n", g_reduction_percentage);
            max_size = (size_t)((double)max_size * (double)g_reduction_percentage/100.0);
        }

        // Round to nearest MB.
        max_size &= (size_t)(0xFFFFFFFFFF00000ULL);

        log_info("** Target allocation size (rounded to nearest MB) is: %llu bytes (%gMB).\n", llu( max_size ), toMB(max_size));

        if (test_to_run == BUFFER || test_to_run == BUFFER_NON_BLOCKING) log_info("** Allocating buffer(s) to size %gMB.\n", toMB(max_size));
        else if (test_to_run == IMAGE_READ || test_to_run == IMAGE_READ_NON_BLOCKING) log_info("** Allocating read-only image(s) to size %gMB.\n", toMB(max_size));
        else if (test_to_run == IMAGE_WRITE || test_to_run == IMAGE_WRITE_NON_BLOCKING) log_info("** Allocating write-only image(s) to size %gMB.\n", toMB(max_size));
        else {log_error("Test logic error.\n"); return -1;}

        // Run the test the requested number of times
        for (count = 0; count < g_repetition_count; count++) {
            current_test_size = max_size;
            error = FAILED_TOO_BIG;
            log_info("  => Allocation %d\n", count+1);

            while (error == FAILED_TOO_BIG && current_test_size > max_size/8) {
                // Reset our checksum for each allocation
                checksum = 0;

                // Do the allocation
                error = allocate_size(g_context, &g_queue, g_device_id, g_multiple_allocations, current_test_size, test_to_run, mems, &number_of_mems_used, &final_size, g_write_allocations, seed);

                // If we succeeded and we're supposed to execute a kernel, do so.
                if (error == SUCCEEDED && g_execute_kernel) {
                    log_info("\tExecuting kernel with memory objects.\n");
                    error = execute_kernel(g_context, &g_queue, g_device_id, test_to_run, mems, number_of_mems_used, g_write_allocations);
                }

                // If we failed to allocate more than 1/8th of the requested amount return a failure.
                if (final_size < (size_t)max_size/8) {
                    //          log_error("===> Allocation %d failed to allocate more than 1/8th of the requested size.\n", count+1);
                    failure_counts++;
                }
                // Clean up.
                for (int i=0; i<number_of_mems_used; i++)
                    clReleaseMemObject(mems[i]);

                if (error == FAILED_ABORT) {
                    log_error("  => Allocation %d failed.\n", count+1);
                    failure_counts++;
                }

                if (error == FAILED_TOO_BIG) {
                    current_test_size -= max_size/16;
                    // log_info("\tFailed at this size; trying a smaller size of %gMB.\n", toMB(current_test_size));
                }
            }
            if (error == SUCCEEDED && current_test_size == max_size)
                log_info("\tPASS: Allocation succeeded.\n");
            else if (error == SUCCEEDED && current_test_size > max_size/8)
                log_info("\tPASS: Allocation succeeded at reduced size.\n");
            else {
                log_error("\tFAIL: Allocation failed.\n");
                failure_counts++;
            }
        }
    }

    if (failure_counts)
        log_error("FAILED allocations test.\n");
    else
        log_info("PASSED allocations test.\n");

    test_finish();
    return failure_counts;
}



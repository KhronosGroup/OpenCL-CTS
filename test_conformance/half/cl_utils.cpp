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
#include "cl_utils.h"
#include <stdlib.h>

#if !defined (_WIN32)
#include <sys/mman.h>
#endif

#include "test_config.h"
#include "string.h"
#include "harness/kernelHelpers.h"

#include "harness/testHarness.h"

#define HALF_MIN 1.0p-14


const char *vector_size_name_extensions[kVectorSizeCount+kStrangeVectorSizeCount] = { "", "2", "4", "8", "16", "3" };
const char *vector_size_strings[kVectorSizeCount+kStrangeVectorSizeCount] = { "1", "2", "4", "8", "16", "3" };
const char *align_divisors[kVectorSizeCount+kStrangeVectorSizeCount] = { "1", "2", "4", "8", "16", "4" };
const char *align_types[kVectorSizeCount+kStrangeVectorSizeCount] = { "half", "int", "int2", "int4", "int8", "int2" };


void *gIn_half = NULL;
void *gOut_half = NULL;
void *gOut_half_reference = NULL;
void *gOut_half_reference_double = NULL;
void *gIn_single = NULL;
void *gOut_single = NULL;
void *gOut_single_reference = NULL;
void *gIn_double = NULL;
// void *gOut_double = NULL;
// void *gOut_double_reference = NULL;
cl_mem gInBuffer_half = NULL;
cl_mem gOutBuffer_half = NULL;
cl_mem gInBuffer_single = NULL;
cl_mem gOutBuffer_single = NULL;
cl_mem gInBuffer_double = NULL;
// cl_mem gOutBuffer_double = NULL;

cl_context gContext = NULL;
cl_command_queue gQueue = NULL;
uint32_t gDeviceFrequency = 0;
uint32_t gComputeDevices = 0;
size_t gMaxThreadGroupSize = 0;
size_t gWorkGroupSize = 0;
bool gWimpyMode = false;
int gWimpyReductionFactor = 512;
int gTestDouble = 0;
bool gHostReset = false;

#if defined( __APPLE__ )
int gReportTimes = 1;
#else
int gReportTimes = 0;
#endif

#pragma mark -

test_status InitCL( cl_device_id device )
{
    size_t configSize = sizeof( gComputeDevices );
    int error;

#if MULTITHREAD
    if( (error = clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS,  configSize, &gComputeDevices, NULL )) )
#endif
    gComputeDevices = 1;

    configSize = sizeof( gMaxThreadGroupSize );
    if( (error = clGetDeviceInfo( device, CL_DEVICE_MAX_WORK_GROUP_SIZE, configSize, &gMaxThreadGroupSize,  NULL )) )
        gMaxThreadGroupSize = 1;

    // Use only one-eighth the work group size
    if (gMaxThreadGroupSize > 8)
        gWorkGroupSize = gMaxThreadGroupSize / 8;
    else
        gWorkGroupSize = gMaxThreadGroupSize;

    configSize = sizeof( gDeviceFrequency );
    if( (error = clGetDeviceInfo( device, CL_DEVICE_MAX_CLOCK_FREQUENCY, configSize, &gDeviceFrequency,  NULL )) )
        gDeviceFrequency = 1;

    // Check extensions
    int hasDouble = is_extension_available(device, "cl_khr_fp64");
    gTestDouble ^= hasDouble;

    vlog( "%d compute devices at %f GHz\n", gComputeDevices, (double) gDeviceFrequency / 1000. );
    vlog("Max thread group size is %zu.\n", gMaxThreadGroupSize);

    gContext = clCreateContext( NULL, 1, &device, notify_callback, NULL, &error );
    if( NULL == gContext )
    {
        vlog_error( "clCreateDeviceGroup failed. (%d)\n", error );
        return TEST_FAIL;
    }

    gQueue = clCreateCommandQueue(gContext, device, 0, &error);
    if( NULL == gQueue )
    {
        vlog_error( "clCreateCommandQueue failed. (%d)\n", error );
        return TEST_FAIL;
    }

#if defined( __APPLE__ )
    // FIXME: use clProtectedArray
#endif
    //Allocate buffers
    gIn_half   = malloc( getBufferSize(device)/2  );
    gOut_half = malloc( BUFFER_SIZE/2  );
    gOut_half_reference = malloc( BUFFER_SIZE/2  );
    gOut_half_reference_double = malloc( BUFFER_SIZE/2  );
    gIn_single   = malloc( BUFFER_SIZE );
    gOut_single = malloc( getBufferSize(device)  );
    gOut_single_reference = malloc( getBufferSize(device)  );
    gIn_double   = malloc( 2*BUFFER_SIZE  );
    // gOut_double = malloc( (2*getBufferSize(device))  );
    // gOut_double_reference = malloc( (2*getBufferSize(device))  );

    if ( NULL == gIn_half ||
     NULL == gOut_half ||
     NULL == gOut_half_reference ||
     NULL == gOut_half_reference_double ||
         NULL == gIn_single ||
     NULL == gOut_single ||
     NULL == gOut_single_reference ||
         NULL == gIn_double // || NULL == gOut_double || NULL == gOut_double_reference
         )
        return TEST_FAIL;

    gInBuffer_half = clCreateBuffer(gContext, CL_MEM_READ_ONLY, getBufferSize(device) / 2, NULL, &error);
    if( gInBuffer_half == NULL )
    {
        vlog_error( "clCreateArray failed for input (%d)\n", error );
        return TEST_FAIL;
    }

    gInBuffer_single = clCreateBuffer(gContext, CL_MEM_READ_ONLY, BUFFER_SIZE, NULL, &error );
    if( gInBuffer_single == NULL )
    {
        vlog_error( "clCreateArray failed for input (%d)\n", error );
        return TEST_FAIL;
    }

    gInBuffer_double = clCreateBuffer(gContext, CL_MEM_READ_ONLY, BUFFER_SIZE*2, NULL, &error );
    if( gInBuffer_double == NULL )
    {
        vlog_error( "clCreateArray failed for input (%d)\n", error );
        return TEST_FAIL;
    }

    gOutBuffer_half = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY, BUFFER_SIZE/2, NULL, &error );
    if( gOutBuffer_half == NULL )
    {
        vlog_error( "clCreateArray failed for output (%d)\n", error );
        return TEST_FAIL;
    }

    gOutBuffer_single = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY, getBufferSize(device), NULL, &error );
    if( gOutBuffer_single == NULL )
    {
        vlog_error( "clCreateArray failed for output (%d)\n", error );
        return TEST_FAIL;
    }

#if 0
    gOutBuffer_double = clCreateBuffer(gContext, CL_MEM_WRITE_ONLY, (size_t)(2*getBufferSize(device)), NULL, &error );
    if( gOutBuffer_double == NULL )
    {
        vlog_error( "clCreateArray failed for output (%d)\n", error );
        return TEST_FAIL;
    }
#endif

    char string[16384];
    vlog( "\nCompute Device info:\n" );
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(string), string, NULL);
    vlog( "\tDevice Name: %s\n", string );
    error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(string), string, NULL);
    vlog( "\tVendor: %s\n", string );
    error = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(string), string, NULL);
    vlog( "\tDevice Version: %s\n", string );
    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(string), string, NULL);
    vlog( "\tOpenCL C Version: %s\n", string );
    error = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(string), string, NULL);
    vlog( "\tDriver Version: %s\n", string );
    vlog( "\tProcessing with %d devices\n", gComputeDevices );
    vlog( "\tDevice Frequency: %d MHz\n", gDeviceFrequency );
    vlog( "\tHas double? %s\n", hasDouble ? "YES" : "NO" );
    vlog( "\tTest double? %s\n", gTestDouble ? "YES" : "NO" );

    return TEST_PASS;
}

cl_program MakeProgram( cl_device_id device, const char *source[], int count )
{
    int error;
    int i;

    //create the program
    cl_program program;
    error = create_single_kernel_helper_create_program(gContext, &program, (cl_uint)count, source);
    if( NULL == program )
    {
        vlog_error( "\t\tFAILED -- Failed to create program. (%d)\n", error );
        return NULL;
    }

    // build it
    if( (error = clBuildProgram( program, 1, &device, NULL, NULL, NULL )) )
    {
        size_t  len;
        char    buffer[16384];

        vlog_error("\t\tFAILED -- clBuildProgramExecutable() failed:\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        vlog_error("Log: %s\n", buffer);
        vlog_error("Source :\n");
        for(i = 0; i < count; ++i) {
            vlog_error("%s", source[i]);
        }
        vlog_error("\n");

        clReleaseProgram( program );
        return NULL;
    }

    return program;
}

void ReleaseCL(void)
{
    clReleaseMemObject(gInBuffer_half);
    clReleaseMemObject(gOutBuffer_half);
    clReleaseMemObject(gInBuffer_single);
    clReleaseMemObject(gOutBuffer_single);
    clReleaseMemObject(gInBuffer_double);
    // clReleaseMemObject(gOutBuffer_double);
    clReleaseCommandQueue(gQueue);
    clReleaseContext(gContext);

    free(gIn_half);
    free(gOut_half);
    free(gOut_half_reference);
    free(gOut_half_reference_double);
    free(gIn_single);
    free(gOut_single);
    free(gOut_single_reference);
    free(gIn_double);
}

cl_uint numVecs(cl_uint count, int vectorSizeIdx, bool aligned) {
    if(aligned && g_arrVecSizes[vectorSizeIdx] == 3) {
        return count/4;
    }
    return  (count + g_arrVecSizes[vectorSizeIdx] - 1)/
    ( (g_arrVecSizes[vectorSizeIdx]) );
}

cl_uint runsOverBy(cl_uint count, int vectorSizeIdx, bool aligned) {
    if(aligned || g_arrVecSizes[vectorSizeIdx] != 3) { return -1; }
    return count% (g_arrVecSizes[vectorSizeIdx]);
}

void printSource(const char * src[], int len) {
    int i;
    for(i = 0; i < len; ++i) {
        vlog("%s", src[i]);
    }
}

int RunKernel( cl_device_id device, cl_kernel kernel, void *inBuf, void *outBuf, uint32_t blockCount , int extraArg)
{
    size_t localCount = blockCount;
    size_t wg_size;
    int error;

    error = clSetKernelArg(kernel, 0, sizeof inBuf, &inBuf);
    error |= clSetKernelArg(kernel, 1, sizeof outBuf, &outBuf);

    if(extraArg >= 0) {
        error |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &extraArg);
    }

    if( error )
    {
        vlog_error( "FAILED -- could not set kernel args\n" );
        return -3;
    }

    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof( wg_size ), &wg_size, NULL);
    if (error)
    {
        vlog_error( "FAILED -- could not get kernel work group info\n" );
        return -4;
    }

    wg_size = (wg_size > gWorkGroupSize) ? gWorkGroupSize : wg_size;
    while( localCount % wg_size )
        wg_size--;

    if( (error = clEnqueueNDRangeKernel( gQueue, kernel, 1, NULL, &localCount, &wg_size, 0, NULL, NULL )) )
    {
        vlog_error( "FAILED -- could not execute kernel\n" );
        return -5;
    }

    return 0;
}

#if defined (__APPLE__ )

#include <mach/mach_time.h>

uint64_t ReadTime( void )
{
    return mach_absolute_time();        // returns time since boot.  Ticks have better than microsecond precsion.
}

double SubtractTime( uint64_t endTime, uint64_t startTime )
{
    static double conversion = 0.0;

    if(  0.0 == conversion )
    {
        mach_timebase_info_data_t   info;
        kern_return_t err = mach_timebase_info( &info );
        if( 0 == err )
            conversion = 1e-9 * (double) info.numer / (double) info.denom;
    }

    return (double) (endTime - startTime) * conversion;
}

#elif defined( _WIN32 ) && defined (_MSC_VER)

// functions are defined in compat.h

#else

//
//  Please feel free to substitute your own timing facility here.
//

#warning  Times are meaningless. No timing facility in place for this platform.
uint64_t ReadTime( void )
{
    return 0ULL;
}

// return the difference between two times obtained from ReadTime in seconds
double SubtractTime( uint64_t endTime, uint64_t startTime )
{
    return INFINITY;
}

#endif

size_t getBufferSize(cl_device_id device_id)
{
    static int s_initialized = 0;
    static cl_device_id s_device_id;
    static cl_ulong s_result = 64*1024;

    if(s_initialized == 0 || s_device_id != device_id)
    {
        cl_ulong result, maxGlobalSize;
        cl_int err = clGetDeviceInfo (device_id,
                                      CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                      sizeof(result), (void *)&result,
                                      NULL);
        if(err)
        {
            vlog_error("clGetDeviceInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) failed\n");
            s_result = 64*1024;
            goto exit;
        }
        if (result > BUFFER_SIZE)
            result = BUFFER_SIZE;
        log_info("Using const buffer size 0x%lx (%lu)\n", (unsigned long)result, (unsigned long)result);
        err = clGetDeviceInfo (device_id,
                               CL_DEVICE_GLOBAL_MEM_SIZE,
                               sizeof(maxGlobalSize), (void *)&maxGlobalSize,
                               NULL);
        if(err)
        {
            vlog_error("clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed\n");
            goto exit;
        }
        result = result / 2;
        if(maxGlobalSize < result * 10)
            result = result / 10;
        s_initialized = 1;
        s_device_id = device_id;
        s_result = result;
    }

exit:
    if( s_result > SIZE_MAX )
    {
        vlog_error( "ERROR: clGetDeviceInfo is reporting a CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE larger than addressable memory on the host.\n It seems highly unlikely that this is usable, due to the API design.\n" );
        fflush(stdout);
        abort();
    }

    return (size_t) s_result;
}

cl_ulong getBufferCount(cl_device_id device_id, size_t vecSize, size_t typeSize)
{
    cl_ulong tmp = getBufferSize(device_id);
    if(vecSize == 3)
    {
        return tmp/(cl_ulong)(4*typeSize);
    }
    return tmp/(cl_ulong)(vecSize*typeSize);
}

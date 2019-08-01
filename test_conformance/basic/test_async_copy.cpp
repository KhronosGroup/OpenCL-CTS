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



#include "procs.h"
#include "harness/conversions.h"

static const char *async_global_to_local_kernel =
"%s\n" // optional pragma string
"__kernel void test_fn( const __global %s *src, __global %s *dst, __local %s *localBuffer, int copiesPerWorkgroup, int copiesPerWorkItem )\n"
"{\n"
" int i;\n"
// Zero the local storage first
" for(i=0; i<copiesPerWorkItem; i++)\n"
"     localBuffer[ get_local_id( 0 )*copiesPerWorkItem+i ] = (%s)(%s)0;\n"
// Do this to verify all kernels are done zeroing the local buffer before we try the copy
"    barrier( CLK_LOCAL_MEM_FENCE );\n"
"    event_t event;\n"
"    event = async_work_group_copy( (__local %s*)localBuffer, (__global const %s*)(src+copiesPerWorkgroup*get_group_id(0)), (size_t)copiesPerWorkgroup, 0 );\n"
// Wait for the copy to complete, then verify by manually copying to the dest
"    wait_group_events( 1, &event );\n"
" for(i=0; i<copiesPerWorkItem; i++)\n"
"  dst[ get_global_id( 0 )*copiesPerWorkItem+i ] = localBuffer[ get_local_id( 0 )*copiesPerWorkItem+i ];\n"
"}\n" ;

static const char *async_local_to_global_kernel =
"%s\n" // optional pragma string
"__kernel void test_fn( const __global %s *src, __global %s *dst, __local %s *localBuffer, int copiesPerWorkgroup, int copiesPerWorkItem )\n"
"{\n"
" int i;\n"
// Zero the local storage first
" for(i=0; i<copiesPerWorkItem; i++)\n"
"  localBuffer[ get_local_id( 0 )*copiesPerWorkItem+i ] = (%s)(%s)0;\n"
// Do this to verify all kernels are done zeroing the local buffer before we try the copy
"    barrier( CLK_LOCAL_MEM_FENCE );\n"
" for(i=0; i<copiesPerWorkItem; i++)\n"
"  localBuffer[ get_local_id( 0 )*copiesPerWorkItem+i ] = src[ get_global_id( 0 )*copiesPerWorkItem+i ];\n"
// Do this to verify all kernels are done copying to the local buffer before we try the copy
"    barrier( CLK_LOCAL_MEM_FENCE );\n"
"    event_t event;\n"
"    event = async_work_group_copy((__global %s*)(dst+copiesPerWorkgroup*get_group_id(0)), (__local const %s*)localBuffer, (size_t)copiesPerWorkgroup, 0 );\n"
"    wait_group_events( 1, &event );\n"
"}\n" ;


static const char *prefetch_kernel =
"%s\n" // optional pragma string
"__kernel void test_fn( const __global %s *src, __global %s *dst, __local %s *localBuffer, int copiesPerWorkgroup, int copiesPerWorkItem )\n"
"{\n"
" // Ignore this: %s%s%s\n"
" int i;\n"
" prefetch( (const __global %s*)(src+copiesPerWorkItem*get_global_id(0)), copiesPerWorkItem);\n"
" for(i=0; i<copiesPerWorkItem; i++)\n"
"  dst[ get_global_id( 0 )*copiesPerWorkItem+i ] = src[ get_global_id( 0 )*copiesPerWorkItem+i ];\n"
"}\n" ;



int test_copy(cl_device_id deviceID, cl_context context, cl_command_queue queue, const char *kernelCode,
              ExplicitType vecType, int vecSize
              )
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 2 ];
    size_t threads[ 1 ], localThreads[ 1 ];
    void *inBuffer, *outBuffer;
    MTdata d;
    char vecNameString[64]; vecNameString[0] = 0;
    if (vecSize == 1)
        sprintf(vecNameString, "%s", get_explicit_type_name(vecType));
    else
        sprintf(vecNameString, "%s%d", get_explicit_type_name(vecType), vecSize);


    size_t elementSize = get_explicit_type_size(vecType)*vecSize;
    log_info("Testing %s\n", vecNameString);

    cl_long max_local_mem_size;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error( error, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.");

    unsigned int num_of_compute_devices;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_of_compute_devices), &num_of_compute_devices, NULL);
    test_error( error, "clGetDeviceInfo for CL_DEVICE_MAX_COMPUTE_UNITS failed.");

    char programSource[4096]; programSource[0]=0;
    char *programPtr;

    sprintf(programSource, kernelCode,
            vecType == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
            vecNameString, vecNameString, vecNameString, vecNameString, get_explicit_type_name(vecType), vecNameString, vecNameString);
    //log_info("program: %s\n", programSource);
    programPtr = programSource;

    error = create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "test_fn" );
    test_error( error, "Unable to create testing kernel" );

    size_t max_workgroup_size;
    error = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_workgroup_size), &max_workgroup_size, NULL);
    test_error (error, "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE.");

    size_t max_local_workgroup_size[3];
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
    test_error (error, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Pick the minimum of the device and the kernel
    if (max_workgroup_size > max_local_workgroup_size[0])
        max_workgroup_size = max_local_workgroup_size[0];

    size_t numberOfCopiesPerWorkitem = 13;
    elementSize = get_explicit_type_size(vecType)* ((vecSize == 3) ? 4 : vecSize);
    size_t localStorageSpacePerWorkitem = numberOfCopiesPerWorkitem*elementSize;
    size_t maxLocalWorkgroupSize = (((int)max_local_mem_size/2)/localStorageSpacePerWorkitem);

    // Calculation can return 0 on embedded devices due to 1KB local mem limit
    if(maxLocalWorkgroupSize == 0)
    {
        maxLocalWorkgroupSize = 1;
    }

    size_t localWorkgroupSize = maxLocalWorkgroupSize;
    if (maxLocalWorkgroupSize > max_workgroup_size)
        localWorkgroupSize = max_workgroup_size;

    size_t localBufferSize = localWorkgroupSize*elementSize*numberOfCopiesPerWorkitem;
    size_t numberOfLocalWorkgroups = 1111;
    size_t globalBufferSize = numberOfLocalWorkgroups*localBufferSize;
    size_t globalWorkgroupSize = numberOfLocalWorkgroups*localWorkgroupSize;

    inBuffer = (void*)malloc(globalBufferSize);
    outBuffer = (void*)malloc(globalBufferSize);
    memset(outBuffer, 0, globalBufferSize);

    cl_int copiesPerWorkItemInt, copiesPerWorkgroup;
    copiesPerWorkItemInt = (int)numberOfCopiesPerWorkitem;
    copiesPerWorkgroup = (int)(numberOfCopiesPerWorkitem*localWorkgroupSize);

    log_info("Global: %d, local %d, local buffer %db, global buffer %db, each work group will copy %d elements and each work item item will copy %d elements.\n",
             (int) globalWorkgroupSize, (int)localWorkgroupSize, (int)localBufferSize, (int)globalBufferSize, copiesPerWorkgroup, copiesPerWorkItemInt);

    threads[0] = globalWorkgroupSize;
    localThreads[0] = localWorkgroupSize;

    d = init_genrand( gRandomSeed );
    generate_random_data( vecType, globalBufferSize/get_explicit_type_size(vecType), d, inBuffer );
    free_mtdata(d); d = NULL;

    streams[ 0 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, globalBufferSize, inBuffer, &error );
    test_error( error, "Unable to create input buffer" );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, globalBufferSize, outBuffer, &error );
    test_error( error, "Unable to create output buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 2, localBufferSize, NULL );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 3, sizeof(copiesPerWorkgroup), &copiesPerWorkgroup );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 4, sizeof(copiesPerWorkItemInt), &copiesPerWorkItemInt );
    test_error( error, "Unable to set kernel argument" );

    // Enqueue
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to queue kernel" );

    // Read
    error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0, globalBufferSize, outBuffer, 0, NULL, NULL );
    test_error( error, "Unable to read results" );

    // Verify
    int failuresPrinted = 0;
    if( memcmp( inBuffer, outBuffer, globalBufferSize ) != 0 )
    {
        size_t typeSize = get_explicit_type_size(vecType)* vecSize;
        unsigned char * inchar = (unsigned char*)inBuffer;
        unsigned char * outchar = (unsigned char*)outBuffer;
        for (int i=0; i< (int)globalBufferSize; i+=(int)elementSize) {
            if (memcmp( ((char *)inchar)+i, ((char *)outchar)+i, typeSize) != 0 )
            {
                char values[4096];
                values[0] = 0;
                if ( failuresPrinted == 0 ) {
                    // Print first failure message
                    log_error( "ERROR: Results of copy did not validate!\n" );
                }
                sprintf(values + strlen( values), "%d -> [", i);
                for (int j=0; j<(int)elementSize; j++)
                    sprintf(values + strlen( values), "%2x ", inchar[i+j]);
                sprintf(values + strlen(values), "] != [");
                for (int j=0; j<(int)elementSize; j++)
                    sprintf(values + strlen( values), "%2x ", outchar[i+j]);
                sprintf(values + strlen(values), "]");
                log_error("%s\n", values);
                failuresPrinted++;
            }

            if (failuresPrinted > 5) {
                log_error("Not printing further failures...\n");
                break;
            }
        }
    }

    free(inBuffer);
    free(outBuffer);

    return failuresPrinted ? -1 : 0;
}

int test_copy_all_types(cl_device_id deviceID, cl_context context, cl_command_queue queue, const char *kernelCode) {
    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int size, typeIndex;

    int errors = 0;

    for( typeIndex = 0; vecType[ typeIndex ] != kNumExplicitTypes; typeIndex++ )
    {
        if( vecType[ typeIndex ] == kDouble && !is_extension_available( deviceID, "cl_khr_fp64" ) )
            continue;

        if (( vecType[ typeIndex ] == kLong || vecType[ typeIndex ] == kULong ) && !gHasLong )
            continue;

        for( size = 0; vecSizes[ size ] != 0; size++ )
        {
            if (test_copy( deviceID, context, queue, kernelCode, vecType[typeIndex],vecSizes[size] )) {
                errors++;
            }
        }
    }
    if (errors)
        return -1;
    return 0;
}




int test_async_copy_global_to_local(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_copy_all_types( deviceID, context, queue, async_global_to_local_kernel );
}

int test_async_copy_local_to_global(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_copy_all_types( deviceID, context, queue, async_local_to_global_kernel );
}

int test_prefetch(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_copy_all_types( deviceID, context, queue, prefetch_kernel );
}


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
#include "compat.h"
#include "threadTesting.h"
#include "errorHelpers.h"
#include <stdio.h>
#include <string.h>

#if !defined(_WIN32)
#include <pthread.h>
#endif

#if 0 // Disabed for now

typedef struct
{
    basefn            mFunction;
    cl_device_id    mDevice;
    cl_context        mContext;
    int                mNumElements;
} TestFnArgs;

////////////////////////////////////////////////////////////////////////////////
// Thread-based testing. Spawns a new thread to run the given test function,
// then waits for it to complete. The entire idea is that, if the thread crashes,
// we can catch it and report it as a failure instead of crashing the entire suite
////////////////////////////////////////////////////////////////////////////////

void *test_thread_wrapper( void *data )
{
    TestFnArgs *args;
    int retVal;
    cl_context context;

    args = (TestFnArgs *)data;

    /* Create a new context to use (contexts can't cross threads) */
    context = clCreateContext(NULL, args->mDeviceGroup);
    if( context == NULL )
    {
        log_error("clCreateContext failed for new thread\n");
        return (void *)(-1);
    }

    /* Call function */
    retVal = args->mFunction( args->mDeviceGroup, args->mDevice, context, args->mNumElements );

    clReleaseContext( context );

    return (void *)retVal;
}

int test_threaded_function( basefn fnToTest, cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    int error;
    pthread_t threadHdl;
    void *retVal;
    TestFnArgs args;


    args.mFunction = fnToTest;
    args.mDeviceGroup = deviceGroup;
    args.mDevice = device;
    args.mContext = context;
    args.mNumElements = numElements;


    error = pthread_create( &threadHdl, NULL, test_thread_wrapper, (void *)&args );
    if( error != 0 )
    {
        log_error( "ERROR: Unable to create thread for testing!\n" );
        return -1;
    }

    /* Thread has been started, now just wait for it to complete (or crash) */
    error = pthread_join( threadHdl, &retVal );
    if( error != 0 )
    {
        log_error( "ERROR: Unable to join testing thread!\n" );
        return -1;
    }

    return (int)((intptr_t)retVal);
}
#endif



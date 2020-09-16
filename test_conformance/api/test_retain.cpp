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
#if !defined(_WIN32)
#include <unistd.h>
#endif // !_WIN32

// Note: According to spec, the various functions to get instance counts should return an error when passed in an object
// that has already been released. However, the spec is out of date. If it gets re-updated to allow such action, re-enable
// this define.
//#define VERIFY_AFTER_RELEASE    1

#define GET_QUEUE_INSTANCE_COUNT(p) numInstances = ( (err = clGetCommandQueueInfo(p, CL_QUEUE_REFERENCE_COUNT, sizeof( numInstances ), &numInstances, NULL)) == CL_SUCCESS ? numInstances : 0 )
#define GET_MEM_INSTANCE_COUNT(p) numInstances = ( (err = clGetMemObjectInfo(p, CL_MEM_REFERENCE_COUNT, sizeof( numInstances ), &numInstances, NULL)) == CL_SUCCESS ? numInstances : 0 )

#define VERIFY_INSTANCE_COUNT(c,rightValue) if( c != rightValue ) { \
log_error( "ERROR: Instance count for test object is not valid! (should be %d, really is %d)\n", rightValue, c ); \
return -1;    }

int test_retain_queue_single(cl_device_id deviceID, cl_context context, cl_command_queue queueNotUsed, int num_elements)
{
    cl_command_queue queue;
    cl_uint numInstances;
    int err;


    /* Create a test queue */
    queue = clCreateCommandQueue( context, deviceID, 0, &err );
    test_error( err, "Unable to create command queue to test with" );

    /* Test the instance count */
    GET_QUEUE_INSTANCE_COUNT( queue );
    test_error( err, "Unable to get queue instance count" );
    VERIFY_INSTANCE_COUNT( numInstances, 1 );

    /* Now release the program */
    clReleaseCommandQueue( queue );
#ifdef VERIFY_AFTER_RELEASE
    /* We're not allowed to get the instance count after the object has been completely released. But that's
     exactly how we can tell the release worked--by making sure getting the instance count fails! */
    GET_QUEUE_INSTANCE_COUNT( queue );
    if( err != CL_INVALID_COMMAND_QUEUE )
    {
        print_error( err, "Command queue was not properly released" );
        return -1;
    }
#endif

    return 0;
}

int test_retain_queue_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queueNotUsed, int num_elements)
{
    cl_command_queue queue;
    unsigned int numInstances, i;
    int err;


    /* Create a test program */
    queue = clCreateCommandQueue( context, deviceID, 0, &err );
    test_error( err, "Unable to create command queue to test with" );

    /* Increment 9 times, which should bring the count to 10 */
    for( i = 0; i < 9; i++ )
    {
        clRetainCommandQueue( queue );
    }

    /* Test the instance count */
    GET_QUEUE_INSTANCE_COUNT( queue );
    test_error( err, "Unable to get queue instance count" );
    VERIFY_INSTANCE_COUNT( numInstances, 10 );

    /* Now release 5 times, which should take us to 5 */
    for( i = 0; i < 5; i++ )
    {
        clReleaseCommandQueue( queue );
    }

    GET_QUEUE_INSTANCE_COUNT( queue );
    test_error( err, "Unable to get queue instance count" );
    VERIFY_INSTANCE_COUNT( numInstances, 5 );

    /* Retain again three times, which should take us to 8 */
    for( i = 0; i < 3; i++ )
    {
        clRetainCommandQueue( queue );
    }

    GET_QUEUE_INSTANCE_COUNT( queue );
    test_error( err, "Unable to get queue instance count" );
    VERIFY_INSTANCE_COUNT( numInstances, 8 );

    /* Release 7 times, which should take it to 1 */
    for( i = 0; i < 7; i++ )
    {
        clReleaseCommandQueue( queue );
    }

    GET_QUEUE_INSTANCE_COUNT( queue );
    test_error( err, "Unable to get queue instance count" );
    VERIFY_INSTANCE_COUNT( numInstances, 1 );

    /* And one last one */
    clReleaseCommandQueue( queue );

#ifdef VERIFY_AFTER_RELEASE
    /* We're not allowed to get the instance count after the object has been completely released. But that's
     exactly how we can tell the release worked--by making sure getting the instance count fails! */
    GET_QUEUE_INSTANCE_COUNT( queue );
    if( err != CL_INVALID_COMMAND_QUEUE )
    {
        print_error( err, "Command queue was not properly released" );
        return -1;
    }
#endif

    return 0;
}

int test_retain_mem_object_single(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem object;
    cl_uint numInstances;
    int err;


    /* Create a test object */
    object = clCreateBuffer( context, CL_MEM_READ_ONLY, 32, NULL, &err );
    test_error( err, "Unable to create buffer to test with" );

    /* Test the instance count */
    GET_MEM_INSTANCE_COUNT( object );
    test_error( err, "Unable to get mem object count" );
    VERIFY_INSTANCE_COUNT( numInstances, 1 );

    /* Now release the program */
    clReleaseMemObject( object );
#ifdef VERIFY_AFTER_RELEASE
    /* We're not allowed to get the instance count after the object has been completely released. But that's
     exactly how we can tell the release worked--by making sure getting the instance count fails! */
    GET_MEM_INSTANCE_COUNT( object );
    if( err != CL_INVALID_MEM_OBJECT )
    {
        print_error( err, "Mem object was not properly released" );
        return -1;
    }
#endif

    return 0;
}

int test_retain_mem_object_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem object;
    unsigned int numInstances, i;
    int err;


    /* Create a test object */
    object = clCreateBuffer( context, CL_MEM_READ_ONLY, 32, NULL, &err );
    test_error( err, "Unable to create buffer to test with" );

    /* Increment 9 times, which should bring the count to 10 */
    for( i = 0; i < 9; i++ )
    {
        clRetainMemObject( object );
    }

    /* Test the instance count */
    GET_MEM_INSTANCE_COUNT( object );
    test_error( err, "Unable to get mem object count" );
    VERIFY_INSTANCE_COUNT( numInstances, 10 );

    /* Now release 5 times, which should take us to 5 */
    for( i = 0; i < 5; i++ )
    {
        clReleaseMemObject( object );
    }

    GET_MEM_INSTANCE_COUNT( object );
    test_error( err, "Unable to get mem object count" );
    VERIFY_INSTANCE_COUNT( numInstances, 5 );

    /* Retain again three times, which should take us to 8 */
    for( i = 0; i < 3; i++ )
    {
        clRetainMemObject( object );
    }

    GET_MEM_INSTANCE_COUNT( object );
    test_error( err, "Unable to get mem object count" );
    VERIFY_INSTANCE_COUNT( numInstances, 8 );

    /* Release 7 times, which should take it to 1 */
    for( i = 0; i < 7; i++ )
    {
        clReleaseMemObject( object );
    }

    GET_MEM_INSTANCE_COUNT( object );
    test_error( err, "Unable to get mem object count" );
    VERIFY_INSTANCE_COUNT( numInstances, 1 );

    /* And one last one */
    clReleaseMemObject( object );

#ifdef VERIFY_AFTER_RELEASE
    /* We're not allowed to get the instance count after the object has been completely released. But that's
     exactly how we can tell the release worked--by making sure getting the instance count fails! */
    GET_MEM_INSTANCE_COUNT( object );
    if( err != CL_INVALID_MEM_OBJECT )
    {
        print_error( err, "Mem object was not properly released" );
        return -1;
    }
#endif

    return 0;
}

int test_retain_mem_object_set_kernel_arg(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int err;
    cl_mem buffer = nullptr;
    cl_program program;
    cl_kernel kernel;
    static volatile uint32_t sValue;
    sValue = 0;
    auto callback = []( cl_mem, void * ) {
      ++sValue;
    };
    const char *testProgram[] = { "__kernel void sample_test(__global int *data){}" };

    buffer = clCreateBuffer( context, CL_MEM_READ_ONLY, 32, NULL, &err );
    test_error( err, "Unable to create buffer to test with" );

    err = clSetMemObjectDestructorCallback( buffer, callback, nullptr );
    test_error( err, "Unable to set destructor callback" );

    err = create_single_kernel_helper( context, &program, nullptr, 1, testProgram, nullptr );
    test_error( err, "Unable to build sample program" );

    kernel = clCreateKernel( program, "sample_test", &err );
    test_error( err, "Unable to create sample_test kernel" );

    err = clSetKernelArg( kernel, 0, sizeof(cl_mem), &buffer );
    test_error( err, "Unable to set kernel argument" );

    err = clReleaseMemObject( buffer );
    test_error( err, "Unable to release buffer" );

    // Spin waiting for the release to finish.  If you don't call the mem_destructor_callback, you will not
    // pass the test.  bugzilla 6316
    while (sValue == 0) { }

    clReleaseKernel( kernel );
    clReleaseProgram( program );

    // If we got this far, we succeeded.
    return 0;
}

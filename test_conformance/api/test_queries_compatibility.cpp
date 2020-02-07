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
#include "harness/imageHelpers.h"
#include <stdlib.h>
#include <ctype.h>

int test_get_sampler_info_compatibility(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t size;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    clSamplerWrapper sampler = clCreateSampler( context, CL_TRUE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR, &error );
    test_error( error, "Unable to create sampler to test with" );

    cl_uint refCount;
    error = clGetSamplerInfo( sampler, CL_SAMPLER_REFERENCE_COUNT, sizeof( refCount ), &refCount, &size );
    test_error( error, "Unable to get sampler ref count" );
    if( size != sizeof( refCount ) )
    {
        log_error( "ERROR: Returned size of sampler refcount does not validate! (expected %d, got %d)\n", (int)sizeof( refCount ), (int)size );
        return -1;
    }

    cl_context otherCtx;
    error = clGetSamplerInfo( sampler, CL_SAMPLER_CONTEXT, sizeof( otherCtx ), &otherCtx, &size );
    test_error( error, "Unable to get sampler context" );
    if( otherCtx != context )
    {
        log_error( "ERROR: Sampler context does not validate! (expected %p, got %p)\n", context, otherCtx );
        return -1;
    }
    if( size != sizeof( otherCtx ) )
    {
        log_error( "ERROR: Returned size of sampler context does not validate! (expected %d, got %d)\n", (int)sizeof( otherCtx ), (int)size );
        return -1;
    }

    cl_addressing_mode mode;
    error = clGetSamplerInfo( sampler, CL_SAMPLER_ADDRESSING_MODE, sizeof( mode ), &mode, &size );
    test_error( error, "Unable to get sampler addressing mode" );
    if( mode != CL_ADDRESS_CLAMP )
    {
        log_error( "ERROR: Sampler addressing mode does not validate! (expected %d, got %d)\n", (int)CL_ADDRESS_CLAMP, (int)mode );
        return -1;
    }
    if( size != sizeof( mode ) )
    {
        log_error( "ERROR: Returned size of sampler addressing mode does not validate! (expected %d, got %d)\n", (int)sizeof( mode ), (int)size );
        return -1;
    }

    cl_filter_mode fmode;
    error = clGetSamplerInfo( sampler, CL_SAMPLER_FILTER_MODE, sizeof( fmode ), &fmode, &size );
    test_error( error, "Unable to get sampler filter mode" );
    if( fmode != CL_FILTER_LINEAR )
    {
        log_error( "ERROR: Sampler filter mode does not validate! (expected %d, got %d)\n", (int)CL_FILTER_LINEAR, (int)fmode );
        return -1;
    }
    if( size != sizeof( fmode ) )
    {
        log_error( "ERROR: Returned size of sampler filter mode does not validate! (expected %d, got %d)\n", (int)sizeof( fmode ), (int)size );
        return -1;
    }

    cl_int norm;
    error = clGetSamplerInfo( sampler, CL_SAMPLER_NORMALIZED_COORDS, sizeof( norm ), &norm, &size );
    test_error( error, "Unable to get sampler normalized flag" );
    if( norm != CL_TRUE )
    {
        log_error( "ERROR: Sampler normalized flag does not validate! (expected %d, got %d)\n", (int)CL_TRUE, (int)norm );
        return -1;
    }
    if( size != sizeof( norm ) )
    {
        log_error( "ERROR: Returned size of sampler normalized flag does not validate! (expected %d, got %d)\n", (int)sizeof( norm ), (int)size );
        return -1;
    }

    return 0;
}

#define TEST_COMMAND_QUEUE_PARAM( queue, paramName, val, expected, name, type, cast )    \
error = clGetCommandQueueInfo( queue, paramName, sizeof( val ), &val, &size );        \
test_error( error, "Unable to get command queue " name );                            \
if( val != expected )                                                                \
{                                                                                    \
log_error( "ERROR: Command queue " name " did not validate! (expected " type ", got " type ")\n", (cast)expected, (cast)val );    \
return -1;                                                                        \
}            \
if( size != sizeof( val ) )                \
{                                        \
log_error( "ERROR: Returned size of command queue " name " does not validate! (expected %d, got %d)\n", (int)sizeof( val ), (int)size );    \
return -1;    \
}

int test_get_command_queue_info_compatibility(cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements)
{
    int error;
    size_t size;

    cl_command_queue_properties device_props;
    clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_PROPERTIES, sizeof(device_props), &device_props, NULL);
    log_info("CL_DEVICE_QUEUE_PROPERTIES is %d\n", (int)device_props);

    clCommandQueueWrapper queue = clCreateCommandQueue( context, deviceID, device_props, &error );
    test_error( error, "Unable to create command queue to test with" );

    cl_uint refCount;
    error = clGetCommandQueueInfo( queue, CL_QUEUE_REFERENCE_COUNT, sizeof( refCount ), &refCount, &size );
    test_error( error, "Unable to get command queue reference count" );
    if( size != sizeof( refCount ) )
    {
        log_error( "ERROR: Returned size of command queue reference count does not validate! (expected %d, got %d)\n", (int)sizeof( refCount ), (int)size );
        return -1;
    }

    cl_context otherCtx;
    TEST_COMMAND_QUEUE_PARAM( queue, CL_QUEUE_CONTEXT, otherCtx, context, "context", "%p", cl_context )

    cl_device_id otherDevice;
    error = clGetCommandQueueInfo( queue, CL_QUEUE_DEVICE, sizeof(otherDevice), &otherDevice, &size);
    test_error(error, "clGetCommandQueue failed.");

    if (size != sizeof(cl_device_id)) {
        log_error( " ERROR: Returned size of command queue CL_QUEUE_DEVICE does not validate! (expected %d, got %d)\n", (int)sizeof( otherDevice ), (int)size );
        return -1;
    }

    /* Since the device IDs are opaque types we check the CL_DEVICE_VENDOR_ID which is unique for identical hardware. */
    cl_uint otherDevice_vid, deviceID_vid;
    error = clGetDeviceInfo(otherDevice, CL_DEVICE_VENDOR_ID, sizeof(otherDevice_vid), &otherDevice_vid, NULL );
    test_error( error, "Unable to get device CL_DEVICE_VENDOR_ID" );
    error = clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR_ID, sizeof(deviceID_vid), &deviceID_vid, NULL );
    test_error( error, "Unable to get device CL_DEVICE_VENDOR_ID" );

    if( otherDevice_vid != deviceID_vid )
    {
        log_error( "ERROR: Incorrect device returned for queue! (Expected vendor ID 0x%x, got 0x%x)\n", deviceID_vid, otherDevice_vid );
        return -1;
    }

    cl_command_queue_properties props;
    TEST_COMMAND_QUEUE_PARAM( queue, CL_QUEUE_PROPERTIES, props, (unsigned int)( device_props ), "properties", "%d", unsigned int )

    return 0;
}


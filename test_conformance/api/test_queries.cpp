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

int test_get_platform_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_platform_id platform;
    cl_int error;
    char buffer[ 16384 ];
    size_t length;

    // Get the platform to use
    error = clGetPlatformIDs(1, &platform, NULL);
    test_error( error, "Unable to get platform" );

    // Platform profile should either be FULL_PROFILE or EMBEDDED_PROFILE
    error = clGetPlatformInfo(platform,  CL_PLATFORM_PROFILE, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get platform profile string" );

    log_info("Returned CL_PLATFORM_PROFILE %s.\n", buffer);

    if( strcmp( buffer, "FULL_PROFILE" ) != 0 && strcmp( buffer, "EMBEDDED_PROFILE" ) != 0 )
    {
        log_error( "ERROR: Returned platform profile string is not a valid string by OpenCL 1.2! (Returned: %s)\n", buffer );
        return -1;
    }
    if( strlen( buffer )+1 != length )
    {
        log_error( "ERROR: Returned length of profile string is incorrect (actual length: %d, returned length: %d)\n",
                  (int)strlen( buffer )+1, (int)length );
        return -1;
    }

    // Check just length return
    error = clGetPlatformInfo(platform,  CL_PLATFORM_PROFILE, 0, NULL, &length );
    test_error( error, "Unable to get platform profile length" );
    if( strlen( (char *)buffer )+1 != length )
    {
        log_error( "ERROR: Returned length of profile string is incorrect (actual length: %d, returned length: %d)\n",
                  (int)strlen( (char *)buffer )+1, (int)length );
        return -1;
    }


    // Platform version should fit the regex "OpenCL *[0-9]+\.[0-9]+"
    error = clGetPlatformInfo(platform,  CL_PLATFORM_VERSION, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get platform version string" );

    log_info("Returned CL_PLATFORM_VERSION %s.\n", buffer);

    if( memcmp( buffer, "OpenCL ", strlen( "OpenCL " ) ) != 0 )
    {
        log_error( "ERROR: Initial part of platform version string does not match required format! (returned: %s)\n", (char *)buffer );
        return -1;
    }
    char *p1 = (char *)buffer + strlen( "OpenCL " );
    while( *p1 == ' ' )
        p1++;
    char *p2 = p1;
    while( isdigit( *p2 ) )
        p2++;
    if( *p2 != '.' )
    {
        log_error( "ERROR: Numeric part of platform version string does not match required format! (returned: %s)\n", (char *)buffer );
        return -1;
    }
    char *p3 = p2 + 1;
    while( isdigit( *p3 ) )
        p3++;
    if( *p3 != ' ' )
    {
        log_error( "ERROR: space expected after minor version number! (returned: %s)\n", (char *)buffer );
        return -1;
    }
    *p2 = ' '; // Put in a space for atoi below.
    p2++;

    // make sure it is null terminated
    for( ; p3 != buffer + length; p3++ )
        if( *p3 == '\0' )
            break;
    if( p3 == buffer + length )
    {
        log_error( "ERROR: platform version string is not NUL terminated!\n" );
        return -1;
    }

    int major = atoi( p1 );
    int minor = atoi( p2 );
    int minor_revision = 2;
    if( major * 10 + minor < 10 + minor_revision )
    {
        log_error( "ERROR: OpenCL profile version returned is less than 1.%d!\n", minor_revision );
        return -1;
    }

    // Sanity checks on the returned values
    if( length != strlen( (char *)buffer ) + 1)
    {
        log_error( "ERROR: Returned length of version string does not match actual length (actual: %d, returned: %d)\n", (int)strlen( (char *)buffer )+1, (int)length );
        return -1;
    }

    // Check just length
    error = clGetPlatformInfo(platform,  CL_PLATFORM_VERSION, 0, NULL, &length );
    test_error( error, "Unable to get platform version length" );
    if( length != strlen( (char *)buffer )+1 )
    {
        log_error( "ERROR: Returned length of version string does not match actual length (actual: %d, returned: %d)\n", (int)strlen( buffer )+1, (int)length );
        return -1;
    }

    return 0;
}

int test_get_sampler_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t size;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    cl_sampler_properties properties[] = {
        CL_SAMPLER_NORMALIZED_COORDS, CL_TRUE,
        CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP,
        CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
        0 };
    clSamplerWrapper sampler = clCreateSamplerWithProperties(context, properties, &error);
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

int test_get_command_queue_info(cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements)
{
    int error;
    size_t size;

    cl_queue_properties device_props;
    cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES,0,0};

    clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, sizeof(device_props), &device_props, NULL);
    log_info("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES is %d\n", (int)device_props);

    // Mask off vendor extension properties.  Only test standard OpenCL properties
    device_props &= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE;

    queue_props[1] = device_props;
    clCommandQueueWrapper queue = clCreateCommandQueueWithProperties( context, deviceID, &queue_props[0], &error );
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

int test_get_context_info(cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements)
{
    int error;
    size_t size;
    cl_context_properties props;

    error = clGetContextInfo( context, CL_CONTEXT_PROPERTIES, sizeof( props ), &props, &size );
    test_error( error, "Unable to get context props" );

    if (size == 0) {
        // Valid size
        return 0;
    } else if (size == sizeof(cl_context_properties)) {
        // Data must be NULL
        if (props != 0) {
            log_error("ERROR: Returned properties is no NULL.\n");
            return -1;
        }
        // Valid data and size
        return 0;
    }
    // Size was not 0 or 1
    log_error( "ERROR: Returned size of context props is not valid! (expected 0 or %d, got %d)\n",
              (int)sizeof(cl_context_properties), (int)size );
    return -1;
}

#define TEST_MEM_OBJECT_PARAM( mem, paramName, val, expected, name, type, cast )    \
error = clGetMemObjectInfo( mem, paramName, sizeof( val ), &val, &size );        \
test_error( error, "Unable to get mem object " name );                            \
if( val != expected )                                                                \
{                                                                                    \
log_error( "ERROR: Mem object " name " did not validate! (expected " type ", got " type ")\n", (cast)(expected), (cast)val );    \
return -1;                                                                        \
}            \
if( size != sizeof( val ) )                \
{                                        \
log_error( "ERROR: Returned size of mem object " name " does not validate! (expected %d, got %d)\n", (int)sizeof( val ), (int)size );    \
return -1;    \
}

void CL_CALLBACK mem_obj_destructor_callback( cl_mem, void *data )
{
    free( data );
}

// All possible combinations of valid cl_mem_flags.
static cl_mem_flags all_flags[16] = {
  0,
  CL_MEM_READ_WRITE,
  CL_MEM_READ_ONLY,
  CL_MEM_WRITE_ONLY,
  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
  CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
  CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
  CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
};

#define TEST_DEVICE_PARAM( device, paramName, val, name, type, cast )    \
error = clGetDeviceInfo( device, paramName, sizeof( val ), &val, &size );        \
test_error( error, "Unable to get device " name );                            \
if( size != sizeof( val ) )                \
{                                        \
log_error( "ERROR: Returned size of device " name " does not validate! (expected %d, got %d)\n", (int)sizeof( val ), (int)size );    \
return -1;    \
}                \
log_info( "\tReported device " name " : " type "\n", (cast)val );

#define TEST_DEVICE_PARAM_MEM( device, paramName, val, name, type, div )    \
error = clGetDeviceInfo( device, paramName, sizeof( val ), &val, &size );        \
test_error( error, "Unable to get device " name );                            \
if( size != sizeof( val ) )                \
{                                        \
log_error( "ERROR: Returned size of device " name " does not validate! (expected %d, got %d)\n", (int)sizeof( val ), (int)size );    \
return -1;    \
}                \
log_info( "\tReported device " name " : " type "\n", (int)( val / div ) );

int test_get_device_info(cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements)
{
    int error;
    size_t size;

    cl_uint vendorID;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_VENDOR_ID, vendorID, "vendor ID", "0x%08x", int )

    char extensions[ 10240 ];
    error = clGetDeviceInfo( deviceID, CL_DEVICE_EXTENSIONS, sizeof( extensions ), &extensions, &size );
    test_error( error, "Unable to get device extensions" );
    if( size != strlen( extensions ) + 1 )
    {
        log_error( "ERROR: Returned size of device extensions does not validate! (expected %d, got %d)\n", (int)( strlen( extensions ) + 1 ), (int)size );
        return -1;
    }
    log_info( "\tReported device extensions: %s \n", extensions );

    cl_uint preferred;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, preferred, "preferred vector char width", "%d", int )
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, preferred, "preferred vector short width", "%d", int )
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, preferred, "preferred vector int width", "%d", int )
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, preferred, "preferred vector long width", "%d", int )
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, preferred, "preferred vector float width", "%d", int )
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, preferred, "preferred vector double width", "%d", int )

    // Note that even if cl_khr_fp64, the preferred width for double can be non-zero.  For example, vendors
    // extensions can support double but may not support cl_khr_fp64, which implies math library support.

    cl_uint baseAddrAlign;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, baseAddrAlign, "base address alignment", "%d bits", int )

    cl_uint maxDataAlign;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, maxDataAlign, "min data type alignment", "%d bytes", int )

    cl_device_mem_cache_type cacheType;
    error = clGetDeviceInfo( deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof( cacheType ), &cacheType, &size );
    test_error( error, "Unable to get device global mem cache type" );
    if( size != sizeof( cacheType ) )
    {
        log_error( "ERROR: Returned size of device global mem cache type does not validate! (expected %d, got %d)\n", (int)sizeof( cacheType ), (int)size );
        return -1;
    }
    const char *cacheTypeName = ( cacheType == CL_NONE ) ? "CL_NONE" : ( cacheType == CL_READ_ONLY_CACHE ) ? "CL_READ_ONLY_CACHE" : ( cacheType == CL_READ_WRITE_CACHE ) ? "CL_READ_WRITE_CACHE" : "<unknown>";
    log_info( "\tReported device global mem cache type: %s \n", cacheTypeName );

    cl_uint cachelineSize;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cachelineSize, "global mem cacheline size", "%d bytes", int )

    cl_ulong cacheSize;
    TEST_DEVICE_PARAM_MEM( deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cacheSize, "global mem cache size", "%d KB", 1024 )

    cl_ulong memSize;
    TEST_DEVICE_PARAM_MEM( deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, memSize, "global mem size", "%d MB", ( 1024 * 1024 ) )

    cl_device_local_mem_type localMemType;
    error = clGetDeviceInfo( deviceID, CL_DEVICE_LOCAL_MEM_TYPE, sizeof( localMemType ), &localMemType, &size );
    test_error( error, "Unable to get device local mem type" );
    if( size != sizeof( cacheType ) )
    {
        log_error( "ERROR: Returned size of device local mem type does not validate! (expected %d, got %d)\n", (int)sizeof( localMemType ), (int)size );
        return -1;
    }
    const char *localMemTypeName = ( localMemType == CL_LOCAL ) ? "CL_LOCAL" : ( cacheType == CL_GLOBAL ) ? "CL_GLOBAL" : "<unknown>";
    log_info( "\tReported device local mem type: %s \n", localMemTypeName );


    cl_bool errSupport;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_ERROR_CORRECTION_SUPPORT, errSupport, "error correction support", "%d", int )

    size_t timerResolution;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_PROFILING_TIMER_RESOLUTION, timerResolution, "profiling timer resolution", "%ld nanoseconds", long )

    cl_bool endian;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_ENDIAN_LITTLE, endian, "little endian flag", "%d", int )

    cl_bool avail;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_AVAILABLE, avail, "available flag", "%d", int )

    cl_bool compilerAvail;
    TEST_DEVICE_PARAM( deviceID, CL_DEVICE_COMPILER_AVAILABLE, compilerAvail, "compiler available flag", "%d", int )

    char profile[ 1024 ];
    error = clGetDeviceInfo( deviceID, CL_DEVICE_PROFILE, sizeof( profile ), &profile, &size );
    test_error( error, "Unable to get device profile" );
    if( size != strlen( profile ) + 1 )
    {
        log_error( "ERROR: Returned size of device profile does not validate! (expected %d, got %d)\n", (int)( strlen( profile ) + 1 ), (int)size );
        return -1;
    }
    if( strcmp( profile, "FULL_PROFILE" ) != 0 && strcmp( profile, "EMBEDDED_PROFILE" ) != 0 )
    {
        log_error( "ERROR: Returned profile of device not FULL or EMBEDDED as required by OpenCL 1.2! (Returned %s)\n", profile );
        return -1;
    }
    log_info( "\tReported device profile: %s \n", profile );


    return 0;
}




static const char *sample_compile_size[2] = {
    "__kernel void sample_test(__global int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     dst[tid] = src[tid];\n"
    "\n"
    "}\n",
    "__kernel __attribute__((reqd_work_group_size(%d,%d,%d))) void sample_test(__global int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     dst[tid] = src[tid];\n"
    "\n"
    "}\n" };

int test_kernel_required_group_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t realSize;
    size_t kernel_max_workgroup_size;
    size_t global[] = {64,14,10};
    size_t local[] = {0,0,0};

    cl_uint max_dimensions;

    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_dimensions), &max_dimensions, NULL);
    test_error(error,  "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    log_info("Device reported CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %d.\n", (int)max_dimensions);

    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        error = create_single_kernel_helper( context, &program, &kernel, 1, &sample_compile_size[ 0 ], "sample_test" );
        if( error != 0 )
            return error;

        error = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_max_workgroup_size), &kernel_max_workgroup_size, NULL);
        test_error( error, "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE");
        log_info("The CL_KERNEL_WORK_GROUP_SIZE for the kernel is %d.\n", (int)kernel_max_workgroup_size);

        size_t size[ 3 ];
        error = clGetKernelWorkGroupInfo( kernel, deviceID, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof( size ), size, &realSize );
        test_error( error, "Unable to get work group info" );

        if( size[ 0 ] != 0 || size[ 1 ] != 0 || size[ 2 ] != 0 )
        {
            log_error( "ERROR: Nonzero compile work group size returned for nonspecified size! (returned %d,%d,%d)\n", (int)size[0], (int)size[1], (int)size[2] );
            return -1;
        }

        if( realSize != sizeof( size ) )
        {
            log_error( "ERROR: Returned size of compile work group size not valid! (Expected %d, got %d)\n", (int)sizeof( size ), (int)realSize );
            return -1;
        }

        // Determine some local dimensions to use for the test.
        if (max_dimensions == 1) {
            error = get_max_common_work_group_size(context, kernel, global[0], &local[0]);
            test_error( error, "get_max_common_work_group_size failed");
            log_info("For global dimension %d, kernel will require local dimension %d.\n", (int)global[0], (int)local[0]);
        } else if (max_dimensions == 2) {
            error = get_max_common_2D_work_group_size(context, kernel, global, local);
            test_error( error, "get_max_common_2D_work_group_size failed");
            log_info("For global dimension %d x %d, kernel will require local dimension %d x %d.\n", (int)global[0], (int)global[1], (int)local[0], (int)local[1]);
        } else {
            error = get_max_common_3D_work_group_size(context, kernel, global, local);
            test_error( error, "get_max_common_3D_work_group_size failed");
            log_info("For global dimension %d x %d x %d, kernel will require local dimension %d x %d x %d.\n",
                     (int)global[0], (int)global[1], (int)global[2], (int)local[0], (int)local[1], (int)local[2]);
        }
    }


    {
        clProgramWrapper program;
        clKernelWrapper kernel;
        clMemWrapper in, out;
        //char source[1024];
        char *source = (char*)malloc(1024);
        source[0] = '\0';

        sprintf(source, sample_compile_size[1], local[0], local[1], local[2]);

        error = create_single_kernel_helper( context, &program, &kernel, 1, (const char**)&source, "sample_test" );
        if( error != 0 )
            return error;

        size_t size[ 3 ];
        error = clGetKernelWorkGroupInfo( kernel, deviceID, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof( size ), size, &realSize );
        test_error( error, "Unable to get work group info" );

        if( size[ 0 ] != local[0] || size[ 1 ] != local[1] || size[ 2 ] != local[2] )
        {
            log_error( "ERROR: Incorrect compile work group size returned for specified size! (returned %d,%d,%d, expected %d,%d,%d)\n",
                      (int)size[0], (int)size[1], (int)size[2], (int)local[0], (int)local[1], (int)local[2]);
            return -1;
        }

        // Verify that the kernel will only execute with that size.
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int)*global[0], NULL, &error);
        test_error(error, "clCreateBuffer failed");
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*global[0], NULL, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(in), &in);
        test_error(error, "clSetKernelArg failed");
        error = clSetKernelArg(kernel, 1, sizeof(out), &out);
        test_error(error, "clSetKernelArg failed");

        error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        log_info("kernel_required_group_size may report spurious ERRORS in the conformance log.\n");

        local[0]++;
        error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        if (error != CL_INVALID_WORK_GROUP_SIZE) {
            log_error("Incorrect error returned for executing a kernel with the wrong required local work group size. (used %d,%d,%d, required %d,%d,%d)\n",
                      (int)local[0], (int)local[1], (int)local[2], (int)local[0]-1, (int)local[1], (int)local[2] );
            print_error(error, "Expected: CL_INVALID_WORK_GROUP_SIZE.");
            return -1;
        }

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        if (max_dimensions == 1) {
            free(source);
            return 0;
        }

        local[0]--; local[1]++;
        error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        if (error != CL_INVALID_WORK_GROUP_SIZE) {
            log_error("Incorrect error returned for executing a kernel with the wrong required local work group size. (used %d,%d,%d, required %d,%d,%d)\n",
                      (int)local[0], (int)local[1], (int)local[2], (int)local[0]-1, (int)local[1], (int)local[2]);
            print_error(error, "Expected: CL_INVALID_WORK_GROUP_SIZE.");
            return -1;
        }

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        if (max_dimensions == 2) {
            return 0;
            free(source);
        }

        local[1]--; local[2]++;
        error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        if (error != CL_INVALID_WORK_GROUP_SIZE) {
            log_error("Incorrect error returned for executing a kernel with the wrong required local work group size. (used %d,%d,%d, required %d,%d,%d)\n",
                      (int)local[0], (int)local[1], (int)local[2], (int)local[0]-1, (int)local[1], (int)local[2]);
            print_error(error, "Expected: CL_INVALID_WORK_GROUP_SIZE.");
            return -1;
        }

        error = clFinish(queue);
        test_error(error, "clFinish failed");
        free(source);
    }

    return 0;
}



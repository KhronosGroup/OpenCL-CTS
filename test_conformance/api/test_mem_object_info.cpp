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
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"


#define TEST_MEM_OBJECT_PARAM(mem, paramName, val, expected, name, type, cast) \
    error = clGetMemObjectInfo(mem, paramName, sizeof(val), &val, &size);      \
    test_error(error, "Unable to get mem object " name);                       \
    if (val != expected)                                                       \
    {                                                                          \
        log_error("ERROR: Mem object " name                                    \
                  " did not validate! (expected " type ", got " type           \
                  " from %s:%d)\n",                                            \
                  (cast)expected, (cast)val, __FILE__, __LINE__);              \
        return -1;                                                             \
    }                                                                          \
    if (size != sizeof(val))                                                   \
    {                                                                          \
        log_error("ERROR: Returned size of mem object " name                   \
                  " does not validate! (expected %d, got %d from %s:%d)\n",    \
                  (int)sizeof(val), (int)size, __FILE__, __LINE__);            \
        return -1;                                                             \
    }

static void CL_CALLBACK mem_obj_destructor_callback( cl_mem, void * data )
{
    free( data );
}

static unsigned int
get_image_dim(MTdata *d, unsigned int mod)
{
    unsigned int val = 0;

    do
    {
        val = (unsigned int)genrand_int32(*d) % mod;
    } while (val == 0);

    return val;
}


REGISTER_TEST(get_buffer_info)
{
    int error;
    size_t size;
    void * buffer = NULL;

    clMemWrapper bufferObject;
    clMemWrapper subBufferObject;

    cl_mem_flags bufferFlags[] = {
        CL_MEM_READ_WRITE,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_WRITE_ONLY,
        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
    };

    cl_mem_flags subBufferFlags[] = {
        CL_MEM_READ_WRITE,
        CL_MEM_READ_ONLY,
        CL_MEM_WRITE_ONLY,
        0,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_READ_ONLY | 0,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_WRITE_ONLY | 0,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_NO_ACCESS | 0,
    };


    // Get the address alignment, so we can make sure the sub-buffer test later works properly.
    cl_uint addressAlignBits;
    error = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                            sizeof(addressAlignBits), &addressAlignBits, NULL);

    size_t addressAlign = addressAlignBits/8;
    if ( addressAlign < 128 )
    {
        addressAlign = 128;
    }

    for ( unsigned int i = 0; i < sizeof(bufferFlags) / sizeof(cl_mem_flags); ++i )
    {
        //printf("@@@ bufferFlags[%u]=0x%x\n", i, bufferFlags[ i ]);
        if ( bufferFlags[ i ] & CL_MEM_USE_HOST_PTR )
        {
            // Create a buffer object to test against.
            buffer = malloc( addressAlign * 4 );
            bufferObject = clCreateBuffer( context, bufferFlags[ i ], addressAlign * 4, buffer, &error );
            if ( error )
            {
                free( buffer );
                test_error( error, "Unable to create buffer (CL_MEM_USE_HOST_PTR) to test with" );
            }

            // Make sure buffer is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( bufferObject, mem_obj_destructor_callback, buffer );
            test_error( error, "Unable to set mem object destructor callback" );

            void * ptr;
            TEST_MEM_OBJECT_PARAM( bufferObject, CL_MEM_HOST_PTR, ptr, buffer, "host pointer", "%p", void * )
        }
        else if ( (bufferFlags[ i ] & CL_MEM_ALLOC_HOST_PTR) && (bufferFlags[ i ] & CL_MEM_COPY_HOST_PTR) )
        {
            // Create a buffer object to test against.
            buffer = malloc( addressAlign * 4 );
            bufferObject = clCreateBuffer( context, bufferFlags[ i ], addressAlign * 4, buffer, &error );
            if ( error )
            {
                free( buffer );
                test_error( error, "Unable to create buffer (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure buffer is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( bufferObject, mem_obj_destructor_callback, buffer );
            test_error( error, "Unable to set mem object destructor callback" );
        }
        else if ( bufferFlags[ i ] & CL_MEM_ALLOC_HOST_PTR )
        {
            // Create a buffer object to test against.
            bufferObject = clCreateBuffer( context, bufferFlags[ i ], addressAlign * 4, NULL, &error );
            test_error( error, "Unable to create buffer (CL_MEM_ALLOC_HOST_PTR) to test with" );
        }
        else if ( bufferFlags[ i ] & CL_MEM_COPY_HOST_PTR )
        {
            // Create a buffer object to test against.
            buffer = malloc( addressAlign * 4 );
            bufferObject = clCreateBuffer( context, bufferFlags[ i ], addressAlign * 4, buffer, &error );
            if ( error )
            {
                free( buffer );
                test_error( error, "Unable to create buffer (CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure buffer is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( bufferObject, mem_obj_destructor_callback, buffer );
            test_error( error, "Unable to set mem object destructor callback" );
        }
        else
        {
            // Create a buffer object to test against.
            bufferObject = clCreateBuffer( context, bufferFlags[ i ], addressAlign * 4, NULL, &error );
            test_error( error, "Unable to create buffer to test with" );
            void *ptr;
            TEST_MEM_OBJECT_PARAM(bufferObject, CL_MEM_HOST_PTR, ptr, NULL,
                                  "host pointer", "%p", void *)
        }

        // Perform buffer object queries.
        void *ptr;
        TEST_MEM_OBJECT_PARAM(
            bufferObject, CL_MEM_HOST_PTR, ptr,
            ((bufferFlags[i] & CL_MEM_USE_HOST_PTR) ? buffer : NULL),
            "host pointer", "%p", void *)

        cl_mem_object_type type;
        TEST_MEM_OBJECT_PARAM( bufferObject, CL_MEM_TYPE, type, CL_MEM_OBJECT_BUFFER, "type", "%d", int )

        cl_mem_flags flags;
        TEST_MEM_OBJECT_PARAM( bufferObject, CL_MEM_FLAGS, flags, (unsigned int)bufferFlags[ i ], "flags", "%d", unsigned int )

        size_t sz;
        TEST_MEM_OBJECT_PARAM(bufferObject, CL_MEM_SIZE, sz,
                              (size_t)(addressAlign * 4), "size", "%zu", size_t)

        cl_uint mapCount;
        error = clGetMemObjectInfo( bufferObject, CL_MEM_MAP_COUNT, sizeof( mapCount ), &mapCount, &size );
        test_error( error, "Unable to get mem object map count" );
        if( size != sizeof( mapCount ) )
        {
            log_error( "ERROR: Returned size of mem object map count does not validate! (expected %d, got %d from %s:%d)\n",
                      (int)sizeof( mapCount ), (int)size, __FILE__, __LINE__ );
            return -1;
        }

        cl_uint refCount;
        error = clGetMemObjectInfo( bufferObject, CL_MEM_REFERENCE_COUNT, sizeof( refCount ), &refCount, &size );
        test_error( error, "Unable to get mem object reference count" );
        if( size != sizeof( refCount ) )
        {
            log_error( "ERROR: Returned size of mem object reference count does not validate! (expected %d, got %d from %s:%d)\n",
                      (int)sizeof( refCount ), (int)size, __FILE__, __LINE__ );
            return -1;
        }

        cl_context otherCtx;
        TEST_MEM_OBJECT_PARAM( bufferObject, CL_MEM_CONTEXT, otherCtx, context, "context", "%p", cl_context )

        cl_mem origObj;
        TEST_MEM_OBJECT_PARAM( bufferObject, CL_MEM_ASSOCIATED_MEMOBJECT, origObj, (void *)NULL, "associated mem object", "%p", void * )

        size_t offset;
        TEST_MEM_OBJECT_PARAM(bufferObject, CL_MEM_OFFSET, offset, size_t(0),
                              "offset", "%zu", size_t)

        cl_buffer_region region;
        region.origin = addressAlign;
        region.size = addressAlign;

        // Loop over possible sub-buffer objects to create.
        for ( unsigned int j = 0; j < sizeof(subBufferFlags) / sizeof(cl_mem_flags); ++j )
        {
            if ( subBufferFlags[ j ] & CL_MEM_READ_WRITE )
            {
                if ( !(bufferFlags[ i ] & CL_MEM_READ_WRITE) )
                    continue; // Buffer must be read_write for sub-buffer to be read_write.
            }
            if ( subBufferFlags[ j ] & CL_MEM_READ_ONLY )
            {
                if ( !(bufferFlags[ i ] & CL_MEM_READ_WRITE) && !(bufferFlags[ i ] & CL_MEM_READ_ONLY) )
                    continue; // Buffer must be read_write or read_only for sub-buffer to be read_only
            }
            if ( subBufferFlags[ j ] & CL_MEM_WRITE_ONLY )
            {
                if ( !(bufferFlags[ i ] & CL_MEM_READ_WRITE) && !(bufferFlags[ i ] & CL_MEM_WRITE_ONLY) )
                    continue; // Buffer must be read_write or write_only for sub-buffer to be write_only
            }
            if ( subBufferFlags[ j ] & CL_MEM_HOST_READ_ONLY )
            {
                if ( (bufferFlags[ i ] & CL_MEM_HOST_NO_ACCESS) || (bufferFlags[ i ] & CL_MEM_HOST_WRITE_ONLY) )
                    continue; // Buffer must be host all access or host read_only for sub-buffer to be host read_only
            }
            if ( subBufferFlags[ j ] & CL_MEM_HOST_WRITE_ONLY )
            {
                if ( (bufferFlags[ i ] & CL_MEM_HOST_NO_ACCESS) || (bufferFlags[ i ] & CL_MEM_HOST_READ_ONLY) )
                    continue; // Buffer must be host all access or host write_only for sub-buffer to be host write_only
            }
            //printf("@@@ bufferFlags[%u]=0x%x subBufferFlags[%u]=0x%x\n", i, bufferFlags[ i ], j, subBufferFlags[ j ]);

            subBufferObject = clCreateSubBuffer( bufferObject, subBufferFlags[ j ], CL_BUFFER_CREATE_TYPE_REGION, &region, &error );
            test_error( error, "Unable to create sub-buffer to test against" );

            // Perform sub-buffer object queries.
            cl_mem_object_type type;
            TEST_MEM_OBJECT_PARAM( subBufferObject, CL_MEM_TYPE, type, CL_MEM_OBJECT_BUFFER, "type", "%d", int )

            cl_mem_flags flags;
            cl_mem_flags inheritedFlags = subBufferFlags[ j ];
            if ( (subBufferFlags[ j ] & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY)) == 0 )
            {
              inheritedFlags |= bufferFlags[ i ] & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
            }
            inheritedFlags |= bufferFlags[ i ] & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);
            if ( (subBufferFlags[ j ] & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) == 0)
            {
              inheritedFlags |= bufferFlags[ i ] & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS);
            }
            TEST_MEM_OBJECT_PARAM( subBufferObject, CL_MEM_FLAGS, flags, (unsigned int)inheritedFlags, "flags", "%d", unsigned int )

            TEST_MEM_OBJECT_PARAM(subBufferObject, CL_MEM_SIZE, sz,
                                  (size_t)(addressAlign), "size", "%zu", size_t)

            if ( bufferFlags[ i ] & CL_MEM_USE_HOST_PTR )
            {
                void * ptr;
                void * offsetInBuffer = (char *)buffer + addressAlign;

                TEST_MEM_OBJECT_PARAM( subBufferObject, CL_MEM_HOST_PTR, ptr, offsetInBuffer, "host pointer", "%p", void * )
            }

            cl_uint mapCount;
            error = clGetMemObjectInfo( subBufferObject, CL_MEM_MAP_COUNT, sizeof( mapCount ), &mapCount, &size );
            test_error( error, "Unable to get mem object map count" );
            if( size != sizeof( mapCount ) )
            {
                log_error( "ERROR: Returned size of mem object map count does not validate! (expected %d, got %d from %s:%d)\n",
                          (int)sizeof( mapCount ), (int)size, __FILE__, __LINE__ );
                return -1;
            }

            cl_uint refCount;
            error = clGetMemObjectInfo( subBufferObject, CL_MEM_REFERENCE_COUNT, sizeof( refCount ), &refCount, &size );
            test_error( error, "Unable to get mem object reference count" );
            if( size != sizeof( refCount ) )
            {
                log_error( "ERROR: Returned size of mem object reference count does not validate! (expected %d, got %d from %s:%d)\n",
                          (int)sizeof( refCount ), (int)size, __FILE__, __LINE__ );
                return -1;
            }

            cl_context otherCtx;
            TEST_MEM_OBJECT_PARAM( subBufferObject, CL_MEM_CONTEXT, otherCtx, context, "context", "%p", cl_context )

            TEST_MEM_OBJECT_PARAM( subBufferObject, CL_MEM_ASSOCIATED_MEMOBJECT, origObj, (cl_mem)bufferObject, "associated mem object", "%p", void * )

            TEST_MEM_OBJECT_PARAM(subBufferObject, CL_MEM_OFFSET, offset,
                                  (size_t)(addressAlign), "offset", "%zu",
                                  size_t)
        }
    }

    return CL_SUCCESS;
}


int test_get_imageObject_info( cl_mem * image, cl_mem_flags objectFlags, cl_image_desc *imageInfo, cl_image_format *imageFormat, size_t pixelSize, cl_context context )
{
    int error;
    size_t size;
    cl_mem_object_type type;
    cl_mem_flags flags;
    cl_uint mapCount;
    cl_uint refCount;
    cl_context otherCtx;
    size_t offset;
    size_t sz;

    TEST_MEM_OBJECT_PARAM( *image, CL_MEM_TYPE, type, imageInfo->image_type, "type", "%d", int )

    TEST_MEM_OBJECT_PARAM( *image, CL_MEM_FLAGS, flags, (unsigned int)objectFlags, "flags", "%d", unsigned int )

    error = clGetMemObjectInfo( *image, CL_MEM_SIZE, sizeof( sz ), &sz, NULL );
    test_error( error, "Unable to get mem size" );

    // The size returned is not constrained by the spec.

    error = clGetMemObjectInfo( *image, CL_MEM_MAP_COUNT, sizeof( mapCount ), &mapCount, &size );
    test_error( error, "Unable to get mem object map count" );
    if( size != sizeof( mapCount ) )
    {
        log_error( "ERROR: Returned size of mem object map count does not validate! (expected %d, got %d from %s:%d)\n",
                  (int)sizeof( mapCount ), (int)size, __FILE__, __LINE__ );
        return -1;
    }

    error = clGetMemObjectInfo( *image, CL_MEM_REFERENCE_COUNT, sizeof( refCount ), &refCount, &size );
    test_error( error, "Unable to get mem object reference count" );
    if( size != sizeof( refCount ) )
    {
        log_error( "ERROR: Returned size of mem object reference count does not validate! (expected %d, got %d from %s:%d)\n",
                  (int)sizeof( refCount ), (int)size, __FILE__, __LINE__ );
        return -1;
    }

    TEST_MEM_OBJECT_PARAM( *image, CL_MEM_CONTEXT, otherCtx, context, "context", "%p", cl_context )

    TEST_MEM_OBJECT_PARAM(*image, CL_MEM_OFFSET, offset, size_t(0), "offset",
                          "%zu", size_t)

    return CL_SUCCESS;
}


int test_get_image_info(cl_device_id device, cl_context context,
                        cl_mem_object_type type)
{
    int error;
    size_t size;
    void * image = NULL;

    cl_mem imageObject;
    cl_image_desc imageInfo;

    cl_mem_flags imageFlags[] = {
        CL_MEM_READ_WRITE,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_READ_ONLY,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_WRITE_ONLY,
        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_WRITE_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
        CL_MEM_HOST_NO_ACCESS | CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
    };
    MTdataHolder d_holder(gRandomSeed);
    MTdata d = static_cast<MTdata>(d_holder);

    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    cl_image_format imageFormat;
    size_t pixelSize = 4;

    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNORM_INT8;

    imageInfo.image_width = imageInfo.image_height = imageInfo.image_depth = 1;
    imageInfo.image_array_size = 0;
    imageInfo.num_mip_levels = imageInfo.num_samples = 0;
#ifdef CL_VERSION_2_0
    imageInfo.mem_object = NULL;
#else
    imageInfo.buffer = NULL;
#endif

    for ( unsigned int i = 0; i < sizeof(imageFlags) / sizeof(cl_mem_flags); ++i )
    {
        imageInfo.image_row_pitch = 0;
        imageInfo.image_slice_pitch = 0;

        switch (type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                imageInfo.image_width = get_image_dim(&d, 1023);
                imageInfo.image_type = CL_MEM_OBJECT_IMAGE1D;
                break;

            case CL_MEM_OBJECT_IMAGE2D:
                imageInfo.image_width = get_image_dim(&d, 1023);
                imageInfo.image_height = get_image_dim(&d, 1023);
                imageInfo.image_type = CL_MEM_OBJECT_IMAGE2D;
                break;

            case CL_MEM_OBJECT_IMAGE3D:
                error = checkFor3DImageSupport(device);
                if (error == CL_IMAGE_FORMAT_NOT_SUPPORTED)
                {
                    log_info("Device doesn't support 3D images. Skipping test.\n");
                    return CL_SUCCESS;
                }
                imageInfo.image_width = get_image_dim(&d, 127);
                imageInfo.image_height = get_image_dim(&d, 127);
                imageInfo.image_depth = get_image_dim(&d, 127);
                imageInfo.image_type = CL_MEM_OBJECT_IMAGE3D;
                break;

            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                imageInfo.image_width = get_image_dim(&d, 1023);
                imageInfo.image_array_size = get_image_dim(&d, 1023);
                imageInfo.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
                break;

            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                imageInfo.image_width = get_image_dim(&d, 255);
                imageInfo.image_height = get_image_dim(&d, 255);
                imageInfo.image_array_size = get_image_dim(&d, 255);
                imageInfo.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
                break;
        }

        if ( imageFlags[i] & CL_MEM_USE_HOST_PTR )
        {
            // Create an image object to test against.
            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image with (CL_MEM_USE_HOST_PTR) to test with" );
            }

            // Make sure image is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );

            void * ptr;
            TEST_MEM_OBJECT_PARAM( imageObject, CL_MEM_HOST_PTR, ptr, image, "host pointer", "%p", void * )
            int ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

            // release image object
            clReleaseMemObject(imageObject);

            // Try again with non-zero rowPitch.
            imageInfo.image_row_pitch = imageInfo.image_width * pixelSize;
            switch (type)
            {
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE3D:
                    imageInfo.image_slice_pitch = imageInfo.image_row_pitch * imageInfo.image_height;
                    break;
            }

            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image2d (CL_MEM_USE_HOST_PTR) to test with" );
            }

            // Make sure image2d is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );

            TEST_MEM_OBJECT_PARAM( imageObject, CL_MEM_HOST_PTR, ptr, image, "host pointer", "%p", void * )
            ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

        }
        else if ( (imageFlags[i] & CL_MEM_ALLOC_HOST_PTR) && (imageFlags[i] & CL_MEM_COPY_HOST_PTR) )
        {
            // Create an image object to test against.
            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image with (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure image is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );
            int ret = test_get_imageObject_info( &imageObject, imageFlags[ i ], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

            // release image object
            clReleaseMemObject(imageObject);

            // Try again with non-zero rowPitch.
            imageInfo.image_row_pitch = imageInfo.image_width * pixelSize;
            switch (type)
            {
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE3D:
                    imageInfo.image_slice_pitch = imageInfo.image_row_pitch * imageInfo.image_height;
                    break;
            }

            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image with (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure image is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );
            ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

        }
        else if ( imageFlags[i] & CL_MEM_ALLOC_HOST_PTR )
        {
            // Create an image object to test against.
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, NULL, &error );
            test_error( error, "Unable to create image with (CL_MEM_ALLOC_HOST_PTR) to test with" );
            int ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

        }
        else if ( imageFlags[i] & CL_MEM_COPY_HOST_PTR )
        {
            // Create an image object to test against.
            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image with (CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure image is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );
            int ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

            clReleaseMemObject(imageObject);

            // Try again with non-zero rowPitch.
            imageInfo.image_row_pitch = imageInfo.image_width * pixelSize;
            switch (type)
            {
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE3D:
                    imageInfo.image_slice_pitch = imageInfo.image_row_pitch * imageInfo.image_height;
                    break;
            }

            image = malloc( imageInfo.image_width * imageInfo.image_height * imageInfo.image_depth * pixelSize *
                           ((imageInfo.image_array_size == 0) ? 1 : imageInfo.image_array_size) );
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, image, &error );
            if ( error )
            {
                free( image );
                test_error( error, "Unable to create image with (CL_MEM_COPY_HOST_PTR) to test with" );
            }

            // Make sure image is cleaned up appropriately if we encounter an error in the rest of the calls.
            error = clSetMemObjectDestructorCallback( imageObject, mem_obj_destructor_callback, image );
            test_error( error, "Unable to set mem object destructor callback" );
            ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

        }
        else
        {
            // Create an image object to test against.
            imageObject = clCreateImage( context, imageFlags[i], &imageFormat, &imageInfo, NULL, &error );
            test_error( error, "Unable to create image to test with" );
            int ret = test_get_imageObject_info( &imageObject, imageFlags[i], &imageInfo, &imageFormat, pixelSize, context );
            if (ret)
                return ret;

        }

        clReleaseMemObject( imageObject );
    }

    return CL_SUCCESS;
}


REGISTER_TEST(get_image2d_info)
{
    return test_get_image_info(device, context, CL_MEM_OBJECT_IMAGE2D);
}

REGISTER_TEST(get_image3d_info)
{
    return test_get_image_info(device, context, CL_MEM_OBJECT_IMAGE3D);
}

REGISTER_TEST(get_image1d_info)
{
    return test_get_image_info(device, context, CL_MEM_OBJECT_IMAGE1D);
}

REGISTER_TEST(get_image1d_array_info)
{
    return test_get_image_info(device, context, CL_MEM_OBJECT_IMAGE1D_ARRAY);
}

REGISTER_TEST(get_image2d_array_info)
{
    return test_get_image_info(device, context, CL_MEM_OBJECT_IMAGE2D_ARRAY);
}

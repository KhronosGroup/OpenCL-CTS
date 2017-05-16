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
#include "typeWrappers.h"
#include "kernelHelpers.h"
#include "errorHelpers.h"
#include <stdlib.h>
#include "clImageHelper.h"

#define ROUND_SIZE_UP( _size, _align )      (((size_t)(_size) + (size_t)(_align) - 1) & -((size_t)(_align)))

#if defined( __APPLE__ )
    #define kPageSize       4096
    #include <sys/mman.h>
    #include <stdlib.h>
#elif defined(__linux__)
    #include <unistd.h>
    #define kPageSize  (getpagesize())
#endif

clProtectedImage::clProtectedImage( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, cl_int *errcode_ret )
{
    cl_int err = Create( context, mem_flags, fmt, width );
    if( errcode_ret != NULL )
        *errcode_ret = err;
}

cl_int clProtectedImage::Create( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width )
{
    cl_int error;
#if defined( __APPLE__ )
    int protect_pages = 1;
    cl_device_id devices[16];
    size_t number_of_devices;
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &number_of_devices);
    test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

    number_of_devices /= sizeof(cl_device_id);
    for (int i=0; i<(int)number_of_devices; i++) {
        cl_device_type type;
        error = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed");
        if (type == CL_DEVICE_TYPE_GPU) {
            protect_pages = 0;
            break;
        }
    }

    if (protect_pages) {
        size_t pixelBytes = get_pixel_bytes(fmt);
        size_t rowBytes = ROUND_SIZE_UP( width * pixelBytes, kPageSize );
        size_t rowStride = rowBytes + kPageSize;

        // create backing store
        backingStoreSize = rowStride + 8 * rowStride;
        backingStore = mmap(0, backingStoreSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

        // add guard pages
        size_t row;
        char *p = (char*) backingStore;
        char *imagePtr = (char*) backingStore + 4 * rowStride;
        for( row = 0; row < 4; row++ )
        {
            mprotect( p, rowStride, PROT_NONE );    p += rowStride;
        }
        p += rowBytes;
        mprotect( p, kPageSize, PROT_NONE );        p += rowStride;
        p -= rowBytes;
        for( row = 0; row < 4; row++ )
        {
            mprotect( p, rowStride, PROT_NONE );    p += rowStride;
        }

        if(  getenv( "CL_ALIGN_RIGHT" ) )
        {
            static int spewEnv = 1;
            if(spewEnv)
            {
                log_info( "***CL_ALIGN_RIGHT is set. Aligning images at right edge of page\n" );
                spewEnv = 0;
            }
            imagePtr += rowBytes - pixelBytes * width;
        }

        image = create_image_1d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, rowStride, imagePtr, NULL, &error );
    } else {
        backingStore = NULL;
        image = create_image_1d( context, mem_flags, fmt, width, 0, NULL, NULL, &error );

    }
#else

    backingStore = NULL;
    image = create_image_1d( context, mem_flags, fmt, width, 0, NULL, NULL, &error );

#endif
    return error;
}


clProtectedImage::clProtectedImage( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height, cl_int *errcode_ret )
{
    cl_int err = Create( context, mem_flags, fmt, width, height );
    if( errcode_ret != NULL )
        *errcode_ret = err;
}

cl_int clProtectedImage::Create( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height )
{
    cl_int error;
#if defined( __APPLE__ )
  int protect_pages = 1;
  cl_device_id devices[16];
  size_t number_of_devices;
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &number_of_devices);
  test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

  number_of_devices /= sizeof(cl_device_id);
  for (int i=0; i<(int)number_of_devices; i++) {
    cl_device_type type;
    error = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed");
    if (type == CL_DEVICE_TYPE_GPU) {
      protect_pages = 0;
      break;
    }
  }

  if (protect_pages) {
    size_t pixelBytes = get_pixel_bytes(fmt);
    size_t rowBytes = ROUND_SIZE_UP( width * pixelBytes, kPageSize );
    size_t rowStride = rowBytes + kPageSize;

    // create backing store
    backingStoreSize = height * rowStride + 8 * rowStride;
    backingStore = mmap(0, backingStoreSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

    // add guard pages
    size_t row;
    char *p = (char*) backingStore;
    char *imagePtr = (char*) backingStore + 4 * rowStride;
    for( row = 0; row < 4; row++ )
    {
        mprotect( p, rowStride, PROT_NONE );    p += rowStride;
    }
    p += rowBytes;
    for( row = 0; row < height; row++ )
    {
        mprotect( p, kPageSize, PROT_NONE );    p += rowStride;
    }
    p -= rowBytes;
    for( row = 0; row < 4; row++ )
    {
        mprotect( p, rowStride, PROT_NONE );    p += rowStride;
    }

    if(  getenv( "CL_ALIGN_RIGHT" ) )
    {
      static int spewEnv = 1;
      if(spewEnv)
      {
        log_info( "***CL_ALIGN_RIGHT is set. Aligning images at right edge of page\n" );
        spewEnv = 0;
      }
      imagePtr += rowBytes - pixelBytes * width;
    }

      image = create_image_2d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, height, rowStride, imagePtr, &error );
  } else {
    backingStore = NULL;
      image = create_image_2d( context, mem_flags, fmt, width, height, 0, NULL, &error );

  }
#else

  backingStore = NULL;
  image = create_image_2d( context, mem_flags, fmt, width, height, 0, NULL, &error );

#endif
    return error;
}

clProtectedImage::clProtectedImage( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, cl_int *errcode_ret )
{
    cl_int err = Create( context, mem_flags, fmt, width, height, depth );
    if( errcode_ret != NULL )
        *errcode_ret = err;
}

cl_int clProtectedImage::Create( cl_context context, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth )
{
    cl_int error;

#if defined( __APPLE__ )
  int protect_pages = 1;
  cl_device_id devices[16];
  size_t number_of_devices;
  error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &number_of_devices);
  test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

  number_of_devices /= sizeof(cl_device_id);
  for (int i=0; i<(int)number_of_devices; i++) {
    cl_device_type type;
    error = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed");
    if (type == CL_DEVICE_TYPE_GPU) {
      protect_pages = 0;
      break;
    }
  }

  if (protect_pages) {
    size_t pixelBytes = get_pixel_bytes(fmt);
    size_t rowBytes = ROUND_SIZE_UP( width * pixelBytes, kPageSize );
    size_t rowStride = rowBytes + kPageSize;

    // create backing store
    backingStoreSize = height * depth * rowStride + 8 * rowStride;
    backingStore = mmap(0, backingStoreSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

    // add guard pages
    size_t row;
    char *p = (char*) backingStore;
    char *imagePtr = (char*) backingStore + 4 * rowStride;
    for( row = 0; row < 4; row++ )
    {
        mprotect( p, rowStride, PROT_NONE );    p += rowStride;
    }
    p += rowBytes;
    for( row = 0; row < height*depth; row++ )
    {
        mprotect( p, kPageSize, PROT_NONE );    p += rowStride;
    }
    p -= rowBytes;
    for( row = 0; row < 4; row++ )
    {
        mprotect( p, rowStride, PROT_NONE );    p += rowStride;
    }

    if(  getenv( "CL_ALIGN_RIGHT" ) )
    {
        static int spewEnv = 1;
        if(spewEnv)
        {
            log_info( "***CL_ALIGN_RIGHT is set. Aligning images at right edge of page\n" );
            spewEnv = 0;
        }
        imagePtr += rowBytes - pixelBytes * width;
    }

    image = create_image_3d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, height, depth, rowStride, height*rowStride, imagePtr, &error );
  } else {
    backingStore = NULL;
    image = create_image_3d( context, mem_flags, fmt, width, height, depth, 0, 0, NULL, &error );
  }
#else

    backingStore = NULL;
    image = create_image_3d( context, mem_flags, fmt, width, height, depth, 0, 0, NULL, &error );

#endif

    return error;
}


clProtectedImage::clProtectedImage( cl_context context, cl_mem_object_type imageType, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, size_t arraySize, cl_int *errcode_ret )
{
    cl_int err = Create( context, imageType, mem_flags, fmt, width, height, depth, arraySize );
    if( errcode_ret != NULL )
        *errcode_ret = err;
}

cl_int clProtectedImage::Create( cl_context context, cl_mem_object_type imageType, cl_mem_flags mem_flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, size_t arraySize )
{
    cl_int error;
#if defined( __APPLE__ )
    int protect_pages = 1;
    cl_device_id devices[16];
    size_t number_of_devices;
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &number_of_devices);
    test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

    number_of_devices /= sizeof(cl_device_id);
    for (int i=0; i<(int)number_of_devices; i++) {
        cl_device_type type;
        error = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed");
        if (type == CL_DEVICE_TYPE_GPU) {
            protect_pages = 0;
            break;
        }
    }

    if (protect_pages) {
        size_t pixelBytes = get_pixel_bytes(fmt);
        size_t rowBytes = ROUND_SIZE_UP( width * pixelBytes, kPageSize );
        size_t rowStride = rowBytes + kPageSize;

        // create backing store
        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                backingStoreSize = rowStride + 8 * rowStride;
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                backingStoreSize = height * rowStride + 8 * rowStride;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                backingStoreSize = height * depth * rowStride + 8 * rowStride;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                backingStoreSize = arraySize * rowStride + 8 * rowStride;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                backingStoreSize = height * arraySize * rowStride + 8 * rowStride;
                break;
        }
        backingStore = mmap(0, backingStoreSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

        // add guard pages
        size_t row;
        char *p = (char*) backingStore;
        char *imagePtr = (char*) backingStore + 4 * rowStride;
        for( row = 0; row < 4; row++ )
        {
            mprotect( p, rowStride, PROT_NONE );    p += rowStride;
        }
        p += rowBytes;
        size_t sz = (height > 0 ? height : 1) * (depth > 0 ? depth : 1) * (arraySize > 0 ? arraySize : 1);
        for( row = 0; row < sz; row++ )
        {
            mprotect( p, kPageSize, PROT_NONE );    p += rowStride;
        }
        p -= rowBytes;
        for( row = 0; row < 4; row++ )
        {
            mprotect( p, rowStride, PROT_NONE );    p += rowStride;
        }

        if(  getenv( "CL_ALIGN_RIGHT" ) )
        {
            static int spewEnv = 1;
            if(spewEnv)
            {
                log_info( "***CL_ALIGN_RIGHT is set. Aligning images at right edge of page\n" );
                spewEnv = 0;
            }
            imagePtr += rowBytes - pixelBytes * width;
        }

        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                image = create_image_1d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, rowStride, imagePtr, NULL, &error );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                image = create_image_2d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, height, rowStride, imagePtr, &error );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                image = create_image_3d( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, height, depth, rowStride, height*rowStride, imagePtr, &error );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                image = create_image_1d_array( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, arraySize, rowStride, rowStride, imagePtr, &error );
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                image = create_image_2d_array( context, mem_flags | CL_MEM_USE_HOST_PTR, fmt, width, height, arraySize, rowStride, height*rowStride, imagePtr, &error );
                break;
        }
    } else {
        backingStore = NULL;
        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                image = create_image_1d( context, mem_flags, fmt, width, 0, NULL, NULL, &error );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                image = create_image_2d( context, mem_flags, fmt, width, height, 0, NULL, &error );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                image = create_image_3d( context, mem_flags, fmt, width, height, depth, 0, 0, NULL, &error );;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                image = create_image_1d_array( context, mem_flags, fmt, width, arraySize, 0, 0, NULL, &error );
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                image = create_image_2d_array( context, mem_flags, fmt, width, height, arraySize, 0, 0, NULL, &error );
                break;
        }

    }
#else

    backingStore = NULL;
    switch (imageType)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            image = create_image_1d( context, mem_flags, fmt, width, 0, NULL, NULL, &error );
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            image = create_image_2d( context, mem_flags, fmt, width, height, 0, NULL, &error );
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            image = create_image_3d( context, mem_flags, fmt, width, height, depth, 0, 0, NULL, &error );;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            image = create_image_1d_array( context, mem_flags, fmt, width, arraySize, 0, 0, NULL, &error );
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            image = create_image_2d_array( context, mem_flags, fmt, width, height, arraySize, 0, 0, NULL, &error );
            break;
    }
#endif
    return error;
}



/*******
 * clProtectedArray implementation
 *******/
clProtectedArray::clProtectedArray()
{
    mBuffer = mValidBuffer = NULL;
}

clProtectedArray::clProtectedArray( size_t sizeInBytes )
{
    mBuffer = mValidBuffer = NULL;
    Allocate( sizeInBytes );
}

clProtectedArray::~clProtectedArray()
{
    if( mBuffer != NULL ) {
#if defined( __APPLE__ )
        int error = munmap( mBuffer, mRealSize );
      if (error) log_error("WARNING: munmap failed in clProtectedArray.\n");
#else
    free( mBuffer );
#endif
  }
}

void clProtectedArray::Allocate( size_t sizeInBytes )
{

#if defined( __APPLE__ )

    // Allocate enough space to: round up our actual allocation to an even number of pages
    // and allocate two pages on either side
    mRoundedSize = ROUND_SIZE_UP( sizeInBytes, kPageSize );
    mRealSize = mRoundedSize + kPageSize * 2;

    // Use mmap here to ensure we start on a page boundary, so the mprotect calls will work OK
    mBuffer = (char *)mmap(0, mRealSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, 0, 0);

    mValidBuffer = mBuffer + kPageSize;

    // Protect guard area from access
    mprotect( mValidBuffer - kPageSize, kPageSize, PROT_NONE );
    mprotect( mValidBuffer + mRoundedSize, kPageSize, PROT_NONE );
#else
  mRoundedSize = mRealSize = sizeInBytes;
  mBuffer = mValidBuffer = (char *)calloc(1, mRealSize);
#endif
}



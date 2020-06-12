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
#ifndef _typeWrappers_h
#define _typeWrappers_h

#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include "compat.h"
#include <stdio.h>
#include "mt19937.h"
#include "errorHelpers.h"
#include "kernelHelpers.h"

/* cl_context wrapper */

class clContextWrapper
{
    public:
        clContextWrapper() { mContext = NULL; }
        clContextWrapper( cl_context program ) { mContext = program; }
        ~clContextWrapper() { if( mContext != NULL ) clReleaseContext( mContext ); }

        clContextWrapper & operator=( const cl_context &rhs ) { mContext = rhs; return *this; }
        operator cl_context() const { return mContext; }

        cl_context * operator&() { return &mContext; }

        bool operator==( const cl_context &rhs ) { return mContext == rhs; }

    protected:

        cl_context mContext;
};

/* cl_program wrapper */

class clProgramWrapper
{
    public:
        clProgramWrapper() { mProgram = NULL; }
        clProgramWrapper( cl_program program ) { mProgram = program; }
        ~clProgramWrapper() { if( mProgram != NULL ) clReleaseProgram( mProgram ); }

        clProgramWrapper & operator=( const cl_program &rhs ) { mProgram = rhs; return *this; }
        operator cl_program() const { return mProgram; }

        cl_program * operator&() { return &mProgram; }

        bool operator==( const cl_program &rhs ) { return mProgram == rhs; }

    protected:

        cl_program mProgram;
};

/* cl_kernel wrapper */

class clKernelWrapper
{
    public:
        clKernelWrapper() { mKernel = NULL; }
        clKernelWrapper( cl_kernel kernel ) { mKernel = kernel; }
        ~clKernelWrapper() { if( mKernel != NULL ) clReleaseKernel( mKernel ); }

        clKernelWrapper & operator=( const cl_kernel &rhs ) { mKernel = rhs; return *this; }
        operator cl_kernel() const { return mKernel; }

        cl_kernel * operator&() { return &mKernel; }

        bool operator==( const cl_kernel &rhs ) { return mKernel == rhs; }

    protected:

        cl_kernel mKernel;
};

/* cl_mem (stream) wrapper */

class clMemWrapper
{
    public:
        clMemWrapper() { mMem = NULL; }
        clMemWrapper( cl_mem mem ) { mMem = mem; }
        ~clMemWrapper() { if( mMem != NULL ) clReleaseMemObject( mMem ); }

        clMemWrapper & operator=( const cl_mem &rhs ) { mMem = rhs; return *this; }
        operator cl_mem() const { return mMem; }

        cl_mem * operator&() { return &mMem; }

        bool operator==( const cl_mem &rhs ) { return mMem == rhs; }

    protected:

        cl_mem mMem;
};

class clProtectedImage
{
    public:
        clProtectedImage() { image = NULL; backingStore = NULL; }
        clProtectedImage( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width, cl_int *errcode_ret );
        clProtectedImage( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height, cl_int *errcode_ret );
        clProtectedImage( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, cl_int *errcode_ret );
        clProtectedImage( cl_context context, cl_mem_object_type imageType, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, size_t arraySize, cl_int *errcode_ret );
        ~clProtectedImage()
        {
            if( image != NULL )
                clReleaseMemObject( image );

#if defined( __APPLE__ )
            if(backingStore)
                munmap(backingStore, backingStoreSize);
#endif
        }

        cl_int Create( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width );
        cl_int Create( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height );
        cl_int Create( cl_context context, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth );
        cl_int Create( cl_context context, cl_mem_object_type imageType, cl_mem_flags flags, const cl_image_format *fmt, size_t width, size_t height, size_t depth, size_t arraySize );

        clProtectedImage & operator=( const cl_mem &rhs ) { image = rhs; backingStore = NULL; return *this; }
        operator cl_mem() { return image; }

        cl_mem * operator&() { return &image; }

        bool operator==( const cl_mem &rhs ) { return image == rhs; }

    protected:
        void *backingStore;
        size_t backingStoreSize;
        cl_mem  image;
};

/* cl_command_queue wrapper */
class clCommandQueueWrapper
{
    public:
        clCommandQueueWrapper() { mMem = NULL; }
        clCommandQueueWrapper( cl_command_queue mem ) { mMem = mem; }
  ~clCommandQueueWrapper() { if( mMem != NULL ) { clReleaseCommandQueue( mMem ); } }

        clCommandQueueWrapper & operator=( const cl_command_queue &rhs ) { mMem = rhs; return *this; }
        operator cl_command_queue() const { return mMem; }

        cl_command_queue * operator&() { return &mMem; }

        bool operator==( const cl_command_queue &rhs ) { return mMem == rhs; }

    protected:

        cl_command_queue mMem;
};

/* cl_sampler wrapper */
class clSamplerWrapper
{
    public:
        clSamplerWrapper() { mMem = NULL; }
        clSamplerWrapper( cl_sampler mem ) { mMem = mem; }
        ~clSamplerWrapper() { if( mMem != NULL ) clReleaseSampler( mMem ); }

        clSamplerWrapper & operator=( const cl_sampler &rhs ) { mMem = rhs; return *this; }
        operator cl_sampler() const { return mMem; }

        cl_sampler * operator&() { return &mMem; }

        bool operator==( const cl_sampler &rhs ) { return mMem == rhs; }

    protected:

        cl_sampler mMem;
};

/* cl_event wrapper */
class clEventWrapper
{
    public:
        clEventWrapper() { mMem = NULL; }
        clEventWrapper( cl_event mem ) { mMem = mem; }
        ~clEventWrapper() { if( mMem != NULL ) clReleaseEvent( mMem ); }

        clEventWrapper & operator=( const cl_event &rhs ) { mMem = rhs; return *this; }
        operator cl_event() const { return mMem; }

        cl_event * operator&() { return &mMem; }

        bool operator==( const cl_event &rhs ) { return mMem == rhs; }

    protected:

        cl_event mMem;
};

/* Generic protected memory buffer, for verifying access within bounds */
class clProtectedArray
{
    public:
        clProtectedArray();
        clProtectedArray( size_t sizeInBytes );
        virtual ~clProtectedArray();

        void    Allocate( size_t sizeInBytes );

        operator void *()        { return (void *)mValidBuffer; }
        operator const void *() const { return (const void *)mValidBuffer; }

    protected:

         char *    mBuffer;
         char * mValidBuffer;
        size_t    mRealSize, mRoundedSize;
};

class RandomSeed
{
    public:
        RandomSeed( cl_uint seed  ){ if(seed) log_info( "(seed = %10.10u) ", seed ); mtData = init_genrand(seed); }
        ~RandomSeed()
        {
            if( gReSeed )
                gRandomSeed = genrand_int32( mtData );
            free_mtdata(mtData);
        }

        operator MTdata ()     {return mtData;}

    protected:
        MTdata mtData;
};


template <typename T> class BufferOwningPtr
{
  BufferOwningPtr(BufferOwningPtr const &); // do not implement
    void operator=(BufferOwningPtr const &);  // do not implement

    void *ptr;
    void *map;
  size_t mapsize;   // Bytes allocated total, pointed to by map.
  size_t allocsize; // Bytes allocated in unprotected pages, pointed to by ptr.
  bool aligned;
  public:
  explicit BufferOwningPtr(void *p = 0) : ptr(p), map(0), mapsize(0), allocsize(0), aligned(false) {}
  explicit BufferOwningPtr(void *p, void *m, size_t s)
    : ptr(p), map(m), mapsize(s), allocsize(0), aligned(false)
      {
#if ! defined( __APPLE__ )
        if(m)
        {
            log_error( "ERROR: unhandled code path. BufferOwningPtr allocated with mapped buffer!" );
            abort();
        }
#endif
      }
    ~BufferOwningPtr() {
      if (map) {
#if defined( __APPLE__ )
        int error = munmap(map, mapsize);
        if (error) log_error("WARNING: munmap failed in BufferOwningPtr.\n");
#endif
      } else {
          if ( aligned )
          {
              align_free(ptr);
          }
          else
          {
            free(ptr);
          }
      }
    }
  void reset(void *p, void *m = 0, size_t mapsize_ = 0, size_t allocsize_ = 0, bool aligned_ = false) {
      if (map){
#if defined( __APPLE__ )
        int error = munmap(map, mapsize);
        if (error) log_error("WARNING: munmap failed in BufferOwningPtr.\n");
#else
        log_error( "ERROR: unhandled code path. BufferOwningPtr reset with mapped buffer!" );
        abort();
#endif
      } else {
          if ( aligned )
          {
              align_free(ptr);
          }
          else
          {
            free(ptr);
          }
      }
      ptr = p;
      map = m;
      mapsize = mapsize_;
      allocsize =  (ptr != NULL) ? allocsize_ : 0; // Force allocsize to zero if ptr is NULL.
      aligned = aligned_;
#if ! defined( __APPLE__ )
        if(m)
        {
            log_error( "ERROR: unhandled code path. BufferOwningPtr allocated with mapped buffer!" );
            abort();
        }
#endif
    }
    operator T*() { return (T*)ptr; }

      size_t getSize() const { return allocsize; };
};

#endif // _typeWrappers_h


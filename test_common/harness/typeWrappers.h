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

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include "compat.h"
#include "mt19937.h"
#include "errorHelpers.h"
#include "kernelHelpers.h"

#include <cstdlib>
#include <type_traits>

namespace wrapper_details {

// clRetain*() and clRelease*() functions share the same type.
template <typename T> // T should be cl_context, cl_program, ...
using RetainReleaseType = cl_int CL_API_CALL(T);

// A generic wrapper class that follows OpenCL retain/release semantics.
//
// This Wrapper class implement copy and move semantics, which makes it
// compatible with standard containers for example.
//
// Template parameters:
//  - T is the cl_* type (e.g. cl_context, cl_program, ...)
//  - Retain is the clRetain* function (e.g. clRetainContext, ...)
//  - Release is the clRelease* function (e.g. clReleaseContext, ...)
template <typename T, RetainReleaseType<T> Retain, RetainReleaseType<T> Release>
class Wrapper {
    static_assert(std::is_pointer<T>::value, "T should be a pointer type.");
    T object = nullptr;

    void retain()
    {
        if (!object) return;

        auto err = Retain(object);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clRetain*() failed");
            std::abort();
        }
    }

    void release()
    {
        if (!object) return;

        auto err = Release(object);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clRelease*() failed");
            std::abort();
        }
    }

public:
    Wrapper() = default;

    // On initialisation, assume the object has a refcount of one.
    Wrapper(T object): object(object) {}

    // On assignment, assume the object has a refcount of one.
    Wrapper &operator=(T rhs)
    {
        reset(rhs);
        return *this;
    }

    // Copy semantics, increase retain count.
    Wrapper(Wrapper const &w) { *this = w; }
    Wrapper &operator=(Wrapper const &w)
    {
        reset(w.object);
        retain();
        return *this;
    }

    // Move semantics, directly take ownership.
    Wrapper(Wrapper &&w) { *this = std::move(w); }
    Wrapper &operator=(Wrapper &&w)
    {
        reset(w.object);
        w.object = nullptr;
        return *this;
    }

    ~Wrapper() { reset(); }

    // Release the existing object, if any, and own the new one, if any.
    void reset(T new_object = nullptr)
    {
        release();
        object = new_object;
    }

    operator T() const { return object; }

    // Ideally this function should not exist as it breaks encapsulation by
    // allowing external mutation of the Wrapper internal state. However, too
    // much code currently relies on this. For example, instead of using T* as
    // output parameters, existing code can be updated to use Wrapper& instead.
    T *operator&() { return &object; }
};

} // namespace wrapper_details

using clContextWrapper =
    wrapper_details::Wrapper<cl_context, clRetainContext, clReleaseContext>;

using clProgramWrapper =
    wrapper_details::Wrapper<cl_program, clRetainProgram, clReleaseProgram>;

using clKernelWrapper =
    wrapper_details::Wrapper<cl_kernel, clRetainKernel, clReleaseKernel>;

using clMemWrapper =
    wrapper_details::Wrapper<cl_mem, clRetainMemObject, clReleaseMemObject>;

using clCommandQueueWrapper =
    wrapper_details::Wrapper<cl_command_queue, clRetainCommandQueue,
                             clReleaseCommandQueue>;

using clSamplerWrapper =
    wrapper_details::Wrapper<cl_sampler, clRetainSampler, clReleaseSampler>;

using clEventWrapper =
    wrapper_details::Wrapper<cl_event, clRetainEvent, clReleaseEvent>;

class clSVMWrapper {
    void *Ptr = nullptr;
    cl_context Ctx = nullptr;

public:
    clSVMWrapper() = default;

    clSVMWrapper(cl_context C, size_t Size,
                 cl_svm_mem_flags F = CL_MEM_READ_WRITE)
        : Ctx(C)
    {
        Ptr = clSVMAlloc(C, F, Size, 0);
    }

    clSVMWrapper &operator=(void *other) = delete;
    clSVMWrapper(clSVMWrapper const &other) = delete;
    clSVMWrapper &operator=(clSVMWrapper const &other) = delete;
    clSVMWrapper(clSVMWrapper &&other)
    {
        Ptr = other.Ptr;
        Ctx = other.Ctx;
        other.Ptr = nullptr;
        other.Ctx = nullptr;
    }
    clSVMWrapper &operator=(clSVMWrapper &&other)
    {
        Ptr = other.Ptr;
        Ctx = other.Ctx;
        other.Ptr = nullptr;
        other.Ctx = nullptr;
        return *this;
    }

    ~clSVMWrapper()
    {
        if (Ptr) clSVMFree(Ctx, Ptr);
    }

    void *operator()() const { return Ptr; }
};


class clProtectedImage {
public:
    clProtectedImage()
    {
        image = NULL;
        backingStore = NULL;
    }
    clProtectedImage(cl_context context, cl_mem_flags flags,
                     const cl_image_format *fmt, size_t width,
                     cl_int *errcode_ret);
    clProtectedImage(cl_context context, cl_mem_flags flags,
                     const cl_image_format *fmt, size_t width, size_t height,
                     cl_int *errcode_ret);
    clProtectedImage(cl_context context, cl_mem_flags flags,
                     const cl_image_format *fmt, size_t width, size_t height,
                     size_t depth, cl_int *errcode_ret);
    clProtectedImage(cl_context context, cl_mem_object_type imageType,
                     cl_mem_flags flags, const cl_image_format *fmt,
                     size_t width, size_t height, size_t depth,
                     size_t arraySize, cl_int *errcode_ret);
    ~clProtectedImage()
    {
        if (image != NULL) clReleaseMemObject(image);

#if defined(__APPLE__)
        if (backingStore) munmap(backingStore, backingStoreSize);
#endif
    }

    cl_int Create(cl_context context, cl_mem_flags flags,
                  const cl_image_format *fmt, size_t width);
    cl_int Create(cl_context context, cl_mem_flags flags,
                  const cl_image_format *fmt, size_t width, size_t height);
    cl_int Create(cl_context context, cl_mem_flags flags,
                  const cl_image_format *fmt, size_t width, size_t height,
                  size_t depth);
    cl_int Create(cl_context context, cl_mem_object_type imageType,
                  cl_mem_flags flags, const cl_image_format *fmt, size_t width,
                  size_t height, size_t depth, size_t arraySize);

    clProtectedImage &operator=(const cl_mem &rhs)
    {
        image = rhs;
        backingStore = NULL;
        return *this;
    }
    operator cl_mem() { return image; }

    cl_mem *operator&() { return &image; }

protected:
    void *backingStore;
    size_t backingStoreSize;
    cl_mem image;
};

/* Generic protected memory buffer, for verifying access within bounds */
class clProtectedArray {
public:
    clProtectedArray();
    clProtectedArray(size_t sizeInBytes);
    virtual ~clProtectedArray();

    void Allocate(size_t sizeInBytes);

    operator void *() { return (void *)mValidBuffer; }
    operator const void *() const { return (const void *)mValidBuffer; }

protected:
    char *mBuffer;
    char *mValidBuffer;
    size_t mRealSize, mRoundedSize;
};

class RandomSeed {
public:
    RandomSeed(cl_uint seed)
    {
        if (seed) log_info("(seed = %10.10u) ", seed);
        mtData = init_genrand(seed);
    }
    ~RandomSeed()
    {
        if (gReSeed) gRandomSeed = genrand_int32(mtData);
        free_mtdata(mtData);
    }

    operator MTdata() { return mtData; }

protected:
    MTdata mtData;
};


template <typename T> class BufferOwningPtr {
    BufferOwningPtr(BufferOwningPtr const &); // do not implement
    void operator=(BufferOwningPtr const &); // do not implement

    void *ptr;
    void *map;
    // Bytes allocated total, pointed to by map:
    size_t mapsize;
    // Bytes allocated in unprotected pages, pointed to by ptr:
    size_t allocsize;
    bool aligned;

public:
    explicit BufferOwningPtr(void *p = 0)
        : ptr(p), map(0), mapsize(0), allocsize(0), aligned(false)
    {}
    explicit BufferOwningPtr(void *p, void *m, size_t s)
        : ptr(p), map(m), mapsize(s), allocsize(0), aligned(false)
    {
#if !defined(__APPLE__)
        if (m)
        {
            log_error("ERROR: unhandled code path. BufferOwningPtr allocated "
                      "with mapped buffer!");
            abort();
        }
#endif
    }
    ~BufferOwningPtr()
    {
        if (map)
        {
#if defined(__APPLE__)
            int error = munmap(map, mapsize);
            if (error)
                log_error("WARNING: munmap failed in BufferOwningPtr.\n");
#endif
        }
        else
        {
            if (aligned)
            {
                align_free(ptr);
            }
            else
            {
                free(ptr);
            }
        }
    }
    void reset(void *p, void *m = 0, size_t mapsize_ = 0, size_t allocsize_ = 0,
               bool aligned_ = false)
    {
        if (map)
        {
#if defined(__APPLE__)
            int error = munmap(map, mapsize);
            if (error)
                log_error("WARNING: munmap failed in BufferOwningPtr.\n");
#else
            log_error("ERROR: unhandled code path. BufferOwningPtr reset with "
                      "mapped buffer!");
            abort();
#endif
        }
        else
        {
            if (aligned)
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
        // Force allocsize to zero if ptr is NULL:
        allocsize = (ptr != NULL) ? allocsize_ : 0;
        aligned = aligned_;
#if !defined(__APPLE__)
        if (m)
        {
            log_error("ERROR: unhandled code path. BufferOwningPtr allocated "
                      "with mapped buffer!");
            abort();
        }
#endif
    }
    operator T *() { return (T *)ptr; }

    size_t getSize() const { return allocsize; };
};

#endif // _typeWrappers_h

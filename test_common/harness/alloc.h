//
// Copyright (c) 2020 The Khronos Group Inc.
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

#ifndef HARNESS_ALLOC_H_
#define HARNESS_ALLOC_H_

#if defined(__linux__) || defined (linux) || defined(__APPLE__)
#if defined(__ANDROID__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#endif

#if defined(__MINGW32__)
#include "mingw_compat.h"
#endif

static void * align_malloc(size_t size, size_t alignment)
{
#if defined(_WIN32) && defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif  defined(__linux__) || defined (linux) || defined(__APPLE__)
    void * ptr = NULL;
#if defined(__ANDROID__)
    ptr = memalign(alignment, size);
    if ( ptr )
        return ptr;
#else
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    if (0 == posix_memalign(&ptr, alignment, size))
        return ptr;
#endif
    return NULL;
#elif defined(__MINGW32__)
    return __mingw_aligned_malloc(size, alignment);
#else
    #error "Please add support OS for aligned malloc"
#endif
}

static void align_free(void * ptr)
{
#if defined(_WIN32) && defined(_MSC_VER)
    _aligned_free(ptr);
#elif  defined(__linux__) || defined (linux) || defined(__APPLE__)
    return  free(ptr);
#elif defined(__MINGW32__)
    return __mingw_aligned_free(ptr);
#else
    #error "Please add support OS for aligned free"
#endif
}

#endif // #ifndef HARNESS_ALLOC_H_


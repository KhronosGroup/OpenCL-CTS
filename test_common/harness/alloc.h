//
// Copyright (c) 2020 - 2024 The Khronos Group Inc.
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

#if defined(__linux__) || defined(linux) || defined(__APPLE__)
#if defined(__ANDROID__)
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#endif
#include <stdint.h>

#if defined(__MINGW32__)
#include "mingw_compat.h"
#endif

#if defined(_WIN32)
#include <cstdlib>
#endif

inline void* align_malloc(size_t size, size_t alignment)
{
#if defined(_WIN32) && defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
    void* ptr = NULL;
#if defined(__ANDROID__)
    ptr = memalign(alignment, size);
    if (ptr) return ptr;
#else
    if (alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }
    if (0 == posix_memalign(&ptr, alignment, size)) return ptr;
#endif
    return NULL;
#elif defined(__MINGW32__)
    return __mingw_aligned_malloc(size, alignment);
#else
#error "Please add support OS for aligned malloc"
#endif
}

inline void align_free(void* ptr)
{
#if defined(_WIN32) && defined(_MSC_VER)
    _aligned_free(ptr);
#elif defined(__linux__) || defined(linux) || defined(__APPLE__)
    return free(ptr);
#elif defined(__MINGW32__)
    return __mingw_aligned_free(ptr);
#else
#error "Please add support OS for aligned free"
#endif
}

enum class dma_buf_heap_type
{
    SYSTEM
};

/**
 * @brief Allocate a DMA buffer.
 *
 * On systems that support it, use the DMA buffer heaps to allocate a DMA buffer
 * of the requested size, using the requested heap type. The heap type defaults
 * to using the system heap if no type is specified.
 *
 * A heap type will use a default path if one exists, and can be overriden using
 * an environment variable for each type, as follows:
 *
 * SYSTEM:
 *     * Default path: /dev/dma_heap/system
 *     * Environment variable: OCL_CTS_DMA_HEAP_PATH_SYSTEM
 *
 * DMA buffer heaps require a minimum Linux kernel version 5.6. A compile-time
 * warning is issued on older systems, as well as an error message at runtime.
 *
 * @param size [in]           The requested buffer size in bytes.
 * @param heap_type [in,opt]  The heap type to use for the allocation.
 *
 * @retrun A file descriptor representing the allocated DMA buffer on success,
 * -1 otherwise. Failure to open the DMA device returns TEST_SKIPPED_ITSELF so
 * it can be handled separately to other failures.
 */
int allocate_dma_buf(uint64_t size,
                     dma_buf_heap_type heap_type = dma_buf_heap_type::SYSTEM);

#endif // #ifndef HARNESS_ALLOC_H_

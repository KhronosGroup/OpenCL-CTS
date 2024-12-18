//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include "alloc.h"
#include "errorHelpers.h"

#if defined(linux) || defined(__linux__) || defined(__ANDROID__)
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#include <linux/dma-heap.h>
#endif

struct dma_buf_heap_helper_t
{
    dma_buf_heap_type heap_type;
    const char* env_var = nullptr;
    const char* default_path = nullptr;

    constexpr dma_buf_heap_helper_t(dma_buf_heap_type heap_type,
                                    const char* env_var,
                                    const char* default_path)
        : heap_type(heap_type), env_var(env_var), default_path(default_path)
    {}
};

constexpr dma_buf_heap_helper_t DMA_BUF_HEAP_TABLE[] = {
    { dma_buf_heap_type::SYSTEM, "OCL_CTS_DMA_HEAP_PATH_SYSTEM",
      "/dev/dma_heap/system" },
};

static dma_buf_heap_helper_t lookup_dma_heap(dma_buf_heap_type heap_type)
{
    for (const auto& entry : DMA_BUF_HEAP_TABLE)
    {
        if (heap_type == entry.heap_type)
        {
            return entry;
        }
    }

    assert(false
           && "DMA heap type does not have an entry in DMA_BUF_HEAP_TABLE");
    return DMA_BUF_HEAP_TABLE[0];
}

int allocate_dma_buf(uint64_t size, dma_buf_heap_type heap_type)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
    constexpr int DMA_HEAP_FLAGS = O_RDWR | O_CLOEXEC;

    const auto entry = lookup_dma_heap(heap_type);
    const auto override_path = getenv(entry.env_var);
    const auto dma_heap_path =
        (override_path == nullptr) ? entry.default_path : override_path;

    const int dma_heap_fd = open(dma_heap_path, DMA_HEAP_FLAGS);
    if (dma_heap_fd == -1)
    {
        log_error(
            "Opening the DMA heap device: %s failed with error: %d (%s)\n",
            dma_heap_path, errno, strerror(errno));

        return -1;
    }

    dma_heap_allocation_data dma_heap_data = { 0 };
    dma_heap_data.len = size;
    dma_heap_data.fd_flags = O_RDWR | O_CLOEXEC;

    int result = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &dma_heap_data);
    if (result != 0)
    {
        log_error("DMA heap allocation IOCTL call failed, error: %d\n", result);

        close(dma_heap_fd);
        return -1;
    }

    result = close(dma_heap_fd);
    if (result == -1)
    {
        log_info("Failed to close the DMA heap device: %s\n", dma_heap_path);
    }

    return dma_heap_data.fd;
#else
#warning                                                                       \
    "Kernel version doesn't support DMA buffer heaps (at least v5.6.0 is required)."
    return -1;
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
}

#else
int allocate_dma_buf(uint64_t size, dma_buf_heap_type heap_type)
{
    log_error(
        "OS doesn't have DMA buffer heaps (only Linux and Android do).\n");

    return -1;
}
#endif // defined(linux) || defined(__linux__) || defined(__ANDROID__)

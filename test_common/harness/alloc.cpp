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
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#include <linux/dma-heap.h>
#endif

#include <sys/ioctl.h>
#include <unistd.h>

int allocate_dma_buf(uint64_t size)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
    constexpr const char* DMA_HEAP_PATH = "/dev/dma_heap/system";
    constexpr int DMA_HEAP_FLAGS = O_RDWR | O_CLOEXEC;

    const int dma_heap_fd = open(DMA_HEAP_PATH, DMA_HEAP_FLAGS);
    if (dma_heap_fd == -1)
    {
        log_error(
            "Opening the DMA heap device: %s failed with error: %d (%s)\n",
            DMA_HEAP_PATH, errno, strerror(errno));

        return -1;
    }

    dma_heap_allocation_data dma_heap_data = { 0 };
    dma_heap_data.len = size;
    dma_heap_data.fd_flags = O_RDWR | O_CLOEXEC;

    int result = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &dma_heap_data);
    if (result != 0)
    {
        log_error("DMA heap allocation IOCTL call failed, error: %d", result);

        close(dma_heap_fd);
        return -1;
    }

    result = close(dma_heap_fd);
    if (result == -1)
    {
        log_info("Failed to close the DMA heap device: %s", DMA_HEAP_PATH);
    }

    return dma_heap_data.fd;
#else
#warning                                                                       \
    "Kernel version doesn't support DMA-BUF heaps (at least v5.6.0 is required)."
    return -1;
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
}

#else
int allocate_dma_buf(uint64_t size)
{
    log_error("OS doesn't have DMA-BUF heaps (only Linux and Android do).");

    return -1;
}
#endif // defined(linux) || defined(__linux__) || defined(__ANDROID__)

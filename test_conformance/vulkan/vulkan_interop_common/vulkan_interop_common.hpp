#ifndef _vulkan_interop_common_hpp_
#define _vulkan_interop_common_hpp_

#include "vulkan_wrapper_types.hpp"
#include "vulkan_wrapper.hpp"
#include "vulkan_list_map.hpp"
#include "vulkan_utility.hpp"
#include "opencl_vulkan_wrapper.hpp"

// Number of iterations for loops within tests (default value 5)
extern unsigned int innerIterations;
// Number of iterations for loops within perf tests (default value 100)
extern unsigned int perfIterations;
// Number of iterations for loops within stress tests (default value 1000)
extern unsigned int stressIterations;
// Number of CPU threads per GPU (default value 3)
extern size_t cpuThreadsPerGpu;
// Number of command queues (default value 1)
extern unsigned int numCQ;
// Enable Multi-import of vulkan device memory
extern bool multiImport;
// Enable Multi-import of vulkan device memory under different context
extern bool multiCtx;
// Enable additional debug info logging
extern bool debug_trace;

extern bool useSingleImageKernel;
extern bool useDeviceLocal;
extern bool disableNTHandleType;
// Enable offset for multiImport of vulkan device memory
extern bool enableOffset;

#endif // _vulkan_interop_common_hpp_
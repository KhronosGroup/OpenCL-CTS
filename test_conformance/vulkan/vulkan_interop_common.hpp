//
// Copyright (c) 2022 The Khronos Group Inc.
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

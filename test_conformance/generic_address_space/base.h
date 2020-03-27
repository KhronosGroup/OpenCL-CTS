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
#include "harness/testHarness.h"

#include <string>

class CTest  {
public:
    virtual int Execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) = 0;
};

#define NL "\n"

namespace common {
    static const std::string CONFORMANCE_VERIFY_FENCE =
        NL
        NL "// current spec says get_fence can return any valid fence"
        NL "bool isFenceValid(cl_mem_fence_flags fence) {"
        NL "    if ((fence == 0) || (fence == CLK_GLOBAL_MEM_FENCE) || (fence == CLK_LOCAL_MEM_FENCE) "
        NL "        || (fence == (CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)))"
        NL "        return true;"
        NL "    else"
        NL "        return false;"
        NL "}"
        NL;

    static std::string GLOBAL_KERNEL_FUNCTION = CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(uint *ptr, uint tid) {"
        NL "    if (!isFenceValid(get_fence(ptr)))"
        NL "        return false;"
        NL
        NL "    if (*ptr != tid)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results, __global uint *buf) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    results[tid] = helperFunction(&buf[tid], tid);"
        NL "}"
        NL;

    static std::string LOCAL_KERNEL_FUNCTION = CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(uint *ptr, uint tid) {"
        NL "    if (!isFenceValid(get_fence(ptr)))"
        NL "        return false;"
        NL
        NL "    if (*ptr != tid)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results, __local uint *buf) {"
        NL "    uint tid = get_global_id(0);"
        NL "    if (get_local_id(0) == 0) {"
        NL "        for (uint i = 0; i < get_local_size(0); ++i) {"
        NL "            uint idx = get_local_size(0) * get_group_id(0) + i;"
        NL "            buf[idx] = idx;"
        NL "        }"
        NL "    }"
        NL
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL "    results[tid] = helperFunction(&buf[tid], tid);"
        NL "}"
        NL;
}

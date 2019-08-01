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
}

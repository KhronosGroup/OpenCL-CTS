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

#include "harness/compat.h"
#include "harness/testHarness.h"

#include "testBase.h"

const cl_mem_flags flag_set[] = { CL_MEM_ALLOC_HOST_PTR,
                                  CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                                  CL_MEM_USE_HOST_PTR,
                                  CL_MEM_COPY_HOST_PTR,
                                  0,
                                  CL_MEM_IMMUTABLE_EXT | CL_MEM_USE_HOST_PTR,
                                  CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR,
                                  CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR
                                      | CL_MEM_ALLOC_HOST_PTR };
const char* flag_set_names[] = {
    "CL_MEM_ALLOC_HOST_PTR",
    "CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR",
    "CL_MEM_USE_HOST_PTR",
    "CL_MEM_COPY_HOST_PTR",
    "0",
    "CL_MEM_IMMUTABLE_EXT | CL_MEM_USE_HOST_PTR",
    "CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR",
    "CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR",
};

int main( int argc, const char *argv[] )
{
    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}

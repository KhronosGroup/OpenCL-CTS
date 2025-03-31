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

#include <stdio.h>
#include <string.h>

#if !defined(__APPLE__)
#include <CL/cl.h>
#endif

#include "harness/testHarness.h"
#include "harness/imageHelpers.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

int main(int argc, const char *argv[])
{
    log_info("1st part, non gl-sharing objects...\n");
    gTestRounding = true;
    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}

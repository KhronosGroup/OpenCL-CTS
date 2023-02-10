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

#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST(mem_host_read_only_buffer),
    ADD_TEST(mem_host_read_only_subbuffer),
    ADD_TEST(mem_host_write_only_buffer),
    ADD_TEST(mem_host_write_only_subbuffer),
    ADD_TEST(mem_host_no_access_buffer),
    ADD_TEST(mem_host_no_access_subbuffer),
    ADD_TEST(mem_host_read_only_image),
    ADD_TEST(mem_host_write_only_image),
    ADD_TEST(mem_host_no_access_image),
};

const int test_num = ARRAY_SIZE(test_list);

int main(int argc, const char *argv[])
{
    log_info("1st part, non gl-sharing objects...\n");
    gTestRounding = true;
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}

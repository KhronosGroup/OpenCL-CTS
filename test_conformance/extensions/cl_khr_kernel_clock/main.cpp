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
#include "procs.h"
#include "harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST(device_scope),
    ADD_TEST(workgroup_scope),
    ADD_TEST(subgroup_scope),
};


int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, ARRAY_SIZE(test_list), test_list, false,
                          0);
}

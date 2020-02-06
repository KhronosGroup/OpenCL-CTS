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
#include "../common.hpp"

#include "test_read.hpp"
#include "test_sample.hpp"
#include "test_write.hpp"

// FIXME: To use certain functions in test_common/harness/imageHelpers.h
// (for example, generate_random_image_data()), the tests are required to declare
// the following variable (hangover from code specific to Apple's implementation):

int main(int argc, const char *argv[])
{
    auto& tests = autotest::test_suite::global_test_suite().test_defs;
    return runTestHarness(argc, argv, tests.size(), tests.data(), false, false, 0);
}

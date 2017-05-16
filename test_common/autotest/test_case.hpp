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
#ifndef TEST_COMMON_AUTOTEST_TEST_CASE_HPP
#define TEST_COMMON_AUTOTEST_TEST_CASE_HPP

#include <string>

#include "../../test_common/harness/threadTesting.h"

namespace autotest
{

struct test_case {
    // Test case name
    const std::string name;
    // Pointer to test function.
    const basefn function_pointer;

    test_case(const std::string& name, const basefn function_ptr)
        : name(name), function_pointer(function_ptr)
    {

    }
};

} // end namespace autotest

#endif // TEST_COMMON_AUTOTEST_TEST_CASE_HPP

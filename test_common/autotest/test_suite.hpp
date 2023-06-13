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
#ifndef TEST_COMMON_AUTOTEST_TEST_SUITE_HPP
#define TEST_COMMON_AUTOTEST_TEST_SUITE_HPP

#include <vector>
#include <string> 

namespace autotest {

struct test_suite {
    test_suite(const std::string& name)
        : name(name)
    {

    } 

    void add(const test_definition& td)
    {
        test_defs.push_back(td);
    }    

    // List of test definitions
    std::vector<test_definition> test_defs;
    // Test suite name
    const std::string name;

    static test_suite& global_test_suite()
    {
        static test_suite global_test_suite("global");
        return global_test_suite;
    }
};

namespace detail {

struct test_case_registration
{
    test_case_registration(const std::string& name,
                           const test_function_pointer ptr)
    {
        ::autotest::test_suite::global_test_suite().add(
            test_definition({ ptr, strdup(name.c_str()) }));
    }
};

} // end detail namespace
} // end autotest namespace

#endif // TEST_COMMON_AUTOTEST_TEST_SUITE_HPP

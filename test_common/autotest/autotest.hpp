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
#ifndef TEST_COMMON_AUTOTEST_AUTOTEST_HPP
#define TEST_COMMON_AUTOTEST_AUTOTEST_HPP

#include "test_suite.hpp"
#include "test_case.hpp"

namespace autotest {
    inline std::vector<const char*> get_strings_ptrs(const std::vector<std::string>& list)
    {
        std::vector<const char*> v;
        for(auto& s : list)
        {
            v.push_back(s.c_str());
        }
        return v;
    }
}

#define STR_JOIN( X, Y ) STR_DO_JOIN( X, Y )
#define STR_DO_JOIN( X, Y ) STR_DO_JOIN_2(X,Y)
#define STR_DO_JOIN_2( X, Y ) X##Y


// How to use AUTO_TEST_CASE macro:
//
// AUTO_TEST_CASE(<test_case_name>)(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
// {
//      (test case code...)
// }
//
// It automatically registers created test case to global test_suite object. Test functions
// names and pointers to those functions can be later retrieved this way: 
// - std::vector<basefn> test_functions_list = autotest::test_suite::get_test_functions();
// - std::vector<std::string> test_functions_names = autotest::test_suite::get_test_names();
//
// Helper function which constructs vector of const char pointers to test functions names:
// - std::vector<const char *> test_functions_names_c_str = autotest::get_strings_ptrs(test_functions_names);
#define AUTO_TEST_CASE(name) \
    struct name { static int run_test(cl_device_id, cl_context, cl_command_queue, int); }; \
    static autotest::detail::test_case_registration STR_JOIN(name, STR_JOIN(_registration, __LINE__)) (#name, name::run_test); \
    int name::run_test

#endif //TEST_COMMON_AUTOTEST_AUTOTEST_HPP
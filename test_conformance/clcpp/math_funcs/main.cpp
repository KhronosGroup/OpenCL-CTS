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
#include <limits>

#include "../common.hpp"

#include "comparison_funcs.hpp"
#include "exponential_funcs.hpp"
#include "floating_point_funcs.hpp"
#include "half_math_funcs.hpp"
#include "logarithmic_funcs.hpp"
#include "other_funcs.hpp"
#include "power_funcs.hpp"
#include "trigonometric_funcs.hpp"

int main(int argc, const char *argv[])
{
    // Check if cl_float (float) and cl_double (double) fulfill the requirements of
    // IEC 559 (IEEE 754) standard. This is required for the tests to run correctly.
    if(!std::numeric_limits<cl_float>::is_iec559)
    {
        RETURN_ON_ERROR_MSG(-1,
            "cl_float (float) does not fulfill the requirements of IEC 559 (IEEE 754) standard. "
            "Tests won't run correctly."
        );
    }
    if(!std::numeric_limits<cl_double>::is_iec559)
    {
        RETURN_ON_ERROR_MSG(-1,
            "cl_double (double) does not fulfill the requirements of IEC 559 (IEEE 754) standard. "
            "Tests won't run correctly."
        );
    }

    // Get list to all test functions
    std::vector<basefn> testfn_list = autotest::test_suite::get_test_functions();
    // Get names of all test functions
    std::vector<std::string> testfn_names = autotest::test_suite::get_test_names();
    // Create a vector of pointers to the names test functions
    std::vector<const char *> testfn_names_c_str = autotest::get_strings_ptrs(testfn_names);
    return runTestHarness(argc, argv, testfn_list.size(), testfn_list.data(), testfn_names_c_str.data(), false, false, 0);
}

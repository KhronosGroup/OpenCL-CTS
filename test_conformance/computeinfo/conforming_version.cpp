
//
// Copyright (c) 2020 The Khronos Group Inc.
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

#include <regex>
#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

REGISTER_TEST_VERSION(conformance_version, Version(3, 0))
{
    std::string version_string{ get_device_info_string(
        device, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED) };

    // Latest conformance version passed should match vYYYY-MM-DD-XX, where XX
    // is a number
    std::regex valid_format("^v\\d{4}-(((0)[1-9])|((1)[0-2]))-((0)[1-9]|[1-2]["
                            "0-9]|(3)[0-1])-\\d{2}$");
    test_assert_error(
        std::regex_match(version_string, valid_format),
        "CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED does not return "
        "valid format vYYYY-MM-DD-XX");

    return TEST_PASS;
}

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

#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

test_status InitCL(cl_device_id device)
{
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(3, 0);

    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    if (!is_extension_available(device, "cl_ext_buffer_device_address"))
    {
        log_info("The device does not support the "
                 "cl_ext_buffer_device_address extension.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_version ext_version =
        get_extension_version(device, "cl_ext_buffer_device_address");
    if (ext_version != CL_MAKE_VERSION(1, 0, 2))
    {
        log_info("The test is written against cl_ext_buffer_device_address "
                 "extension version 1.0.2, device supports version: %u.%u.%u\n",
                 CL_VERSION_MAJOR(ext_version), CL_VERSION_MINOR(ext_version),
                 CL_VERSION_PATCH(ext_version));
        return TEST_SKIPPED_ITSELF;
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

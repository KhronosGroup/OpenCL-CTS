//
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

#include "common.h"
#include <cinttypes>

REGISTER_TEST(unified_svm_consistency)
{
    if (!is_extension_available(device, "cl_khr_unified_svm"))
    {
        log_info("cl_khr_unified_svm is not supported, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int err;

    cl_platform_id platformID;
    err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                          (void *)(&platformID), nullptr);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");

    cl_uint numDevices = 0;
    err =
        clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    test_error(err, "clGetDeviceIDs failed to get number of devices");

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, numDevices,
                         devices.data(), nullptr);
    test_error(err, "clGetDeviceIDs failed to get device IDs");

    // For each device in the platform, check that the platform and device
    // report the same number of SVM capability combinations.

    size_t platformSize{};
    err = clGetPlatformInfo(platformID, CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR,
                            0, nullptr, &platformSize);
    test_error(err,
               "clGetPlatformInfo failed for "
               "CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR");

    if (platformSize % sizeof(cl_svm_capabilities_khr) != 0)
    {
        test_fail(
            "Unexpected platform SVM type capabilities size: %zu bytes.\n",
            platformSize);
    }

    for (auto device : devices)
    {
        size_t deviceSize{};
        err = clGetDeviceInfo(device, CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR, 0,
                              nullptr, &deviceSize);
        test_error(
            err,
            "clGetDeviceInfo failed for CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR");

        if (deviceSize % sizeof(cl_svm_capabilities_khr) != 0)
        {
            test_fail("Unexpected device SVM type capabilities size: "
                      "%zu bytes.\n",
                      deviceSize);
        }
        if (platformSize != deviceSize)
        {
            test_fail("Platform and device report different number of "
                      "SVM type capability combinations.\n");
        }
    }

    // For each SVM capability combination reported by the platform, check that
    // the reported platform capabilities at an index are the intersection of
    // all non-zero device capabilities at the same index.

    size_t capabilityCount = platformSize / sizeof(cl_svm_capabilities_khr);

    std::vector<cl_svm_capabilities_khr> platformCapabilities(capabilityCount);
    err = clGetPlatformInfo(platformID, CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR,
                            platformSize, platformCapabilities.data(), nullptr);
    test_error(err,
               "clGetPlatformInfo failed for "
               "CL_PLATFORM_SVM_TYPE_CAPABILITIES_KHR");

    for (int i = 0; i < capabilityCount; i++)
    {
        cl_svm_capabilities_khr check = 0;
        for (auto device : devices)
        {
            std::vector<cl_svm_capabilities_khr> deviceCapabilities(
                capabilityCount);
            err = clGetDeviceInfo(device, CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR,
                                  platformSize, deviceCapabilities.data(),
                                  nullptr);
            test_error(
                err,
                "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES_KHR");

            if (deviceCapabilities[i] != 0)
            {
                if (check == 0)
                {
                    check = deviceCapabilities[i];
                }
                else
                {
                    check &= deviceCapabilities[i];
                }
            }
        }
        if (platformCapabilities[i] != check)
        {
            test_fail("Platform SVM type capabilities at index %d: 0x%" PRIx64
                      " do not match the intersection of device capabilities "
                      "0x%" PRIx64 ".\n",
                      i, platformCapabilities[i], check);
        }
    }

    // For each SVM capability combination reported by the test device, check
    // that the device SVM capabilities are either a super-set of the platform
    // SVM capabilities or are zero, indicating that this SVM type is not
    // supported.

    std::vector<cl_svm_capabilities_khr> deviceCapabilities(capabilityCount);
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR,
                          platformSize, deviceCapabilities.data(), nullptr);
    test_error(err,
               "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES_KHR");

    for (int i = 0; i < capabilityCount; i++)
    {
        bool consistent = (deviceCapabilities[i] & platformCapabilities[i])
                == platformCapabilities[i]
            || deviceCapabilities[i] == 0;
        if (!consistent)
        {
            test_fail(
                "Device SVM type capabilities at index %d: 0x%" PRIx64
                " are not consistent with platform SVM type capabilities: "
                "0x%" PRIx64 ".\n",
                i, deviceCapabilities[i], platformCapabilities[i]);
        }
    }

    return TEST_PASS;
}

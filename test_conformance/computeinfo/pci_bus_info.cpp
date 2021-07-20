//
// Copyright (c) 2021 The Khronos Group Inc.
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

#include <array>
#include <bitset>

#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

int test_pci_bus_info(cl_device_id deviceID, cl_context context,
                      cl_command_queue ignoreQueue, int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_pci_bus_info"))
    {
        log_info("cl_khr_pci_bus_info not supported. Skipping test...\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error;

    cl_device_pci_bus_info_khr info;

    size_t size_ret;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_PCI_BUS_INFO_KHR, 0, NULL,
                            &size_ret);
    test_error(error, "Unable to query CL_DEVICE_PCI_BUS_INFO_KHR size");
    test_assert_error(
        size_ret == sizeof(info),
        "Query for CL_DEVICE_PCI_BUS_INFO_KHR returned an unexpected size");

    error = clGetDeviceInfo(deviceID, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(info),
                            &info, NULL);
    test_error(error, "Unable to query CL_DEVICE_PCI_BUS_INFO_KHR");

    log_info("\tPCI Bus Info: %04x:%02x:%02x.%x\n", info.pci_domain,
             info.pci_bus, info.pci_device, info.pci_function);

    return TEST_PASS;
}

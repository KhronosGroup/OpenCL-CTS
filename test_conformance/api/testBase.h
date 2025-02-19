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
#ifndef _testBase_h
#define _testBase_h

#include "harness/compat.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/imageHelpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

// scope guard helper to ensure proper releasing of sub devices
struct SubDevicesScopeGuarded
{
    SubDevicesScopeGuarded(const cl_int dev_count)
    {
        sub_devices.resize(dev_count);
    }
    cl_int erase(std::vector<cl_device_id>::iterator it)
    {
        if (it != sub_devices.end())
        {
            cl_int err = clReleaseDevice(*it);
            test_error(err, "\n Releasing sub-device failed \n");
            sub_devices.erase(it);
        }
        return CL_SUCCESS;
    }
    ~SubDevicesScopeGuarded()
    {
        for (auto &device : sub_devices)
        {
            cl_int err = clReleaseDevice(device);
            if (err != CL_SUCCESS)
                log_error("\n Releasing sub-device failed \n");
        }
    }

    std::vector<cl_device_id> sub_devices;
};

#endif // _testBase_h

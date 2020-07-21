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
//

#include "errorHelpers.h"
#include "version.h"

#include <cstring>
#include <vector>

Version get_device_cl_version(cl_device_id device)
{
    size_t str_size;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &str_size);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    std::vector<char> str(str_size);
    err =
        clGetDeviceInfo(device, CL_DEVICE_VERSION, str_size, str.data(), NULL);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    if (strstr(str.data(), "OpenCL 1.0") != NULL)
        return Version(1, 0);
    else if (strstr(str.data(), "OpenCL 1.1") != NULL)
        return Version(1, 1);
    else if (strstr(str.data(), "OpenCL 1.2") != NULL)
        return Version(1, 2);
    else if (strstr(str.data(), "OpenCL 2.0") != NULL)
        return Version(2, 0);
    else if (strstr(str.data(), "OpenCL 2.1") != NULL)
        return Version(2, 1);
    else if (strstr(str.data(), "OpenCL 2.2") != NULL)
        return Version(2, 2);
    else if (strstr(str.data(), "OpenCL 3.0") != NULL)
        return Version(3, 0);

    throw std::runtime_error(std::string("Unknown OpenCL version: ")
                             + str.data());
}

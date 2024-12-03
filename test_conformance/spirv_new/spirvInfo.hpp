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

#pragma once

#include "harness/compat.h"

#include <string>

extern bool gVersionSkip;

static bool is_spirv_version_supported(cl_device_id deviceID,
                                       const char* version)
{
    std::string ilVersions = get_device_il_version_string(deviceID);

    if (gVersionSkip)
    {
        log_info("    Skipping version check for %s.\n", version);
        return true;
    }
    else if (ilVersions.find(version) == std::string::npos)
    {
        return false;
    }

    return true;
}

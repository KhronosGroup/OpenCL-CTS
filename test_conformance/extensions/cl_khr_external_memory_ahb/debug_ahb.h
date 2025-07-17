//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include <android/hardware_buffer.h>
#include <string>
#include <vector>
#include <numeric>

#define CHECK_AHARDWARE_BUFFER_SUPPORT(ahardwareBuffer_Desc, format)           \
    if (!AHardwareBuffer_isSupported(&ahardwareBuffer_Desc))                   \
    {                                                                          \
        const std::string usage_string =                                       \
            ahardwareBufferDecodeUsageFlagsToString(                           \
                static_cast<AHardwareBuffer_UsageFlags>(                       \
                    ahardwareBuffer_Desc.usage));                              \
        log_info("Unsupported format %s:\n   Usage flags %s\n   Size (%u, "    \
                 "%u, layers = %u)\n",                                         \
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)   \
                     .c_str(),                                                 \
                 usage_string.c_str(), ahardwareBuffer_Desc.width,             \
                 ahardwareBuffer_Desc.height, ahardwareBuffer_Desc.layers);    \
        continue;                                                              \
    }

std::string ahardwareBufferFormatToString(AHardwareBuffer_Format format);
std::string ahardwareBufferUsageFlagToString(AHardwareBuffer_UsageFlags flag);
std::string
ahardwareBufferDecodeUsageFlagsToString(AHardwareBuffer_UsageFlags flags);
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

#include "debug_ahb.h"

constexpr AHardwareBuffer_UsageFlags flag_list[] = {
    AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
    AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK,
    AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
    AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER,
    AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY,
    AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT,
    AHARDWAREBUFFER_USAGE_VIDEO_ENCODE,
    AHARDWAREBUFFER_USAGE_SENSOR_DIRECT_DATA,
    AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER,
    AHARDWAREBUFFER_USAGE_GPU_CUBE_MAP,
    AHARDWAREBUFFER_USAGE_GPU_MIPMAP_COMPLETE,
    AHARDWAREBUFFER_USAGE_FRONT_BUFFER,
};

std::string
ahardwareBufferDecodeUsageFlagsToString(const AHardwareBuffer_UsageFlags flags)
{
    if (flags == 0)
    {
        return "UNKNOWN FLAG";
    }

    std::vector<std::string> active_flags;
    for (const auto flag : flag_list)
    {
        if (flag & flags)
        {
            active_flags.push_back(ahardwareBufferUsageFlagToString(flag));
        }
    }

    if (active_flags.empty())
    {
        return "UNKNOWN FLAG";
    }

    return std::accumulate(active_flags.begin() + 1, active_flags.end(),
                           active_flags.front(),
                           [](std::string acc, const std::string& flag) {
                               return std::move(acc) + "|" + flag;
                           });
}

std::string
ahardwareBufferUsageFlagToString(const AHardwareBuffer_UsageFlags flag)
{
    std::string result;
    switch (flag)
    {
        case AHARDWAREBUFFER_USAGE_CPU_READ_NEVER:
            result = "AHARDWAREBUFFER_USAGE_CPU_READ_NEVER";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_READ_RARELY:
            result = "AHARDWAREBUFFER_USAGE_CPU_READ_RARELY";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN:
            result = "AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_READ_MASK:
            result = "AHARDWAREBUFFER_USAGE_CPU_READ_MASK";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY:
            result = "AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN:
            result = "AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN";
            break;
        case AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK:
            result = "AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK";
            break;
        case AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE:
            result = "AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE";
            break;
        case AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER:
            result = "AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER";
            break;
        case AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY:
            result = "AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY";
            break;
        case AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT:
            result = "AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT";
            break;
        case AHARDWAREBUFFER_USAGE_VIDEO_ENCODE:
            result = "AHARDWAREBUFFER_USAGE_VIDEO_ENCODE";
            break;
        case AHARDWAREBUFFER_USAGE_SENSOR_DIRECT_DATA:
            result = "AHARDWAREBUFFER_USAGE_SENSOR_DIRECT_DATA";
            break;
        case AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER:
            result = "AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER";
            break;
        case AHARDWAREBUFFER_USAGE_GPU_CUBE_MAP:
            result = "AHARDWAREBUFFER_USAGE_GPU_CUBE_MAP";
            break;
        case AHARDWAREBUFFER_USAGE_GPU_MIPMAP_COMPLETE:
            result = "AHARDWAREBUFFER_USAGE_GPU_MIPMAP_COMPLETE";
            break;
        default: result = "Unknown flag";
    }
    return result;
}

std::string ahardwareBufferFormatToString(AHardwareBuffer_Format format)
{
    std::string result;
    switch (format)
    {
        case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT:
            result = "AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT";
            break;
        case AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_BLOB:
            result = "AHARDWAREBUFFER_FORMAT_BLOB";
            break;
        case AHARDWAREBUFFER_FORMAT_D16_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_D16_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_D24_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_D24_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT:
            result = "AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_D32_FLOAT:
            result = "AHARDWAREBUFFER_FORMAT_D32_FLOAT";
            break;
        case AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT:
            result = "AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_S8_UINT:
            result = "AHARDWAREBUFFER_FORMAT_S8_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420:
            result = "AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420";
            break;
        case AHARDWAREBUFFER_FORMAT_YCbCr_P010:
            result = "AHARDWAREBUFFER_FORMAT_YCbCr_P010";
            break;
        case AHARDWAREBUFFER_FORMAT_YCbCr_P210:
            result = "AHARDWAREBUFFER_FORMAT_YCbCr_P210";
            break;
        case AHARDWAREBUFFER_FORMAT_R8_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R8_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R16_UINT:
            result = "AHARDWAREBUFFER_FORMAT_R16_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_R16G16_UINT:
            result = "AHARDWAREBUFFER_FORMAT_R16G16_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM";
            break;
    }
    return result;
}

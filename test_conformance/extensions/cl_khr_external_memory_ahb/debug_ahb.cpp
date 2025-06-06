//
// Created by joshkell on 8/18/24.
//

#include <cstring>
#include <cstdlib>
#include "debug_ahb.h"

AHardwareBuffer_UsageFlags flag_list[] = {
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
    AHARDWAREBUFFER_USAGE_FRONT_BUFFER, // This is not in older NDK 25
};

char *ahardwareBufferDecodeUsageFlagsToString(AHardwareBuffer_UsageFlags flags)
{
    size_t flags_len = 0;
    size_t num_flags = 0;
    const char *separator = "|";

    for (uint64_t f : flag_list)
    {
        if (((f & flags) != 0) && ((f & flags) == f))
        {
            flags_len += strlen(ahardwareBufferUsageFlagToString(
                static_cast<AHardwareBuffer_UsageFlags>(f)));
            num_flags++;
        }
    }

    if (num_flags == 0)
    {
        const char *unknown_flag = "UNKNOWN_FLAG";
        size_t res_size = strlen(unknown_flag) + 1;
        char *result = new char[res_size];
        strlcat(result, unknown_flag, res_size);
        return result;
    }

    size_t string_len = flags_len + ((num_flags - 1) * strlen(separator)) + 1;
    char *result = new char[string_len];
    memset(result, 0, string_len);

    size_t flag_counter = 0;
    for (uint64_t f : flag_list)
    {
        if (((f & flags) != 0) && ((f & flags) == f))
        {
            flag_counter++;
            strlcat(result,
                    ahardwareBufferUsageFlagToString(
                        static_cast<AHardwareBuffer_UsageFlags>(f)),
                    string_len);
            if (flag_counter < num_flags)
            {
                strlcat(result, separator, string_len);
            }
        }
    }

    return result;
}

const char *ahardwareBufferUsageFlagToString(AHardwareBuffer_UsageFlags flag)
{
    const char *result = "";
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

const char *ahardwareBufferFormatToString(AHardwareBuffer_Format format)
{
    const char *result = "";
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
        case AHARDWAREBUFFER_FORMAT_R8_UNORM:
            result = "AHARDWAREBUFFER_FORMAT_R8_UNORM";
            break;
        case AHARDWAREBUFFER_FORMAT_R16_UINT: // This is not in older NDK 25
            result = "AHARDWAREBUFFER_FORMAT_R16_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_R16G16_UINT: // This is not in older NDK 25
            result = "AHARDWAREBUFFER_FORMAT_R16G16_UINT";
            break;
        case AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM: // This is not in older
                                                        // NDK 25
            result = "AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM";
            break;
    }
    return result;
}
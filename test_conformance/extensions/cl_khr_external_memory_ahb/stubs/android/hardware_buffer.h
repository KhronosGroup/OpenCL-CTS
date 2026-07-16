#ifndef STUBS_ANDROID_HARDWARE_BUFFER_H
#define STUBS_ANDROID_HARDWARE_BUFFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AHardwareBuffer AHardwareBuffer;

typedef struct AHardwareBuffer_Desc
{
    uint32_t width;
    uint32_t height;
    uint32_t layers;
    uint32_t format;
    uint64_t usage;
    uint32_t stride;
    uint32_t rfu0;
    uint64_t rfu1;
} AHardwareBuffer_Desc;

enum AHardwareBuffer_Format
{
    AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM = 1,
    AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM = 2,
    AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM = 3,
    AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM = 4,
    AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT = 0x16,
    AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM = 0x2b,
    AHARDWAREBUFFER_FORMAT_BLOB = 0x21,
    AHARDWAREBUFFER_FORMAT_D16_UNORM = 0x30,
    AHARDWAREBUFFER_FORMAT_D24_UNORM = 0x31,
    AHARDWAREBUFFER_FORMAT_D24_UNORM_S8_UINT = 0x32,
    AHARDWAREBUFFER_FORMAT_D32_FLOAT = 0x33,
    AHARDWAREBUFFER_FORMAT_D32_FLOAT_S8_UINT = 0x34,
    AHARDWAREBUFFER_FORMAT_S8_UINT = 0x35,
    AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420 = 0x23,
    AHARDWAREBUFFER_FORMAT_YCbCr_P010 = 0x36,
    AHARDWAREBUFFER_FORMAT_YCbCr_P210 = 0x37,
    AHARDWAREBUFFER_FORMAT_R8_UNORM = 0x38,
    AHARDWAREBUFFER_FORMAT_R16_UINT = 0x39,
    AHARDWAREBUFFER_FORMAT_R16G16_UINT = 0x3a,
    AHARDWAREBUFFER_FORMAT_R10G10B10A10_UNORM = 0x3b,
};

typedef enum AHardwareBuffer_UsageFlags
{
    AHARDWAREBUFFER_USAGE_CPU_READ_NEVER = 0UL,
    AHARDWAREBUFFER_USAGE_CPU_READ_RARELY = 2UL,
    AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN = 3UL,
    AHARDWAREBUFFER_USAGE_CPU_READ_MASK = 0xfUL,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER = 0UL << 4,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY = 2UL << 4,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN = 3UL << 4,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK = 0xfUL << 4,
    AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE = 1UL << 8,
    AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER = 1UL << 9,
    AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY = 1UL << 11,
    AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT = 1UL << 14,
    AHARDWAREBUFFER_USAGE_VIDEO_ENCODE = 1UL << 16,
    AHARDWAREBUFFER_USAGE_SENSOR_DIRECT_DATA = 1UL << 23,
    AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER = 1UL << 24,
    AHARDWAREBUFFER_USAGE_GPU_CUBE_MAP = 1UL << 25,
    AHARDWAREBUFFER_USAGE_GPU_MIPMAP_COMPLETE = 1UL << 26,
    AHARDWAREBUFFER_USAGE_FRONT_BUFFER = 1ULL << 32,
} AHardwareBuffer_UsageFlags;

typedef struct ARect
{
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
} ARect;

int AHardwareBuffer_allocate(const AHardwareBuffer_Desc* desc,
                             AHardwareBuffer** outBuffer);
void AHardwareBuffer_acquire(AHardwareBuffer* buffer);
void AHardwareBuffer_release(AHardwareBuffer* buffer);
void AHardwareBuffer_describe(const AHardwareBuffer* buffer,
                              AHardwareBuffer_Desc* outDesc);
int AHardwareBuffer_lock(AHardwareBuffer* buffer, uint64_t usage, int32_t fence,
                         const ARect* rect, void** outVirtualAddress);
int AHardwareBuffer_unlock(AHardwareBuffer* buffer, int32_t* fence);
bool AHardwareBuffer_isSupported(const AHardwareBuffer_Desc* desc);

#ifdef __cplusplus
}
#endif

#endif // STUBS_ANDROID_HARDWARE_BUFFER_H

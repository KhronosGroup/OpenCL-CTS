#include <android/hardware_buffer.h>
#include "harness/errorHelpers.h"

extern "C" {

int AHardwareBuffer_allocate(const AHardwareBuffer_Desc* desc,
                             AHardwareBuffer** outBuffer)
{
    log_error("Stub AHardwareBuffer_allocate called\n");
    return -1;
}

void AHardwareBuffer_acquire(AHardwareBuffer* buffer)
{
    log_info("Stub AHardwareBuffer_acquire called\n");
}

void AHardwareBuffer_release(AHardwareBuffer* buffer)
{
    log_info("Stub AHardwareBuffer_release called\n");
}

void AHardwareBuffer_describe(const AHardwareBuffer* buffer,
                              AHardwareBuffer_Desc* outDesc)
{
    log_error("Stub AHardwareBuffer_describe called\n");
    if (outDesc)
    {
        *outDesc = AHardwareBuffer_Desc{};
    }
}

int AHardwareBuffer_lock(AHardwareBuffer* buffer, uint64_t usage, int32_t fence,
                         const ARect* rect, void** outVirtualAddress)
{
    log_error("Stub AHardwareBuffer_lock called\n");
    return -1;
}

int AHardwareBuffer_unlock(AHardwareBuffer* buffer, int32_t* fence)
{
    log_error("Stub AHardwareBuffer_unlock called\n");
    return -1;
}

bool AHardwareBuffer_isSupported(const AHardwareBuffer_Desc* desc)
{
    log_info("Stub AHardwareBuffer_isSupported called\n");
    return false;
}
}

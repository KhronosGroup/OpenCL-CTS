#pragma once

#include <android/hardware_buffer.h>

const char * ahardwareBufferFormatToString(AHardwareBuffer_Format format);
const char * ahardwareBufferUsageFlagToString(AHardwareBuffer_UsageFlags flag);
char * ahardwareBufferDecodeUsageFlagsToString(AHardwareBuffer_UsageFlags flags);
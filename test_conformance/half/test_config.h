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
#ifndef TEST_CONFIG_H
#define TEST_CONFIG_H

#include <CL/cl.h>

#define MULTITHREAD 1

#define kVectorSizeCount    5
#define kStrangeVectorSizeCount 1
#define kMinVectorSize      0
#define kLargestVectorSize      (1 << (kVectorSizeCount-1))

#define kLastVectorSizeToTest (kVectorSizeCount + kStrangeVectorSizeCount)

#define BUFFER_SIZE ((size_t)2 * 1024 * 1024)

extern size_t getBufferSize(cl_device_id device_id);
extern cl_ulong getBufferCount(cl_device_id device_id, size_t vecSize, size_t typeSize);
// could call
// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
#define kPageSize       4096

extern int g_arrVecSizes[kVectorSizeCount+kStrangeVectorSizeCount];
extern int g_arrVecAligns[kLargestVectorSize+1];

#endif /* TEST_CONFIG_H */



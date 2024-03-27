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
#include "harness/kernelHelpers.h"
#include "harness/clImageHelper.h"
#include "harness/imageHelpers.h"

extern bool gDebugTrace;
extern bool gTestSmallImages;
extern bool gEnablePitch;
extern bool gTestMaxImages;
extern bool gTestMipmaps;

// Amount to offset pixels for checking normalized reads
#define NORM_OFFSET 0.1f

enum TypesToTest
{
    kTestInt = ( 1 << 0 ),
    kTestUInt = ( 1 << 1 ),
    kTestFloat = ( 1 << 2 ),
    kTestAllTypes = kTestInt | kTestUInt | kTestFloat
};

// For the clCopyImage test
enum MethodsToTest
{
    k1D = (1 << 0),
    k2D = (1 << 1),
    k1DArray = (1 << 2),
    k2DArray = (1 << 3),
    k3D = (1 << 4),
    k2DTo3D = (1 << 5),
    k3DTo2D = (1 << 6),
    k2DArrayTo2D = (1 << 7),
    k2DTo2DArray = (1 << 8),
    k2DArrayTo3D = (1 << 9),
    k3DTo2DArray = (1 << 10),
    k1DBuffer = (1 << 11),
    k1DTo1DBuffer = (1 << 12),
    k1DBufferTo1D = (1 << 13),
};


enum TestTypes
{
    kReadTests = 1 << 0 ,
    kWriteTests = 1 << 1,
    kReadWriteTests = 1 << 2,
    kAllTests = ( kReadTests | kWriteTests | kReadWriteTests )
};

typedef int (*test_format_set_fn)(
    cl_device_id device, cl_context context, cl_command_queue queue,
    const std::vector<cl_image_format> &formatList,
    const std::vector<bool> &filterFlags, image_sampler_data *imageSampler,
    ExplicitType outputType, cl_mem_object_type imageType);

extern int test_read_image_formats(
    cl_device_id device, cl_context context, cl_command_queue queue,
    const std::vector<cl_image_format> &formatList,
    const std::vector<bool> &filterFlags, image_sampler_data *imageSampler,
    ExplicitType outputType, cl_mem_object_type imageType);
extern int test_write_image_formats(
    cl_device_id device, cl_context context, cl_command_queue queue,
    const std::vector<cl_image_format> &formatList,
    const std::vector<bool> &filterFlags, image_sampler_data *imageSampler,
    ExplicitType outputType, cl_mem_object_type imageType);

#endif // _testBase_h




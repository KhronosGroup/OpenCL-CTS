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
#ifndef IMAGES_COMMON_H
#define IMAGES_COMMON_H

#include "testBase.h"
#include "harness/kernelHelpers.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"

#include <array>

extern cl_channel_type gChannelTypeToUse;
extern cl_channel_order gChannelOrderToUse;

extern cl_channel_type floatFormats[];
extern cl_channel_type intFormats[];
extern cl_channel_type uintFormats[];

struct ImageTestTypes
{
    TypesToTest type;
    ExplicitType explicitType;
    cl_channel_type *channelTypes;
    const char *name;
};

extern std::array<ImageTestTypes, 3> imageTestTypes;

const char *convert_image_type_to_string(cl_mem_object_type imageType);
int filter_formats(cl_image_format *formatList, bool *filterFlags,
                   unsigned int formatCount,
                   cl_channel_type *channelDataTypesToFilter,
                   bool testMipmaps = false);
int get_format_list(cl_context context, cl_mem_object_type imageType,
                    cl_image_format *&outFormatList,
                    unsigned int &outFormatCount, cl_mem_flags flags);
size_t random_in_ranges(size_t minimum, size_t rangeA, size_t rangeB, MTdata d);

#endif // IMAGES_COMMON_H

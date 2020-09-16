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
#include "common.h"

cl_channel_type floatFormats[] = {
    CL_UNORM_SHORT_565,
    CL_UNORM_SHORT_555,
    CL_UNORM_INT_101010,
#ifdef OBSOLETE_FORAMT
    CL_UNORM_SHORT_565_REV,
    CL_UNORM_SHORT_555_REV,
    CL_UNORM_INT_8888,
    CL_UNORM_INT_8888_REV,
    CL_UNORM_INT_101010_REV,
#endif
#ifdef CL_SFIXED14_APPLE
    CL_SFIXED14_APPLE,
#endif
    CL_UNORM_INT8,
    CL_SNORM_INT8,
    CL_UNORM_INT16,
    CL_SNORM_INT16,
    CL_FLOAT,
    CL_HALF_FLOAT,
    (cl_channel_type)-1,
};

cl_channel_type intFormats[] = {
    CL_SIGNED_INT8,
    CL_SIGNED_INT16,
    CL_SIGNED_INT32,
    (cl_channel_type)-1,
};

cl_channel_type uintFormats[] = {
    CL_UNSIGNED_INT8,
    CL_UNSIGNED_INT16,
    CL_UNSIGNED_INT32,
    (cl_channel_type)-1,
};

std::array<ImageTestTypes, 3> imageTestTypes = { {
    { kTestInt, kInt, intFormats, "int" },
    { kTestUInt, kUInt, uintFormats, "uint" },
    { kTestFloat, kFloat, floatFormats, "float" },
} };

const char *convert_image_type_to_string(cl_mem_object_type image_type)
{
    switch (image_type)
    {
        case CL_MEM_OBJECT_IMAGE1D: return "1D";
        case CL_MEM_OBJECT_IMAGE2D: return "2D";
        case CL_MEM_OBJECT_IMAGE3D: return "3D";
        case CL_MEM_OBJECT_IMAGE1D_ARRAY: return "1D array";
        case CL_MEM_OBJECT_IMAGE2D_ARRAY: return "2D array";
        case CL_MEM_OBJECT_IMAGE1D_BUFFER: return "1D image buffer";
        default: return "unrecognized object type";
    }
}

int filter_formats(cl_image_format *formatList, bool *filterFlags,
                   unsigned int formatCount,
                   cl_channel_type *channelDataTypesToFilter,
                   bool testMipmaps /*=false*/)
{
    int numSupported = 0;
    for (unsigned int j = 0; j < formatCount; j++)
    {
        // If this format has been previously filtered, remove the filter
        if (filterFlags[j]) filterFlags[j] = false;

        // skip mipmap tests for CL_DEPTH formats (re# Khronos Bug 13762)
        if (testMipmaps && (formatList[j].image_channel_order == CL_DEPTH))
        {
            log_info("Skip mipmap tests for CL_DEPTH format\n");
            filterFlags[j] = true;
            continue;
        }

        // Have we already discarded the channel type via the command line?
        if (gChannelTypeToUse != (cl_channel_type)-1
            && gChannelTypeToUse != formatList[j].image_channel_data_type)
        {
            filterFlags[j] = true;
            continue;
        }

        // Have we already discarded the channel order via the command line?
        if (gChannelOrderToUse != (cl_channel_order)-1
            && gChannelOrderToUse != formatList[j].image_channel_order)
        {
            filterFlags[j] = true;
            continue;
        }

        // Is given format standard channel order and type given by spec. We
        // don't want to test it if this is vendor extension
        if (!IsChannelOrderSupported(formatList[j].image_channel_order)
            || !IsChannelTypeSupported(formatList[j].image_channel_data_type))
        {
            filterFlags[j] = true;
            continue;
        }

        if (!channelDataTypesToFilter)
        {
            numSupported++;
            continue;
        }

        // Is the format supported?
        int i;
        for (i = 0; channelDataTypesToFilter[i] != (cl_channel_type)-1; i++)
        {
            if (formatList[j].image_channel_data_type
                == channelDataTypesToFilter[i])
            {
                numSupported++;
                break;
            }
        }
        if (channelDataTypesToFilter[i] == (cl_channel_type)-1)
        {
            // Format is NOT supported, so mark it as such
            filterFlags[j] = true;
        }
    }
    return numSupported;
}

int get_format_list(cl_context context, cl_mem_object_type imageType,
                    cl_image_format *&outFormatList,
                    unsigned int &outFormatCount, cl_mem_flags flags)
{
    int error = clGetSupportedImageFormats(context, flags, imageType, 0, NULL,
                                           &outFormatCount);
    test_error(error, "Unable to get count of supported image formats");

    outFormatList =
        (outFormatCount > 0) ? new cl_image_format[outFormatCount] : NULL;

    error = clGetSupportedImageFormats(context, flags, imageType,
                                       outFormatCount, outFormatList, NULL);
    test_error(error, "Unable to get list of supported image formats");
    return 0;
}

size_t random_in_ranges(size_t minimum, size_t rangeA, size_t rangeB, MTdata d)
{
    if (rangeB < rangeA) rangeA = rangeB;
    if (rangeA < minimum) return rangeA;
    return (size_t)random_in_range((int)minimum, (int)rangeA - 1, d);
}

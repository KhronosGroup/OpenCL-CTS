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
#include "../testBase.h"
#include "../common.h"

extern cl_filter_mode gFilterModeToUse;
extern cl_addressing_mode gAddressModeToUse;
extern int gNormalizedModeToUse;
extern int gTypesToTest;
extern int gtestTypesToRun;

extern int test_read_image_set_1D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  const cl_image_format *format,
                                  image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType);
extern int test_read_image_set_2D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  const cl_image_format *format,
                                  image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType);
extern int test_read_image_set_3D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  const cl_image_format *format,
                                  image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType);
extern int test_read_image_set_1D_array(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        const cl_image_format *format,
                                        image_sampler_data *imageSampler,
                                        bool floatCoords,
                                        ExplicitType outputType);
extern int test_read_image_set_2D_array(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        const cl_image_format *format,
                                        image_sampler_data *imageSampler,
                                        bool floatCoords,
                                        ExplicitType outputType);

int test_read_image_type(cl_device_id device, cl_context context,
                         cl_command_queue queue, const cl_image_format *format,
                         bool floatCoords, image_sampler_data *imageSampler,
                         ExplicitType outputType, cl_mem_object_type imageType)
{
    int ret = 0;
    cl_addressing_mode *addressModes = NULL;

    // The sampler-less read image functions behave exactly as the corresponding
    // read image functions described in section 6.13.14.2 that take integer
    // coordinates and a sampler with filter mode set to CLK_FILTER_NEAREST,
    // normalized coordinates set to CLK_NORMALIZED_COORDS_FALSE and addressing
    // mode to CLK_ADDRESS_NONE
    cl_addressing_mode addressModes_rw[] = { CL_ADDRESS_NONE,
                                             (cl_addressing_mode)-1 };
    cl_addressing_mode addressModes_ro[] = {
        /* CL_ADDRESS_CLAMP_NONE,*/ CL_ADDRESS_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP,
        CL_ADDRESS_REPEAT, CL_ADDRESS_MIRRORED_REPEAT, (cl_addressing_mode)-1
    };

    if (gtestTypesToRun & kReadWriteTests)
    {
        addressModes = addressModes_rw;
    }
    else
    {
        addressModes = addressModes_ro;
    }

#if defined(__APPLE__)
    // According to the OpenCL specification, we do not guarantee the precision
    // of operations for linear filtering on the GPU.  We do not test linear
    // filtering for the CL_RGB CL_UNORM_INT_101010 image format; however, we
    // test it internally for a set of other image formats.
    if ((gDeviceType & CL_DEVICE_TYPE_GPU)
        && (imageSampler->filter_mode == CL_FILTER_LINEAR)
        && (format->image_channel_order == CL_RGB)
        && (format->image_channel_data_type == CL_UNORM_INT_101010))
    {
        log_info("--- Skipping CL_RGB CL_UNORM_INT_101010 format with "
                 "CL_FILTER_LINEAR on GPU.\n");
        return 0;
    }
#endif

    for (int adMode = 0; addressModes[adMode] != (cl_addressing_mode)-1;
         adMode++)
    {
        imageSampler->addressing_mode = addressModes[adMode];

        if ((addressModes[adMode] == CL_ADDRESS_REPEAT
             || addressModes[adMode] == CL_ADDRESS_MIRRORED_REPEAT)
            && !(imageSampler->normalized_coords))
            continue; // Repeat doesn't make sense for non-normalized coords

        // Use this run if we were told to only run a certain filter mode
        if (gAddressModeToUse != (cl_addressing_mode)-1
            && imageSampler->addressing_mode != gAddressModeToUse)
            continue;

        /*
         Remove redundant check to see if workaround still necessary
         // Check added in because this case was leaking through causing a crash
         on CPU if( ! imageSampler->normalized_coords &&
         imageSampler->addressing_mode == CL_ADDRESS_REPEAT ) continue; //repeat
         mode requires normalized coordinates
         */
        print_read_header(format, imageSampler, false);

        gTestCount++;

        int retCode = 0;
        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                retCode = test_read_image_set_1D(device, context, queue, format,
                                                 imageSampler, floatCoords,
                                                 outputType);
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                retCode = test_read_image_set_1D_array(device, context, queue,
                                                       format, imageSampler,
                                                       floatCoords, outputType);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                retCode = test_read_image_set_2D(device, context, queue, format,
                                                 imageSampler, floatCoords,
                                                 outputType);
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                retCode = test_read_image_set_2D_array(device, context, queue,
                                                       format, imageSampler,
                                                       floatCoords, outputType);
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                retCode = test_read_image_set_3D(device, context, queue, format,
                                                 imageSampler, floatCoords,
                                                 outputType);
                break;
        }
        if (retCode != 0)
        {
            gFailCount++;
            log_error("FAILED: ");
            print_read_header(format, imageSampler, true);
            log_info("\n");
        }
        ret |= retCode;
    }

    return ret;
}

int test_read_image_formats(cl_device_id device, cl_context context,
                            cl_command_queue queue,
                            const std::vector<cl_image_format> &formatList,
                            const std::vector<bool> &filterFlags,
                            image_sampler_data *imageSampler,
                            ExplicitType outputType,
                            cl_mem_object_type imageType)
{
    int ret = 0;
    bool flipFlop[2] = { false, true };
    int normalizedIdx, floatCoordIdx;

    if (gTestMipmaps)
    {
        if (0 == is_extension_available(device, "cl_khr_mipmap_image"))
        {
            log_info("-----------------------------------------------------\n");
            log_info("This device does not support "
                     "cl_khr_mipmap_image.\nSkipping mipmapped image test. \n");
            log_info(
                "-----------------------------------------------------\n\n");
            return 0;
        }
    }

    // Use this run if we were told to only run a certain filter mode
    if (gFilterModeToUse != (cl_filter_mode)-1
        && imageSampler->filter_mode != gFilterModeToUse)
        return 0;

    // Test normalized/non-normalized
    for (normalizedIdx = 0; normalizedIdx < 2; normalizedIdx++)
    {
        imageSampler->normalized_coords = flipFlop[normalizedIdx];
        if (gNormalizedModeToUse != 7
            && gNormalizedModeToUse != (int)imageSampler->normalized_coords)
            continue;

        for (floatCoordIdx = 0; floatCoordIdx < 2; floatCoordIdx++)
        {
            // Checks added in because this case was leaking through causing a
            // crash on CPU
            if (!flipFlop[floatCoordIdx])
                if (imageSampler->filter_mode != CL_FILTER_NEAREST
                    || // integer coords can only be used with nearest
                    flipFlop[normalizedIdx]) // Normalized integer coords makes
                                             // no sense (they'd all be zero)
                    continue;

            if (flipFlop[floatCoordIdx] && (gtestTypesToRun & kReadWriteTests))
                // sampler-less read in read_write tests run only integer coord
                continue;


            log_info("read_image (%s coords, %s results) "
                     "*****************************\n",
                     flipFlop[floatCoordIdx] ? (imageSampler->normalized_coords
                                                    ? "normalized float"
                                                    : "unnormalized float")
                                             : "integer",
                     get_explicit_type_name(outputType));

            for (unsigned int i = 0; i < formatList.size(); i++)
            {
                if (filterFlags[i]) continue;

                const cl_image_format &imageFormat = formatList[i];

                ret |=
                    test_read_image_type(device, context, queue, &imageFormat,
                                         flipFlop[floatCoordIdx], imageSampler,
                                         outputType, imageType);
            }
        }
    }
    return ret;
}


int test_image_set(cl_device_id device, cl_context context,
                   cl_command_queue queue, test_format_set_fn formatTestFn,
                   cl_mem_object_type imageType)
{
    int ret = 0;
    static int printedFormatList = -1;


    if ((imageType == CL_MEM_OBJECT_IMAGE3D)
        && (formatTestFn == test_write_image_formats))
    {
        if (0 == is_extension_available(device, "cl_khr_3d_image_writes"))
        {
            log_info("-----------------------------------------------------\n");
            log_info(
                "This device does not support "
                "cl_khr_3d_image_writes.\nSkipping 3d image write test. \n");
            log_info(
                "-----------------------------------------------------\n\n");
            return 0;
        }
    }

    if (gTestMipmaps)
    {
        if (0 == is_extension_available(device, "cl_khr_mipmap_image"))
        {
            log_info("-----------------------------------------------------\n");
            log_info("This device does not support "
                     "cl_khr_mipmap_image.\nSkipping mipmapped image test. \n");
            log_info(
                "-----------------------------------------------------\n\n");
            return 0;
        }
        if ((0 == is_extension_available(device, "cl_khr_mipmap_image_writes"))
            && (formatTestFn == test_write_image_formats))
        {
            log_info("-----------------------------------------------------\n");
            log_info("This device does not support "
                     "cl_khr_mipmap_image_writes.\nSkipping mipmapped image "
                     "write test. \n");
            log_info(
                "-----------------------------------------------------\n\n");
            return 0;
        }
    }

    int version_check = (get_device_cl_version(device) < Version(1, 2));
    if (version_check != 0)
    {
        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                test_missing_feature(version_check, "image_1D");
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                test_missing_feature(version_check, "image_1D_array");
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                test_missing_feature(version_check, "image_2D_array");
        }
    }

    // This flag is only for querying the list of supported formats
    // The flag for creating image will be set explicitly in test functions
    cl_mem_flags flags;
    const char *flagNames;
    if (formatTestFn == test_read_image_formats)
    {
        if (gtestTypesToRun & kReadTests)
        {
            flags = CL_MEM_READ_ONLY;
            flagNames = "read";
        }
        else
        {
            flags = CL_MEM_KERNEL_READ_AND_WRITE;
            flagNames = "read_write";
        }
    }
    else
    {
        if (gtestTypesToRun & kWriteTests)
        {
            flags = CL_MEM_WRITE_ONLY;
            flagNames = "write";
        }
        else
        {
            flags = CL_MEM_KERNEL_READ_AND_WRITE;
            flagNames = "read_write";
        }
    }

    // Grab the list of supported image formats for integer reads
    std::vector<cl_image_format> formatList;
    if (get_format_list(context, imageType, formatList, flags)) return -1;

    // First time through, we'll go ahead and print the formats supported,
    // regardless of type
    int test = imageType
        | (formatTestFn == test_read_image_formats ? (1 << 16) : (1 << 17));
    if (printedFormatList != test)
    {
        log_info("---- Supported %s %s formats for this device ---- \n",
                 convert_image_type_to_string(imageType), flagNames);
        for (unsigned int f = 0; f < formatList.size(); f++)
        {
            if (IsChannelOrderSupported(formatList[f].image_channel_order)
                && IsChannelTypeSupported(
                    formatList[f].image_channel_data_type))
                log_info(
                    "  %-7s %-24s %d\n",
                    GetChannelOrderName(formatList[f].image_channel_order),
                    GetChannelTypeName(formatList[f].image_channel_data_type),
                    (int)get_format_channel_count(&formatList[f]));
        }
        log_info("------------------------------------------- \n");
        printedFormatList = test;
    }

    image_sampler_data imageSampler;

    for (auto test : imageTestTypes)
    {
        if (gTypesToTest & test.type)
        {
            std::vector<bool> filterFlags(formatList.size(), false);
            if (filter_formats(formatList, filterFlags, test.channelTypes,
                               gTestMipmaps)
                == 0)
            {
                log_info("No formats supported for %s type\n", test.name);
            }
            else
            {
                imageSampler.filter_mode = CL_FILTER_NEAREST;
                ret += formatTestFn(device, context, queue, formatList,
                                    filterFlags, &imageSampler,
                                    test.explicitType, imageType);

                // Linear filtering is only supported with floats
                if (test.type == kTestFloat)
                {
                    imageSampler.filter_mode = CL_FILTER_LINEAR;
                    ret += formatTestFn(device, context, queue, formatList,
                                        filterFlags, &imageSampler,
                                        test.explicitType, imageType);
                }
            }
        }
    }
    return ret;
}

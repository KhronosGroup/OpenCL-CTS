//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "test_cl_ext_image_buffer.hpp"

static int test_image_set(cl_device_id device, cl_context context,
                          cl_command_queue queue, cl_mem_object_type imageType,
                          const context_t &ctx)
{
    int ret = 0;

    // Grab the list of supported image formats for integer reads
    std::vector<cl_image_format> formatList = {
        { CL_R, CL_UNSIGNED_INT_RAW10_EXT }, { CL_R, CL_UNSIGNED_INT_RAW12_EXT }
    };

    // First time through, we'll go ahead and print the formats supported,
    // regardless of type
    log_info("---- Supported %s %s formats for this device for "
             "cl_ext_image_raw10_raw12---- \n",
             convert_image_type_to_string(imageType), "read");
    log_info("  %-7s %-24s %d\n", "CL_R", "CL_UNSIGNED_INT_RAW10_EXT", 1);
    log_info("  %-7s %-24s %d\n", "CL_R", "CL_UNSIGNED_INT_RAW12_EXT", 1);
    log_info("------------------------------------------- \n");

    image_sampler_data imageSampler;
    ImageTestTypes test{ kTestUInt, kUInt, uintFormats, "uint" };

    if (ctx.typesToTest & test.type)
    {
        std::vector<bool> filterFlags(formatList.size(), false);
        imageSampler.filter_mode = CL_FILTER_NEAREST;
        // `CL_UNSIGNED_INT_RAW10_EXT` and `CL_UNSIGNED_INT_RAW12_EXT` image
        // channel data types are unnormalised
        imageSampler.normalized_coords = false;
        ret = test_read_image_formats(device, context, queue, formatList,
                                      filterFlags, &imageSampler,
                                      test.explicitType, imageType, ctx);
    }
    return ret;
}

int ext_image_raw10_raw12(cl_device_id device, cl_context context,
                          cl_command_queue queue, const context_t &ctx)
{
    int ret = 0;

    if (true != ctx.normalizedModeToUse)
    {
        if (0 == is_extension_available(device, "cl_ext_image_raw10_raw12"))
        {
            log_info("-----------------------------------------------------\n");
            log_info("This device does not support "
                     "cl_ext_image_raw10_raw12.\n");
            log_info("Skipping cl_ext_image_raw10_raw12 "
                     "image test.\n");
            log_info(
                "-----------------------------------------------------\n\n");
            ret = TEST_SKIPPED_ITSELF;
        }
        else
        {
            context_t sub_ctx = ctx;
            sub_ctx.testTypesToRun = kReadTests;
            ret += test_image_set(device, context, queue, CL_MEM_OBJECT_IMAGE2D,
                                  sub_ctx);
        }
    }
    else
    {
        // skip the test if it is forced to be NORMALIZED from the command line
        // argument i.e. ctx.normalizedModeToUse is true
        log_info("cl_ext_image_raw10_raw12 does not support normalized channel "
                 "components. Skipping the test.\n");
        ret = TEST_SKIPPED_ITSELF;
    }

    return ret;
}

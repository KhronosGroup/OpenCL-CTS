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
#include "test_copy_generic.h"
#include <algorithm>

int test_image_type(cl_device_id device, cl_context context,
                    cl_command_queue queue,
                    const struct TestConfigs& test_config,
                    const image_test_context_t& ctx)
{
    if (ctx.testMipmaps)
    {
        if ( 0 == is_extension_available( device, "cl_khr_mipmap_image" ))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image.\nSkipping mipmapped image test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return TEST_SKIPPED_ITSELF;
        }
    }


    int ret = 0;
    {
        if (ctx.testMipmaps)
            log_info("Running mipmapped %s tests...\n",
                     test_config.name.c_str());
        else
            log_info("Running %s tests...\n", test_config.name.c_str());

        // Grab the list of supported image formats for integer reads
        std::vector<cl_image_format> formatList;
        {
            std::vector<cl_image_format> srcFormatList;
            std::vector<cl_image_format> dstFormatList;
            if (get_format_list(context, test_config.src_type, srcFormatList,
                                test_config.src_flags))
                return TEST_FAIL;
            if (get_format_list(context, test_config.dst_type, dstFormatList,
                                test_config.dst_flags))
                return TEST_FAIL;

            for (auto src_format : srcFormatList)
            {
                const bool src_format_in_dst =
                    std::find_if(dstFormatList.begin(), dstFormatList.end(),
                                 [src_format](cl_image_format fmt) {
                                     return src_format.image_channel_data_type
                                         == fmt.image_channel_data_type
                                         && src_format.image_channel_order
                                         == fmt.image_channel_order;
                                 })
                    != dstFormatList.end();
                if (src_format_in_dst)
                {
                    formatList.push_back(src_format);
                }
            }
        }

        std::vector<bool> filterFlags(formatList.size(), false);
        filter_formats(formatList, filterFlags, nullptr,
                       test_config.channel_type, ctx.channelOrderToUse);

        // Run the format list
        for (unsigned int i = 0; i < formatList.size(); i++)
        {
            int test_return = 0;
            if (filterFlags[i])
            {
                continue;
            }

            print_header(&formatList[i], false);

            gTestCount++;

            test_return =
                test_config.func(device, context, queue, test_config.src_flags,
                                 test_config.src_type, test_config.dst_flags,
                                 test_config.dst_type, &formatList[i], ctx);

            if (test_return)
            {
                gFailCount++;
                log_error("FAILED: ");
                print_header(&formatList[i], true);
                log_info("\n");
            }

            ret += test_return;
        }
    }

    return ret;
}

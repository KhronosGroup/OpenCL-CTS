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

#include <stdio.h>
#include <string.h>
#include "../common.h"
#include "../harness/compat.h"
#include "../harness/testHarness.h"
#include "test_copy_generic.h"

static image_test_context_t ctx;

static std::vector<TestConfigs> test_configs;

static int test_image_set(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements, void *arg)
{
    return test_image_type(device, context, queue, test_configs[(uintptr_t)arg],
                           ctx);
}

static test_status parseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{
    help = R"(        test_mipmaps - Test with mipmapped images
        debug_trace - Enables additional debug info logging
        small_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes
        max_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128
        randomize - Use random seed
        use_pitches - Enables row and slice pitches
)";

    cl_channel_order chanOrder;

    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    // Parse arguments
    for (int i = 1; i < argc; i++)
    {
        removed_args.push_back(argv[i]);
        if (strcmp(argv[i], "test_mipmaps") == 0)
        {
            ctx.testMipmaps = true;
            // Don't test pitches with mipmaps, at least currently.
            ctx.enablePitch = false;
        }
        else if (strcmp(argv[i], "debug_trace") == 0)
            ctx.debugTrace = true;
        else if (strcmp(argv[i], "small_images") == 0)
            ctx.testSmallImages = true;
        else if (strcmp(argv[i], "max_images") == 0)
            ctx.testMaxImages = true;
        else if (strcmp(argv[i], "use_pitches") == 0)
            ctx.enablePitch = true;
        else if ((chanOrder = get_channel_order_from_name(argv[i]))
                 != (cl_channel_order)-1)
            ctx.channelOrderToUse = chanOrder;
        else
        {
            removed_args.pop_back();
            argList.push_back(argv[i]);
        }
    }

    if (ctx.testSmallImages) log_info("Note: Using small test images\n");

    update_argc_argv_from_args_list(argList, argc, argv);

    for (auto channel_type : channel_types)
    {
        for (auto testMethod :
             { k1D, k2D, k3D, k1DBuffer, k1DTo1DBuffer, k1DBufferTo1D, k1DArray,
               k2DArray, k2DTo3D, k3DTo2D, k2DArrayTo2D, k2DTo2DArray,
               k2DArrayTo3D, k3DTo2DArray })
        {
            switch (testMethod)
            {
                case k1D:
                    test_configs.emplace_back(
                        "1D", CL_MEM_OBJECT_IMAGE1D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE1D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_1D, channel_type);
                    break;
                case k2D:
                    test_configs.emplace_back(
                        "2D", CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D, channel_type);
                    break;
                case k3D:
                    test_configs.emplace_back(
                        "3D", CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_3D, channel_type);
                    break;
                case k1DArray:
                    test_configs.emplace_back(
                        "1Darray", CL_MEM_OBJECT_IMAGE1D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE1D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_1D_array, channel_type);
                    break;
                case k2DArray:
                    test_configs.emplace_back(
                        "2Darray", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D_array, channel_type);
                    break;
                case k2DTo3D:
                    test_configs.emplace_back(
                        "2DTo3D", CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D_3D, channel_type);
                    break;
                case k3DTo2D:
                    test_configs.emplace_back(
                        "3DTo2D", CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D_3D, channel_type);
                    break;
                case k2DArrayTo2D:
                    test_configs.emplace_back(
                        "2DArrayTo2D", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D_2D_array, channel_type);
                    break;
                case k2DTo2DArray:
                    test_configs.emplace_back(
                        "2DTo2DArray", CL_MEM_OBJECT_IMAGE2D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_2D_2D_array, channel_type);
                    break;
                case k2DArrayTo3D:
                    test_configs.emplace_back(
                        "2DArrayTo3D", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_3D_2D_array, channel_type);
                    break;
                case k3DTo2DArray:
                    test_configs.emplace_back(
                        "3DTo2DArray", CL_MEM_OBJECT_IMAGE3D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_3D_2D_array, channel_type);
                    break;
                case k1DBuffer:
                    test_configs.emplace_back(
                        "1DBuffer", CL_MEM_OBJECT_IMAGE1D_BUFFER,
                        CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D_BUFFER,
                        CL_MEM_READ_ONLY, test_copy_image_set_1D_buffer,
                        channel_type);
                    break;
                case k1DTo1DBuffer:
                    test_configs.emplace_back(
                        "1DTo1DBuffer", CL_MEM_OBJECT_IMAGE1D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_READ_ONLY,
                        test_copy_image_set_1D_1D_buffer, channel_type);
                    break;
                case k1DBufferTo1D:
                    test_configs.emplace_back(
                        "1DBufferTo1D", CL_MEM_OBJECT_IMAGE1D_BUFFER,
                        CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D,
                        ctx.enablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                                        : CL_MEM_READ_ONLY,
                        test_copy_image_set_1D_1D_buffer, channel_type);
                    break;
            }
        }
    }
    for (uint32_t i = 0; i < test_configs.size(); i++)
    {
        test_registry::getInstance().add_test(
            test_image_set, test_configs[i].name.c_str(), Version(1, 2),
            (void *)(uintptr_t)i);
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(argc, argv, false, 0,
                                           verifyImageSupport, parseArgs);
}

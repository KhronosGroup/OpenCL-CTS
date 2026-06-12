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
#include "../testBase.h"
#include "../harness/compat.h"
#include "../harness/testHarness.h"

static context_t ctx;

extern int test_image_set(cl_device_id device, cl_context context,
                          cl_command_queue queue, MethodsToTest testMethod,
                          const context_t &ctx);

REGISTER_TEST(1D) { return test_image_set(device, context, queue, k1D, ctx); }
REGISTER_TEST(2D) { return test_image_set(device, context, queue, k2D, ctx); }
REGISTER_TEST(3D) { return test_image_set(device, context, queue, k3D, ctx); }

REGISTER_TEST(1Dbuffer)
{
    return test_image_set(device, context, queue, k1DBuffer, ctx);
}
REGISTER_TEST(1DTo1Dbuffer)
{
    return test_image_set(device, context, queue, k1DTo1DBuffer, ctx);
}
REGISTER_TEST(1DbufferTo1D)
{
    return test_image_set(device, context, queue, k1DBufferTo1D, ctx);
}
REGISTER_TEST(1Darray)
{
    return test_image_set(device, context, queue, k1DArray, ctx);
}
REGISTER_TEST(2Darray)
{
    return test_image_set(device, context, queue, k2DArray, ctx);
}
REGISTER_TEST(2Dto3D)
{
    return test_image_set(device, context, queue, k2DTo3D, ctx);
}
REGISTER_TEST(3Dto2D)
{
    return test_image_set(device, context, queue, k3DTo2D, ctx);
}
REGISTER_TEST(2Darrayto2D)
{
    return test_image_set(device, context, queue, k2DArrayTo2D, ctx);
}
REGISTER_TEST(2Dto2Darray)
{
    return test_image_set(device, context, queue, k2DTo2DArray, ctx);
}
REGISTER_TEST(2Darrayto3D)
{
    return test_image_set(device, context, queue, k2DArrayTo3D, ctx);
}
REGISTER_TEST(3Dto2Darray)
{
    return test_image_set(device, context, queue, k3DTo2DArray, ctx);
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

    cl_channel_type chanType;
    cl_channel_order chanOrder;

    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    init_context(ctx);

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
        else if ((chanType = get_channel_type_from_name(argv[i]))
                 != (cl_channel_type)-1)
            ctx.channelTypeToUse = chanType;
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
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0,
        verifyImageSupport, parseArgs);
}

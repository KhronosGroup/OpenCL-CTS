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
#include "../harness/fpcontrol.h"

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif

static context_t ctx;

extern int test_image_set(cl_device_id device, cl_context context,
                          cl_command_queue queue,
                          test_format_set_fn formatTestFn,
                          cl_mem_object_type imageType, const context_t &ctx);

extern int cl_image_requirements_size_ext_negative(cl_device_id device,
                                                   cl_context context,
                                                   cl_command_queue queue);
extern int cl_image_requirements_size_ext_consistency(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue);
extern int clGetImageRequirementsInfoEXT_negative(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue);
extern int cl_image_requirements_max_val_ext_negative(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue);
extern int cl_image_requirements_max_val_ext_positive(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue);

extern int image2d_from_buffer_positive(cl_device_id device, cl_context context,
                                        cl_command_queue queue);
extern int memInfo_image_from_buffer_positive(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue);
extern int imageInfo_image_from_buffer_positive(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue);
extern int image_from_buffer_alignment_negative(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue);
extern int image_from_small_buffer_negative(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue);
extern int image_from_buffer_fill_positive(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue);
extern int image_from_buffer_read_positive(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue);
extern int ext_image_raw10_raw12(cl_device_id device, cl_context context,
                                 cl_command_queue queue, const context_t &ctx);

static int doTest(cl_device_id device, cl_context context,
                  cl_command_queue queue, cl_mem_object_type imageType,
                  const context_t &ctx)
{
    int ret = 0;
    bool is_2d_image = imageType == CL_MEM_OBJECT_IMAGE2D;

    if (ctx.testTypesToRun & kReadTests)
    {
        context_t sub_ctx = ctx;
        sub_ctx.testTypesToRun = kReadTests;
        ret += test_image_set(device, context, queue, test_read_image_formats,
                              imageType, sub_ctx);

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info( "Testing read_image{f | i | ui} for 2D image from buffer\n" );

            // NOTE: for 2D image from buffer test, ctx.testSmallImages,
            // ctx.testMaxImages, gTestRounding and ctx.testMipmaps must be
            // false
            if (ctx.testSmallImages == false && ctx.testMaxImages == false
                && gTestRounding == false && ctx.testMipmaps == false)
            {

                sub_ctx.testImage2DFromBuffer = true;
                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this
                // for 2.0
                sub_ctx.memFlagsToUse = CL_MEM_COPY_HOST_PTR;

                ret +=
                    test_image_set(device, context, queue,
                                   test_read_image_formats, imageType, sub_ctx);
            }
        }
    }

    if (ctx.testTypesToRun & kWriteTests)
    {
        context_t sub_ctx = ctx;
        sub_ctx.testTypesToRun = kWriteTests;
        ret += test_image_set(device, context, queue, test_write_image_formats,
                              imageType, sub_ctx);

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info( "Testing write_image{f | i | ui} for 2D image from buffer\n" );

            // NOTE: for 2D image from buffer test, ctx.testSmallImages,
            // ctx.testMaxImages,gTestRounding and ctx.testMipmaps must be false
            if (ctx.testSmallImages == false && ctx.testMaxImages == false
                && gTestRounding == false && ctx.testMipmaps == false)
            {
                sub_ctx.enablePitch = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this for 2.0
                sub_ctx.memFlagsToUse = CL_MEM_COPY_HOST_PTR;
                sub_ctx.testImage2DFromBuffer = true;

                ret += test_image_set(device, context, queue,
                                      test_write_image_formats, imageType,
                                      sub_ctx);
            }
        }
    }

    if ((ctx.testTypesToRun & kReadWriteTests)
        && checkForReadWriteImageSupport(device))
    {
        return ret;
    }

    if ((ctx.testTypesToRun & kReadWriteTests) && !ctx.testMipmaps)
    {
        context_t sub_ctx = ctx;
        sub_ctx.testTypesToRun = kReadWriteTests;
        // mipmap images only support sampler read built-in while read_write
        // images only support sampler-less read built-in. Hence we cannot test
        // mipmap for read_write image.
        sub_ctx.testMipmaps = false;
        // Read_write images are read by sampler-less read which does not handle
        // out-of-bound read It's application responsibility to make sure that
        // the read happens in-bound Therefore we should not enable offset in
        // testing read_write images because it will cause out-of-bound
        sub_ctx.disableOffsets = true;
        // The sampler-less read image functions behave exactly as the
        // corresponding read image functions
        sub_ctx.normalizedModeToUse = false;
        sub_ctx.filterModeToUse = CL_FILTER_NEAREST;

        ret += test_image_set(device, context, queue, test_read_image_formats,
                              imageType, sub_ctx);

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            // NOTE: for 2D image from buffer test, ctx.testSmallImages,
            // ctx.testMaxImages, gTestRounding and ctx.testMipmaps must be
            // false
            if (ctx.testSmallImages == false && ctx.testMaxImages == false
                && gTestRounding == false && ctx.testMipmaps == false)
            {
                log_info("Testing read_image{f | i | ui} for 2D image from "
                         "buffer\n");

                context_t sub_sub_ctx = sub_ctx;
                sub_sub_ctx.testImage2DFromBuffer = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this
                // for 2.0
                sub_sub_ctx.memFlagsToUse = CL_MEM_COPY_HOST_PTR;

                ret += test_image_set(device, context, queue,
                                      test_read_image_formats, imageType,
                                      sub_sub_ctx);
            }
        }

        ret += test_image_set(device, context, queue, test_write_image_formats,
                              imageType, sub_ctx);

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            // NOTE: for 2D image from buffer test, ctx.testSmallImages,
            // ctx.testMaxImages,gTestRounding and ctx.testMipmaps must be false
            if (ctx.testSmallImages == false && ctx.testMaxImages == false
                && gTestRounding == false && ctx.testMipmaps == false)
            {
                log_info("Testing write_image{f | i | ui} for 2D image from "
                         "buffer\n");

                context_t sub_sub_ctx = sub_ctx;
                sub_sub_ctx.enablePitch = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this
                // for 2.0
                sub_sub_ctx.memFlagsToUse = CL_MEM_COPY_HOST_PTR;
                sub_sub_ctx.testImage2DFromBuffer = true;

                ret += test_image_set(device, context, queue,
                                      test_write_image_formats, imageType,
                                      sub_sub_ctx);
            }
        }
    }

    return ret;
}

REGISTER_TEST(1D)
{
    return doTest(device, context, queue, CL_MEM_OBJECT_IMAGE1D, ctx);
}
REGISTER_TEST(2D)
{
    return doTest(device, context, queue, CL_MEM_OBJECT_IMAGE2D, ctx);
}
REGISTER_TEST(3D)
{
    return doTest(device, context, queue, CL_MEM_OBJECT_IMAGE3D, ctx);
}
REGISTER_TEST(1Darray)
{
    return doTest(device, context, queue, CL_MEM_OBJECT_IMAGE1D_ARRAY, ctx);
}
REGISTER_TEST(2Darray)
{
    return doTest(device, context, queue, CL_MEM_OBJECT_IMAGE2D_ARRAY, ctx);
}

REGISTER_TEST_VERSION(cl_image_requirements_size_ext_negative, Version(3, 0))
{
    return cl_image_requirements_size_ext_negative(device, context, queue);
}
REGISTER_TEST_VERSION(cl_image_requirements_size_ext_consistency, Version(3, 0))
{
    return cl_image_requirements_size_ext_consistency(device, context, queue);
}
REGISTER_TEST_VERSION(clGetImageRequirementsInfoEXT_negative, Version(3, 0))
{
    return clGetImageRequirementsInfoEXT_negative(device, context, queue);
}
REGISTER_TEST_VERSION(cl_image_requirements_max_val_ext_negative, Version(3, 0))
{
    return cl_image_requirements_max_val_ext_negative(device, context, queue);
}
REGISTER_TEST_VERSION(cl_image_requirements_max_val_ext_positive, Version(3, 0))
{
    return cl_image_requirements_max_val_ext_positive(device, context, queue);
}

REGISTER_TEST_VERSION(image2d_from_buffer_positive, Version(3, 0))
{
    return image2d_from_buffer_positive(device, context, queue);
}
REGISTER_TEST_VERSION(memInfo_image_from_buffer_positive, Version(3, 0))
{
    return memInfo_image_from_buffer_positive(device, context, queue);
}
REGISTER_TEST_VERSION(imageInfo_image_from_buffer_positive, Version(3, 0))
{
    return imageInfo_image_from_buffer_positive(device, context, queue);
}
REGISTER_TEST_VERSION(image_from_buffer_alignment_negative, Version(3, 0))
{
    return image_from_buffer_alignment_negative(device, context, queue);
}
REGISTER_TEST_VERSION(image_from_small_buffer_negative, Version(3, 0))
{
    return image_from_small_buffer_negative(device, context, queue);
}
REGISTER_TEST_VERSION(image_from_buffer_fill_positive, Version(3, 0))
{
    return image_from_buffer_fill_positive(device, context, queue);
}
REGISTER_TEST_VERSION(image_from_buffer_read_positive, Version(3, 0))
{
    return image_from_buffer_read_positive(device, context, queue);
}

REGISTER_TEST_VERSION(cl_ext_image_raw10_raw12, Version(1, 2))
{
    return ext_image_raw10_raw12(device, context, queue, ctx);
}

static test_status parseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{

    help =
        R"(        The following flags specify what kinds of operations to test. They can be combined; if none are specified, all are tested:
          read - Tests reading from an image
          write - Tests writing to an image (can be specified with read to run both; default is both)

        The following flags specify the types to test. They can be combined; if none are specified, all are tested:
          int - Test integer I/O (read_imagei, write_imagei)
          uint - Test unsigned integer I/O (read_imageui, write_imageui)
          float - Test float I/O (read_imagef, write_imagef)

        CL_FILTER_LINEAR - Only tests formats with CL_FILTER_LINEAR filtering
        CL_FILTER_NEAREST - Only tests formats with CL_FILTER_NEAREST filtering

        NORMALIZED - Only tests formats with NORMALIZED coordinates
        UNNORMALIZED - Only tests formats with UNNORMALIZED coordinates

        CL_ADDRESS_CLAMP - Only tests formats with CL_ADDRESS_CLAMP addressing
        CL_ADDRESS_CLAMP_TO_EDGE - Only tests formats with CL_ADDRESS_CLAMP_TO_EDGE addressing
        CL_ADDRESS_REPEAT - Only tests formats with CL_ADDRESS_REPEAT addressing
        CL_ADDRESS_MIRRORED_REPEAT - Only tests formats with CL_ADDRESS_MIRRORED_REPEAT addressing

        You may also use appropriate CL_ channel type and ordering constants.

        local_samplers - Use samplers declared in the kernel functions instead of passed in as arguments

        The following specify to use the specific flag to allocate images to use in the tests:
          CL_MEM_COPY_HOST_PTR
          CL_MEM_USE_HOST_PTR (default)
          CL_MEM_ALLOC_HOST_PTR
          NO_HOST_PTR - Specifies to use none of the above flags

        The following modify the types of images tested:
          small_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes
          max_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128
          rounding - Runs every format through a single image filled with every possible value for that image format, to verify rounding works properly

        no_offsets - Disables offsets when testing reads (can be good for diagnosing address repeating/clamping problems)
        debug_trace - Enables additional debug info logging
        extra_validate - Enables additional validation failure debug information
        use_pitches - Enables row and slice pitches
        test_mipmaps - Enables mipmapped images
)";

    cl_channel_type chanType;
    cl_channel_order chanOrder;

    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    init_context(ctx);

    // Parse arguments
    for( int i = 1; i < argc; i++ )
    {
        removed_args.push_back(argv[i]);
        if( strcmp( argv[i], "debug_trace" ) == 0 )
            ctx.debugTrace = true;

        else if( strcmp( argv[i], "CL_FILTER_NEAREST" ) == 0 || strcmp( argv[i], "NEAREST" ) == 0 )
            ctx.filterModeToUse = CL_FILTER_NEAREST;
        else if( strcmp( argv[i], "CL_FILTER_LINEAR" ) == 0 || strcmp( argv[i], "LINEAR" ) == 0 )
            ctx.filterModeToUse = CL_FILTER_LINEAR;

        else if( strcmp( argv[i], "CL_ADDRESS_NONE" ) == 0 )
            ctx.addressModeToUse = CL_ADDRESS_NONE;
        else if( strcmp( argv[i], "CL_ADDRESS_CLAMP" ) == 0 )
            ctx.addressModeToUse = CL_ADDRESS_CLAMP;
        else if( strcmp( argv[i], "CL_ADDRESS_CLAMP_TO_EDGE" ) == 0 )
            ctx.addressModeToUse = CL_ADDRESS_CLAMP_TO_EDGE;
        else if( strcmp( argv[i], "CL_ADDRESS_REPEAT" ) == 0 )
            ctx.addressModeToUse = CL_ADDRESS_REPEAT;
        else if( strcmp( argv[i], "CL_ADDRESS_MIRRORED_REPEAT" ) == 0 )
            ctx.addressModeToUse = CL_ADDRESS_MIRRORED_REPEAT;

        else if( strcmp( argv[i], "NORMALIZED" ) == 0 )
            ctx.normalizedModeToUse = true;
        else if( strcmp( argv[i], "UNNORMALIZED" ) == 0 )
            ctx.normalizedModeToUse = false;


        else if( strcmp( argv[i], "no_offsets" ) == 0 )
            ctx.disableOffsets = true;
        else if( strcmp( argv[i], "small_images" ) == 0 )
            ctx.testSmallImages = true;
        else if( strcmp( argv[i], "max_images" ) == 0 )
            ctx.testMaxImages = true;
        else if( strcmp( argv[i], "use_pitches" ) == 0 )
            ctx.enablePitch = true;
        else if( strcmp( argv[i], "rounding" ) == 0 )
            gTestRounding = true;
        else if( strcmp( argv[i], "extra_validate" ) == 0 )
            ctx.extraValidateInfo = true;
        else if( strcmp( argv[i], "test_mipmaps" ) == 0 ) {
            // 2.0 Spec does not allow using mem flags, unnormalized coordinates with mipmapped images
            ctx.testMipmaps = true;
            ctx.memFlagsToUse = 0;
            ctx.normalizedModeToUse = true;
        }

        else if( strcmp( argv[i], "read" ) == 0 )
            ctx.testTypesToRun |= kReadTests;
        else if( strcmp( argv[i], "write" ) == 0 )
            ctx.testTypesToRun |= kWriteTests;
        else if( strcmp( argv[i], "read_write" ) == 0 )
        {
            ctx.testTypesToRun |= kReadWriteTests;
        }

        else if( strcmp( argv[i], "local_samplers" ) == 0 )
            ctx.useKernelSamplers = true;

        else if( strcmp( argv[i], "int" ) == 0 )
            ctx.typesToTest |= kTestInt;
        else if( strcmp( argv[i], "uint" ) == 0 )
            ctx.typesToTest |= kTestUInt;
        else if( strcmp( argv[i], "float" ) == 0 )
            ctx.typesToTest |= kTestFloat;

        else if( strcmp( argv[i], "CL_MEM_COPY_HOST_PTR" ) == 0 || strcmp( argv[i], "COPY_HOST_PTR" ) == 0 )
            ctx.memFlagsToUse = CL_MEM_COPY_HOST_PTR;
        else if( strcmp( argv[i], "CL_MEM_USE_HOST_PTR" ) == 0 || strcmp( argv[i], "USE_HOST_PTR" ) == 0 )
            ctx.memFlagsToUse = CL_MEM_USE_HOST_PTR;
        else if( strcmp( argv[i], "CL_MEM_ALLOC_HOST_PTR" ) == 0 || strcmp( argv[i], "ALLOC_HOST_PTR" ) == 0 )
            ctx.memFlagsToUse = CL_MEM_ALLOC_HOST_PTR;
        else if( strcmp( argv[i], "NO_HOST_PTR" ) == 0 )
            ctx.memFlagsToUse = 0;

        else if( ( chanType = get_channel_type_from_name( argv[i] ) ) != (cl_channel_type)-1 )
            ctx.channelTypeToUse = chanType;

        else if( ( chanOrder = get_channel_order_from_name( argv[i] ) ) != (cl_channel_order)-1 )
            ctx.channelOrderToUse = chanOrder;
        else
        {
            removed_args.pop_back();
            argList.push_back(argv[i]);
        }
    }

    if (ctx.testTypesToRun == 0) ctx.testTypesToRun = kAllTests;
    if (ctx.typesToTest == 0) ctx.typesToTest = kTestAllTypes;

    if (ctx.testSmallImages) log_info("Note: Using small test images\n");

    update_argc_argv_from_args_list(argList, argc, argv);
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{

    // On most platforms which support denorm, default is FTZ off. However,
    // on some hardware where the reference is computed, default might be flush denorms to zero e.g. arm.
    // This creates issues in result verification. Since spec allows the implementation to either flush or
    // not flush denorms to zero, an implementation may choose not to flush i.e. return denorm result whereas
    // reference result may be zero (flushed denorm). Hence we need to disable denorm flushing on host side
    // where reference is being computed to make sure we get non-flushed reference result. If implementation
    // returns flushed result, we correctly take care of that in verification code.

    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);

    int ret = runTestHarnessWithCheckAndParse(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0,
        verifyImageSupport, parseArgs);

    // Restore FP state before leaving
    RestoreFPState(&oldMode);

    return ret;
}

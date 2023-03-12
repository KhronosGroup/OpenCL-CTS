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
#include "common.h"

#include <algorithm>

using namespace std;

struct image_kernel_data
{
    cl_int width;
    cl_int height;
    cl_int depth;
    cl_int arraySize;
    cl_int widthDim;
    cl_int heightDim;
    cl_int channelType;
    cl_int channelOrder;
    cl_int expectedChannelType;
    cl_int expectedChannelOrder;
    cl_int numSamples;
};

// clang-format off
static const char *methodTestKernelPattern = 
"%s"
"typedef struct {\n"
"    int width;\n"
"    int height;\n"
"    int depth;\n"
"    int arraySize;\n"
"    int widthDim;\n"
"    int heightDim;\n"
"    int channelType;\n"
"    int channelOrder;\n"
"    int expectedChannelType;\n"
"    int expectedChannelOrder;\n"
"    int numSamples;\n"
" } image_kernel_data;\n"
"__kernel void sample_kernel( read_only %s input, __global image_kernel_data *outData )\n"
"{\n"
"%s%s%s%s%s%s%s%s%s%s%s"
"}\n";
// clang-format on

static const char *arraySizeKernelLine =
    "   outData->arraySize = get_image_array_size( input );\n";
static const char *imageWidthKernelLine =
    "   outData->width = get_image_width( input );\n";
static const char *imageHeightKernelLine =
    "   outData->height = get_image_height( input );\n";
static const char *imageDimKernelLine =
    "   int2 dim = get_image_dim( input );\n";
static const char *imageWidthDimKernelLine = "   outData->widthDim = dim.x;\n";
static const char *imageHeightDimKernelLine =
    "   outData->heightDim = dim.y;\n";
static const char *channelTypeKernelLine =
    "   outData->channelType = get_image_channel_data_type( input );\n";
static const char *channelTypeConstLine =
    "   outData->expectedChannelType = CLK_%s;\n";
static const char *channelOrderKernelLine =
    "   outData->channelOrder = get_image_channel_order( input );\n";
static const char *channelOrderConstLine =
    "   outData->expectedChannelOrder = CLK_%s;\n";
static const char *numSamplesKernelLine =
    "   outData->numSamples = get_image_num_samples( input );\n";
static const char *enableMSAAKernelLine =
    "#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable\n";

static int verify(cl_int input, cl_int kernelOutput, const char *description)
{
    if (kernelOutput != input)
    {
        log_error("ERROR: %s did not validate (expected %d, got %d)\n",
                  description, input, kernelOutput);
        return -1;
    }
    return 0;
}

extern int supportsMsaa(cl_context context, bool *supports_msaa);
extern int supportsDepth(cl_context context, bool *supports_depth);

int test_image_format_methods(cl_device_id device, cl_context context,
                              cl_command_queue queue, size_t width,
                              size_t height, size_t arraySize, size_t samples,
                              GLenum target, format format, MTdata d)
{
    int error, result = 0;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper image, outDataBuffer;
    char programSrc[10240];

    image_kernel_data outKernelData;

#ifdef GL_VERSION_3_2
    if (get_base_gl_target(target) == GL_TEXTURE_2D_MULTISAMPLE
        || get_base_gl_target(target) == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    {
        bool supports_msaa;
        error = supportsMsaa(context, &supports_msaa);
        if (error != 0) return error;
        if (!supports_msaa) return 0;
    }
    if (format.formattype == GL_DEPTH_COMPONENT
        || format.formattype == GL_DEPTH_STENCIL)
    {
        bool supports_depth;
        error = supportsDepth(context, &supports_depth);
        if (error != 0) return error;
        if (!supports_depth) return 0;
    }
#endif
    DetectFloatToHalfRoundingMode(queue);

    glTextureWrapper glTexture;
    switch (get_base_gl_target(target))
    {
        case GL_TEXTURE_2D:
            CreateGLTexture2D(width, height, target, format.formattype,
                              format.internal, format.datatype, format.type,
                              &glTexture, &error, false, d);
            break;
        case GL_TEXTURE_2D_ARRAY:
            CreateGLTexture2DArray(width, height, arraySize, target,
                                   format.formattype, format.internal,
                                   format.datatype, format.type, &glTexture,
                                   &error, false, d);
            break;
        case GL_TEXTURE_2D_MULTISAMPLE:
            CreateGLTexture2DMultisample(width, height, samples, target,
                                         format.formattype, format.internal,
                                         format.datatype, format.type,
                                         &glTexture, &error, false, d, false);
            break;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            CreateGLTexture2DArrayMultisample(
                width, height, arraySize, samples, target, format.formattype,
                format.internal, format.datatype, format.type, &glTexture,
                &error, false, d, false);
            break;

        default:
            log_error("Unsupported GL tex target (%s) passed to write test: "
                      "%s (%s):%d",
                      GetGLTargetName(target), __FUNCTION__, __FILE__,
                      __LINE__);
    }

    // Check to see if the texture could not be created for some other reason
    // like GL_FRAMEBUFFER_UNSUPPORTED
    if (error == GL_FRAMEBUFFER_UNSUPPORTED)
    {
        return 0;
    }

    // Construct testing source
    log_info(" - Creating image %d by %d...\n", width, height);
    // Create a CL image from the supplied GL texture
    image = (*clCreateFromGLTexture_ptr)(context, CL_MEM_READ_ONLY, target, 0,
                                         glTexture, &error);

    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to create CL image from GL texture");
        GLint fmt;
        glGetTexLevelParameteriv(target, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
        log_error("    Supplied GL texture was base format %s and internal "
                  "format %s\n",
                  GetGLBaseFormatName(fmt), GetGLFormatName(fmt));
        return error;
    }

    cl_image_format imageFormat;
    error = clGetImageInfo(image, CL_IMAGE_FORMAT, sizeof(imageFormat),
                           &imageFormat, NULL);
    test_error(error, "Failed to get image format");

    const char *imageType = 0;
    bool doArraySize = false;
    bool doImageWidth = false;
    bool doImageHeight = false;
    bool doImageChannelDataType = false;
    bool doImageChannelOrder = false;
    bool doImageDim = false;
    bool doNumSamples = false;
    bool doMSAA = false;
    switch (target)
    {
        case GL_TEXTURE_2D:
            imageType = "image2d_depth_t";
            doImageWidth = true;
            doImageHeight = true;
            doImageChannelDataType = true;
            doImageChannelOrder = true;
            doImageDim = true;
            break;
        case GL_TEXTURE_2D_ARRAY:
            imageType = "image2d_array_depth_t";
            doImageWidth = true;
            doImageHeight = true;
            doArraySize = true;
            doImageChannelDataType = true;
            doImageChannelOrder = true;
            doImageDim = true;
            doArraySize = true;
            break;
        case GL_TEXTURE_2D_MULTISAMPLE:
            doNumSamples = true;
            doMSAA = true;
            if (format.formattype == GL_DEPTH_COMPONENT)
            {
                doImageWidth = true;
                imageType = "image2d_msaa_depth_t";
            }
            else
            {
                imageType = "image2d_msaa_t";
            }
            break;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            doMSAA = true;
            if (format.formattype == GL_DEPTH_COMPONENT)
            {
                doImageWidth = true;
                imageType = "image2d_msaa_array_depth_t";
            }
            else
            {
                imageType = "image2d_array_msaa_t";
            }
            break;
    }


    char channelTypeConstKernelLine[512] = { 0 };
    char channelOrderConstKernelLine[512] = { 0 };
    const char *channelTypeName = 0;
    const char *channelOrderName = 0;
    if (doImageChannelDataType)
    {
        channelTypeName =
            GetChannelTypeName(imageFormat.image_channel_data_type);
        if (channelTypeName && strlen(channelTypeName))
        {
            // replace CL_* with CLK_*
            sprintf(channelTypeConstKernelLine, channelTypeConstLine,
                    &channelTypeName[3]);
        }
    }
    if (doImageChannelOrder)
    {
        channelOrderName = GetChannelOrderName(imageFormat.image_channel_order);
        if (channelOrderName && strlen(channelOrderName))
        {
            // replace CL_* with CLK_*
            sprintf(channelOrderConstKernelLine, channelOrderConstLine,
                    &channelOrderName[3]);
        }
    }

    // Create a program to run against
    sprintf(programSrc, methodTestKernelPattern,
            (doMSAA) ? enableMSAAKernelLine : "", imageType,
            (doArraySize) ? arraySizeKernelLine : "",
            (doImageWidth) ? imageWidthKernelLine : "",
            (doImageHeight) ? imageHeightKernelLine : "",
            (doImageChannelDataType) ? channelTypeKernelLine : "",
            (doImageChannelDataType) ? channelTypeConstKernelLine : "",
            (doImageChannelOrder) ? channelOrderKernelLine : "",
            (doImageChannelOrder) ? channelOrderConstKernelLine : "",
            (doImageDim) ? imageDimKernelLine : "",
            (doImageDim && doImageWidth) ? imageWidthDimKernelLine : "",
            (doImageDim && doImageHeight) ? imageHeightDimKernelLine : "",
            (doNumSamples) ? numSamplesKernelLine : "");


    // log_info("-----------------------------------\n%s\n", programSrc);
    error = clFinish(queue);
    if (error) print_error(error, "clFinish failed.\n");
    const char *ptr = programSrc;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    test_error(error, "Unable to create kernel to test against");

    // Create an output buffer
    outDataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(outKernelData), NULL, &error);
    test_error(error, "Unable to create output buffer");

    // Set up arguments and run
    error = clSetKernelArg(kernel, 0, sizeof(image), &image);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 1, sizeof(outDataBuffer), &outDataBuffer);
    test_error(error, "Unable to set kernel argument");

    // Finish and Acquire.
    glFinish();
    error = (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &image, 0, NULL, NULL);
    test_error(error, "Unable to acquire GL obejcts");

    size_t threads[1] = { 1 }, localThreads[1] = { 1 };

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to run kernel");

    error = clEnqueueReadBuffer(queue, outDataBuffer, CL_TRUE, 0,
                                sizeof(outKernelData), &outKernelData, 0, NULL,
                                NULL);
    test_error(error, "Unable to read data buffer");

    // Verify the results now
    if (doImageWidth) result |= verify(width, outKernelData.width, "width");
    if (doImageHeight) result |= verify(height, outKernelData.height, "height");
    if (doImageDim && doImageWidth)
        result |=
            verify(width, outKernelData.widthDim, "width from get_image_dim");
    if (doImageDim && doImageHeight)
        result |= verify(height, outKernelData.heightDim,
                         "height from get_image_dim");
    if (doImageChannelDataType)
        result |= verify(outKernelData.channelType,
                         outKernelData.expectedChannelType, channelTypeName);
    if (doImageChannelOrder)
        result |= verify(outKernelData.channelOrder,
                         outKernelData.expectedChannelOrder, channelOrderName);
    if (doArraySize)
        result |= verify(arraySize, outKernelData.arraySize, "array size");
    if (doNumSamples)
        result |= verify(samples, outKernelData.numSamples, "samples");
    if (result)
    {
        log_error("Test image methods failed");
    }

    clEventWrapper event;
    error = (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &image, 0, NULL, &event);
    test_error(error, "clEnqueueReleaseGLObjects failed");

    error = clWaitForEvents(1, &event);
    test_error(error, "clWaitForEvents failed");

    return result;
}

int test_image_methods_depth(cl_device_id device, cl_context context,
                             cl_command_queue queue, int numElements)
{
    if (!is_extension_available(device, "cl_khr_gl_depth_images"))
    {
        log_info("Test not run because 'cl_khr_gl_depth_images' extension is "
                 "not supported by the tested device\n");
        return 0;
    }

    int result = 0;
    GLenum depth_targets[] = { GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY };
    size_t ntargets = sizeof(depth_targets) / sizeof(depth_targets[0]);
    size_t nformats = sizeof(depth_formats) / sizeof(depth_formats[0]);

    const size_t nsizes = 5;
    sizevec_t sizes[nsizes];
    // Need to limit texture size according to GL device properties
    GLint maxTextureSize = 4096, maxTextureRectangleSize = 4096,
          maxTextureLayers = 16, size;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
    glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT, &maxTextureRectangleSize);
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxTextureLayers);

    size = min(maxTextureSize, maxTextureRectangleSize);

    RandomSeed seed(gRandomSeed);

    // Generate some random sizes (within reasonable ranges)
    for (size_t i = 0; i < nsizes; i++)
    {
        sizes[i].width = random_in_range(2, min(size, 1 << (i + 4)), seed);
        sizes[i].height = random_in_range(2, min(size, 1 << (i + 4)), seed);
        sizes[i].depth =
            random_in_range(2, min(maxTextureLayers, 1 << (i + 4)), seed);
    }

    for (size_t i = 0; i < nsizes; i++)
    {
        for (size_t itarget = 0; itarget < ntargets; ++itarget)
        {
            for (size_t iformat = 0; iformat < nformats; ++iformat)
                result |= test_image_format_methods(
                    device, context, queue, sizes[i].width, sizes[i].height,
                    (depth_targets[itarget] == GL_TEXTURE_2D_ARRAY)
                        ? sizes[i].depth
                        : 1,
                    0, depth_targets[itarget], depth_formats[iformat], seed);
        }
    }
    return result;
}

int test_image_methods_multisample(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int numElements)
{
    if (!is_extension_available(device, "cl_khr_gl_msaa_sharing"))
    {
        log_info("Test not run because 'cl_khr_gl_msaa_sharing' extension is "
                 "not supported by the tested device\n");
        return 0;
    }

    int result = 0;
    GLenum targets[] = { GL_TEXTURE_2D_MULTISAMPLE,
                         GL_TEXTURE_2D_MULTISAMPLE_ARRAY };
    size_t ntargets = sizeof(targets) / sizeof(targets[0]);
    size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

    const size_t nsizes = 5;
    sizevec_t sizes[nsizes];
    GLint maxTextureLayers = 16, maxTextureSize = 4096;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxTextureLayers);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    RandomSeed seed(gRandomSeed);

    // Generate some random sizes (within reasonable ranges)
    for (size_t i = 0; i < nsizes; i++)
    {
        sizes[i].width =
            random_in_range(2, min(maxTextureSize, 1 << (i + 4)), seed);
        sizes[i].height =
            random_in_range(2, min(maxTextureSize, 1 << (i + 4)), seed);
        sizes[i].depth =
            random_in_range(2, min(maxTextureLayers, 1 << (i + 4)), seed);
    }

    glEnable(GL_MULTISAMPLE);

    for (size_t i = 0; i < nsizes; i++)
    {
        for (size_t itarget = 0; itarget < ntargets; ++itarget)
        {
            for (size_t iformat = 0; iformat < nformats; ++iformat)
            {
                GLint samples = get_gl_max_samples(
                    targets[itarget], common_formats[iformat].internal);
                result |= test_image_format_methods(
                    device, context, queue, sizes[i].width, sizes[i].height,
                    (targets[ntargets] == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
                        ? sizes[i].depth
                        : 1,
                    samples, targets[itarget], common_formats[iformat], seed);
            }
        }
    }
    return result;
}

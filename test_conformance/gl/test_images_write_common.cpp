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
#include "testBase.h"
#include "common.h"
#include <limits.h>

#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#include <CL/cl_gl.h>
#endif

#pragma mark -
#pragma mark Write test kernels

// clang-format off
static const char *kernelpattern_image_write_1D =
"__kernel void sample_test( __global %s4 *source, write_only image1d_t dest )\n"
"{\n"
"    uint index = get_global_id(0);\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, index, %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_1D_half =
"__kernel void sample_test( __global half4 *source, write_only image1d_t dest )\n"
"{\n"
"    uint index = get_global_id(0);\n"
"    write_imagef( dest, index, vload_half4(index, (__global half *)source));\n"
"}\n";

static const char *kernelpattern_image_write_1D_buffer =
"__kernel void sample_test( __global %s4 *source, write_only image1d_buffer_t dest )\n"
"{\n"
"    uint index = get_global_id(0);\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, index, %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_1D_buffer_half =
"__kernel void sample_test( __global half4 *source, write_only image1d_buffer_t dest )\n"
"{\n"
"    uint index = get_global_id(0);\n"
"    write_imagef( dest, index, vload_half4(index, (__global half *)source));\n"
"}\n";

static const char *kernelpattern_image_write_2D =
"__kernel void sample_test( __global %s4 *source, write_only image2d_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, (int2)( tidX, tidY ), %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_2D_half =
"__kernel void sample_test( __global half4 *source, write_only image2d_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    write_imagef( dest, (int2)( tidX, tidY ), vload_half4(index, (__global half *)source));\n"
"}\n";

static const char *kernelpattern_image_write_1Darray =
"__kernel void sample_test( __global %s4 *source, write_only image1d_array_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, (int2)( tidX, tidY ), %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_1Darray_half =
"__kernel void sample_test( __global half4 *source, write_only image1d_array_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    write_imagef( dest, (int2)( tidX, tidY ), vload_half4(index, (__global half *)source));\n"
"}\n";

static const char *kernelpattern_image_write_3D =
"#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n"
"__kernel void sample_test( __global %s4 *source, write_only image3d_t dest )\n"
"{\n"
"    int  tidX   = get_global_id(0);\n"
"    int  tidY   = get_global_id(1);\n"
"    int  tidZ   = get_global_id(2);\n"
"    int  width  = get_image_width( dest );\n"
"    int  height = get_image_height( dest );\n"
"    int  index = tidZ * width * height + tidY * width + tidX;\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, (int4)( tidX, tidY, tidZ, 0 ), %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_3D_half =
"#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable\n"
"__kernel void sample_test( __global half4 *source, write_only image3d_t dest )\n"
"{\n"
"    int  tidX   = get_global_id(0);\n"
"    int  tidY   = get_global_id(1);\n"
"    int  tidZ   = get_global_id(2);\n"
"    int  width  = get_image_width( dest );\n"
"    int  height = get_image_height( dest );\n"
"    int  index = tidZ * width * height + tidY * width + tidX;\n"
"    write_imagef( dest, (int4)( tidX, tidY, tidZ, 0 ), vload_half4(index, (__global half *)source));\n"
"}\n";

static const char *kernelpattern_image_write_2Darray =
"__kernel void sample_test( __global %s4 *source, write_only image2d_array_t dest )\n"
"{\n"
"    int  tidX   = get_global_id(0);\n"
"    int  tidY   = get_global_id(1);\n"
"    int  tidZ   = get_global_id(2);\n"
"    int  width  = get_image_width( dest );\n"
"    int  height = get_image_height( dest );\n"
"    int  index = tidZ * width * height + tidY * width + tidX;\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, (int4)( tidX, tidY, tidZ, 0 ), %s(value));\n"
"}\n";

static const char *kernelpattern_image_write_2Darray_half =
"__kernel void sample_test( __global half4 *source, write_only image2d_array_t dest )\n"
"{\n"
"    int  tidX   = get_global_id(0);\n"
"    int  tidY   = get_global_id(1);\n"
"    int  tidZ   = get_global_id(2);\n"
"    int  width  = get_image_width( dest );\n"
"    int  height = get_image_height( dest );\n"
"    int  index = tidZ * width * height + tidY * width + tidX;\n"
"    write_imagef( dest, (int4)( tidX, tidY, tidZ, 0 ), vload_half4(index, (__global half *)source));\n"
"}\n";

#ifdef GL_VERSION_3_2

static const char * kernelpattern_image_write_2D_depth =
"__kernel void sample_test( __global %s *source, write_only image2d_depth_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    float value = source[index];\n"
"    write_imagef( dest, (int2)( tidX, tidY ), value);\n"
"}\n";

static const char * kernelpattern_image_write_2D_array_depth =
"__kernel void sample_test( __global %s *source, write_only image2d_array_depth_t dest )\n"
"{\n"
"    int  tidX   = get_global_id(0);\n"
"    int  tidY   = get_global_id(1);\n"
"    int  tidZ   = get_global_id(2);\n"
"    int  width  = get_image_width( dest );\n"
"    int  height = get_image_height( dest );\n"
"    int  index = tidZ * width * height + tidY * width + tidX;\n"
"    %s value = source[index];\n"
"    write_image%s( dest, (int4)( tidX, tidY, tidZ, 0 ), %s(value));\n"
"}\n";


#endif
// clang-format on

#pragma mark -
#pragma mark Utility functions

static const char *get_appropriate_write_kernel(GLenum target,
                                                ExplicitType type,
                                                cl_channel_order channel_order)
{
    switch (get_base_gl_target(target))
    {
        case GL_TEXTURE_1D:

            if (type == kHalf)
                return kernelpattern_image_write_1D_half;
            else
                return kernelpattern_image_write_1D;
            break;
        case GL_TEXTURE_BUFFER:
            if (type == kHalf)
                return kernelpattern_image_write_1D_buffer_half;
            else
                return kernelpattern_image_write_1D_buffer;
            break;
        case GL_TEXTURE_1D_ARRAY:
            if (type == kHalf)
                return kernelpattern_image_write_1Darray_half;
            else
                return kernelpattern_image_write_1Darray;
            break;
        case GL_COLOR_ATTACHMENT0:
        case GL_RENDERBUFFER:
        case GL_TEXTURE_RECTANGLE_EXT:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_CUBE_MAP:
#ifdef GL_VERSION_3_2
            if (channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
                return kernelpattern_image_write_2D_depth;
#endif
            if (type == kHalf)
                return kernelpattern_image_write_2D_half;
            else
                return kernelpattern_image_write_2D;
            break;

        case GL_TEXTURE_2D_ARRAY:
#ifdef GL_VERSION_3_2
            if (channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
                return kernelpattern_image_write_2D_array_depth;
#endif
            if (type == kHalf)
                return kernelpattern_image_write_2Darray_half;
            else
                return kernelpattern_image_write_2Darray;
            break;

        case GL_TEXTURE_3D:
            if (type == kHalf)
                return kernelpattern_image_write_3D_half;
            else
                return kernelpattern_image_write_3D;
            break;

        default:
            log_error("Unsupported GL tex target (%s) passed to write test: "
                      "%s (%s):%d",
                      GetGLTargetName(target), __FUNCTION__, __FILE__,
                      __LINE__);
            return NULL;
    }
}

void set_dimensions_by_target(GLenum target, size_t *dims, size_t sizes[3],
                              size_t width, size_t height, size_t depth)
{
    switch (get_base_gl_target(target))
    {
        case GL_TEXTURE_1D:
            sizes[0] = width;
            *dims = 1;
            break;

        case GL_TEXTURE_BUFFER:
            sizes[0] = width;
            *dims = 1;
            break;

        case GL_TEXTURE_1D_ARRAY:
            sizes[0] = width;
            sizes[1] = height;
            *dims = 2;
            break;

        case GL_COLOR_ATTACHMENT0:
        case GL_RENDERBUFFER:
        case GL_TEXTURE_RECTANGLE_EXT:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_CUBE_MAP:

            sizes[0] = width;
            sizes[1] = height;
            *dims = 2;
            break;

        case GL_TEXTURE_2D_ARRAY:
            sizes[0] = width;
            sizes[1] = height;
            sizes[2] = depth;
            *dims = 3;
            break;

        case GL_TEXTURE_3D:
            sizes[0] = width;
            sizes[1] = height;
            sizes[2] = depth;
            *dims = 3;
            break;

        default:
            log_error("Unsupported GL tex target (%s) passed to write test: "
                      "%s (%s):%d",
                      GetGLTargetName(target), __FUNCTION__, __FILE__,
                      __LINE__);
    }
}

int test_cl_image_write(cl_context context, cl_command_queue queue,
                        GLenum target, cl_mem clImage, size_t width,
                        size_t height, size_t depth, cl_image_format *outFormat,
                        ExplicitType *outType, void **outSourceBuffer, MTdata d,
                        bool supports_half)
{
    size_t global_dims, global_sizes[3];
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper inStream;
    char *programPtr;
    int error;
    char kernelSource[2048];

    // What CL format did we get from the texture?

    error = clGetImageInfo(clImage, CL_IMAGE_FORMAT, sizeof(cl_image_format),
                           outFormat, NULL);
    test_error(error, "Unable to get the CL image format");

    // Create the kernel source.  The target and the data type will influence
    // which particular kernel we choose.

    *outType = get_write_kernel_type(outFormat);
    size_t channelSize = get_explicit_type_size(*outType);

    const char *appropriateKernel = get_appropriate_write_kernel(
        target, *outType, outFormat->image_channel_order);
    if (*outType == kHalf && !supports_half)
    {
        log_info("cl_khr_fp16 isn't supported. Skip this test.\n");
        return 0;
    }

    const char *suffix = get_kernel_suffix(outFormat);
    const char *convert = get_write_conversion(outFormat, *outType);

    sprintf(kernelSource, appropriateKernel, get_explicit_type_name(*outType),
            get_explicit_type_name(*outType), suffix, convert);

    programPtr = kernelSource;
    if (create_single_kernel_helper_with_build_options(
            context, &program, &kernel, 1, (const char **)&programPtr,
            "sample_test", ""))
    {
        return -1;
    }

    // Create an appropriately-sized output buffer.

    // Check to see if the output buffer will fit on the device
    size_t bytes = channelSize * 4 * width * height * depth;
    cl_ulong alloc_size = 0;

    cl_device_id device = NULL;
    error = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device),
                                  &device, NULL);
    test_error(error, "Unable to query command queue for device");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(alloc_size), &alloc_size, NULL);
    test_error(error, "Unable to device for max mem alloc size");

    if (bytes > alloc_size)
    {
        log_info("  Skipping: Buffer size (%lu) is greater than "
                 "CL_DEVICE_MAX_MEM_ALLOC_SIZE (%lu)\n",
                 bytes, alloc_size);
        *outSourceBuffer = NULL;
        return 0;
    }

    *outSourceBuffer =
        CreateRandomData(*outType, width * height * depth * 4, d);

    inStream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              channelSize * 4 * width * height * depth,
                              *outSourceBuffer, &error);
    test_error(error, "Unable to create output buffer");

    clSamplerWrapper sampler = clCreateSampler(
        context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
    test_error(error, "Unable to create sampler");

    error = clSetKernelArg(kernel, 0, sizeof(inStream), &inStream);
    test_error(error, "Unable to set kernel arguments");

    error = clSetKernelArg(kernel, 1, sizeof(clImage), &clImage);
    test_error(error, "Unable to set kernel arguments");

    // Flush and Acquire.

    glFinish();

    error = (*clEnqueueAcquireGLObjects_ptr)(queue, 1, &clImage, 0, NULL, NULL);
    test_error(error, "Unable to acquire GL obejcts");

    // Execute ( letting OpenCL choose the local size )

    // Setup the global dimensions and sizes based on the target type.
    set_dimensions_by_target(target, &global_dims, global_sizes, width, height,
                             depth);

    error = clEnqueueNDRangeKernel(queue, kernel, global_dims, NULL,
                                   global_sizes, NULL, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    clEventWrapper event;
    error =
        (*clEnqueueReleaseGLObjects_ptr)(queue, 1, &clImage, 0, NULL, &event);
    test_error(error, "clEnqueueReleaseGLObjects failed");

    error = clWaitForEvents(1, &event);
    test_error(error, "clWaitForEvents failed");

    return 0;
}

static int test_image_write(cl_context context, cl_command_queue queue,
                            GLenum glTarget, GLuint glTexture, size_t width,
                            size_t height, size_t depth,
                            cl_image_format *outFormat, ExplicitType *outType,
                            void **outSourceBuffer, MTdata d,
                            bool supports_half)
{
    int error;

    // Create a CL image from the supplied GL texture
    clMemWrapper image = (*clCreateFromGLTexture_ptr)(
        context, CL_MEM_WRITE_ONLY, glTarget, 0, glTexture, &error);

    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to create CL image from GL texture");
        GLint fmt;
        glGetTexLevelParameteriv(glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
        log_error("    Supplied GL texture was base format %s and internal "
                  "format %s\n",
                  GetGLBaseFormatName(fmt), GetGLFormatName(fmt));
        return error;
    }

    return test_cl_image_write(context, queue, glTarget, image, width, height,
                               depth, outFormat, outType, outSourceBuffer, d,
                               supports_half);
}

int supportsHalf(cl_context context, bool *supports_half)
{
    int error;
    cl_uint numDev;

    error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint),
                             &numDev, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_NUM_DEVICES failed");

    cl_device_id *devices = new cl_device_id[numDev];
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             numDev * sizeof(cl_device_id), devices, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

    *supports_half = is_extension_available(devices[0], "cl_khr_fp16");
    delete[] devices;

    return error;
}

int supportsMsaa(cl_context context, bool *supports_msaa)
{
    int error;
    cl_uint numDev;

    error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint),
                             &numDev, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_NUM_DEVICES failed");

    cl_device_id *devices = new cl_device_id[numDev];
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             numDev * sizeof(cl_device_id), devices, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

    *supports_msaa =
        is_extension_available(devices[0], "cl_khr_gl_msaa_sharing");
    delete[] devices;

    return error;
}

int supportsDepth(cl_context context, bool *supports_depth)
{
    int error;
    cl_uint numDev;

    error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint),
                             &numDev, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_NUM_DEVICES failed");

    cl_device_id *devices = new cl_device_id[numDev];
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             numDev * sizeof(cl_device_id), devices, NULL);
    test_error(error, "clGetContextInfo for CL_CONTEXT_DEVICES failed");

    *supports_depth =
        is_extension_available(devices[0], "cl_khr_gl_depth_images");
    delete[] devices;

    return error;
}

static int test_image_format_write(cl_context context, cl_command_queue queue,
                                   size_t width, size_t height, size_t depth,
                                   GLenum target, GLenum format,
                                   GLenum internalFormat, GLenum glType,
                                   ExplicitType type, MTdata d)
{
    int error;
    // If we're testing a half float format, then we need to determine the
    // rounding mode of this machine.  Punt if we fail to do so.

    if (type == kHalf)
        if (DetectFloatToHalfRoundingMode(queue)) return 1;

    // Create an appropriate GL texture or renderbuffer, given the target.

    glTextureWrapper glTexture;
    glBufferWrapper glBuf;
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    switch (get_base_gl_target(target))
    {
        case GL_TEXTURE_1D:
            CreateGLTexture1D(width, target, format, internalFormat, glType,
                              type, &glTexture, &error, false, d);
            break;
        case GL_TEXTURE_BUFFER:
            CreateGLTextureBuffer(width, target, format, internalFormat, glType,
                                  type, &glTexture, &glBuf, &error, false, d);
            break;
        case GL_TEXTURE_1D_ARRAY:
            CreateGLTexture1DArray(width, height, target, format,
                                   internalFormat, glType, type, &glTexture,
                                   &error, false, d);
            break;
        case GL_TEXTURE_RECTANGLE_EXT:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_CUBE_MAP:
            CreateGLTexture2D(width, height, target, format, internalFormat,
                              glType, type, &glTexture, &error, false, d);
            break;
        case GL_COLOR_ATTACHMENT0:
        case GL_RENDERBUFFER:
            CreateGLRenderbuffer(width, height, target, format, internalFormat,
                                 glType, type, &glFramebuffer, &glRenderbuffer,
                                 &error, d, false);
        case GL_TEXTURE_2D_ARRAY:
            CreateGLTexture2DArray(width, height, depth, target, format,
                                   internalFormat, glType, type, &glTexture,
                                   &error, false, d);
            break;
        case GL_TEXTURE_3D:
            CreateGLTexture3D(width, height, depth, target, format,
                              internalFormat, glType, type, &glTexture, &error,
                              d, false);
            break;

        default:
            log_error("Unsupported GL tex target (%s) passed to write test: "
                      "%s (%s):%d",
                      GetGLTargetName(target), __FUNCTION__, __FILE__,
                      __LINE__);
    }

    // If there was a problem during creation, make sure it isn't a known
    // cause, and then complain.
    if (error == -2)
    {
        log_info("OpenGL texture couldn't be created, because a texture is too "
                 "big. Skipping test.\n");
        return 0;
    }

    if (error != 0)
    {
        if ((format == GL_RGBA_INTEGER_EXT)
            && (!CheckGLIntegerExtensionSupport()))
        {
            log_info("OpenGL version does not support GL_RGBA_INTEGER_EXT. "
                     "Skipping test.\n");
            return 0;
        }
        else
        {
            return error;
        }
    }

    // Run and get the results
    cl_image_format clFormat;
    ExplicitType sourceType;
    ExplicitType validationType;
    void *outSourceBuffer = NULL;

    GLenum globj = glTexture;
    if (target == GL_RENDERBUFFER || target == GL_COLOR_ATTACHMENT0)
    {
        globj = glRenderbuffer;
    }

    bool supports_half = false;
    error = supportsHalf(context, &supports_half);
    if (error != 0) return error;

    error = test_image_write(context, queue, target, globj, width, height,
                             depth, &clFormat, &sourceType,
                             (void **)&outSourceBuffer, d, supports_half);

    if (error != 0 || ((sourceType == kHalf) && !supports_half))
    {
        if (outSourceBuffer) free(outSourceBuffer);
        return error;
    }

    if (!outSourceBuffer) return 0;

    // If actual source type was half, convert to float for validation.

    if (sourceType == kHalf)
        validationType = kFloat;
    else
        validationType = sourceType;

    BufferOwningPtr<char> validationSource;

    if (clFormat.image_channel_data_type == CL_UNORM_INT_101010)
    {
        validationSource.reset(outSourceBuffer);
    }
    else
    {
        validationSource.reset(convert_to_expected(
            outSourceBuffer, width * height * depth, sourceType, validationType,
            get_channel_order_channel_count(clFormat.image_channel_order)));
        free(outSourceBuffer);
    }

    log_info(
        "- Write for %s [%4ld x %4ld x %4ld] : GL Texture : %s : %s : %s =>"
        " CL Image : %s : %s \n",
        GetGLTargetName(target), width, height, depth, GetGLFormatName(format),
        GetGLFormatName(internalFormat), GetGLTypeName(glType),
        GetChannelOrderName(clFormat.image_channel_order),
        GetChannelTypeName(clFormat.image_channel_data_type));

    // Read the results from the GL texture.

    ExplicitType readType = type;
    BufferOwningPtr<char> glResults(
        ReadGLTexture(target, glTexture, glBuf, width, format, internalFormat,
                      glType, readType, /* unused */ 1, 1));
    if (glResults == NULL) return -1;

    // We have to convert our input buffer to the returned type, so we can
    // validate.
    BufferOwningPtr<char> convertedGLResults;
    if (clFormat.image_channel_data_type != CL_UNORM_INT_101010)
    {
        convertedGLResults.reset(convert_to_expected(
            glResults, width * height * depth, readType, validationType,
            get_channel_order_channel_count(clFormat.image_channel_order),
            glType));
    }

    // Validate.

    int valid = 0;
    if (convertedGLResults)
    {
        if (sourceType == kFloat || sourceType == kHalf)
        {
            if (clFormat.image_channel_data_type == CL_UNORM_INT_101010)
            {
                valid = validate_float_results_rgb_101010(
                    validationSource, glResults, width, height, depth, 1);
            }
            else
            {
                valid =
                    validate_float_results(validationSource, convertedGLResults,
                                           width, height, depth, 1,
                                           get_channel_order_channel_count(
                                               clFormat.image_channel_order));
            }
        }
        else
        {
            valid = validate_integer_results(
                validationSource, convertedGLResults, width, height, depth, 1,
                get_explicit_type_size(readType));
        }
    }

    return valid;
}

#pragma mark -
#pragma mark Write test common entry point

// This is the main loop for all of the write tests.  It iterates over the
// given formats & targets, testing a variety of sizes against each
// combination.

int test_images_write_common(cl_device_id device, cl_context context,
                             cl_command_queue queue, const format *formats,
                             size_t nformats, GLenum *targets, size_t ntargets,
                             sizevec_t *sizes, size_t nsizes)
{
    int err = 0;
    int error = 0;
    RandomSeed seed(gRandomSeed);

    // First, ensure this device supports images.

    if (checkForImageSupport(device))
    {
        log_info("Device does not support images.  Skipping test.\n");
        return 0;
    }

    // Get the value of CL_DEVICE_MAX_MEM_ALLOC_SIZE
    cl_ulong max_individual_allocation_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(max_individual_allocation_size),
                          &max_individual_allocation_size, NULL);
    if (err)
    {
        log_error("ERROR: clGetDeviceInfo failed for "
                  "CL_DEVICE_MAX_MEM_ALLOC_SIZE.\n");
        error++;
        return error;
    }

    size_t total_allocation_size;
    size_t fidx, tidx, sidx;

    for (fidx = 0; fidx < nformats; fidx++)
    {
        for (tidx = 0; tidx < ntargets; tidx++)
        {

            // Texture buffer only takes an internal format, so the level data
            // passed by the test and used for verification must match the
            // internal format
            if ((targets[tidx] == GL_TEXTURE_BUFFER)
                && (GetGLFormat(formats[fidx].internal)
                    != formats[fidx].formattype))
                continue;

            if (formats[fidx].datatype == GL_UNSIGNED_INT_2_10_10_10_REV)
            {
                // Check if the RGB 101010 format is supported
                if (is_rgb_101010_supported(context, targets[tidx]) == 0)
                    continue; // skip
            }

            if (formats[fidx].datatype == GL_UNSIGNED_INT_24_8)
            {
                // check if a implementation supports writing to the depth
                // stencil formats
                cl_image_format imageFormat = { CL_DEPTH_STENCIL,
                                                CL_UNORM_INT24 };
                if (!is_image_format_supported(
                        context, CL_MEM_WRITE_ONLY,
                        (targets[tidx] == GL_TEXTURE_2D
                         || targets[tidx] == GL_TEXTURE_RECTANGLE)
                            ? CL_MEM_OBJECT_IMAGE2D
                            : CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        &imageFormat))
                    continue;
            }

            if (formats[fidx].datatype == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
            {
                // check if a implementation supports writing to the depth
                // stencil formats
                cl_image_format imageFormat = { CL_DEPTH_STENCIL, CL_FLOAT };
                if (!is_image_format_supported(
                        context, CL_MEM_WRITE_ONLY,
                        (targets[tidx] == GL_TEXTURE_2D
                         || targets[tidx] == GL_TEXTURE_RECTANGLE)
                            ? CL_MEM_OBJECT_IMAGE2D
                            : CL_MEM_OBJECT_IMAGE2D_ARRAY,
                        &imageFormat))
                    continue;
            }

            if (targets[tidx] != GL_TEXTURE_BUFFER)
                log_info(
                    "Testing image write for GL format %s : %s : %s : %s\n",
                    GetGLTargetName(targets[tidx]),
                    GetGLFormatName(formats[fidx].internal),
                    GetGLBaseFormatName(formats[fidx].formattype),
                    GetGLTypeName(formats[fidx].datatype));
            else
                log_info("Testing image write for GL format %s : %s\n",
                         GetGLTargetName(targets[tidx]),
                         GetGLFormatName(formats[fidx].internal));


            for (sidx = 0; sidx < nsizes; sidx++)
            {

                // All tested formats are 4-channel formats
                total_allocation_size = sizes[sidx].width * sizes[sidx].height
                    * sizes[sidx].depth * 4
                    * get_explicit_type_size(formats[fidx].type);

                if (total_allocation_size > max_individual_allocation_size)
                {
                    log_info("The requested allocation size (%gMB) is larger "
                             "than the "
                             "maximum individual allocation size (%gMB)\n",
                             total_allocation_size / (1024.0 * 1024.0),
                             max_individual_allocation_size
                                 / (1024.0 * 1024.0));
                    log_info("Skipping write test for %s : %s : %s : %s "
                             " and size (%ld, %ld, %ld)\n",
                             GetGLTargetName(targets[tidx]),
                             GetGLFormatName(formats[fidx].internal),
                             GetGLBaseFormatName(formats[fidx].formattype),
                             GetGLTypeName(formats[fidx].datatype),
                             sizes[sidx].width, sizes[sidx].height,
                             sizes[sidx].depth);
                    continue;
                }
#ifdef GL_VERSION_3_2
                if (get_base_gl_target(targets[tidx])
                        == GL_TEXTURE_2D_MULTISAMPLE
                    || get_base_gl_target(targets[tidx])
                        == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
                {
                    bool supports_msaa;
                    int errorInGetInfo = supportsMsaa(context, &supports_msaa);
                    if (errorInGetInfo != 0) return errorInGetInfo;
                    if (!supports_msaa) return 0;
                }
                if (formats[fidx].formattype == GL_DEPTH_COMPONENT
                    || formats[fidx].formattype == GL_DEPTH_STENCIL)
                {
                    bool supports_depth;
                    int errorInGetInfo =
                        supportsDepth(context, &supports_depth);
                    if (errorInGetInfo != 0) return errorInGetInfo;
                    if (!supports_depth) return 0;
                }
#endif

                if (test_image_format_write(
                        context, queue, sizes[sidx].width, sizes[sidx].height,
                        sizes[sidx].depth, targets[tidx],
                        formats[fidx].formattype, formats[fidx].internal,
                        formats[fidx].datatype, formats[fidx].type, seed))
                {
                    log_error(
                        "ERROR: Image write test failed for %s : %s : %s : %s "
                        " and size (%ld, %ld, %ld)\n\n",
                        GetGLTargetName(targets[tidx]),
                        GetGLFormatName(formats[fidx].internal),
                        GetGLBaseFormatName(formats[fidx].formattype),
                        GetGLTypeName(formats[fidx].datatype),
                        sizes[sidx].width, sizes[sidx].height,
                        sizes[sidx].depth);

                    error++;
                    break; // Skip other sizes for this combination
                }
            }

            // If we passed all sizes (check versus size loop count):

            if (sidx == nsizes)
            {
                log_info(
                    "passed: Image write for GL format  %s : %s : %s : %s\n\n",
                    GetGLTargetName(targets[tidx]),
                    GetGLFormatName(formats[fidx].internal),
                    GetGLBaseFormatName(formats[fidx].formattype),
                    GetGLTypeName(formats[fidx].datatype));
            }
        }
    }

    return error;
}

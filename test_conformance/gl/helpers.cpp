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
#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

const char *get_kernel_suffix(cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_UNORM_INT24:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
        case CL_UNORM_INT_101010: return "f";
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32: return "i";
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32: return "ui";
        default:
            log_error("Test error: unsupported kernel suffix for "
                      "image_channel_data_type 0x%X\n",
                      format->image_channel_data_type);
            return "";
    }
}

ExplicitType get_read_kernel_type(cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_UNORM_INT24:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
        case CL_UNORM_INT_101010:
#ifdef GL_VERSION_3_2
        case CL_DEPTH:
#endif
            return kFloat;
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32: return kInt;
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32: return kUInt;
        default:
            log_error("Test error: unsupported kernel suffix for "
                      "image_channel_data_type 0x%X\n",
                      format->image_channel_data_type);
            return kNumExplicitTypes;
    }
}

ExplicitType get_write_kernel_type(cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_UNORM_INT8: return kFloat;
        case CL_UNORM_INT16: return kFloat;
        case CL_UNORM_INT24: return kFloat;
        case CL_SNORM_INT8: return kFloat;
        case CL_SNORM_INT16: return kFloat;
        case CL_HALF_FLOAT: return kHalf;
        case CL_FLOAT: return kFloat;
        case CL_SIGNED_INT8: return kChar;
        case CL_SIGNED_INT16: return kShort;
        case CL_SIGNED_INT32: return kInt;
        case CL_UNSIGNED_INT8: return kUChar;
        case CL_UNSIGNED_INT16: return kUShort;
        case CL_UNSIGNED_INT32: return kUInt;
        case CL_UNORM_INT_101010: return kFloat;
#ifdef GL_VERSION_3_2
        case CL_DEPTH: return kFloat;
#endif
        default: return kInt;
    }
}

const char *get_write_conversion(cl_image_format *format, ExplicitType type)
{
    switch (format->image_channel_data_type)
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
        case CL_UNORM_INT_101010:
        case CL_UNORM_INT24:
            if (type != kFloat) return "convert_float4";
            break;
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
            if (type != kInt) return "convert_int4";
            break;
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
            if (type != kUInt) return "convert_uint4";
            break;
        default: return "";
    }
    return "";
}

// The only three input types to this function are kInt, kUInt and kFloat, due
// to the way we set up our tests The output types, though, are pretty much
// anything valid for GL to receive

#define DOWNSCALE_INTEGER_CASE(enum, type, bitShift)                           \
    case enum: {                                                               \
        cl_##type *dst = new cl_##type[numPixels * 4];                         \
        for (size_t i = 0; i < numPixels * 4; i++) dst[i] = src[i];            \
        return (char *)dst;                                                    \
    }

#define UPSCALE_FLOAT_CASE(enum, type, typeMax)                                \
    case enum: {                                                               \
        cl_##type *dst = new cl_##type[numPixels * 4];                         \
        for (size_t i = 0; i < numPixels * 4; i++)                             \
            dst[i] = (cl_##type)(src[i] * typeMax);                            \
        return (char *)dst;                                                    \
    }

char *convert_to_expected(void *inputBuffer, size_t numPixels,
                          ExplicitType inType, ExplicitType outType,
                          size_t channelNum, GLenum glDataType)
{
#ifdef DEBUG
    log_info("- Converting from input type '%s' to output type '%s'\n",
             get_explicit_type_name(inType), get_explicit_type_name(outType));
#endif

    if (inType == outType)
    {
        char *outData =
            new char[numPixels * channelNum
                     * get_explicit_type_size(outType)]; // sizeof( cl_int ) ];
        if (glDataType == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
        {
            for (size_t i = 0; i < numPixels; ++i)
            {
                ((cl_float *)outData)[i] = ((cl_float *)inputBuffer)[2 * i];
            }
        }
        else
        {
            memcpy(outData, inputBuffer,
                   numPixels * channelNum * get_explicit_type_size(inType));
        }
        return outData;
    }
    else if (inType == kChar)
    {
        cl_char *src = (cl_char *)inputBuffer;

        switch (outType)
        {
            case kInt: {
                cl_int *outData = new cl_int[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_int)((src[i]));
                }
                return (char *)outData;
            }
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_float)src[i] / 127.0f;
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kUChar)
    {
        cl_uchar *src = (cl_uchar *)inputBuffer;

        switch (outType)
        {
            case kUInt: {
                cl_uint *outData = new cl_uint[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_uint)((src[i]));
                }
                return (char *)outData;
            }
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_float)(src[i]) / 256.0f;
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kShort)
    {
        cl_short *src = (cl_short *)inputBuffer;

        switch (outType)
        {
            case kInt: {
                cl_int *outData = new cl_int[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_int)((src[i]));
                }
                return (char *)outData;
            }
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_float)src[i] / 32768.0f;
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kUShort)
    {
        cl_ushort *src = (cl_ushort *)inputBuffer;

        switch (outType)
        {
            case kUInt: {
                cl_uint *outData = new cl_uint[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_uint)((src[i]));
                }
                return (char *)outData;
            }
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_float)(src[i]) / 65535.0f;
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kInt)
    {
        cl_int *src = (cl_int *)inputBuffer;

        switch (outType)
        {
            DOWNSCALE_INTEGER_CASE(kShort, short, 16)
            DOWNSCALE_INTEGER_CASE(kChar, char, 24)
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] =
                        (cl_float)fmaxf((float)src[i] / 2147483647.f, -1.f);
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kUInt)
    {
        cl_uint *src = (cl_uint *)inputBuffer;

        switch (outType)
        {
            DOWNSCALE_INTEGER_CASE(kUShort, ushort, 16)
            DOWNSCALE_INTEGER_CASE(kUChar, uchar, 24)
            case kFloat: {
                // If we're converting to float, then CL decided that we should
                // be normalized
                cl_float *outData = new cl_float[numPixels * channelNum];
                const cl_float MaxValue = (glDataType == GL_UNSIGNED_INT_24_8)
                    ? 16777215.f
                    : 4294967295.f;
                const cl_uint ShiftBits =
                    (glDataType == GL_UNSIGNED_INT_24_8) ? 8 : 0;
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = (cl_float)(src[i] >> ShiftBits) / MaxValue;
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else if (inType == kHalf)
    {
        cl_half *src = (cl_half *)inputBuffer;

        switch (outType)
        {
            case kFloat: {
                cl_float *outData = new cl_float[numPixels * channelNum];
                for (size_t i = 0; i < numPixels * channelNum; i++)
                {
                    outData[i] = cl_half_to_float(src[i]);
                }
                return (char *)outData;
            }
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }
    else
    {
        cl_float *src = (cl_float *)inputBuffer;

        switch (outType)
        {
            UPSCALE_FLOAT_CASE(kChar, char, 127.f)
            UPSCALE_FLOAT_CASE(kUChar, uchar, 255.f)
            UPSCALE_FLOAT_CASE(kShort, short, 32767.f)
            UPSCALE_FLOAT_CASE(kUShort, ushort, 65535.f)
            UPSCALE_FLOAT_CASE(kInt, int, 2147483647.f)
            UPSCALE_FLOAT_CASE(kUInt, uint, 4294967295.f)
            default:
                log_error("ERROR: Unsupported conversion from %s to %s!\n",
                          get_explicit_type_name(inType),
                          get_explicit_type_name(outType));
                return NULL;
        }
    }

    return NULL;
}

int validate_integer_results(void *expectedResults, void *actualResults,
                             size_t width, size_t height, size_t sampleNum,
                             size_t typeSize)
{
    return validate_integer_results(expectedResults, actualResults, width,
                                    height, sampleNum, 0, typeSize);
}

int validate_integer_results(void *expectedResults, void *actualResults,
                             size_t width, size_t height, size_t depth,
                             size_t sampleNum, size_t typeSize)
{
    char *expected = (char *)expectedResults;
    char *actual = (char *)actualResults;
    for (size_t s = 0; s < sampleNum; s++)
    {
        for (size_t z = 0; z < ((depth == 0) ? 1 : depth); z++)
        {
            for (size_t y = 0; y < height; y++)
            {
                for (size_t x = 0; x < width; x++)
                {
                    if (memcmp(expected, actual, typeSize * 4) != 0)
                    {
                        char scratch[1024];

                        if (depth == 0)
                            log_error("ERROR: Data sample %d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)s);
                        else
                            log_error("ERROR: Data sample %d,%d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)z, (int)s);
                        log_error("\tExpected: %s\n",
                                  GetDataVectorString(expected, typeSize, 4,
                                                      scratch));
                        log_error(
                            "\t  Actual: %s\n",
                            GetDataVectorString(actual, typeSize, 4, scratch));
                        return -1;
                    }
                    expected += typeSize * 4;
                    actual += typeSize * 4;
                }
            }
        }
    }

    return 0;
}

int validate_float_results(void *expectedResults, void *actualResults,
                           size_t width, size_t height, size_t sampleNum,
                           size_t channelNum)
{
    return validate_float_results(expectedResults, actualResults, width, height,
                                  sampleNum, 0, channelNum);
}

int validate_float_results(void *expectedResults, void *actualResults,
                           size_t width, size_t height, size_t depth,
                           size_t sampleNum, size_t channelNum)
{
    cl_float *expected = (cl_float *)expectedResults;
    cl_float *actual = (cl_float *)actualResults;
    for (size_t s = 0; s < sampleNum; s++)
    {
        for (size_t z = 0; z < ((depth == 0) ? 1 : depth); z++)
        {
            for (size_t y = 0; y < height; y++)
            {
                for (size_t x = 0; x < width; x++)
                {
                    float err = 0.f;
                    for (size_t i = 0; i < channelNum; i++)
                    {
                        float error = fabsf(expected[i] - actual[i]);
                        if (error > err) err = error;
                    }

                    if (err > 1.f / 127.f) // Max expected range of error if we
                                           // converted from an 8-bit integer to
                                           // a normalized float
                    {
                        if (depth == 0)
                            log_error("ERROR: Data sample %d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)s);
                        else
                            log_error("ERROR: Data sample %d,%d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)z, (int)s);

                        if (channelNum == 4)
                        {
                            log_error("\tExpected: %f %f %f %f\n", expected[0],
                                      expected[1], expected[2], expected[3]);
                            log_error("\t        : %a %a %a %a\n", expected[0],
                                      expected[1], expected[2], expected[3]);
                            log_error("\t  Actual: %f %f %f %f\n", actual[0],
                                      actual[1], actual[2], actual[3]);
                            log_error("\t        : %a %a %a %a\n", actual[0],
                                      actual[1], actual[2], actual[3]);
                        }
                        else if (channelNum == 1)
                        {
                            log_error("\tExpected: %f\n", expected[0]);
                            log_error("\t        : %a\n", expected[0]);
                            log_error("\t  Actual: %f\n", actual[0]);
                            log_error("\t        : %a\n", actual[0]);
                        }
                        return -1;
                    }
                    expected += channelNum;
                    actual += channelNum;
                }
            }
        }
    }

    return 0;
}

int validate_float_results_rgb_101010(void *expectedResults,
                                      void *actualResults, size_t width,
                                      size_t height, size_t sampleNum)
{
    return validate_float_results_rgb_101010(expectedResults, actualResults,
                                             width, height, sampleNum, 0);
}

int validate_float_results_rgb_101010(void *expectedResults,
                                      void *actualResults, size_t width,
                                      size_t height, size_t depth,
                                      size_t sampleNum)
{
    cl_float *expected = (cl_float *)expectedResults;
    cl_float *actual = (cl_float *)actualResults;
    for (size_t s = 0; s < sampleNum; s++)
    {
        for (size_t z = 0; z < ((depth == 0) ? 1 : depth); z++)
        {
            for (size_t y = 0; y < height; y++)
            {
                for (size_t x = 0; x < width; x++)
                {
                    float err = 0.f;
                    for (size_t i = 0; i < 3; i++) // skip the fourth channel
                    {
                        float error = fabsf(expected[i] - actual[i]);
                        if (error > err) err = error;
                    }

                    if (err > 1.f / 127.f) // Max expected range of error if we
                                           // converted from an 8-bit integer to
                                           // a normalized float
                    {
                        if (depth == 0)
                            log_error("ERROR: Data sample %d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)s);
                        else
                            log_error("ERROR: Data sample %d,%d,%d,%d did not "
                                      "validate!\n",
                                      (int)x, (int)y, (int)z, (int)s);
                        log_error("\tExpected: %f %f %f\n", expected[0],
                                  expected[1], expected[2]);
                        log_error("\t        : %a %a %a\n", expected[0],
                                  expected[1], expected[2]);
                        log_error("\t  Actual: %f %f %f\n", actual[0],
                                  actual[1], actual[2]);
                        log_error("\t        : %a %a %a\n", actual[0],
                                  actual[1], actual[2]);
                        return -1;
                    }
                    expected += 4;
                    actual += 4;
                }
            }
        }
    }

    return 0;
}

int CheckGLObjectInfo(cl_mem mem, cl_gl_object_type expected_cl_gl_type,
                      GLuint expected_gl_name,
                      GLenum expected_cl_gl_texture_target,
                      GLint expected_cl_gl_mipmap_level)
{
    cl_gl_object_type object_type;
    GLuint object_name;
    GLenum texture_target;
    GLint mipmap_level;
    int error;

    error = (*clGetGLObjectInfo_ptr)(mem, &object_type, &object_name);
    test_error(error, "clGetGLObjectInfo failed");
    if (object_type != expected_cl_gl_type)
    {
        log_error("clGetGLObjectInfo did not return expected object type: "
                  "expected %d, got %d.\n",
                  expected_cl_gl_type, object_type);
        return -1;
    }
    if (object_name != expected_gl_name)
    {
        log_error("clGetGLObjectInfo did not return expected object name: "
                  "expected %d, got %d.\n",
                  expected_gl_name, object_name);
        return -1;
    }

    // If we're dealing with a buffer or render buffer, we are done.

    if (object_type == CL_GL_OBJECT_BUFFER
        || object_type == CL_GL_OBJECT_RENDERBUFFER)
    {
        return 0;
    }

    // Otherwise, it's a texture-based object and requires a bit more checking.

    error = (*clGetGLTextureInfo_ptr)(mem, CL_GL_TEXTURE_TARGET,
                                      sizeof(texture_target), &texture_target,
                                      NULL);
    test_error(error, "clGetGLTextureInfo for CL_GL_TEXTURE_TARGET failed");

    if (texture_target != expected_cl_gl_texture_target)
    {
        log_error("clGetGLTextureInfo did not return expected texture target: "
                  "expected %d, got %d.\n",
                  expected_cl_gl_texture_target, texture_target);
        return -1;
    }

    error = (*clGetGLTextureInfo_ptr)(
        mem, CL_GL_MIPMAP_LEVEL, sizeof(mipmap_level), &mipmap_level, NULL);
    test_error(error, "clGetGLTextureInfo for CL_GL_MIPMAP_LEVEL failed");

    if (mipmap_level != expected_cl_gl_mipmap_level)
    {
        log_error("clGetGLTextureInfo did not return expected mipmap level: "
                  "expected %d, got %d.\n",
                  expected_cl_gl_mipmap_level, mipmap_level);
        return -1;
    }

    return 0;
}

bool CheckGLIntegerExtensionSupport()
{
    // Get the OpenGL version and supported extensions
    const GLubyte *glVersion = glGetString(GL_VERSION);
    const GLubyte *glExtensionList = glGetString(GL_EXTENSIONS);

    // Check if the OpenGL vrsion is 3.0 or grater or GL_EXT_texture_integer is
    // supported
    return (
        ((glVersion[0] - '0') >= 3)
        || (strstr((const char *)glExtensionList, "GL_EXT_texture_integer")));
}

int is_rgb_101010_supported(cl_context context, GLenum gl_target)
{
    cl_image_format formatList[128];
    cl_uint formatCount = 0;
    unsigned int i;
    int error;

    cl_mem_object_type image_type;

    switch (get_base_gl_target(gl_target))
    {
        case GL_TEXTURE_1D: image_type = CL_MEM_OBJECT_IMAGE1D;
        case GL_TEXTURE_BUFFER:
            image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            break;
        case GL_TEXTURE_RECTANGLE_EXT:
        case GL_TEXTURE_2D:
        case GL_COLOR_ATTACHMENT0:
        case GL_RENDERBUFFER:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            image_type = CL_MEM_OBJECT_IMAGE2D;
            break;
        case GL_TEXTURE_3D: image_type = CL_MEM_OBJECT_IMAGE3D;
        case GL_TEXTURE_1D_ARRAY: image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
        case GL_TEXTURE_2D_ARRAY:
            image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            break;
        default: image_type = CL_MEM_OBJECT_IMAGE2D;
    }

    if ((error =
             clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, image_type,
                                        128, formatList, &formatCount)))
    {
        return error;
    }

    // Check if the RGB 101010 format is supported
    for (i = 0; i < formatCount; i++)
    {
        if (formatList[i].image_channel_data_type == CL_UNORM_INT_101010)
        {
            return 1;
        }
    }

    return 0;
}

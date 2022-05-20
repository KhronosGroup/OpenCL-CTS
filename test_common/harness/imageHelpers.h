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
#ifndef _imageHelpers_h
#define _imageHelpers_h

#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <time.h>

#include "errorHelpers.h"

#include "conversions.h"
#include "typeWrappers.h"
#include "kernelHelpers.h"
#include "errorHelpers.h"
#include "mt19937.h"
#include "rounding_mode.h"
#include "clImageHelper.h"

#include <CL/cl_half.h>

extern cl_device_type gDeviceType;
extern bool gTestRounding;

// Number of iterations per image format to test if not testing max images,
// rounding, or small images
#define NUM_IMAGE_ITERATIONS 3


// Definition for our own sampler type, to mirror the cl_sampler internals
#define MAX_sRGB_TO_lRGB_CONVERSION_ERROR 0.5
#define MAX_lRGB_TO_sRGB_CONVERSION_ERROR 0.6

// Definition for our own sampler type, to mirror the cl_sampler internals
typedef struct
{
    cl_addressing_mode addressing_mode;
    cl_filter_mode filter_mode;
    bool normalized_coords;
} image_sampler_data;

cl_int round_to_even(float v);

#define NORMALIZE(v, max) (v < 0 ? 0 : (v > 1.f ? max : round_to_even(v * max)))
#define NORMALIZE_UNROUNDED(v, max) (v < 0 ? 0 : (v > 1.f ? max : v * max))
#define NORMALIZE_SIGNED(v, min, max)                                          \
    (v < -1.0f ? min : (v > 1.f ? max : round_to_even(v * max)))
#define NORMALIZE_SIGNED_UNROUNDED(v, min, max)                                \
    (v < -1.0f ? min : (v > 1.f ? max : v * max))
#define CONVERT_INT(v, min, max, max_val)                                      \
    (v < min ? min : (v > max ? max_val : round_to_even(v)))
#define CONVERT_UINT(v, max, max_val)                                          \
    (v < 0 ? 0 : (v > max ? max_val : round_to_even(v)))

extern void print_read_header(const cl_image_format *format,
                              image_sampler_data *sampler, bool err = false,
                              int t = 0);
extern void print_write_header(const cl_image_format *format, bool err);
extern void print_header(const cl_image_format *format, bool err);
extern bool find_format(cl_image_format *formatList, unsigned int numFormats,
                        cl_image_format *formatToFind);
extern bool is_image_format_required(cl_image_format format, cl_mem_flags flags,
                                     cl_mem_object_type image_type,
                                     cl_device_id device);
extern void
build_required_image_formats(cl_mem_flags flags, cl_mem_object_type image_type,
                             cl_device_id device,
                             std::vector<cl_image_format> &formatsToSupport);

extern uint32_t get_format_type_size(const cl_image_format *format);
extern uint32_t get_channel_data_type_size(cl_channel_type channelType);
extern uint32_t get_format_channel_count(const cl_image_format *format);
extern uint32_t get_channel_order_channel_count(cl_channel_order order);
cl_channel_type get_channel_type_from_name(const char *name);
cl_channel_order get_channel_order_from_name(const char *name);
extern int is_format_signed(const cl_image_format *format);
extern uint32_t get_pixel_size(const cl_image_format *format);

/* Helper to get any ol image format as long as it is 8-bits-per-channel */
extern int get_8_bit_image_format(cl_context context,
                                  cl_mem_object_type objType,
                                  cl_mem_flags flags, size_t channelCount,
                                  cl_image_format *outFormat);

/* Helper to get any ol image format as long as it is 32-bits-per-channel */
extern int get_32_bit_image_format(cl_context context,
                                   cl_mem_object_type objType,
                                   cl_mem_flags flags, size_t channelCount,
                                   cl_image_format *outFormat);

int random_in_range(int minV, int maxV, MTdata d);
int random_log_in_range(int minV, int maxV, MTdata d);

typedef struct
{
    size_t width;
    size_t height;
    size_t depth;
    size_t rowPitch;
    size_t slicePitch;
    size_t arraySize;
    const cl_image_format *format;
    cl_mem buffer;
    cl_mem_object_type type;
    cl_uint num_mip_levels;
} image_descriptor;

typedef struct
{
    float p[4];
} FloatPixel;

void print_first_pixel_difference_error(size_t where, const char *sourcePixel,
                                        const char *destPixel,
                                        image_descriptor *imageInfo, size_t y,
                                        size_t thirdDim);

size_t compare_scanlines(const image_descriptor *imageInfo, const char *aPtr,
                         const char *bPtr);

void get_max_sizes(size_t *numberOfSizes, const int maxNumberOfSizes,
                   size_t sizes[][3], size_t maxWidth, size_t maxHeight,
                   size_t maxDepth, size_t maxArraySize,
                   const cl_ulong maxIndividualAllocSize,
                   const cl_ulong maxTotalAllocSize,
                   cl_mem_object_type image_type, const cl_image_format *format,
                   int usingMaxPixelSize = 0);
extern size_t get_format_max_int(const cl_image_format *format);

extern cl_ulong get_image_size(image_descriptor const *imageInfo);
extern cl_ulong get_image_size_mb(image_descriptor const *imageInfo);

extern char *generate_random_image_data(image_descriptor *imageInfo,
                                        BufferOwningPtr<char> &Owner, MTdata d);

extern int debug_find_vector_in_image(void *imagePtr,
                                      image_descriptor *imageInfo,
                                      void *vectorToFind, size_t vectorSize,
                                      int *outX, int *outY, int *outZ,
                                      size_t lod = 0);

extern int debug_find_pixel_in_image(void *imagePtr,
                                     image_descriptor *imageInfo,
                                     unsigned int *valuesToFind, int *outX,
                                     int *outY, int *outZ, int lod = 0);
extern int debug_find_pixel_in_image(void *imagePtr,
                                     image_descriptor *imageInfo,
                                     int *valuesToFind, int *outX, int *outY,
                                     int *outZ, int lod = 0);
extern int debug_find_pixel_in_image(void *imagePtr,
                                     image_descriptor *imageInfo,
                                     float *valuesToFind, int *outX, int *outY,
                                     int *outZ, int lod = 0);

extern void copy_image_data(image_descriptor *srcImageInfo,
                            image_descriptor *dstImageInfo, void *imageValues,
                            void *destImageValues, const size_t sourcePos[],
                            const size_t destPos[], const size_t regionSize[]);

int has_alpha(const cl_image_format *format);

extern bool is_sRGBA_order(cl_channel_order image_channel_order);

inline float calculate_array_index(float coord, float extent);

cl_uint compute_max_mip_levels(size_t width, size_t height, size_t depth);
cl_ulong compute_mipmapped_image_size(image_descriptor imageInfo);
size_t compute_mip_level_offset(image_descriptor *imageInfo, size_t lod);

template <class T>
void read_image_pixel(void *imageData, image_descriptor *imageInfo, int x,
                      int y, int z, T *outData, int lod)
{
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth,
           slice_pitch_lod = 0 /*imageInfo->slicePitch*/,
           row_pitch_lod = 0 /*imageInfo->rowPitch*/;
    width_lod = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;

    if (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY
        && imageInfo->type != CL_MEM_OBJECT_IMAGE1D)
        height_lod =
            (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;

    if (imageInfo->type == CL_MEM_OBJECT_IMAGE3D)
        depth_lod = (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
    row_pitch_lod = (imageInfo->num_mip_levels > 0)
        ? (width_lod * get_pixel_size(imageInfo->format))
        : imageInfo->rowPitch;
    slice_pitch_lod = (imageInfo->num_mip_levels > 0)
        ? (row_pitch_lod * height_lod)
        : imageInfo->slicePitch;

    // correct depth_lod and height_lod for array image types in order to avoid
    // return
    if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY && height_lod == 1
        && depth_lod == 1)
    {
        depth_lod = 0;
        height_lod = 0;
    }

    if (imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY && depth_lod == 1)
    {
        depth_lod = 0;
    }

    if (x < 0 || x >= (int)width_lod
        || (height_lod != 0 && (y < 0 || y >= (int)height_lod))
        || (depth_lod != 0 && (z < 0 || z >= (int)depth_lod))
        || (imageInfo->arraySize != 0
            && (z < 0 || z >= (int)imageInfo->arraySize)))
    {
        // Border color
        if (imageInfo->format->image_channel_order == CL_DEPTH)
        {
            outData[0] = 1;
        }
        else
        {
            outData[0] = outData[1] = outData[2] = outData[3] = 0;
            if (!has_alpha(imageInfo->format)) outData[3] = 1;
        }
        return;
    }

    const cl_image_format *format = imageInfo->format;

    unsigned int i;
    T tempData[4];

    // Advance to the right spot
    char *ptr = (char *)imageData;
    size_t pixelSize = get_pixel_size(format);

    ptr += z * slice_pitch_lod + y * row_pitch_lod + x * pixelSize;

    // OpenCL only supports reading floats from certain formats
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNORM_INT8: {
            cl_uchar *dPtr = (cl_uchar *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_SIGNED_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT8: {
            cl_uchar *dPtr = (cl_uchar *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_SNORM_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNORM_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_SIGNED_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_HALF_FLOAT: {
            cl_half *dPtr = (cl_half *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)cl_half_to_float(dPtr[i]);
            break;
        }

        case CL_SIGNED_INT32: {
            cl_int *dPtr = (cl_int *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT32: {
            cl_uint *dPtr = (cl_uint *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }

        case CL_UNORM_SHORT_565: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            tempData[0] = (T)(dPtr[0] >> 11);
            tempData[1] = (T)((dPtr[0] >> 5) & 63);
            tempData[2] = (T)(dPtr[0] & 31);
            break;
        }

#ifdef OBSOLETE_FORMAT
        case CL_UNORM_SHORT_565_REV: {
            unsigned short *dPtr = (unsigned short *)ptr;
            tempData[2] = (T)(dPtr[0] >> 11);
            tempData[1] = (T)((dPtr[0] >> 5) & 63);
            tempData[0] = (T)(dPtr[0] & 31);
            break;
        }

        case CL_UNORM_SHORT_555_REV: {
            unsigned short *dPtr = (unsigned short *)ptr;
            tempData[2] = (T)((dPtr[0] >> 10) & 31);
            tempData[1] = (T)((dPtr[0] >> 5) & 31);
            tempData[0] = (T)(dPtr[0] & 31);
            break;
        }

        case CL_UNORM_INT_8888: {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[3] = (T)(dPtr[0] >> 24);
            tempData[2] = (T)((dPtr[0] >> 16) & 0xff);
            tempData[1] = (T)((dPtr[0] >> 8) & 0xff);
            tempData[0] = (T)(dPtr[0] & 0xff);
            break;
        }
        case CL_UNORM_INT_8888_REV: {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[0] = (T)(dPtr[0] >> 24);
            tempData[1] = (T)((dPtr[0] >> 16) & 0xff);
            tempData[2] = (T)((dPtr[0] >> 8) & 0xff);
            tempData[3] = (T)(dPtr[0] & 0xff);
            break;
        }

        case CL_UNORM_INT_101010_REV: {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[2] = (T)((dPtr[0] >> 20) & 0x3ff);
            tempData[1] = (T)((dPtr[0] >> 10) & 0x3ff);
            tempData[0] = (T)(dPtr[0] & 0x3ff);
            break;
        }
#endif
        case CL_UNORM_SHORT_555: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            tempData[0] = (T)((dPtr[0] >> 10) & 31);
            tempData[1] = (T)((dPtr[0] >> 5) & 31);
            tempData[2] = (T)(dPtr[0] & 31);
            break;
        }

        case CL_UNORM_INT_101010: {
            cl_uint *dPtr = (cl_uint *)ptr;
            tempData[0] = (T)((dPtr[0] >> 20) & 0x3ff);
            tempData[1] = (T)((dPtr[0] >> 10) & 0x3ff);
            tempData[2] = (T)(dPtr[0] & 0x3ff);
            break;
        }

        case CL_FLOAT: {
            cl_float *dPtr = (cl_float *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i];
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: {
            cl_float *dPtr = (cl_float *)ptr;
            for (i = 0; i < get_format_channel_count(format); i++)
                tempData[i] = (T)dPtr[i] + 0x4000;
            break;
        }
#endif
    }


    outData[0] = outData[1] = outData[2] = 0;
    outData[3] = 1;

    if (format->image_channel_order == CL_A)
    {
        outData[3] = tempData[0];
    }
    else if (format->image_channel_order == CL_R)
    {
        outData[0] = tempData[0];
    }
    else if (format->image_channel_order == CL_Rx)
    {
        outData[0] = tempData[0];
    }
    else if (format->image_channel_order == CL_RA)
    {
        outData[0] = tempData[0];
        outData[3] = tempData[1];
    }
    else if (format->image_channel_order == CL_RG)
    {
        outData[0] = tempData[0];
        outData[1] = tempData[1];
    }
    else if (format->image_channel_order == CL_RGx)
    {
        outData[0] = tempData[0];
        outData[1] = tempData[1];
    }
    else if ((format->image_channel_order == CL_RGB)
             || (format->image_channel_order == CL_sRGB))
    {
        outData[0] = tempData[0];
        outData[1] = tempData[1];
        outData[2] = tempData[2];
    }
    else if ((format->image_channel_order == CL_RGBx)
             || (format->image_channel_order == CL_sRGBx))
    {
        outData[0] = tempData[0];
        outData[1] = tempData[1];
        outData[2] = tempData[2];
        outData[3] = 0;
    }
    else if ((format->image_channel_order == CL_RGBA)
             || (format->image_channel_order == CL_sRGBA))
    {
        outData[0] = tempData[0];
        outData[1] = tempData[1];
        outData[2] = tempData[2];
        outData[3] = tempData[3];
    }
    else if (format->image_channel_order == CL_ARGB)
    {
        outData[0] = tempData[1];
        outData[1] = tempData[2];
        outData[2] = tempData[3];
        outData[3] = tempData[0];
    }
    else if ((format->image_channel_order == CL_BGRA)
             || (format->image_channel_order == CL_sBGRA))
    {
        outData[0] = tempData[2];
        outData[1] = tempData[1];
        outData[2] = tempData[0];
        outData[3] = tempData[3];
    }
    else if (format->image_channel_order == CL_INTENSITY)
    {
        outData[0] = tempData[0];
        outData[1] = tempData[0];
        outData[2] = tempData[0];
        outData[3] = tempData[0];
    }
    else if (format->image_channel_order == CL_LUMINANCE)
    {
        outData[0] = tempData[0];
        outData[1] = tempData[0];
        outData[2] = tempData[0];
    }
    else if (format->image_channel_order == CL_DEPTH)
    {
        outData[0] = tempData[0];
    }
#ifdef CL_1RGB_APPLE
    else if (format->image_channel_order == CL_1RGB_APPLE)
    {
        outData[0] = tempData[1];
        outData[1] = tempData[2];
        outData[2] = tempData[3];
        outData[3] = 0xff;
    }
#endif
#ifdef CL_BGR1_APPLE
    else if (format->image_channel_order == CL_BGR1_APPLE)
    {
        outData[0] = tempData[2];
        outData[1] = tempData[1];
        outData[2] = tempData[0];
        outData[3] = 0xff;
    }
#endif
    else
    {
        log_error("Invalid format:");
        print_header(format, true);
    }
}

template <class T>
void read_image_pixel(void *imageData, image_descriptor *imageInfo, int x,
                      int y, int z, T *outData)
{
    read_image_pixel<T>(imageData, imageInfo, x, y, z, outData, 0);
}

// Stupid template rules
bool get_integer_coords(float x, float y, float z, size_t width, size_t height,
                        size_t depth, image_sampler_data *imageSampler,
                        image_descriptor *imageInfo, int &outX, int &outY,
                        int &outZ);
bool get_integer_coords_offset(float x, float y, float z, float xAddressOffset,
                               float yAddressOffset, float zAddressOffset,
                               size_t width, size_t height, size_t depth,
                               image_sampler_data *imageSampler,
                               image_descriptor *imageInfo, int &outX,
                               int &outY, int &outZ);


template <class T>
void sample_image_pixel_offset(void *imageData, image_descriptor *imageInfo,
                               float x, float y, float z, float xAddressOffset,
                               float yAddressOffset, float zAddressOffset,
                               image_sampler_data *imageSampler, T *outData,
                               int lod)
{
    int iX = 0, iY = 0, iZ = 0;

    float max_w = imageInfo->width;
    float max_h;
    float max_d;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            max_h = imageInfo->arraySize;
            max_d = 0;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            max_h = imageInfo->height;
            max_d = imageInfo->arraySize;
            break;
        default:
            max_h = imageInfo->height;
            max_d = imageInfo->depth;
            break;
    }

    if (/*gTestMipmaps*/ imageInfo->num_mip_levels > 1)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                max_d = (float)((imageInfo->depth >> lod)
                                    ? (imageInfo->depth >> lod)
                                    : 1);
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                max_h = (float)((imageInfo->height >> lod)
                                    ? (imageInfo->height >> lod)
                                    : 1);
                break;
            default:;
        }
        max_w =
            (float)((imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1);
    }
    get_integer_coords_offset(x, y, z, xAddressOffset, yAddressOffset,
                              zAddressOffset, max_w, max_h, max_d, imageSampler,
                              imageInfo, iX, iY, iZ);

    read_image_pixel<T>(imageData, imageInfo, iX, iY, iZ, outData, lod);
}

template <class T>
void sample_image_pixel_offset(void *imageData, image_descriptor *imageInfo,
                               float x, float y, float z, float xAddressOffset,
                               float yAddressOffset, float zAddressOffset,
                               image_sampler_data *imageSampler, T *outData)
{
    sample_image_pixel_offset<T>(imageData, imageInfo, x, y, z, xAddressOffset,
                                 yAddressOffset, zAddressOffset, imageSampler,
                                 outData, 0);
}

template <class T>
void sample_image_pixel(void *imageData, image_descriptor *imageInfo, float x,
                        float y, float z, image_sampler_data *imageSampler,
                        T *outData)
{
    return sample_image_pixel_offset<T>(imageData, imageInfo, x, y, z, 0.0f,
                                        0.0f, 0.0f, imageSampler, outData);
}

FloatPixel
sample_image_pixel_float(void *imageData, image_descriptor *imageInfo, float x,
                         float y, float z, image_sampler_data *imageSampler,
                         float *outData, int verbose, int *containsDenorms);

FloatPixel sample_image_pixel_float(void *imageData,
                                    image_descriptor *imageInfo, float x,
                                    float y, float z,
                                    image_sampler_data *imageSampler,
                                    float *outData, int verbose,
                                    int *containsDenorms, int lod);

FloatPixel sample_image_pixel_float_offset(
    void *imageData, image_descriptor *imageInfo, float x, float y, float z,
    float xAddressOffset, float yAddressOffset, float zAddressOffset,
    image_sampler_data *imageSampler, float *outData, int verbose,
    int *containsDenorms);
FloatPixel sample_image_pixel_float_offset(
    void *imageData, image_descriptor *imageInfo, float x, float y, float z,
    float xAddressOffset, float yAddressOffset, float zAddressOffset,
    image_sampler_data *imageSampler, float *outData, int verbose,
    int *containsDenorms, int lod);


extern void pack_image_pixel(unsigned int *srcVector,
                             const cl_image_format *imageFormat, void *outData);
extern void pack_image_pixel(int *srcVector, const cl_image_format *imageFormat,
                             void *outData);
extern void pack_image_pixel(float *srcVector,
                             const cl_image_format *imageFormat, void *outData);
extern void pack_image_pixel_error(const float *srcVector,
                                   const cl_image_format *imageFormat,
                                   const void *results, float *errors);

extern char *create_random_image_data(ExplicitType dataType,
                                      image_descriptor *imageInfo,
                                      BufferOwningPtr<char> &P, MTdata d,
                                      bool image2DFromBuffer = false);

// deprecated
// extern bool clamp_image_coord( image_sampler_data *imageSampler, float value,
// size_t max, int &outValue );

extern void get_sampler_kernel_code(image_sampler_data *imageSampler,
                                    char *outLine);
extern float get_max_absolute_error(const cl_image_format *format,
                                    image_sampler_data *sampler);
extern float get_max_relative_error(const cl_image_format *format,
                                    image_sampler_data *sampler, int is3D,
                                    int isLinearFilter);


#define errMax(_x, _y) ((_x) != (_x) ? (_x) : (_x) > (_y) ? (_x) : (_y))

static inline cl_uint abs_diff_uint(cl_uint x, cl_uint y)
{
    return y > x ? y - x : x - y;
}

static inline cl_uint abs_diff_int(cl_int x, cl_int y)
{
    return (cl_uint)(y > x ? y - x : x - y);
}

static inline cl_float relative_error(float test, float expected)
{
    // 0-0/0 is 0 in this case, not NaN
    if (test == 0.0f && expected == 0.0f) return 0.0f;

    return (test - expected) / expected;
}

extern float random_float(float low, float high);

class CoordWalker {
public:
    CoordWalker(void *coords, bool useFloats, size_t vecSize);
    ~CoordWalker();

    cl_float Get(size_t idx, size_t el);

protected:
    cl_float *mFloatCoords;
    cl_int *mIntCoords;
    size_t mVecSize;
};

extern cl_half convert_float_to_half(float f);
extern int DetectFloatToHalfRoundingMode(
    cl_command_queue); // Returns CL_SUCCESS on success

// sign bit: don't care, exponent: maximum value, significand: non-zero
static int inline is_half_nan(cl_half half) { return (half & 0x7fff) > 0x7c00; }

// sign bit: don't care, exponent: zero, significand: non-zero
static int inline is_half_denorm(cl_half half) { return IsHalfSubnormal(half); }

// sign bit: don't care, exponent: zero, significand: zero
static int inline is_half_zero(cl_half half) { return (half & 0x7fff) == 0; }

extern double sRGBmap(float fc);

extern const char *convert_image_type_to_string(cl_mem_object_type imageType);


#endif // _imageHelpers_h

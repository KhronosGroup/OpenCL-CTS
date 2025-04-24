//
// Copyright (c) 2020 The Khronos Group Inc.
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

cl_channel_type floatFormats[] = {
    CL_UNORM_SHORT_565,
    CL_UNORM_SHORT_555,
    CL_UNORM_INT_101010,
    CL_UNORM_INT_101010_2,
    CL_UNORM_INT_2_101010_EXT,
#ifdef CL_SFIXED14_APPLE
    CL_SFIXED14_APPLE,
#endif
    CL_UNORM_INT8,
    CL_SNORM_INT8,
    CL_UNORM_INT16,
    CL_SNORM_INT16,
    CL_FLOAT,
    CL_HALF_FLOAT,
    (cl_channel_type)-1,
};

cl_channel_type intFormats[] = {
    CL_SIGNED_INT8,
    CL_SIGNED_INT16,
    CL_SIGNED_INT32,
    (cl_channel_type)-1,
};

cl_channel_type uintFormats[] = {
    CL_UNSIGNED_INT8,
    CL_UNSIGNED_INT16,
    CL_UNSIGNED_INT32,
    (cl_channel_type)-1,
};

std::array<ImageTestTypes, 3> imageTestTypes = { {
    { kTestInt, kInt, intFormats, "int" },
    { kTestUInt, kUInt, uintFormats, "uint" },
    { kTestFloat, kFloat, floatFormats, "float" },
} };

int filter_formats(const std::vector<cl_image_format> &formatList,
                   std::vector<bool> &filterFlags,
                   cl_channel_type *channelDataTypesToFilter,
                   bool testMipmaps /*=false*/)
{
    int numSupported = 0;
    for (unsigned int j = 0; j < formatList.size(); j++)
    {
        // If this format has been previously filtered, remove the filter
        if (filterFlags[j]) filterFlags[j] = false;

        // skip mipmap tests for CL_DEPTH formats (re# Khronos Bug 13762)
        if (testMipmaps && (formatList[j].image_channel_order == CL_DEPTH))
        {
            log_info("Skip mipmap tests for CL_DEPTH format\n");
            filterFlags[j] = true;
            continue;
        }

        // Have we already discarded the channel type via the command line?
        if (gChannelTypeToUse != (cl_channel_type)-1
            && gChannelTypeToUse != formatList[j].image_channel_data_type)
        {
            filterFlags[j] = true;
            continue;
        }

        // Have we already discarded the channel order via the command line?
        if (gChannelOrderToUse != (cl_channel_order)-1
            && gChannelOrderToUse != formatList[j].image_channel_order)
        {
            filterFlags[j] = true;
            continue;
        }

        // Is given format standard channel order and type given by spec. We
        // don't want to test it if this is vendor extension
        if (!IsChannelOrderSupported(formatList[j].image_channel_order)
            || !IsChannelTypeSupported(formatList[j].image_channel_data_type))
        {
            filterFlags[j] = true;
            continue;
        }

        if (!channelDataTypesToFilter)
        {
            numSupported++;
            continue;
        }

        // Is the format supported?
        int i;
        for (i = 0; channelDataTypesToFilter[i] != (cl_channel_type)-1; i++)
        {
            if (formatList[j].image_channel_data_type
                == channelDataTypesToFilter[i])
            {
                numSupported++;
                break;
            }
        }
        if (channelDataTypesToFilter[i] == (cl_channel_type)-1)
        {
            // Format is NOT supported, so mark it as such
            filterFlags[j] = true;
        }
    }
    return numSupported;
}

int get_format_list(cl_context context, cl_mem_object_type imageType,
                    std::vector<cl_image_format> &outFormatList,
                    cl_mem_flags flags)
{
    flags &= CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY
        | CL_MEM_KERNEL_READ_AND_WRITE | CL_MEM_IMMUTABLE_EXT;

    cl_uint formatCount;
    int error = clGetSupportedImageFormats(context, flags, imageType, 0, NULL,
                                           &formatCount);
    test_error(error, "Unable to get count of supported image formats");

    outFormatList.resize(formatCount);

    error = clGetSupportedImageFormats(context, flags, imageType, formatCount,
                                       outFormatList.data(), NULL);
    test_error(error, "Unable to get list of supported image formats");
    return 0;
}

size_t random_in_ranges(size_t minimum, size_t rangeA, size_t rangeB, MTdata d)
{
    if (rangeB < rangeA) rangeA = rangeB;
    if (rangeA < minimum) return rangeA;
    return (size_t)random_in_range((int)minimum, (int)rangeA - 1, d);
}

using free_function_t = void (*)(void *);
struct pitch_buffer_data
{
    void *buf;
    free_function_t free_fn;

    static void CL_CALLBACK free_buffer(cl_mem, void *data)
    {
        pitch_buffer_data *d = static_cast<pitch_buffer_data *>(data);
        d->free_fn(d->buf);
        delete d;
    }
};

static void CL_CALLBACK release_cl_buffer(cl_mem image, void *buf)
{
    clReleaseMemObject((cl_mem)buf);
}

clMemWrapper create_image(cl_context context, cl_command_queue queue,
                          BufferOwningPtr<char> &data,
                          image_descriptor *imageInfo, bool enable_pitch,
                          bool create_mipmaps, int *error)
{
    cl_mem img;
    cl_image_desc imageDesc;
    void *host_ptr = nullptr;
    bool is_host_ptr_aligned = false;

    memset(&imageDesc, 0x0, sizeof(cl_image_desc));
    imageDesc.image_type = imageInfo->type;
    imageDesc.image_width = imageInfo->width;
    imageDesc.image_height = imageInfo->height;
    imageDesc.image_depth = imageInfo->depth;
    imageDesc.image_array_size = imageInfo->arraySize;
    imageDesc.image_row_pitch = enable_pitch ? imageInfo->rowPitch : 0;
    imageDesc.image_slice_pitch = enable_pitch ? imageInfo->slicePitch : 0;
    imageDesc.num_mip_levels = create_mipmaps ? imageInfo->num_mip_levels : 0;

    Version version;
    cl_device_id device;
    {
        cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                                           sizeof(device), &device, nullptr);
        if (err != CL_SUCCESS)
        {
            log_error("Error: Could not get CL_QUEUE_DEVICE from queue");
            return nullptr;
        }
        version = get_device_cl_version(device);
    }

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            if (gDebugTrace)
                log_info(" - Creating 1D image %d ...\n",
                         (int)imageInfo->width);
            if (enable_pitch) host_ptr = malloc(imageInfo->rowPitch);
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            if (gDebugTrace)
                log_info(" - Creating 2D image %d by %d ...\n",
                         (int)imageInfo->width, (int)imageInfo->height);
            if (enable_pitch)
                host_ptr = malloc(imageInfo->height * imageInfo->rowPitch);
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            if (gDebugTrace)
                log_info(" - Creating 3D image %d by %d by %d...\n",
                         (int)imageInfo->width, (int)imageInfo->height,
                         (int)imageInfo->depth);
            if (enable_pitch)
                host_ptr = malloc(imageInfo->depth * imageInfo->slicePitch);
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            if (gDebugTrace)
                log_info(" - Creating 1D image array %d by %d...\n",
                         (int)imageInfo->width, (int)imageInfo->arraySize);
            if (enable_pitch)
                host_ptr = malloc(imageInfo->arraySize * imageInfo->slicePitch);
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            if (gDebugTrace)
                log_info(" - Creating 2D image array %d by %d by %d...\n",
                         (int)imageInfo->width, (int)imageInfo->height,
                         (int)imageInfo->arraySize);
            if (enable_pitch)
                host_ptr = malloc(imageInfo->arraySize * imageInfo->slicePitch);
            break;
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            if (gDebugTrace)
                log_info(" - Creating 1D buffer image %d ...\n",
                         (int)imageInfo->width);
            {
                cl_int err;
                cl_mem_flags buffer_flags = CL_MEM_READ_WRITE;
                if (enable_pitch)
                {
                    if (version.major() == 1)
                    {
                        host_ptr = malloc(imageInfo->rowPitch);
                    }
                    else
                    {
                        cl_uint base_address_alignment = 0;
                        err = clGetDeviceInfo(
                            device, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
                            sizeof(base_address_alignment),
                            &base_address_alignment, nullptr);
                        if (err != CL_SUCCESS)
                        {
                            log_error("ERROR: Could not get "
                                      "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT "
                                      "from device");
                            return nullptr;
                        }
                        host_ptr = align_malloc(imageInfo->rowPitch,
                                                base_address_alignment);
                        is_host_ptr_aligned = true;
                    }
                    buffer_flags |= CL_MEM_USE_HOST_PTR;
                }

                cl_mem buffer = clCreateBuffer(
                    context, buffer_flags, imageInfo->rowPitch, host_ptr, &err);
                if (err != CL_SUCCESS)
                {
                    log_error("ERROR: Could not create buffer for 1D buffer "
                              "image. %zu bytes\n",
                              imageInfo->width);
                    if (host_ptr)
                    {
                        if (is_host_ptr_aligned)
                        {
                            align_free(host_ptr);
                        }
                        else
                        {
                            free(host_ptr);
                        }
                    }
                    return nullptr;
                }
                imageDesc.buffer = buffer;
            }
            break;
    }

    if (gDebugTrace && create_mipmaps)
        log_info(" - with %llu mip levels\n",
                 (unsigned long long)imageInfo->num_mip_levels);

    if (enable_pitch)
    {
        if (nullptr == host_ptr)
        {
            log_error("ERROR: Unable to create backing store for pitched 3D "
                      "image. %zu bytes\n",
                      imageInfo->depth * imageInfo->slicePitch);
            return nullptr;
        }
    }

    if (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_BUFFER)
    {
        img = clCreateImage(context, imageInfo->mem_flags, imageInfo->format,
                            &imageDesc, host_ptr, error);
    }
    else
    {
        img = clCreateImage(context, imageInfo->mem_flags, imageInfo->format,
                            &imageDesc, nullptr, error);
    }

    if (enable_pitch)
    {
        free_function_t free_fn = is_host_ptr_aligned ? align_free : free;
        if (*error == CL_SUCCESS)
        {
            pitch_buffer_data *buf_data = new pitch_buffer_data;
            buf_data->buf = host_ptr;
            buf_data->free_fn = free_fn;

            int callbackError = clSetMemObjectDestructorCallback(
                img, pitch_buffer_data::free_buffer, buf_data);
            if (CL_SUCCESS != callbackError)
            {
                pitch_buffer_data::free_buffer(img, buf_data);
                log_error("ERROR: Unable to attach destructor callback to "
                          "pitched 3D image. Err: %d\n",
                          callbackError);
                clReleaseMemObject(img);
                return nullptr;
            }
        }
        else
        {
            free_fn(host_ptr);
        }
    }

    if (imageDesc.buffer != nullptr)
    {
        int callbackError = clSetMemObjectDestructorCallback(
            img, release_cl_buffer, imageDesc.buffer);
        if (callbackError != CL_SUCCESS)
        {
            log_error("Error: Unable to attach destructor callback to 1d "
                      "buffer image. Err: %d\n",
                      callbackError);
            clReleaseMemObject(imageDesc.buffer);
            clReleaseMemObject(img);
            return nullptr;
        }
    }

    if (*error != CL_SUCCESS)
    {
        long long unsigned imageSize = get_image_size_mb(imageInfo);
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                log_error("ERROR: Unable to create 1D image of size %d (%llu "
                          "MB):(%s)",
                          (int)imageInfo->width, imageSize,
                          IGetErrorString(*error));
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_error("ERROR: Unable to create 2D image of size %d x %d "
                          "(%llu MB):(%s)",
                          (int)imageInfo->width, (int)imageInfo->height,
                          imageSize, IGetErrorString(*error));
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                log_error("ERROR: Unable to create 3D image of size %d x %d x "
                          "%d (%llu MB):(%s)",
                          (int)imageInfo->width, (int)imageInfo->height,
                          (int)imageInfo->depth, imageSize,
                          IGetErrorString(*error));
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_error("ERROR: Unable to create 1D image array of size %d x "
                          "%d (%llu MB):(%s)",
                          (int)imageInfo->width, (int)imageInfo->arraySize,
                          imageSize, IGetErrorString(*error));
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_error("ERROR: Unable to create 2D image array of size %d x "
                          "%d x %d (%llu MB):(%s)",
                          (int)imageInfo->width, (int)imageInfo->height,
                          (int)imageInfo->arraySize, imageSize,
                          IGetErrorString(*error));
                break;
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                log_error(
                    "ERROR: Unable to create 1D buffer image of size %d (%llu "
                    "MB):(%s)",
                    (int)imageInfo->width, imageSize, IGetErrorString(*error));
                break;
        }
        log_error("ERROR: and %llu mip levels\n",
                  (unsigned long long)imageInfo->num_mip_levels);
        return nullptr;
    }

    // Copy the specified data to the image via a Map operation.
    size_t mappedRow, mappedSlice;
    size_t width = imageInfo->width;
    size_t height = 1;
    size_t depth = 1;
    size_t row_pitch_lod, slice_pitch_lod;
    row_pitch_lod = imageInfo->rowPitch;
    slice_pitch_lod = imageInfo->slicePitch;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            height = imageInfo->arraySize;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        case CL_MEM_OBJECT_IMAGE1D: height = depth = 1; break;
        case CL_MEM_OBJECT_IMAGE2D:
            height = imageInfo->height;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            height = imageInfo->height;
            depth = imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            height = imageInfo->height;
            depth = imageInfo->depth;
            break;
        default:
            log_error("ERROR Invalid imageInfo->type = %d\n", imageInfo->type);
            height = 0;
            depth = 0;
            return nullptr;
            break;
    }

    size_t origin[4] = { 0, 0, 0, 0 };
    size_t region[3] = { imageInfo->width, height, depth };

    for (size_t lod = 0; (create_mipmaps && (lod < imageInfo->num_mip_levels))
         || (!create_mipmaps && (lod < 1));
         lod++)
    {
        // Map the appropriate miplevel to copy the specified data.
        if (create_mipmaps)
        {
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                    origin[0] = origin[1] = origin[2] = 0;
                    origin[3] = lod;
                    break;
                case CL_MEM_OBJECT_IMAGE2D:
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                    origin[0] = origin[1] = origin[3] = 0;
                    origin[2] = lod;
                    break;
                case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                case CL_MEM_OBJECT_IMAGE1D:
                    origin[0] = origin[2] = origin[3] = 0;
                    origin[1] = lod;
                    break;
            }

            // Adjust image dimensions as per miplevel
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                    depth = (imageInfo->depth >> lod)
                        ? (imageInfo->depth >> lod)
                        : 1;
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D:
                    height = (imageInfo->height >> lod)
                        ? (imageInfo->height >> lod)
                        : 1;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                case CL_MEM_OBJECT_IMAGE1D:
                    width = (imageInfo->width >> lod)
                        ? (imageInfo->width >> lod)
                        : 1;
            }
            row_pitch_lod = width * get_pixel_size(imageInfo->format);
            slice_pitch_lod = row_pitch_lod * height;
            region[0] = width;
            region[1] = height;
            region[2] = depth;
        }

        char *mapped = static_cast<char *>(clEnqueueMapImage(
            queue, img, CL_TRUE, CL_MAP_WRITE, origin, region, &mappedRow,
            &mappedSlice, 0, nullptr, nullptr, error));
        if (*error != CL_SUCCESS || !mapped)
        {
            log_error("ERROR: Unable to map image for writing: %s\n",
                      IGetErrorString(*error));
            return nullptr;
        }
        size_t mappedSlicePad = mappedSlice - (mappedRow * height);

        // For 1Darray, the height variable actually contains the arraysize,
        // so it can't be used for calculating the slice padding.
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            mappedSlicePad = mappedSlice - (mappedRow * 1);

        // Copy the image.
        size_t scanlineSize = row_pitch_lod;
        size_t sliceSize = slice_pitch_lod - scanlineSize * height;
        size_t imageSize = scanlineSize * height * depth;
        size_t data_lod_offset = 0;
        if (create_mipmaps)
        {
            data_lod_offset = compute_mip_level_offset(imageInfo, lod);
        }

        char *src = static_cast<char *>(data) + data_lod_offset;
        char *dst = mapped;

        if ((mappedRow == scanlineSize)
            && (mappedSlicePad == 0
                || (imageInfo->depth == 0 && imageInfo->arraySize == 0)))
        {
            // Copy the whole image.
            memcpy(dst, src, imageSize);
        }
        else
        {
            // Else copy one scan line at a time.
            size_t dstPitch2D = 0;
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D: dstPitch2D = mappedRow; break;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE1D:
                case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                    dstPitch2D = mappedSlice;
                    break;
            }
            for (size_t z = 0; z < depth; z++)
            {
                for (size_t y = 0; y < height; y++)
                {
                    memcpy(dst, src, scanlineSize);
                    dst += dstPitch2D;
                    src += scanlineSize;
                }

                // mappedSlicePad is incorrect for 2D images here, but we will
                // exit the z loop before this is a problem.
                dst += mappedSlicePad;
                src += sliceSize;
            }
        }

        // Unmap the image.
        *error =
            clEnqueueUnmapMemObject(queue, img, mapped, 0, nullptr, nullptr);
        if (*error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to unmap image after writing: %s\n",
                      IGetErrorString(*error));
            return nullptr;
        }
    }
    return img;
}

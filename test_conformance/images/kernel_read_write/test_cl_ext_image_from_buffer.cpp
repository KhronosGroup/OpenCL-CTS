//
// Copyright (c) 2022 The Khronos Group Inc.
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

static int get_image_requirement_alignment(
    cl_device_id device, cl_context context, cl_mem_flags flags,
    const cl_image_format* image_format, const cl_image_desc* image_desc,
    size_t* row_pitch_alignment, size_t* slice_pitch_alignment,
    size_t* base_address_alignment)
{
    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    cl_int err = CL_SUCCESS;
    if (nullptr != row_pitch_alignment)
    {
        err = clGetImageRequirementsInfoEXT(
            context, nullptr, flags, image_format, image_desc,
            CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
            sizeof(*row_pitch_alignment), row_pitch_alignment, nullptr);
        test_error(err, "Error getting alignment");
    }

    if (nullptr != slice_pitch_alignment && CL_SUCCESS == err)
    {
        err = clGetImageRequirementsInfoEXT(
            context, nullptr, flags, image_format, image_desc,
            CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT,
            sizeof(*slice_pitch_alignment), slice_pitch_alignment, nullptr);
        test_error(err, "Error getting alignment");
    }

    if (nullptr != base_address_alignment && CL_SUCCESS == err)
    {
        err = clGetImageRequirementsInfoEXT(
            context, nullptr, flags, image_format, image_desc,
            CL_IMAGE_REQUIREMENTS_BASE_ADDRESS_ALIGNMENT_EXT,
            sizeof(*base_address_alignment), base_address_alignment, nullptr);
        test_error(err, "Error getting alignment");
    }

    return TEST_PASS;
}

/**
 * Consistency with alignment requirements as returned by
 * cl_khr_image2d_from_buffer Check that the returned values for
 * CL_DEVICE_IMAGE_PITCH_ALIGNMENT and CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
 * are correct.
 */
int image2d_from_buffer_positive(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_khr_image2d_from_buffer"))
    {
        printf("Extension cl_khr_image2d_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            cl_uint row_pitch_alignment_2d = 0;
            cl_int err =
                clGetDeviceInfo(device, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                                sizeof(row_pitch_alignment_2d),
                                &row_pitch_alignment_2d, nullptr);
            test_error(err, "Error clGetDeviceInfo");

            cl_uint base_address_alignment_2d = 0;
            err =
                clGetDeviceInfo(device, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
                                sizeof(base_address_alignment_2d),
                                &base_address_alignment_2d, nullptr);
            test_error(err, "Error clGetDeviceInfo");

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                size_t row_pitch_alignment = 0;
                size_t base_address_alignment = 0;

                int get_error = get_image_requirement_alignment(
                    device, context, 0, &format, &image_desc,
                    &row_pitch_alignment, nullptr, &base_address_alignment);
                if (TEST_PASS != get_error)
                {
                    return get_error;
                }

                const size_t element_size =
                    get_format_size(context, &format, imageType, flag);

                /*  Alignements in pixels vs bytes */
                if (base_address_alignment
                    > base_address_alignment_2d * element_size)
                {
                    test_fail("Unexpected base_address_alignment");
                }

                if (row_pitch_alignment > row_pitch_alignment_2d * element_size)
                {
                    test_fail("Unexpected row_pitch_alignment");
                }
            }
        }
    }

    return TEST_PASS;
}

/**
 * Test clGetMemObjectInfo
 * Check that CL_MEM_ASSOCIATED_MEMOBJECT correctly returns the buffer that was
 * used.
 */
int memInfo_image_from_buffer_positive(cl_device_id device, cl_context context,
                                       cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                size_t row_pitch_alignment = 0;
                size_t slice_pitch_alignment = 0;

                int get_error = get_image_requirement_alignment(
                    device, context, 0, &format, &image_desc,
                    &row_pitch_alignment, &slice_pitch_alignment, nullptr);
                if (TEST_PASS != get_error)
                {
                    return get_error;
                }

                const size_t element_size =
                    get_format_size(context, &format, imageType, flag);

                const size_t row_pitch = aligned_size(
                    TEST_IMAGE_SIZE * element_size, row_pitch_alignment);
                const size_t slice_pitch = aligned_size(
                    row_pitch * TEST_IMAGE_SIZE, slice_pitch_alignment);

                const size_t buffer_size = slice_pitch * TEST_IMAGE_SIZE;

                cl_int err = CL_SUCCESS;
                cl_mem buffer =
                    clCreateBuffer(context, flag, buffer_size, nullptr, &err);
                test_error(err, "Unable to create buffer");

                image_desc.buffer = buffer;

                cl_mem image_buffer = clCreateImage(context, flag, &format,
                                                    &image_desc, nullptr, &err);
                test_error(err, "Unable to create image");

                cl_mem returned_buffer;
                err = clGetMemObjectInfo(
                    image_buffer, CL_MEM_ASSOCIATED_MEMOBJECT,
                    sizeof(returned_buffer), &returned_buffer, nullptr);
                test_error(err, "Error clGetMemObjectInfo");

                if (returned_buffer != buffer)
                {
                    test_fail("Unexpected CL_MEM_ASSOCIATED_MEMOBJECT buffer");
                }

                err = clReleaseMemObject(buffer);
                test_error(err, "Unable to release buffer");

                err = clReleaseMemObject(image_buffer);
                test_error(err, "Unable to release image");
            }
        }
    }

    return TEST_PASS;
}

/**
 * Test clGetImageInfo
 * Check that the returned values for CL_IMAGE_ROW_PITCH and
 * CL_IMAGE_SLICE_PITCH are correct.
 */
int imageInfo_image_from_buffer_positive(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                size_t row_pitch_alignment = 0;
                size_t slice_pitch_alignment = 0;

                int get_error = get_image_requirement_alignment(
                    device, context, 0, &format, &image_desc,
                    &row_pitch_alignment, &slice_pitch_alignment, nullptr);
                if (TEST_PASS != get_error)
                {
                    return get_error;
                }

                const size_t element_size =
                    get_format_size(context, &format, imageType, flag);

                const size_t row_pitch = aligned_size(
                    TEST_IMAGE_SIZE * element_size, row_pitch_alignment);
                const size_t slice_pitch = aligned_size(
                    row_pitch * TEST_IMAGE_SIZE, slice_pitch_alignment);

                const size_t buffer_size = slice_pitch * TEST_IMAGE_SIZE;

                cl_int err = CL_SUCCESS;
                cl_mem buffer =
                    clCreateBuffer(context, flag, buffer_size, nullptr, &err);
                test_error(err, "Unable to create buffer");

                image_desc.buffer = buffer;

                if (imageType == CL_MEM_OBJECT_IMAGE2D
                    || imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                {
                    image_desc.image_row_pitch = row_pitch;
                }
                else if (imageType == CL_MEM_OBJECT_IMAGE3D
                         || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    image_desc.image_row_pitch = row_pitch;
                    image_desc.image_slice_pitch = slice_pitch;
                }

                cl_mem image_buffer = clCreateImage(context, flag, &format,
                                                    &image_desc, nullptr, &err);
                test_error(err, "Unable to create image");

                if (imageType == CL_MEM_OBJECT_IMAGE3D
                    || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY
                    || imageType == CL_MEM_OBJECT_IMAGE2D
                    || imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                {
                    size_t returned_row_pitch = 0;
                    err = clGetImageInfo(image_buffer, CL_IMAGE_ROW_PITCH,
                                         sizeof(returned_row_pitch),
                                         &returned_row_pitch, nullptr);
                    test_error(err, "Error clGetImageInfo");

                    if (returned_row_pitch != row_pitch)
                    {
                        test_fail(
                            "Unexpected row pitch "
                            "CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT");
                    }
                }

                if (imageType == CL_MEM_OBJECT_IMAGE3D
                    || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    size_t returned_slice_pitch = 0;
                    err = clGetImageInfo(image_buffer, CL_IMAGE_SLICE_PITCH,
                                         sizeof(returned_slice_pitch),
                                         &returned_slice_pitch, nullptr);
                    test_error(err, "Error clGetImageInfo");

                    if (returned_slice_pitch != slice_pitch)
                    {
                        test_fail(
                            "Unexpected row pitch "
                            "CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT");
                    }
                }

                err = clReleaseMemObject(buffer);
                test_error(err, "Unable to release buffer");

                err = clReleaseMemObject(image_buffer);
                test_error(err, "Unable to release image");
            }
        }
    }

    return TEST_PASS;
}

/**
 * Negative testing for clCreateImage and wrong alignment
 * - Create an image from a buffer with invalid row pitch (not a multiple of
 * required alignment) and check that CL_INVALID_IMAGE_DESCRIPTOR is returned.
 * - Create an image from a buffer with invalid slice pitch (not a multiple of
 * required alignment) and check that CL_INVALID_IMAGE_DESCRIPTOR is returned.
 * - Create an image from a buffer with invalid base address alignment (not a
 * multiple of required alignment) and check that CL_INVALID_IMAGE_DESCRIPTOR is
 * returned
 */
int image_from_buffer_alignment_negative(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                size_t row_pitch_alignment = 0;
                size_t slice_pitch_alignment = 0;
                size_t base_address_alignment = 0;

                int get_error = get_image_requirement_alignment(
                    device, context, 0, &format, &image_desc,
                    &row_pitch_alignment, &slice_pitch_alignment,
                    &base_address_alignment);
                if (TEST_PASS != get_error)
                {
                    return get_error;
                }

                const size_t element_size =
                    get_format_size(context, &format, imageType, flag);

                const size_t row_pitch = aligned_size(
                    TEST_IMAGE_SIZE * element_size, row_pitch_alignment);
                const size_t slice_pitch = aligned_size(
                    row_pitch * TEST_IMAGE_SIZE, slice_pitch_alignment);

                const size_t buffer_size = (slice_pitch + 1)
                    * TEST_IMAGE_SIZE; /* For bigger row/slice pitch */

                cl_int err = CL_SUCCESS;
                cl_mem buffer =
                    clCreateBuffer(context, flag, buffer_size, nullptr, &err);
                test_error(err, "Unable to create buffer");

                /* Test Row pitch images */
                if (imageType == CL_MEM_OBJECT_IMAGE2D
                    || imageType == CL_MEM_OBJECT_IMAGE3D
                    || imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY
                    || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    image_desc.buffer = buffer;
                    image_desc.image_row_pitch =
                        row_pitch + 1; /* wrong row pitch */

                    clCreateImage(context, flag, &format, &image_desc, nullptr,
                                  &err);
                    test_failure_error(err, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                                       "Unexpected clCreateImage return");
                }

                /* Test Slice pitch images */
                if (imageType == CL_MEM_OBJECT_IMAGE3D
                    || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    image_desc.buffer = buffer;
                    image_desc.image_row_pitch = row_pitch;
                    image_desc.image_slice_pitch =
                        slice_pitch + 1; /* wrong slice pitch */

                    clCreateImage(context, flag, &format, &image_desc, nullptr,
                                  &err);
                    test_failure_error(err, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                                       "Unexpected clCreateImage return");
                }

                /* Test buffer from host ptr to test base address alignment */
                const size_t aligned_buffer_size =
                    aligned_size(buffer_size, base_address_alignment);
                /* Create buffer with host ptr and additional size for the wrong
                 * alignment */
                void* const host_ptr =
                    malloc(aligned_buffer_size + base_address_alignment);
                void* non_aligned_host_ptr =
                    (void*)((char*)(aligned_ptr(host_ptr,
                                                base_address_alignment))
                            + 1); /* wrong alignment */

                cl_mem buffer_host = clCreateBuffer(
                    context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                    buffer_size, non_aligned_host_ptr, &err);
                test_error(err, "Unable to create buffer");

                image_desc.buffer = buffer_host;

                clCreateImage(context, flag, &format, &image_desc, nullptr,
                              &err);
                test_failure_error(err, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                                   "Unexpected clCreateImage return");

                free(host_ptr);

                err = clReleaseMemObject(buffer);
                test_error(err, "Unable to release buffer");

                err = clReleaseMemObject(buffer_host);
                test_error(err, "Unable to release buffer");
            }
        }
    }

    return TEST_PASS;
}

/**
 * Negative testing for clCreateImage (buffer size).
 * Create a buffer too small and check that image creation from that buffer is
 * rejected
 */
int image_from_small_buffer_negative(cl_device_id device, cl_context context,
                                     cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,        CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY,  CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                /* Invalid buffer size */
                cl_int err;
                cl_mem buffer = clCreateBuffer(
                    context, flag, TEST_IMAGE_SIZE / 2, nullptr, &err);
                test_error(err, "Unable to create buffer");

                image_desc.buffer = buffer;

                clCreateImage(context, flag, &format, &image_desc, nullptr,
                              &err);
                test_failure_error(err, CL_INVALID_MEM_OBJECT,
                                   "Unexpected clCreateImage return");

                err = clReleaseMemObject(buffer);
                test_error(err, "Unable to release buffer");
            }
        }
    }

    return TEST_PASS;
}

static int image_from_buffer_fill_check(cl_command_queue queue, cl_mem image,
                                        size_t* region, size_t element_size,
                                        char pattern)
{
    /* read the image from buffer and check the pattern */
    const size_t image_size = region[0] * region[1] * region[2] * element_size;
    size_t origin[3] = { 0, 0, 0 };
    std::vector<char> read_buffer(image_size);

    cl_int error =
        clEnqueueReadImage(queue, image, CL_BLOCKING, origin, region, 0, 0,
                           read_buffer.data(), 0, nullptr, nullptr);
    test_error(error, "Error clEnqueueReadImage");

    for (size_t line = 0; line < region[0]; line++)
    {
        for (size_t row = 0; row < region[1]; row++)
        {
            for (size_t depth = 0; depth < region[2]; depth++)
            {
                for (size_t elmt = 0; elmt < element_size; elmt++)
                {
                    size_t index = line * row * depth * elmt;

                    if (read_buffer[index] != pattern)
                    {
                        test_fail("Image pattern check failed");
                    }
                }
            }
        }
    }

    return TEST_PASS;
}

/**
 * Use fill buffer to fill the image from buffer
 */
int image_from_buffer_fill_positive(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    std::vector<cl_mem_flags> flagTypes{ CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                                         CL_MEM_READ_WRITE,
                                         CL_MEM_KERNEL_READ_AND_WRITE };

    for (auto flag : flagTypes)
    {
        for (auto imageType : imageTypes)
        {
            /* Get the list of supported image formats */
            std::vector<cl_image_format> formatList;
            if (TEST_PASS
                    != get_format_list(context, imageType, formatList, flag)
                || formatList.size() == 0)
            {
                test_fail("Failure to get supported formats list");
            }

            for (auto format : formatList)
            {
                cl_image_desc image_desc = { 0 };
                image_desc_init(&image_desc, imageType);

                flag = (flag == CL_MEM_KERNEL_READ_AND_WRITE)
                    ? CL_MEM_READ_WRITE
                    : flag;

                size_t row_pitch_alignment = 0;
                size_t slice_pitch_alignment = 0;

                int get_error = get_image_requirement_alignment(
                    device, context, 0, &format, &image_desc,
                    &row_pitch_alignment, &slice_pitch_alignment, nullptr);
                if (TEST_PASS != get_error)
                {
                    return get_error;
                }

                const size_t element_size =
                    get_format_size(context, &format, imageType, flag);

                const size_t row_pitch = aligned_size(
                    TEST_IMAGE_SIZE * element_size, row_pitch_alignment);
                const size_t slice_pitch = aligned_size(
                    row_pitch * TEST_IMAGE_SIZE, slice_pitch_alignment);

                const size_t buffer_size = slice_pitch * TEST_IMAGE_SIZE;

                cl_int err = CL_SUCCESS;
                cl_mem buffer =
                    clCreateBuffer(context, flag, buffer_size, nullptr, &err);
                test_error(err, "Unable to create buffer");

                /* fill the buffer with a pattern */
                const char pattern = 0x55;
                err = clEnqueueFillBuffer(queue, buffer, &pattern,
                                          sizeof(pattern), 0, buffer_size, 0,
                                          nullptr, nullptr);
                test_error(err, "Error clEnqueueFillBuffer");

                err = clFinish(queue);
                test_error(err, "Error clFinish");

                cl_mem image1d_buffer;
                if (imageType == CL_MEM_OBJECT_IMAGE1D_BUFFER)
                {
                    image1d_buffer = clCreateBuffer(context, flag, buffer_size,
                                                    nullptr, &err);
                    test_error(err, "Unable to create buffer");

                    image_desc.buffer = image1d_buffer;
                }

                cl_mem image = clCreateImage(context, flag, &format,
                                             &image_desc, nullptr, &err);
                test_error(err, "Unable to create image");

                /* Check the image from buffer */
                image_desc.buffer = buffer;

                if (imageType == CL_MEM_OBJECT_IMAGE2D
                    || imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                {
                    image_desc.image_row_pitch = row_pitch;
                }
                else if (imageType == CL_MEM_OBJECT_IMAGE3D
                         || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    image_desc.image_row_pitch = row_pitch;
                    image_desc.image_slice_pitch = slice_pitch;
                }

                cl_mem image_from_buffer = clCreateImage(
                    context, flag, &format, &image_desc, nullptr, &err);
                test_error(err, "Unable to create image");

                size_t origin[3] = { 0, 0, 0 };
                size_t region[3] = { 1, 1, 1 };

                region[0] = TEST_IMAGE_SIZE;
                if (CL_MEM_OBJECT_IMAGE1D_BUFFER != imageType
                    && CL_MEM_OBJECT_IMAGE1D != imageType)
                {
                    region[1] = TEST_IMAGE_SIZE;
                }
                if (CL_MEM_OBJECT_IMAGE3D == imageType
                    || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
                {
                    region[2] = TEST_IMAGE_SIZE;
                }

                /* Check the copy of the image from buffer */
                err =
                    clEnqueueCopyImage(queue, image_from_buffer, image, origin,
                                       origin, region, 0, nullptr, nullptr);
                test_error(err, "Error clEnqueueCopyImage");

                err = clFinish(queue);
                test_error(err, "Error clFinish");

                int fill_error = image_from_buffer_fill_check(
                    queue, image_from_buffer, region, element_size, pattern);
                if (TEST_PASS != fill_error)
                {
                    return fill_error;
                }

                fill_error = image_from_buffer_fill_check(
                    queue, image, region, element_size, pattern);
                if (TEST_PASS != fill_error)
                {
                    return fill_error;
                }

                err = clReleaseMemObject(buffer);
                test_error(err, "Unable to release buffer");

                err = clReleaseMemObject(image);
                test_error(err, "Unable to release image");

                err = clReleaseMemObject(image_from_buffer);
                test_error(err, "Unable to release image");

                if (imageType == CL_MEM_OBJECT_IMAGE1D_BUFFER)
                {
                    err = clReleaseMemObject(image1d_buffer);
                    test_error(err, "Unable to release image");
                }
            }
        }
    }

    return TEST_PASS;
}

static int image_from_buffer_read_check(cl_command_queue queue, cl_mem buffer,
                                        const size_t buffer_size,
                                        size_t* region, size_t element_size,
                                        char pattern, size_t row_pitch,
                                        size_t slice_pitch)
{
    /* read the buffer and check the pattern */
    std::vector<char> host_buffer(buffer_size);
    char* host_ptr = host_buffer.data();
    char* host_ptr_slice = host_ptr;

    cl_int error =
        clEnqueueReadBuffer(queue, buffer, CL_BLOCKING, 0, buffer_size,
                            host_buffer.data(), 0, nullptr, nullptr);
    test_error(error, "Error clEnqueueReadBuffer");

    for (size_t k = 0; k < region[2]; k++)
    {
        for (size_t i = 0; i < region[1]; i++)
        {
            for (size_t j = 0; j < region[0] * element_size; j++)
            {
                if (host_ptr[j] != pattern)
                {
                    test_fail("Image pattern check failed");
                }
            }
            host_ptr = host_ptr + row_pitch;
        }
        host_ptr_slice = host_ptr_slice + slice_pitch;
        host_ptr = host_ptr_slice;
    }

    return TEST_PASS;
}

/**
 * Use fill image to fill the buffer that was used to create the image
 */
int image_from_buffer_read_positive(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_ext_image_from_buffer"))
    {
        printf("Extension cl_ext_image_from_buffer not available");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_mem_object_type> imageTypes{
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    for (auto imageType : imageTypes)
    {
        cl_image_desc image_desc = { 0 };
        image_desc_init(&image_desc, imageType);

        /* Non normalized format so we can read it back directly from
         * clEnqueueFillImage */
        cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };
        const char pattern = 0x55;

        const size_t element_size =
            get_format_size(context, &format, imageType, CL_MEM_READ_WRITE);

        size_t row_pitch_alignment = 0;
        size_t slice_pitch_alignment = 0;

        int get_error = get_image_requirement_alignment(
            device, context, CL_MEM_READ_WRITE, &format, &image_desc,
            &row_pitch_alignment, &slice_pitch_alignment, nullptr);
        if (TEST_PASS != get_error)
        {
            return get_error;
        }

        const size_t row_pitch =
            aligned_size(TEST_IMAGE_SIZE * element_size, row_pitch_alignment);
        const size_t slice_pitch =
            aligned_size(row_pitch * TEST_IMAGE_SIZE, slice_pitch_alignment);

        const size_t buffer_size = slice_pitch * TEST_IMAGE_SIZE;

        cl_int err = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size,
                                       nullptr, &err);
        test_error(err, "Unable to create buffer");

        /* Check the image from buffer */
        image_desc.buffer = buffer;

        if (imageType == CL_MEM_OBJECT_IMAGE2D
            || imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        {
            image_desc.image_row_pitch = row_pitch;
        }
        else if (imageType == CL_MEM_OBJECT_IMAGE3D
                 || imageType == CL_MEM_OBJECT_IMAGE2D_ARRAY)
        {
            image_desc.image_row_pitch = row_pitch;
            image_desc.image_slice_pitch = slice_pitch;
        }

        cl_mem image = clCreateImage(context, CL_MEM_READ_WRITE, &format,
                                     &image_desc, nullptr, &err);
        test_error(err, "Unable to create image");

        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { 1, 1, 1 };

        region[0] = TEST_IMAGE_SIZE;
        if (CL_MEM_OBJECT_IMAGE1D_BUFFER != imageType
            && CL_MEM_OBJECT_IMAGE1D != imageType)
        {
            region[1] = TEST_IMAGE_SIZE;
        }
        if (CL_MEM_OBJECT_IMAGE3D == imageType
            || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
        {
            region[2] = TEST_IMAGE_SIZE;
        }

        /* fill the image with a pattern */
        cl_uint fill_color[4] = { pattern, pattern, pattern, pattern };
        err = clEnqueueFillImage(queue, image, fill_color, origin, region, 0,
                                 nullptr, nullptr);
        test_error(err, "Error clEnqueueFillImage");

        err = clFinish(queue);
        test_error(err, "Error clFinish");

        int read_error = image_from_buffer_read_check(
            queue, buffer, buffer_size, region, element_size, pattern,
            (imageType == CL_MEM_OBJECT_IMAGE1D_ARRAY) ? slice_pitch
                                                       : row_pitch,
            slice_pitch);
        if (TEST_PASS != read_error)
        {
            return read_error;
        }

        err = clReleaseMemObject(buffer);
        test_error(err, "Unable to release buffer");

        err = clReleaseMemObject(image);
        test_error(err, "Unable to release image");
    }

    return TEST_PASS;
}
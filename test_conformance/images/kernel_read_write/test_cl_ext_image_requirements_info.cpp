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

/**
 * Negative tests for {CL_IMAGE_REQUIREMENTS_SIZE_EXT}
 * Check that attempting to perform the {CL_IMAGE_REQUIREMENTS_SIZE_EXT} query
 *  without specifying the _image_format_ results in {CL_INVALID_VALUE} being
 * returned. Check that attempting to perform the
 * {CL_IMAGE_REQUIREMENTS_SIZE_EXT} query without specifying the _image_desc_
 * results in {CL_INVALID_VALUE} being returned.
 */
int cl_image_requirements_size_ext_negative(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    size_t max_size = 0;
    size_t param_val_size = 0;

    cl_image_desc image_desc = { 0 };
    image_desc_init(&image_desc, CL_MEM_OBJECT_IMAGE2D);

    cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT16 };

    /* Check image_format null results in CL_INVALID_VALUE */
    cl_int err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, nullptr, &image_desc,
        CL_IMAGE_REQUIREMENTS_SIZE_EXT, sizeof(max_size), &max_size,
        &param_val_size);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check image_desc null results in CL_INVALID_VALUE */
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, nullptr,
        CL_IMAGE_REQUIREMENTS_SIZE_EXT, sizeof(max_size), &max_size,
        &param_val_size);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    return TEST_PASS;
}

/**
 * Consistency checks for CL_IMAGE_REQUIREMENTS_SIZE_EXT
 * When creating 2D images from a buffer is supported
 * Check that the CL_IMAGE_REQUIREMENTS_SIZE_EXT query can be performed
 * successfully. Create a buffer with the size returned and check that an image
 * can successfully be created from the buffer. Check that the value returned
 * for CL_MEM_SIZE for the image is the same as the value returned for
 * CL_IMAGE_REQUIREMENTS_SIZE_EXT.
 */
int cl_image_requirements_size_ext_consistency(cl_device_id device,
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

    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    size_t max_size = 0;
    size_t param_val_size = 0;

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

                cl_int err = clGetImageRequirementsInfoEXT(
                    context, nullptr, flag, &format, &image_desc,
                    CL_IMAGE_REQUIREMENTS_SIZE_EXT, sizeof(max_size), &max_size,
                    &param_val_size);
                test_error(err, "Error clGetImageRequirementsInfoEXT");

                /* Create buffer */
                cl_mem buffer =
                    clCreateBuffer(context, flag, max_size, nullptr, &err);
                test_error(err, "Unable to create buffer");

                image_desc.buffer = buffer;

                /* 2D Image from buffer */
                cl_mem image_buffer = clCreateImage(context, flag, &format,
                                                    &image_desc, nullptr, &err);
                test_error(err, "Unable to create image");

                size_t size = 0;
                err = clGetMemObjectInfo(image_buffer, CL_MEM_SIZE,
                                         sizeof(size_t), &size, NULL);
                test_error(err, "Error clGetMemObjectInfo");

                if (max_size != size)
                {
                    test_fail("CL_IMAGE_REQUIREMENTS_SIZE_EXT different from "
                              "CL_MEM_SIZE");
                }

                err = clReleaseMemObject(image_buffer);
                test_error(err, "Error clReleaseMemObject");

                err = clReleaseMemObject(buffer);
                test_error(err, "Error clReleaseMemObject");
            }
        }
    }

    return TEST_PASS;
}

/**
 * Negative testing for all testable error codes returned by
 * clGetImageFormatInfoKHR
 */
int clGetImageRequirementsInfoEXT_negative(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    cl_image_desc image_desc = { 0 };
    image_desc_init(&image_desc, CL_MEM_OBJECT_IMAGE3D);

    cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT16 };

    /* Check that CL_INVALID_CONTEXT is returned when passing nullptr as context
     */
    size_t row_pitch_alignment = 0;
    cl_int err = clGetImageRequirementsInfoEXT(
        nullptr, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_CONTEXT,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_VALUE is returned when passing an invalid
     * image_type */
    cl_image_desc invalid_desc = { CL_MEM_OBJECT_BUFFER, TEST_IMAGE_SIZE };
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, &invalid_desc,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_IMAGE_DESCRIPTOR,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_VALUE is returned when passing invalid flags */
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, -1, &format, &image_desc,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_IMAGE_FORMAT_DESCRIPTOR is returned when passing a
     * nullptr image_format */
    cl_image_format invalid_format = { CL_INTENSITY, CL_UNORM_SHORT_555 };
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &invalid_format, &image_desc,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_IMAGE_DESCRIPTOR is returned when passing an
     * image_desc with invalid values */
    cl_image_desc invalid_desc_size = { CL_MEM_OBJECT_IMAGE1D, 0 };
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, &invalid_desc_size,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_IMAGE_DESCRIPTOR,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_VALUE is returned when passing an invalid
     * param_name */
    cl_image_requirements_info_ext invalid_info = CL_IMAGE_FORMAT;
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc, invalid_info,
        sizeof(row_pitch_alignment), &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_VALUE is returned when passing a param_value_size
     * value smaller than the size of the return type */
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
        CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
        sizeof(row_pitch_alignment) - 1, &row_pitch_alignment, nullptr);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    /* Check that CL_INVALID_VALUE is returned when passing a param_value_size
     * value smaller than the size of the return type */
    uint32_t max_height = 0;
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
        CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT, sizeof(max_height) - 1,
        &max_height, nullptr);
    test_failure_error(err, CL_INVALID_VALUE,
                       "Unexpected clGetImageRequirementsInfoEXT return");

    return TEST_PASS;
}

/**
 * Negative tests for {CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT}
 * Attempt to perform the {CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT} query on all
 * image types for which it is not valid Check that
 * {CL_INVALID_IMAGE_DESCRIPTOR} is returned in all cases.
 *
 * Negative testing for {CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT}
 * Attempt to perform the {CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT} query on all
 * image types for which it is not valid Check that
 * {CL_INVALID_IMAGE_DESCRIPTOR} is returned in all cases.
 *
 * Negative testing for {CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT}
 * Attempt to perform the {CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT} query on
 * all image types for which it is not valid Check that
 * {CL_INVALID_IMAGE_DESCRIPTOR} is returned in all cases.
 */
int cl_image_requirements_max_val_ext_negative(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    size_t value = 0;

    std::vector<cl_mem_object_type> imageTypes_height{
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D
    };

    cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT16 };

    for (auto imageType : imageTypes_height)
    {
        cl_image_desc image_desc = { 0 };
        image_desc_init(&image_desc, imageType);

        /* Check image_format null results in CL_INVALID_VALUE */
        cl_int err = clGetImageRequirementsInfoEXT(
            context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
            CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT, sizeof(value), &value,
            nullptr);
        test_failure_error(err, CL_INVALID_IMAGE_DESCRIPTOR,
                           "Unexpected clGetImageRequirementsInfoEXT return");
    }

    std::vector<cl_mem_object_type> imageTypes_depth{
        CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE2D_ARRAY,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE1D
    };

    for (auto imageType : imageTypes_depth)
    {
        cl_image_desc image_desc = { 0 };
        image_desc_init(&image_desc, imageType);

        /* Check image_format null results in CL_INVALID_VALUE */
        cl_int err = clGetImageRequirementsInfoEXT(
            context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
            CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT, sizeof(value), &value,
            nullptr);
        test_failure_error(err, CL_INVALID_IMAGE_DESCRIPTOR,
                           "Unexpected clGetImageRequirementsInfoEXT return");
    }

    std::vector<cl_mem_object_type> imageTypes_array_size{
        CL_MEM_OBJECT_IMAGE3D, CL_MEM_OBJECT_IMAGE2D,
        CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE1D
    };

    for (auto imageType : imageTypes_array_size)
    {
        cl_image_desc image_desc = { 0 };
        image_desc_init(&image_desc, imageType);

        /* Check image_format null results in CL_INVALID_VALUE */
        cl_int err = clGetImageRequirementsInfoEXT(
            context, nullptr, CL_MEM_READ_WRITE, &format, &image_desc,
            CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT, sizeof(value), &value,
            nullptr);
        test_failure_error(err, CL_INVALID_IMAGE_DESCRIPTOR,
                           "Unexpected clGetImageRequirementsInfoEXT return");
    }

    return TEST_PASS;
}

/**
 * Consistency checks for {CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT}
 ** Check that the {CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT} query can be performed
 *successfully
 *
 * Consistency checks for {CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT}
 ** Check that the {CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT} query can be performed
 *successfully
 *
 * Consistency checks for {CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT}
 ** Check that the {CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT} query can be performed
 *successfully
 *
 * Consistency checks for {CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT}
 ** Check that the {CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT} query can be
 *performed successfully
 */
int cl_image_requirements_max_val_ext_positive(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_ext_image_requirements_info"))
    {
        printf("Extension cl_ext_image_requirements_info not available");
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform = getPlatformFromDevice(device);
    GET_EXTENSION_FUNC(platform, clGetImageRequirementsInfoEXT);

    /* CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT */
    cl_image_desc image_desc_1d = { 0 };
    image_desc_init(&image_desc_1d, CL_MEM_OBJECT_IMAGE1D);

    uint32_t max_width = 0;
    cl_int err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, nullptr, &image_desc_1d,
        CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT, sizeof(max_width), &max_width,
        nullptr);
    test_error(err, "Error clGetImageRequirementsInfoEXT");

    size_t width_1d = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                          sizeof(width_1d), &width_1d, NULL);
    test_error(err, "Error clGetDeviceInfo");

    if (!(max_width <= width_1d && max_width > 0))
    {
        test_fail("Unexpected CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT value");
    }

    /* CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT */
    cl_image_desc image_desc_2d = { 0 };
    image_desc_init(&image_desc_2d, CL_MEM_OBJECT_IMAGE2D);

    uint32_t max_height = 0;
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, nullptr, &image_desc_2d,
        CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT, sizeof(max_height), &max_height,
        nullptr);
    test_error(err, "Error clGetImageRequirementsInfoEXT");

    size_t height_2d = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                          sizeof(height_2d), &height_2d, NULL);
    test_error(err, "Error clGetDeviceInfo");

    if (!(max_height <= height_2d && max_height > 0))
    {
        test_fail("Unexpected CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT value");
    }

    /* CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT */
    cl_image_desc image_desc_3d = { 0 };
    image_desc_init(&image_desc_3d, CL_MEM_OBJECT_IMAGE3D);

    uint32_t max_depth = 0;
    err = clGetImageRequirementsInfoEXT(context, nullptr, CL_MEM_READ_WRITE,
                                        nullptr, &image_desc_3d,
                                        CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT,
                                        sizeof(max_depth), &max_depth, nullptr);
    test_error(err, "Error clGetImageRequirementsInfoEXT");

    size_t depth_3d = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(depth_3d),
                          &depth_3d, NULL);
    test_error(err, "Error clGetDeviceInfo");

    if (!(max_depth <= depth_3d && max_depth > 0))
    {
        test_fail("Unexpected CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT value");
    }

    /* CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT */
    cl_image_desc image_desc_array = { 0 };
    image_desc_init(&image_desc_array, CL_MEM_OBJECT_IMAGE2D_ARRAY);

    uint32_t max_array_size = 0;
    err = clGetImageRequirementsInfoEXT(
        context, nullptr, CL_MEM_READ_WRITE, nullptr, &image_desc_array,
        CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT, sizeof(max_array_size),
        &max_array_size, nullptr);
    test_error(err, "Error clGetImageRequirementsInfoEXT");

    size_t array_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
                          sizeof(array_size), &array_size, NULL);
    test_error(err, "Error clGetDeviceInfo");

    if (!(max_array_size <= array_size && max_array_size > 0))
    {
        test_fail("Unexpected CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT value");
    }

    return TEST_PASS;
}
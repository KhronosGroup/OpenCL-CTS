//
// Copyright (c) 2026 The Khronos Group Inc.
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

#include "testBase.h"

namespace {
const cl_mem_flags mem_flags[] = { CL_MEM_KERNEL_READ_AND_WRITE,
                                   CL_MEM_READ_WRITE
                                       | CL_MEM_KERNEL_READ_AND_WRITE };
const char *mem_flag_names[] = {
    "CL_MEM_KERNEL_READ_AND_WRITE",
    "CL_MEM_READ_WRITE | CL_MEM_KERNEL_READ_AND_WRITE"
};
}

REGISTER_TEST_VERSION(negative_create_image_kernel_read_and_write,
                      Version(2, 0))
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)
    const bool with_props = get_device_cl_version(device) >= Version(3, 0);

    cl_int error = CL_SUCCESS;
    const cl_mem_object_type image_types[] = {
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE2D,       CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY,
    };

    cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t element_size = get_pixel_size(&format);
    const size_t img_size = 4;

    clMemWrapper buffer = clCreateBuffer(
        context, CL_MEM_READ_WRITE, element_size * img_size, nullptr, &error);
    test_error(error, "Unable to create buffer");

    cl_image_desc image_desc = {};
    image_desc.image_width = img_size;
    image_desc.image_height = img_size;
    image_desc.image_depth = img_size;
    image_desc.image_array_size = img_size;

    for (size_t k = 0; k < ARRAY_SIZE(mem_flags); k++)
    {
        for (size_t i = 0; i < ARRAY_SIZE(image_types); i++)
        {
            error = CL_SUCCESS;
            image_desc.image_type = image_types[i];
            image_desc.buffer = image_types[i] == CL_MEM_OBJECT_IMAGE1D_BUFFER
                ? buffer
                : nullptr;

            clMemWrapper image = clCreateImage(context, mem_flags[k], &format,
                                               &image_desc, nullptr, &error);

            const std::string desc =
                std::string(convert_image_type_to_string(image_types[i])) + ", "
                + mem_flag_names[k] + ") should fail";

            test_object_failure_ret(image, error, CL_INVALID_VALUE,
                                    ("clCreateImage(" + desc).c_str(),
                                    TEST_FAIL);

            if (with_props)
            {
                error = CL_SUCCESS;
                image = clCreateImageWithProperties(
                    context, nullptr, mem_flags[k], &format, &image_desc,
                    nullptr, &error);
                test_object_failure_ret(
                    image, error, CL_INVALID_VALUE,
                    ("clCreateImageWithProperties(" + desc).c_str(), TEST_FAIL);
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(negative_create_buffer_kernel_read_and_write,
                      Version(2, 0))
{
    const bool with_props = get_device_cl_version(device) >= Version(3, 0);

    cl_int error = CL_SUCCESS;

    for (size_t k = 0; k < ARRAY_SIZE(mem_flags); k++)
    {
        error = CL_SUCCESS;
        clMemWrapper buffer = clCreateBuffer(context, mem_flags[k],
                                             sizeof(cl_int), nullptr, &error);

        test_object_failure_ret(buffer, error, CL_INVALID_VALUE,
                                (std::string("clCreateBuffer with ")
                                 + mem_flag_names[k] + " should fail")
                                    .c_str(),
                                TEST_FAIL);

        if (with_props)
        {
            error = CL_SUCCESS;
            buffer =
                clCreateBufferWithProperties(context, nullptr, mem_flags[k],
                                             sizeof(cl_int), nullptr, &error);

            test_object_failure_ret(
                buffer, error, CL_INVALID_VALUE,
                (std::string("clCreateBufferWithProperties with ")
                 + mem_flag_names[k] + " should fail")
                    .c_str(),
                TEST_FAIL);
        }
    }

    return TEST_PASS;
}

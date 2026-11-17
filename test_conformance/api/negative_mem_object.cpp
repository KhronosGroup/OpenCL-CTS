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
//
#include "testBase.h"
#include "harness/typeWrappers.h"

#include <vector>

namespace {

constexpr cl_mem_flags base_flags[] = {
    0,
    CL_MEM_READ_WRITE,
    CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY,
    CL_MEM_USE_HOST_PTR,
    CL_MEM_ALLOC_HOST_PTR,
    CL_MEM_COPY_HOST_PTR,
    CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
    CL_MEM_HOST_WRITE_ONLY,
    CL_MEM_HOST_READ_ONLY,
    CL_MEM_HOST_NO_ACCESS,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
};

// Enough for the largest image used below: 4x4x4 texels of 16 bytes.
constexpr size_t host_data_size = 4 * 4 * 4 * 16;

constexpr cl_mem_object_type image_types[] = {
    CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE2D,
    CL_MEM_OBJECT_IMAGE3D,       CL_MEM_OBJECT_IMAGE1D_ARRAY,
    CL_MEM_OBJECT_IMAGE2D_ARRAY, CL_MEM_OBJECT_IMAGE1D_BUFFER,
};

constexpr cl_mem_flags access_mask =
    CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY;
constexpr cl_mem_flags host_ptr_mask =
    CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR;

cl_int setupImageDesc(cl_context context, cl_mem_object_type type,
                      cl_image_desc &desc, clMemWrapper &image_buffer)
{
    cl_int error = CL_SUCCESS;

    desc = {};
    desc.image_type = type;
    desc.image_width = 4;

    // to avoid other than CL_MEM_KERNEL_READ_AND_WRITE related errors,
    // setup cl_image_desc only for required parameters
    switch (type)
    {
        case CL_MEM_OBJECT_IMAGE1D: break;
        case CL_MEM_OBJECT_IMAGE2D: desc.image_height = 4; break;
        case CL_MEM_OBJECT_IMAGE3D:
            desc.image_height = 4;
            desc.image_depth = 4;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY: desc.image_array_size = 4; break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            desc.image_height = 4;
            desc.image_array_size = 4;
            break;
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          host_data_size, nullptr, &error);
            test_error(error, "clCreateBuffer failed");
            desc.buffer = image_buffer;
            break;
        default:
            log_error("Unexpected image type: %s\n",
                      convert_image_type_to_string(type));
            return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

cl_int getSupportedFormats(cl_context context, cl_mem_flags base,
                           cl_mem_object_type type,
                           std::vector<cl_image_format> &formats)
{
    // for query try to use the same access flags, otherwise
    // CL_MEM_READ_WRITE
    cl_mem_flags query_flags =
        (base & access_mask) ? (base & access_mask) : CL_MEM_READ_WRITE;

    cl_uint num_formats = 0;
    cl_int error = clGetSupportedImageFormats(context, query_flags, type, 0,
                                              nullptr, &num_formats);
    test_error(error, "clGetSupportedImageFormats failed");

    formats.resize(num_formats);
    if (num_formats == 0) return CL_SUCCESS;

    error = clGetSupportedImageFormats(context, query_flags, type, num_formats,
                                       formats.data(), nullptr);
    test_error(error, "clGetSupportedImageFormats failed");

    return CL_SUCCESS;
}

int test_create_buffer(cl_context context, bool with_props)
{
    const char *api_name =
        with_props ? "clCreateBufferWithProperties" : "clCreateBuffer";

    std::vector<cl_uchar> host_data(host_data_size);

    for (cl_mem_flags base : base_flags)
    {
        cl_mem_flags flags = base | CL_MEM_KERNEL_READ_AND_WRITE;
        void *host_ptr = (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))
            ? host_data.data()
            : nullptr;

        cl_int error = CL_SUCCESS;
        clMemWrapper buffer = with_props
            ? clCreateBufferWithProperties(context, nullptr, flags,
                                           host_data_size, host_ptr, &error)
            : clCreateBuffer(context, flags, host_data_size, host_ptr, &error);

        if (error != CL_INVALID_VALUE || buffer != nullptr)
        {
            log_error(
                "%s(flags: 0x%llx) with CL_MEM_KERNEL_READ_AND_WRITE returned "
                "%p with %s, expected NULL with CL_INVALID_VALUE\n",
                api_name, (unsigned long long)flags, (void *)buffer,
                IGetErrorString(error));
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int test_create_image(cl_context context, bool with_props)
{
    const char *api_name =
        with_props ? "clCreateImageWithProperties" : "clCreateImage";

    cl_int error = CL_SUCCESS;
    std::vector<cl_uchar> host_data(host_data_size);
    std::vector<cl_image_format> formats;

    for (cl_mem_object_type type : image_types)
    {
        clMemWrapper image_buffer;
        cl_image_desc desc;
        error = setupImageDesc(context, type, desc, image_buffer);
        test_error(error, "setupImageDesc failed");

        for (cl_mem_flags base : base_flags)
        {
            // skip host_ptr cases when an image was created from a buffer
            if (type == CL_MEM_OBJECT_IMAGE1D_BUFFER && (base & host_ptr_mask))
                continue;

            error = getSupportedFormats(context, base, type, formats);
            test_error(error, "getSupportedFormats failed");
            if (formats.empty()) continue;

            cl_mem_flags flags = base | CL_MEM_KERNEL_READ_AND_WRITE;
            void *host_ptr =
                (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))
                ? host_data.data()
                : nullptr;

            clMemWrapper image = with_props
                ? clCreateImageWithProperties(context, nullptr, flags,
                                              &formats[0], &desc, host_ptr,
                                              &error)
                : clCreateImage(context, flags, &formats[0], &desc, host_ptr,
                                &error);

            if (error != CL_INVALID_VALUE || image != nullptr)
            {
                log_error("%s(%s, flags: 0x%llx) with "
                          "CL_MEM_KERNEL_READ_AND_WRITE returned %p with %s, "
                          "expected NULL with CL_INVALID_VALUE\n",
                          api_name, convert_image_type_to_string(type),
                          (unsigned long long)flags, (void *)image,
                          IGetErrorString(error));
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

}

// clCreateBuffer must reject CL_MEM_KERNEL_READ_AND_WRITE with
// CL_INVALID_VALUE.
REGISTER_TEST(negative_create_buffer)
{
    return test_create_buffer(context, false);
}

// clCreateBufferWithProperties must reject CL_MEM_KERNEL_READ_AND_WRITE with
// CL_INVALID_VALUE.
REGISTER_TEST_VERSION(negative_create_buffer_with_properties, Version(3, 0))
{
    return test_create_buffer(context, true);
}

// clCreateImage must reject CL_MEM_KERNEL_READ_AND_WRITE with
// CL_INVALID_VALUE for every supported image type.
REGISTER_TEST(negative_create_image)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device);

    return test_create_image(context, false);
}

// clCreateImageWithProperties must reject CL_MEM_KERNEL_READ_AND_WRITE with
// CL_INVALID_VALUE for every supported image type.
REGISTER_TEST_VERSION(negative_create_image_with_properties, Version(3, 0))
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device);

    return test_create_image(context, true);
}

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

#ifndef TEST_CL_EXT_IMAGE_BUFFER
#define TEST_CL_EXT_IMAGE_BUFFER

#define TEST_IMAGE_SIZE 20

#define GET_EXTENSION_FUNC(platform, function_name)                            \
    function_name##_fn function_name = reinterpret_cast<function_name##_fn>(   \
        clGetExtensionFunctionAddressForPlatform(platform, #function_name));   \
    if (function_name == nullptr)                                              \
    {                                                                          \
        return TEST_FAIL;                                                      \
    }                                                                          \
    do                                                                         \
    {                                                                          \
    } while (false)

static inline size_t aligned_size(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

static inline void* aligned_ptr(void* ptr, size_t alignment)
{
    return (void*)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1));
}

static inline size_t get_format_size(cl_context context,
                                     cl_image_format* format,
                                     cl_mem_object_type imageType,
                                     cl_mem_flags flags)
{
    cl_image_desc image_desc = { 0 };
    image_desc.image_type = imageType;

    /* We use a width of 4 to query element size, as this is
       the smallest possible value that satisfies the requirements
       of all image formats (including extensions). */
    image_desc.image_width = 4;
    if (CL_MEM_OBJECT_IMAGE1D_BUFFER != imageType
        && CL_MEM_OBJECT_IMAGE1D != imageType)
    {
        image_desc.image_height = 1;
    }
    if (CL_MEM_OBJECT_IMAGE3D == imageType
        || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
    {
        image_desc.image_depth = 1;
    }
    if (CL_MEM_OBJECT_IMAGE1D_ARRAY == imageType
        || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
    {
        image_desc.image_array_size = 1;
    }

    cl_int error = 0;
    cl_mem buffer;
    if (imageType == CL_MEM_OBJECT_IMAGE1D_BUFFER)
    {
        buffer = clCreateBuffer(context, flags,
                                get_pixel_size(format) * image_desc.image_width,
                                NULL, &error);
        test_error(error, "Unable to create buffer");

        image_desc.buffer = buffer;
    }

    cl_mem image =
        clCreateImage(context, flags, format, &image_desc, nullptr, &error);
    test_error(error, "Unable to create image");

    size_t element_size = 0;
    error = clGetImageInfo(image, CL_IMAGE_ELEMENT_SIZE, sizeof(element_size),
                           &element_size, nullptr);
    test_error(error, "Error clGetImageInfo");

    error = clReleaseMemObject(image);
    test_error(error, "Unable to release image");

    if (imageType == CL_MEM_OBJECT_IMAGE1D_BUFFER)
    {
        error = clReleaseMemObject(buffer);
        test_error(error, "Unable to release buffer");
    }

    return element_size;
}

static inline void image_desc_init(cl_image_desc* desc,
                                   cl_mem_object_type imageType)
{
    desc->image_type = imageType;
    desc->image_width = TEST_IMAGE_SIZE;
    if (CL_MEM_OBJECT_IMAGE1D_BUFFER != imageType
        && CL_MEM_OBJECT_IMAGE1D != imageType)
    {
        desc->image_height = TEST_IMAGE_SIZE;
    }
    if (CL_MEM_OBJECT_IMAGE3D == imageType
        || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
    {
        desc->image_depth = TEST_IMAGE_SIZE;
    }
    if (CL_MEM_OBJECT_IMAGE1D_ARRAY == imageType
        || CL_MEM_OBJECT_IMAGE2D_ARRAY == imageType)
    {
        desc->image_array_size = TEST_IMAGE_SIZE;
    }
}

#endif // TEST_CL_EXT_IMAGE_BUFFER

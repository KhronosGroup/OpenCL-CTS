//
// Copyright (c) 2024 The Khronos Group Inc.
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
#include "harness/clImageHelper.h"

#include <array>
#include <vector>
#include <memory>

static constexpr cl_mem_object_type image_types[] = {
    CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D, CL_MEM_OBJECT_IMAGE2D_ARRAY,
    CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_ARRAY
};

REGISTER_TEST(negative_enqueue_map_image)
{
    constexpr size_t image_dim = 32;

    REQUIRE_EXTENSION(CL_EXT_IMMUTABLE_MEMORY_OBJECTS);

    static constexpr cl_mem_flags mem_flags[]{
        CL_MEM_IMMUTABLE_EXT | CL_MEM_USE_HOST_PTR,
        CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR,
        CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR
    };

    static constexpr const char *mem_flags_string[]{
        "CL_MEM_IMMUTABLE_EXT | CL_MEM_USE_HOST_PTR",
        "CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR",
        "CL_MEM_IMMUTABLE_EXT | CL_MEM_COPY_HOST_PTR | "
        "CL_MEM_ALLOC_HOST_PTR"
    };

    static_assert(ARRAY_SIZE(mem_flags) == ARRAY_SIZE(mem_flags_string),
                  "mem_flags and mem_flags_string must be of the same size");

    using CLUCharPtr = std::unique_ptr<cl_uchar, decltype(&free)>;

    for (size_t index = 0; index < ARRAY_SIZE(mem_flags); ++index)
    {
        cl_mem_flags mem_flag = mem_flags[index];

        log_info("Testing memory flag: %s\n", mem_flags_string[index]);
        for (cl_mem_object_type image_type : image_types)
        {
            // find supported image formats
            cl_uint num_formats = 0;

            cl_int error = clGetSupportedImageFormats(
                context, mem_flag, image_type, 0, nullptr, &num_formats);
            test_error(error,
                       "clGetSupportedImageFormats failed to return supported "
                       "formats");

            std::vector<cl_image_format> formats(num_formats);
            error = clGetSupportedImageFormats(context, mem_flag, image_type,
                                               num_formats, formats.data(),
                                               nullptr);
            test_error(error,
                       "clGetSupportedImageFormats failed to return supported "
                       "formats");

            clMemWrapper image;
            for (cl_image_format &fmt : formats)
            {
                log_info("Testing %s %s\n",
                         GetChannelOrderName(fmt.image_channel_order),
                         GetChannelTypeName(fmt.image_channel_data_type));

                RandomSeed seed(gRandomSeed);
                size_t origin[3] = { 0, 0, 0 };
                size_t region[3] = { image_dim, image_dim, image_dim };
                switch (image_type)
                {
                    case CL_MEM_OBJECT_IMAGE1D: {
                        const size_t pixel_size = get_pixel_size(&fmt);
                        const size_t image_size =
                            image_dim * pixel_size * sizeof(cl_uchar);
                        CLUCharPtr imgptr{ static_cast<cl_uchar *>(
                                               create_random_data(kUChar, seed,
                                                                  image_size)),
                                           free };
                        image =
                            create_image_1d(context, mem_flag, &fmt, image_dim,
                                            0, imgptr.get(), nullptr, &error);
                        region[1] = 1;
                        region[2] = 1;
                        break;
                    }
                    case CL_MEM_OBJECT_IMAGE2D: {
                        const size_t pixel_size = get_pixel_size(&fmt);
                        const size_t image_size = image_dim * image_dim
                            * pixel_size * sizeof(cl_uchar);
                        CLUCharPtr imgptr{ static_cast<cl_uchar *>(
                                               create_random_data(kUChar, seed,
                                                                  image_size)),
                                           free };
                        image =
                            create_image_2d(context, mem_flag, &fmt, image_dim,
                                            image_dim, 0, imgptr.get(), &error);
                        region[2] = 1;
                        break;
                    }
                    case CL_MEM_OBJECT_IMAGE3D: {
                        const size_t pixel_size = get_pixel_size(&fmt);
                        const size_t image_size = image_dim * image_dim
                            * image_dim * pixel_size * sizeof(cl_uchar);
                        CLUCharPtr imgptr{ static_cast<cl_uchar *>(
                                               create_random_data(kUChar, seed,
                                                                  image_size)),
                                           free };
                        image = create_image_3d(context, mem_flag, &fmt,
                                                image_dim, image_dim, image_dim,
                                                0, 0, imgptr.get(), &error);
                        break;
                    }
                    case CL_MEM_OBJECT_IMAGE1D_ARRAY: {
                        const size_t pixel_size = get_pixel_size(&fmt);
                        const size_t image_size = image_dim * image_dim
                            * pixel_size * sizeof(cl_uchar);
                        CLUCharPtr imgptr{ static_cast<cl_uchar *>(
                                               create_random_data(kUChar, seed,
                                                                  image_size)),
                                           free };
                        image = create_image_1d_array(context, mem_flag, &fmt,
                                                      image_dim, image_dim, 0,
                                                      0, imgptr.get(), &error);
                        region[1] = 1;
                        region[2] = 1;
                        break;
                    }
                    case CL_MEM_OBJECT_IMAGE2D_ARRAY: {
                        const size_t pixel_size = get_pixel_size(&fmt);
                        const size_t image_size = image_dim * image_dim
                            * image_dim * pixel_size * sizeof(cl_uchar);
                        CLUCharPtr imgptr{ static_cast<cl_uchar *>(
                                               create_random_data(kUChar, seed,
                                                                  image_size)),
                                           free };
                        image = create_image_2d_array(
                            context, mem_flag, &fmt, image_dim, image_dim,
                            image_dim, 0, 0, imgptr.get(), &error);
                        region[2] = 1;
                        break;
                    }
                }
                test_error(error, "Failed to create image");

                void *map = clEnqueueMapImage(
                    queue, image, CL_TRUE, CL_MAP_WRITE, origin, region,
                    nullptr, nullptr, 0, nullptr, nullptr, &error);

                constexpr const char *write_err_msg =
                    "clEnqueueMapImage should return CL_INVALID_OPERATION "
                    "when: \"image has been created with CL_MEM_IMMUTABLE_EXT "
                    "and CL_MAP_WRITE is set in map_flags\"";
                test_assert_error(map == nullptr, write_err_msg);
                test_failure_error_ret(error, CL_INVALID_OPERATION,
                                       write_err_msg, TEST_FAIL);

                map = clEnqueueMapImage(queue, image, CL_TRUE,
                                        CL_MAP_WRITE_INVALIDATE_REGION, origin,
                                        region, nullptr, nullptr, 0, nullptr,
                                        nullptr, &error);

                constexpr const char *write_invalidate_err_msg =
                    "clEnqueueMapImage should return CL_INVALID_OPERATION "
                    "when: \"image has been created with CL_MEM_IMMUTABLE_EXT "
                    "and CL_MAP_WRITE_INVALIDATE_REGION is set in map_flags\"";
                test_assert_error(map == nullptr, write_invalidate_err_msg);
                test_failure_error_ret(error, CL_INVALID_OPERATION,
                                       write_invalidate_err_msg, TEST_FAIL);
            }
        }
    }

    return TEST_PASS;
}

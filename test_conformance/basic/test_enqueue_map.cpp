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
#include "harness/compat.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "harness/conversions.h"
#include "harness/imageHelpers.h"
#include "harness/typeWrappers.h"

// clang-format off
namespace {

struct ScopeGuard
{
    ScopeGuard(cl_command_queue q, cl_uchar *region, cl_mem mo)
        : queue(q), mapped_region(region), mem_object(mo)
    {}
    ~ScopeGuard()
    {
        if (queue != nullptr && mapped_region!=nullptr && mem_object!=nullptr)
        {
            // Unmap
            cl_int error = clEnqueueUnmapMemObject(queue, mem_object, mapped_region,
                                            0, NULL, NULL);
            if(error!=CL_SUCCESS)
                log_error("Unable to unmap buffer %d\n", error);

            queue=nullptr;
            mapped_region=nullptr;
            mem_object=nullptr;
        }
    }

    cl_command_queue queue;
    cl_uchar *mapped_region;
    cl_mem mem_object;
};

const cl_mem_flags flag_set[] = {
  CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
  CL_MEM_USE_HOST_PTR,
  CL_MEM_COPY_HOST_PTR,
  CL_MEM_USE_HOST_PTR | CL_MEM_IMMUTABLE_EXT,
  CL_MEM_COPY_HOST_PTR | CL_MEM_IMMUTABLE_EXT,
  CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_IMMUTABLE_EXT,
  0
};

const char *flag_set_names[] = {
  "CL_MEM_ALLOC_HOST_PTR",
  "CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR",
  "CL_MEM_USE_HOST_PTR",
  "CL_MEM_COPY_HOST_PTR",
  "CL_MEM_USE_HOST_PTR | CL_MEM_IMMUTABLE_EXT",
  "CL_MEM_COPY_HOST_PTR | CL_MEM_IMMUTABLE_EXT",
  "CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_IMMUTABLE_EXT",
  "0"
};

static constexpr cl_mem_object_type image_types[] = {
    CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D,
    CL_MEM_OBJECT_IMAGE2D_ARRAY, CL_MEM_OBJECT_IMAGE1D,
    CL_MEM_OBJECT_IMAGE1D_ARRAY
};

static constexpr const char *image_types_names[]{
    "CL_MEM_OBJECT_IMAGE2D", "CL_MEM_OBJECT_IMAGE3D",
    "CL_MEM_OBJECT_IMAGE2D_ARRAY", "CL_MEM_OBJECT_IMAGE1D",
    "CL_MEM_OBJECT_IMAGE1D_ARRAY"
};

constexpr size_t max_3D_image_size = 64;
constexpr size_t max_bytes_per_pixel = 4 * sizeof(cl_ulong);
constexpr size_t max_image_data_size =
    max_3D_image_size * max_3D_image_size * max_3D_image_size * max_bytes_per_pixel;
constexpr size_t num_test_per_image_type = 4;
}
// clang-format on

REGISTER_TEST(enqueue_map_buffer)
{
    int error;
    constexpr size_t bufferSize = 256 * 256;
    MTdataHolder d{ gRandomSeed };
    BufferOwningPtr<cl_char> hostPtrData{ malloc(bufferSize) };
    BufferOwningPtr<cl_char> referenceData{ malloc(bufferSize) };
    BufferOwningPtr<cl_char> finalData{ malloc(bufferSize) };

    for (size_t src_flag_id = 0; src_flag_id < ARRAY_SIZE(flag_set);
         src_flag_id++)
    {
        clMemWrapper memObject;
        log_info("Testing with cl_mem_flags src: %s\n",
                 flag_set_names[src_flag_id]);

        if ((flag_set[src_flag_id] & CL_MEM_IMMUTABLE_EXT)
            && !is_extension_available(device,
                                       "cl_ext_immutable_memory_objects"))
        {
            log_info("Device does not support CL_MEM_IMMUTABLE_EXT. "
                     "Skipping the memory flag.\n");
            continue;
        }

        generate_random_data(kChar, (unsigned int)bufferSize, d, hostPtrData);
        memcpy(referenceData, hostPtrData, bufferSize);

        void *hostPtr = nullptr;
        cl_mem_flags flags = flag_set[src_flag_id];
        const bool is_immutable_buffer = flags & CL_MEM_IMMUTABLE_EXT;
        bool hasHostPtr =
            (flags & CL_MEM_USE_HOST_PTR) || (flags & CL_MEM_COPY_HOST_PTR);
        if (hasHostPtr) hostPtr = hostPtrData;
        memObject = clCreateBuffer(context, flags, bufferSize, hostPtr, &error);
        test_error(error, "Unable to create testing buffer");

        if (!hasHostPtr && !is_immutable_buffer)
        {
            error =
                clEnqueueWriteBuffer(queue, memObject, CL_TRUE, 0, bufferSize,
                                     hostPtrData, 0, NULL, NULL);
            test_error(error, "clEnqueueWriteBuffer failed");
        }

        for (int i = 0; i < 128; i++)
        {

            size_t offset = (size_t)random_in_range(0, (int)bufferSize - 1, d);
            size_t length =
                (size_t)random_in_range(1, (int)(bufferSize - offset), d);

            cl_char *mappedRegion = (cl_char *)clEnqueueMapBuffer(
                queue, memObject, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, offset,
                length, 0, NULL, NULL, &error);

            // Mapping should fail if the buffer is immutable
            if (is_immutable_buffer)
            {
                test_failure_error_ret(
                    error, CL_INVALID_OPERATION,
                    "clEnqueueMapBuffer call was expected to fail "
                    "with CL_INVALID_OPERATION",
                    TEST_FAIL);
                continue;
            }
            else if (error != CL_SUCCESS)
            {
                print_error(error, "clEnqueueMapBuffer call failed");
                log_error("\tOffset: %d  Length: %d\n", (int)offset,
                          (int)length);
                return -1;
            }

            // Write into the region
            for (size_t j = 0; j < length; j++)
            {
                cl_char spin = (cl_char)genrand_int32(d);

                // Test read AND write in one swipe
                cl_char value = mappedRegion[j];
                value = spin - value;
                mappedRegion[j] = value;

                // Also update the initial data array
                value = referenceData[offset + j];
                value = spin - value;
                referenceData[offset + j] = value;
            }

            // Unmap
            error = clEnqueueUnmapMemObject(queue, memObject, mappedRegion, 0,
                                            NULL, NULL);
            test_error(error, "Unable to unmap buffer");
        }

        // Final validation: read actual values of buffer and compare against
        // our reference
        error = clEnqueueReadBuffer(queue, memObject, CL_TRUE, 0, bufferSize,
                                    finalData, 0, NULL, NULL);
        test_error(error, "Unable to read results");

        if (is_immutable_buffer && !hasHostPtr)
        {
            continue;
        }

        for (size_t q = 0; q < bufferSize; q++)
        {
            if (referenceData[q] != finalData[q])
            {
                log_error(
                    "ERROR: Sample %d did not validate! Got %d, expected %d\n",
                    (int)q, (int)finalData[q], (int)referenceData[q]);
                return -1;
            }
        }
    } // cl_mem flags

    return 0;
}

cl_int create_image_type(clMemWrapper &memObject, const cl_context context,
                         cl_mem_object_type image_type,
                         const cl_mem_flags mem_flag,
                         const cl_image_format &fmt, size_t *img_region,
                         const cl_uchar *data)
{
    cl_int error = CL_SUCCESS;
    switch (image_type)
    {
        case CL_MEM_OBJECT_IMAGE1D: {
            size_t image_size = std::min(
                size_t(256), max_image_data_size / max_bytes_per_pixel);
            memObject = create_image_1d(context, mem_flag, &fmt, image_size, 0,
                                        (void *)data, nullptr, &error);
            test_error(error, "Unable to create testing buffer");
            img_region[0] = image_size;
            break;
        }
        case CL_MEM_OBJECT_IMAGE2D: {
            size_t image_size = std::min(
                size_t(256),
                (size_t)std::sqrt(max_image_data_size / max_bytes_per_pixel));
            memObject = create_image_2d(context, mem_flag, &fmt, image_size,
                                        image_size, 0, (void *)data, &error);
            img_region[0] = img_region[1] = image_size;
            break;
        }
        case CL_MEM_OBJECT_IMAGE3D: {
            size_t image_size = std::min(
                size_t(256),
                (size_t)std::cbrt(max_image_data_size / max_bytes_per_pixel));
            memObject =
                create_image_3d(context, mem_flag, &fmt, image_size, image_size,
                                image_size, 0, 0, (void *)data, &error);
            img_region[0] = img_region[1] = img_region[2] = image_size;
            break;
        }
        case CL_MEM_OBJECT_IMAGE1D_ARRAY: {
            size_t image_size = std::min(
                size_t(256),
                (size_t)std::sqrt(max_image_data_size / max_bytes_per_pixel));
            memObject =
                create_image_1d_array(context, mem_flag, &fmt, image_size,
                                      image_size, 0, 0, (void *)data, &error);
            img_region[0] = img_region[1] = image_size;
            break;
        }
        case CL_MEM_OBJECT_IMAGE2D_ARRAY: {
            size_t image_size = std::min(
                size_t(256),
                (size_t)std::cbrt(max_image_data_size / max_bytes_per_pixel));
            memObject = create_image_2d_array(
                context, mem_flag, &fmt, image_size, image_size, image_size, 0,
                0, (void *)data, &error);
            img_region[0] = img_region[1] = img_region[2] = image_size;
            break;
        }
    }
    return error;
}

void random_region_coords(size_t *offset, size_t *region,
                          const cl_mem_object_type image_type,
                          const size_t &imageSize, const MTdataHolder &d)
{
    offset[0] = (size_t)random_in_range(0, (int)imageSize - 1, d);
    region[0] = (size_t)random_in_range(1, (int)(imageSize - offset[0] - 1), d);
    offset[1] = (size_t)random_in_range(0, (int)imageSize - 1, d);
    region[1] = (size_t)random_in_range(1, (int)(imageSize - offset[1] - 1), d);
    offset[2] = (size_t)random_in_range(0, (int)imageSize - 1, d);
    region[2] = (size_t)random_in_range(1, (int)(imageSize - offset[2] - 1), d);

    switch (image_type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        case CL_MEM_OBJECT_IMAGE1D: {
            offset[1] = offset[2] = 0;
            region[1] = region[2] = 1;
            break;
        }
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        case CL_MEM_OBJECT_IMAGE2D: {
            offset[2] = 0;
            region[2] = 1;
            break;
        }
    }
}

REGISTER_TEST(enqueue_map_image)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    MTdataHolder d{ gRandomSeed };
    std::vector<cl_uchar> hostPtrData(max_image_data_size, 0);
    std::vector<cl_uchar> referenceData(max_image_data_size, 0);
    std::vector<cl_uchar> finalData(max_image_data_size, 0);

    generate_random_data(kUChar, (unsigned int)max_image_data_size, d,
                         hostPtrData.data());

    for (size_t src_flag_id = 0; src_flag_id < ARRAY_SIZE(flag_set);
         src_flag_id++)
    {
        log_info("Testing with cl_mem_flags src: %s\n",
                 flag_set_names[src_flag_id]);

        for (size_t image_type_ind = 0;
             image_type_ind < ARRAY_SIZE(image_types); image_type_ind++)
        {
            cl_mem_object_type image_type = image_types[image_type_ind];
            log_info("Testing with cl_mem_object_type src: %s\n",
                     image_types_names[image_type_ind]);

            if ((flag_set[src_flag_id] & CL_MEM_IMMUTABLE_EXT)
                && !is_extension_available(device,
                                           "cl_ext_immutable_memory_objects"))
            {
                log_info("Device does not support CL_MEM_IMMUTABLE_EXT. "
                         "Skipping the memory flag.\n");
                continue;
            }

            // find supported image formats
            cl_uint num_formats = 0;
            cl_mem_flags mem_flag = flag_set[src_flag_id];

            bool is_immutable_image = mem_flag & CL_MEM_IMMUTABLE_EXT;
            bool hasHostPtr = (mem_flag & CL_MEM_USE_HOST_PTR)
                || (mem_flag & CL_MEM_COPY_HOST_PTR);

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

            for (cl_image_format &fmt : formats)
            {
                memcpy(referenceData.data(), hostPtrData.data(),
                       max_image_data_size);

                const size_t pixel_size = get_pixel_size(&fmt);

                size_t img_region[3] = { 1, 1, 1 };
                clMemWrapper memObject;
                error =
                    create_image_type(memObject, context, image_type, mem_flag,
                                      fmt, img_region, hostPtrData.data());
                test_error(error, "create_image_type failed");
                size_t image_size = img_region[0];

                if (!hasHostPtr && !is_immutable_image)
                {
                    size_t write_origin[3] = { 0, 0, 0 };
                    error = clEnqueueWriteImage(
                        queue, memObject, CL_TRUE, write_origin, img_region, 0,
                        0, hostPtrData.data(), 0, NULL, NULL);
                    test_error(error, "Unable to write to testing buffer");
                }

                for (int i = 0; i < num_test_per_image_type; i++)
                {
                    size_t offset[3], region[3];
                    random_region_coords(offset, region, image_type, image_size,
                                         d);

                    size_t rowPitch = 0, slicePitch = 0;

                    cl_uchar *mappedRegion = (cl_uchar *)clEnqueueMapImage(
                        queue, memObject, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                        offset, region, &rowPitch, &slicePitch, 0, NULL, NULL,
                        &error);
                    ScopeGuard sg(queue, mappedRegion, memObject);

                    if (is_immutable_image)
                    {
                        test_failure_error_ret(
                            error, CL_INVALID_OPERATION,
                            "clEnqueueMapImage call was expected to fail "
                            "with CL_INVALID_OPERATION",
                            TEST_FAIL);
                        continue;
                    }
                    else if (error != CL_SUCCESS)
                    {
                        print_error(error, "clEnqueueMapImage call failed");
                        log_error("\tOffset: %d,%d  Region: %d,%d\n",
                                  (int)offset[0], (int)offset[1],
                                  (int)region[0], (int)region[1]);
                        return -1;
                    }

                    // Read and write into the region
                    for (size_t z = 0; z < region[2]; z++)
                    {
                        for (size_t y = 0; y < region[1]; y++)
                        {
                            for (size_t x = 0; x < region[0]; x++)
                            {
                                size_t x_off = offset[0] + x;
                                size_t y_off = offset[1] + y;
                                size_t z_off = offset[2] + z;
                                size_t ref_loc =
                                    (z_off * image_size * image_size
                                     + y_off * image_size + x_off);
                                cl_uchar *pixel =
                                    &mappedRegion
                                        [z * slicePitch + y * rowPitch
                                         + x * pixel_size]; // is this correct ?
                                for (size_t i = 0; i < pixel_size; i++)
                                {
                                    if (pixel[i]
                                        != referenceData[ref_loc * pixel_size
                                                         + i])
                                    {
                                        log_error(
                                            "ERROR: Sample %d (coord "
                                            "%zu,%zu,%zu) "
                                            "did not validate! Got "
                                            "%d, expected %d\n",
                                            (int)ref_loc, x, y, z,
                                            (int)pixel[i],
                                            (int)referenceData[ref_loc
                                                                   * pixel_size
                                                               + i]);
                                        return -1;
                                    }

                                    pixel[i] = 0;
                                    referenceData[ref_loc * pixel_size + i] = 0;
                                }
                            }
                        }
                    }
                }

                // Final validation: read actual values of buffer and compare
                // against our reference
                size_t zero_origin[3] = { 0, 0, 0 };
                error = clEnqueueReadImage(queue, memObject, CL_TRUE,
                                           zero_origin, img_region, 0, 0,
                                           finalData.data(), 0, NULL, NULL);
                test_error(error, "Unable to read results");

                if (is_immutable_image && !hasHostPtr)
                {
                    continue;
                }

                for (size_t z = 0; z < img_region[2]; z++)
                {
                    for (size_t y = 0; y < img_region[1]; y++)
                    {
                        for (size_t x = 0; x < img_region[0]; x++)
                        {
                            size_t loc = (z * image_size * image_size
                                          + y * image_size + x);
                            for (size_t i = 0; i < pixel_size; i++)
                            {
                                if (referenceData[loc * pixel_size + i]
                                    != finalData[loc * pixel_size + i])
                                {
                                    log_error(
                                        "Final validation error: sample %d "
                                        "(coord %zu,%zu,%zu) did not validate! "
                                        "Got "
                                        "%d, expected %d\n",
                                        (int)loc, x, y, z, (int)finalData[loc],
                                        (int)referenceData[loc]);
                                    return -1;
                                }
                            }
                        }
                    }
                }
            }
        }
    } // cl_mem_flags

    return 0;
}

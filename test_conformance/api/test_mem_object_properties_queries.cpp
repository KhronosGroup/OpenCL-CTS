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
#include "testBase.h"
#include "harness/propertyHelpers.h"
#include "harness/typeWrappers.h"
#include <vector>
#include <algorithm>

typedef enum
{
    image,
    image_with_properties,
    buffer,
    buffer_with_properties,
    subbuffer,
} test_type;

struct test_data
{
    test_type type;
    std::vector<cl_mem_properties> properties;
    std::string description;
    cl_kernel kernel;
};

static int create_object_and_check_properties(cl_context context,
                                              clMemWrapper& test_object,
                                              test_data test_case,
                                              cl_mem_flags flags,
                                              std::vector<cl_uint> local_data,
                                              cl_uint size_x, cl_uint size_y)
{
    cl_int error = CL_SUCCESS;

    switch (test_case.type)
    {
        case image: {
            cl_image_format format = { 0 };
            format.image_channel_order = CL_RGBA;
            format.image_channel_data_type = CL_UNSIGNED_INT32;
            test_object = clCreateImage2D(context, flags, &format, size_x,
                                          size_y, 0, local_data.data(), &error);
            test_error(error, "clCreateImage2D failed");
        }
        break;
        case image_with_properties: {
            cl_image_format format = { 0 };
            format.image_channel_order = CL_RGBA;
            format.image_channel_data_type = CL_UNSIGNED_INT32;
            cl_image_desc desc = { 0 };
            desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width = size_x;
            desc.image_height = size_y;

            if (test_case.properties.size() == 0)
            {
                test_object = clCreateImageWithProperties(
                    context, NULL, flags, &format, &desc, local_data.data(),
                    &error);
            }
            else
            {
                test_object = clCreateImageWithProperties(
                    context, test_case.properties.data(), flags, &format, &desc,
                    local_data.data(), &error);
            }
            test_error(error, "clCreateImageWithProperties failed");
        }
        break;
        case buffer: {
            test_object = clCreateBuffer(context, flags,
                                         local_data.size() * sizeof(cl_uint),
                                         local_data.data(), &error);
            test_error(error, "clCreateBuffer failed");
        }
        case buffer_with_properties: {
            if (test_case.properties.size() == 0)
            {
                test_object = clCreateBufferWithProperties(
                    context, NULL, flags, local_data.size() * sizeof(cl_uint),
                    local_data.data(), &error);
            }
            else
            {
                test_object = clCreateBufferWithProperties(
                    context, test_case.properties.data(), flags,
                    local_data.size() * sizeof(cl_uint), local_data.data(),
                    &error);
            }
            test_error(error, "clCreateBufferWithProperties failed.");
        }
        break;
        case subbuffer: {
            clMemWrapper parent_object;
            if (test_case.properties.size() == 0)
            {
                parent_object = clCreateBufferWithProperties(
                    context, NULL, flags, local_data.size() * sizeof(cl_uint),
                    local_data.data(), &error);
            }
            else
            {
                parent_object = clCreateBufferWithProperties(
                    context, test_case.properties.data(), flags,
                    local_data.size() * sizeof(cl_uint), local_data.data(),
                    &error);
            }
            test_error(error, "clCreateBufferWithProperties failed.");

            cl_mem_flags subbuffer_flags = flags
                & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);

            cl_buffer_region region = { 0 };
            region.origin = 0;
            region.size = local_data.size() * sizeof(cl_uint);
            test_object = clCreateSubBuffer(parent_object, subbuffer_flags,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region, &error);
            test_error(error, "clCreateSubBuffer failed.");
        }
        break;
        default: log_error("Unknown test type!"); return TEST_FAIL;
    }

    std::vector<cl_mem_properties> check_properties;
    size_t set_size = 0;

    error =
        clGetMemObjectInfo(test_object, CL_MEM_PROPERTIES, 0, NULL, &set_size);
    test_error(error,
               "clGetMemObjectInfo failed asking for CL_MEM_PROPERTIES size.");

    // Buffers, subbuffers, and images must return no properties.
    if (test_case.type == buffer || test_case.type == subbuffer
        || test_case.type == image)
    {
        if (set_size == 0)
        {
            return TEST_PASS;
        }
        else
        {
            log_error("Queried properties must have size equal to zero for "
                      "buffers, subbuffers, and images.");
            return TEST_FAIL;
        }
    }

    if (set_size == 0 && test_case.properties.size() == 0)
    {
        return TEST_PASS;
    }
    if (set_size != test_case.properties.size() * sizeof(cl_mem_properties))
    {
        log_error("ERROR: CL_MEM_PROPERTIES size is %zu, expected %zu.\n",
                  set_size,
                  test_case.properties.size() * sizeof(cl_queue_properties));
        return TEST_FAIL;
    }

    cl_uint number_of_props = set_size / sizeof(cl_mem_properties);
    check_properties.resize(number_of_props);
    error = clGetMemObjectInfo(test_object, CL_MEM_PROPERTIES, set_size,
                               check_properties.data(), NULL);
    test_error(error,
               "clGetMemObjectInfo failed asking for CL_MEM_PROPERTIES.");

    error = compareProperties(check_properties, test_case.properties);
    return error;
}

static int run_test_query_properties(cl_context context, cl_command_queue queue,
                                     test_data test_case)
{
    int error = CL_SUCCESS;
    log_info("\nTC description: %s\n", test_case.description.c_str());

    clMemWrapper obj_src;
    clMemWrapper obj_dst;
    clEventWrapper event;
    MTdata init_generator = init_genrand(gRandomSeed);
    cl_mem_flags flags;
    cl_uint size_x = 4;
    cl_uint size_y = 4;
    size_t size = size_x * size_y * 4;
    size_t global_dim[2] = { size_x, size_y };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { size_x, size_y, 1 };

    std::vector<cl_uint> src_data(size);
    std::vector<cl_uint> dst_data(size);

    generate_random_data(kUInt, size, init_generator, src_data.data());
    generate_random_data(kUInt, size, init_generator, dst_data.data());
    free_mtdata(init_generator);
    init_generator = NULL;

    flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    error = create_object_and_check_properties(context, obj_src, test_case,
                                               flags, src_data, size_x, size_y);
    test_error(error, "create_object_and_check_properties obj_src failed.");

    flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    error = create_object_and_check_properties(context, obj_dst, test_case,
                                               flags, dst_data, size_x, size_y);
    test_error(error, "create_object_and_check_properties obj_dst failed.");

    error = clSetKernelArg(test_case.kernel, 0, sizeof(obj_src), &obj_src);
    test_error(error, "clSetKernelArg 0 failed.");

    error = clSetKernelArg(test_case.kernel, 1, sizeof(obj_dst), &obj_dst);
    test_error(error, "clSetKernelArg 1 failed.");
    switch (test_case.type)
    {
        case image:
        case image_with_properties: {
            error = clEnqueueNDRangeKernel(queue, test_case.kernel, 2, NULL,
                                           global_dim, NULL, 0, NULL, &event);
            test_error(error, "clEnqueueNDRangeKernel failed.");

            error = clWaitForEvents(1, &event);
            test_error(error, "clWaitForEvents failed.");

            error = clEnqueueReadImage(queue, obj_dst, CL_TRUE, origin, region,
                                       0, 0, dst_data.data(), 0, NULL, NULL);
            test_error(error, "clEnqueueReadImage failed.");
        }
        break;
        case buffer:
        case buffer_with_properties:
        case subbuffer: {
            error = clEnqueueNDRangeKernel(queue, test_case.kernel, 1, NULL,
                                           &size, NULL, 0, NULL, &event);
            test_error(error, "clEnqueueNDRangeKernel failed.");

            error = clWaitForEvents(1, &event);
            test_error(error, "clWaitForEvents failed.");

            error = clEnqueueReadBuffer(queue, obj_dst, CL_TRUE, 0,
                                        dst_data.size() * sizeof(cl_uint),
                                        dst_data.data(), 0, NULL, NULL);
            test_error(error, "clEnqueueReadBuffer failed.");
        }
        break;
        default: log_error("Unknown test type!"); return TEST_FAIL;
    }

    for (size_t i = 0; i < size; ++i)
    {
        if (dst_data[i] != src_data[i])
        {
            log_error("ERROR: Output results mismatch.");
            return TEST_FAIL;
        }
    }

    log_info("TC result: passed\n");
    return TEST_PASS;
}

REGISTER_TEST_VERSION(image_properties_queries, Version(3, 0))
{
    int error = CL_SUCCESS;
    cl_bool supports_images = CL_TRUE;

    error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(supports_images), &supports_images, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

    if (supports_images == CL_FALSE)
    {
        log_info("No image support on current device - skipped\n");
        return TEST_SKIPPED_ITSELF;
    }

    clProgramWrapper program;
    clKernelWrapper kernel;

    const char* kernel_src = R"CLC(
        __kernel void data_copy(read_only image2d_t src, write_only image2d_t dst)
        {
            int tid_x = get_global_id(0);
            int tid_y = get_global_id(1);
            int2 coords = (int2)(tid_x, tid_y);
            uint4 val = read_imageui(src, coords);
            write_imageui(dst, coords, val);

        }
        )CLC";

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_src, "data_copy");
    test_error(error, "create_single_kernel_helper failed");

    std::vector<test_data> test_cases;
    test_cases.push_back({ image, {}, "regular image", kernel });
    test_cases.push_back(
        { image_with_properties, { 0 }, "image, 0 properties", kernel });
    test_cases.push_back(
        { image_with_properties, {}, "image, NULL properties", kernel });

    for (auto test_case : test_cases)
    {
        error |= run_test_query_properties(context, queue, test_case);
    }

    return error;
}

REGISTER_TEST_VERSION(buffer_properties_queries, Version(3, 0))
{
    int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    const char* kernel_src = R"CLC(
        __kernel void data_copy(__global int *src, __global int *dst)
        {
            int  tid = get_global_id(0);

            dst[tid] = src[tid];

        }
        )CLC";
    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_src, "data_copy");
    test_error(error, "create_single_kernel_helper failed");

    std::vector<test_data> test_cases;
    test_cases.push_back({ buffer, {}, "regular buffer", kernel });
    test_cases.push_back(
        { buffer_with_properties, { 0 }, "buffer with 0 properties", kernel });
    test_cases.push_back(
        { buffer_with_properties, {}, "buffer with NULL properties", kernel });
    test_cases.push_back(
        { subbuffer, { 0 }, "subbuffer with 0 properties", kernel });
    test_cases.push_back(
        { subbuffer, {}, "subbuffer with NULL properties", kernel });

    for (auto test_case : test_cases)
    {
        error |= run_test_query_properties(context, queue, test_case);
    }

    return error;
}

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
#include "harness/typeWrappers.h"
#include <vector>

using namespace std;

typedef enum
{
    image,
    buffer
} mem_obj_type;

typedef enum
{
    NON_NULL_PATH,
    NULL_PATH
} query_type;

struct test_data
{
    mem_obj_type obj_t;
    query_type query_t;
    std::vector<cl_mem_properties> properties;
    std::string description;
    std::string src;
    std::string kernel_name;
};


int create_object_and_check_properties(cl_context context,
                                       clMemWrapper& test_object,
                                       test_data test_case, cl_mem_flags flags,
                                       std::vector<cl_uint> local_data,
                                       cl_uint size_x, cl_uint size_y)
{
    int error = CL_SUCCESS;
    size_t set_size;
    std::vector<cl_mem_properties> object_properties_check;
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT32;
    cl_image_desc desc;
    memset(&desc, 0x0, sizeof(cl_image_desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = size_x;
    desc.image_height = size_y;

    if (test_case.obj_t == image)
    {
        if (test_case.query_t == NULL_PATH)
        {
            test_object =
                clCreateImageWithProperties(context, NULL, flags, &format,
                                            &desc, local_data.data(), &error);
        }
        else
        {
            test_object = clCreateImageWithProperties(
                context, test_case.properties.data(), flags, &format, &desc,
                local_data.data(), &error);
        }
        test_error(error, "clCreateImageWithProperties failed");
    }
    if (test_case.obj_t == buffer)
    {
        if (test_case.query_t == NULL_PATH)
        {
            test_object = clCreateBufferWithProperties(
                context, NULL, flags, local_data.size() * sizeof(cl_uint),
                local_data.data(), &error);
        }
        else
        {
            test_object = clCreateBufferWithProperties(
                context, test_case.properties.data(), flags,
                local_data.size() * sizeof(cl_uint), local_data.data(), &error);
        }

        test_error(error, "clCreateBufferWithProperties failed.");
    }
    clGetMemObjectInfo(test_object, CL_MEM_PROPERTIES, 0, NULL, &set_size);
    test_error(error,
               "clGetMemObjectInfo failed asking for CL_MEM_PROPERTIES.");
    if (test_case.query_t == NULL_PATH && set_size == 0)
    {
        return TEST_PASS;
    }
    if (test_case.query_t == NON_NULL_PATH && set_size == 0)
    {
        log_error("clGetMemObjectInfo - invalid value returned");
        return TEST_FAIL;
    }
    cl_uint number_of_props = set_size / sizeof(cl_mem_properties);
    object_properties_check.resize(number_of_props);
    clGetMemObjectInfo(test_object, CL_MEM_PROPERTIES, set_size,
                       object_properties_check.data(), NULL);
    test_error(error,
               "clGetMemObjectInfo failed asking for CL_MEM_PROPERTIES.");

    if (object_properties_check != test_case.properties)
    {
        log_error("clGetMemObjectInfo returned different properties than was "
                  "set initially.\n");
        error = TEST_FAIL;
    }
    return error;
}


int run_test_query_properties(cl_context context, cl_command_queue queue,
                              test_data test_case)
{
    int error = CL_SUCCESS;
    log_info("test case description: %s\n", test_case.description.c_str());

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper obj_src;
    clMemWrapper obj_dst;
    clEventWrapper event;
    MTdata d;

    cl_mem_flags flags;
    cl_uint size_x;
    cl_uint size_y;
    size_t size;
    size_t global_dim[2];

    size_x = 4;
    size_y = 4;
    size = size_x * size_y * 4;
    global_dim[0] = size_x;
    global_dim[1] = size_y;
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { size_x, size_y, 1 };

    std::vector<cl_uint> src_data(size);
    std::vector<cl_uint> dst_data(size);
    d = init_genrand(gRandomSeed);

    generate_random_data(kUInt, size, d, src_data.data());
    generate_random_data(kUInt, size, d, dst_data.data());
    free_mtdata(d);
    d = NULL;
    const char* kernel_src = test_case.src.c_str();
    error = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &kernel_src,
        test_case.kernel_name.c_str(), "-cl-std=CL3.0");

    test_error(error, "create_single_kernel_helper failed");

    flags = (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    error = create_object_and_check_properties(context, obj_src, test_case,
                                               flags, src_data, size_x, size_y);
    test_error(error, "create_object_and_check_properties obj_src failed.");

    flags = (cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR);
    error = create_object_and_check_properties(context, obj_dst, test_case,
                                               flags, dst_data, size_x, size_y);
    test_error(error, "create_object_and_check_properties obj_dst failed.");

    error = clSetKernelArg(kernel, 0, sizeof(obj_src), &obj_src);
    test_error(error, "clSetKernelArg 0 failed.");

    error = clSetKernelArg(kernel, 1, sizeof(obj_dst), &obj_dst);
    test_error(error, "clSetKernelArg 1 failed.");
    if (test_case.obj_t == image)
    {
        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_dim, NULL,
                                       0, NULL, &event);
        test_error(error, "clEnqueueNDRangeKernel failed.");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed.");

        error = clEnqueueReadImage(queue, obj_dst, CL_TRUE, origin, region, 0,
                                   0, dst_data.data(), 0, NULL, NULL);
        test_error(error, "clEnqueueReadImage failed.");
    }
    if (test_case.obj_t == buffer)
    {
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0,
                                       NULL, &event);
        test_error(error, "clEnqueueNDRangeKernel failed.");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed.");

        error = clEnqueueReadBuffer(queue, obj_dst, CL_TRUE, 0,
                                    dst_data.size() * sizeof(cl_uint),
                                    dst_data.data(), 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed.");
    }

    for (size_t i = 0; i < size; ++i)
    {
        if (dst_data[i] != src_data[i])
        {
            log_error("ERROR: Output results mismatch.");
            return TEST_FAIL;
        }
    }

    return 0;
}


int test_image_properties_queries(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;
    std::vector<test_data> test_cases;
    std::string test_kernel = { "__kernel void data_copy(read_only image2d_t "
                                "src, write_only image2d_t dst)\n"
                                "{\n"
                                "    int tid_x = get_global_id(0);\n"
                                "    int tid_y = get_global_id(1);\n"
                                "    int2 coords = (int2)(tid_x, tid_y);\n"
                                "    uint4 val = read_imageui(src, coords);\n"
                                "    write_imageui(dst, coords, val);\n"
                                "\n"
                                "}\n" };
    test_cases.push_back({ image,
                           NON_NULL_PATH,
                           { 0 },
                           "image, 0 properties",
                           test_kernel,
                           "data_copy" });
    test_cases.push_back({ image,
                           NULL_PATH,
                           { NULL },
                           "image, NULL properties",
                           test_kernel,
                           "data_copy" });

    for (auto test_case : test_cases)
    {
        error |= run_test_query_properties(context, queue, test_case);
    }

    return error;
}

int test_buffer_properties_queries(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;
    std::vector<test_data> test_cases;
    std::string test_kernel = {
        "__kernel void data_copy(__global int *src, __global int *dst)\n"
        "{\n"
        "    int  tid = get_global_id(0);\n"
        "\n"
        "    dst[tid] = src[tid];\n"
        "\n"
        "}\n"
    };
    test_cases.push_back({ buffer,
                           NON_NULL_PATH,
                           { 0 },
                           "buffer, 0 properties",
                           test_kernel,
                           "data_copy" });
    test_cases.push_back({ buffer,
                           NULL_PATH,
                           { NULL },
                           "buffer, NULL properties",
                           test_kernel,
                           "data_copy" });

    for (auto test_case : test_cases)
    {
        error |= run_test_query_properties(context, queue, test_case);
    }

    return error;
}
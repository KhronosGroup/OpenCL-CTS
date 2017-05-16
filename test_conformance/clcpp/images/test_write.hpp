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
#ifndef TEST_CONFORMANCE_CLCPP_IMAGES_TEST_WRITE_HPP
#define TEST_CONFORMANCE_CLCPP_IMAGES_TEST_WRITE_HPP

#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common.hpp"


namespace test_images_write {

template<cl_mem_object_type ImageType, cl_channel_type ChannelType>
struct image_test : image_test_base<ImageType, ChannelType>
{
    cl_channel_order channel_order;

    image_test(cl_channel_order channel_order) :
        channel_order(channel_order)
    { }
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    std::string generate_source()
    {
        std::stringstream s;
        s << R"(
        typedef )" << type_name<typename image_test::element_type>() << R"( element_type;

        kernel void test(
            write_only )" << image_test::image_type_name() << R"(_t img,
            const global int4 *coords,
            const global element_type *input
        ) {
            const ulong gid = get_global_linear_id();

            write_image)" << image_test::function_suffix() <<
                "(img, coords[gid]." << image_test::coord_accessor() << R"(, input[gid]);
        }
        )";

        return s.str();
    }
#else
    std::string generate_source()
    {
        std::stringstream s;
        s << R"(
        #include <opencl_memory>
        #include <opencl_common>
        #include <opencl_work_item>
        #include <opencl_image>
        using namespace cl;
        )";

        s << R"(
        typedef )" << type_name<typename image_test::element_type>() <<  R"( element_type;

        kernel void test(
            )" << image_test::image_type_name() << R"(<element_type, image_access::write> img,
            const global_ptr<int4[]> coords,
            const global_ptr<element_type[]> input
        ) {
            const ulong gid = get_global_linear_id();

            img.write(coords[gid].)" << image_test::coord_accessor() << R"(, input[gid]);
        }
        )";

        return s.str();
    }
#endif

    int run(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
    {
        int error = CL_SUCCESS;

        cl_program program;
        cl_kernel kernel;

        std::string kernel_name = "test";
        std::string source = generate_source();

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
        error = create_opencl_kernel(
            context, &program, &kernel,
            source, kernel_name
        );
        RETURN_ON_ERROR(error)
        return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
        error = create_opencl_kernel(
            context, &program, &kernel,
            source, kernel_name, "-cl-std=CL2.0", false
        );
        RETURN_ON_ERROR(error)
// Normal run
#else
        error = create_opencl_kernel(
            context, &program, &kernel,
            source, kernel_name
        );
        RETURN_ON_ERROR(error)
#endif

        using element_type = typename image_test::element_type;
        using coord_type = cl_int4;
        using scalar_element_type = typename scalar_type<element_type>::type;
        using channel_type = typename image_test::channel_type;

        cl_image_format image_format;
        image_format.image_channel_order = channel_order;
        image_format.image_channel_data_type = ChannelType;

        const size_t pixel_size = get_pixel_size(&image_format);
        const size_t channel_count = get_channel_order_channel_count(image_format.image_channel_order);

        cl_image_desc image_desc;
        image_desc.image_type = ImageType;
        if (ImageType == CL_MEM_OBJECT_IMAGE1D)
        {
            image_desc.image_width = 2048;
            image_desc.image_height = 1;
            image_desc.image_depth = 1;
        }
        else if (ImageType == CL_MEM_OBJECT_IMAGE2D)
        {
            image_desc.image_width = 256;
            image_desc.image_height = 256;
            image_desc.image_depth = 1;
        }
        else if (ImageType == CL_MEM_OBJECT_IMAGE3D)
        {
            image_desc.image_width = 64;
            image_desc.image_height = 64;
            image_desc.image_depth = 64;
        }
        image_desc.image_array_size = 0;
        image_desc.image_row_pitch = image_desc.image_width * pixel_size;
        image_desc.image_slice_pitch = image_desc.image_row_pitch * image_desc.image_height;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.mem_object = NULL;

        image_descriptor image_info = create_image_descriptor(image_desc, &image_format);

        std::vector<channel_type> random_image_values = generate_input(
            image_desc.image_width * image_desc.image_height * image_desc.image_depth * channel_count,
            image_test::channel_min(), image_test::channel_max(),
            std::vector<channel_type>()
        );

        const size_t count = num_elements;

        std::vector<coord_type> coords = generate_input(
            count,
            detail::make_value<coord_type>(0),
            coord_type {
                static_cast<cl_int>(image_desc.image_width - 1),
                static_cast<cl_int>(image_desc.image_height - 1),
                static_cast<cl_int>(image_desc.image_depth - 1),
                0
            },
            std::vector<coord_type>()
        );

        std::vector<element_type> input(count);
        for (size_t i = 0; i < count; i++)
        {
            const coord_type c = coords[i];

            // Use read_image_pixel from harness/imageHelpers to fill input values
            // (it will deal with correct channels, orders etc.)
            read_image_pixel<scalar_element_type>(static_cast<void *>(random_image_values.data()), &image_info,
                c.s[0], c.s[1], c.s[2],
                input[i].s);
        }

        // image_row_pitch and image_slice_pitch must be 0, when clCreateImage is used with host_ptr = NULL
        image_desc.image_row_pitch = 0;
        image_desc.image_slice_pitch = 0;
        cl_mem img = clCreateImage(context, CL_MEM_WRITE_ONLY,
            &image_format, &image_desc, NULL, &error);
        RETURN_ON_CL_ERROR(error, "clCreateImage")

        cl_mem coords_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(coord_type) * count, static_cast<void *>(coords.data()), &error);
        RETURN_ON_CL_ERROR(error, "clCreateBuffer")

        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(element_type) * count, static_cast<void *>(input.data()), &error);
        RETURN_ON_CL_ERROR(error, "clCreateBuffer")

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 1, sizeof(coords_buffer), &coords_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 2, sizeof(input_buffer), &input_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")

        const size_t global_size = count;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

        std::vector<channel_type> image_values(image_desc.image_width * image_desc.image_height * image_desc.image_depth * channel_count);

        const size_t origin[3] = { 0 };
        const size_t region[3] = { image_desc.image_width, image_desc.image_height, image_desc.image_depth };
        error = clEnqueueReadImage(
            queue, img, CL_TRUE,
            origin, region, 0, 0,
            static_cast<void *>(image_values.data()),
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

        for (size_t i = 0; i < count; i++)
        {
            const coord_type c = coords[i];
            const element_type expected = input[i];

            element_type result;
            read_image_pixel<scalar_element_type>(static_cast<void *>(image_values.data()), &image_info,
                c.s[0], c.s[1], c.s[2],
                result.s);

            if (!are_equal(result, expected))
            {
                RETURN_ON_ERROR_MSG(-1,
                    "Writing to coordinates %s failed. Expected: %s, got: %s",
                    format_value(c).c_str(), format_value(expected).c_str(), format_value(result).c_str()
                );
            }
        }

        clReleaseMemObject(img);
        clReleaseMemObject(coords_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return error;
    }
};

template<cl_mem_object_type ImageType>
int run_test_cases(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    if (!is_test_supported(device))
        return CL_SUCCESS;

    int error = CL_SUCCESS;

    for (auto channel_order : get_channel_orders(device))
    {
        error = image_test<ImageType, CL_SIGNED_INT8>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_SIGNED_INT16>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_SIGNED_INT32>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)

        error = image_test<ImageType, CL_UNSIGNED_INT8>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_UNSIGNED_INT16>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_UNSIGNED_INT32>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)

        error = image_test<ImageType, CL_FLOAT>(channel_order)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
    }

    return error;
}


AUTO_TEST_CASE(test_images_write_1d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE1D>(device, context, queue, num_elements);
}

AUTO_TEST_CASE(test_images_write_2d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE2D>(device, context, queue, num_elements);
}

AUTO_TEST_CASE(test_images_write_3d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE3D>(device, context, queue, num_elements);
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_IMAGES_TEST_WRITE_HPP

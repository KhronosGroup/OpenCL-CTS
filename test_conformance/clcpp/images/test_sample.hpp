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
#ifndef TEST_CONFORMANCE_CLCPP_IMAGES_TEST_SAMPLE_HPP
#define TEST_CONFORMANCE_CLCPP_IMAGES_TEST_SAMPLE_HPP

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common.hpp"


namespace test_images_sample {

enum class sampler_source
{
    param,
    program_scope
};

const sampler_source sampler_sources[] = { sampler_source::param, sampler_source::program_scope };

template<cl_mem_object_type ImageType, cl_channel_type ChannelType>
struct image_test : image_test_base<ImageType, ChannelType>
{
    cl_channel_order channel_order;
    sampler_source source;

    image_test(cl_channel_order channel_order, sampler_source source) :
        channel_order(channel_order),
        source(source)
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
        )";

        std::string sampler;
        if (source == sampler_source::program_scope)
        {
            s << R"(
        constant sampler_t sampler_program_scope = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;
            )";
            sampler = "sampler_program_scope";
        }
        else if (source == sampler_source::param)
        {
            sampler = "sampler_param";
        }

        s << R"(
        kernel void test(
            read_only )" << image_test::image_type_name() << R"(_t img,
            const global int4 *coords,
            global element_type *output,
            sampler_t sampler_param
        ) {
            const ulong gid = get_global_linear_id();

            output[gid] = read_image)" << image_test::function_suffix() <<
                "(img, " << sampler << ", coords[gid]." << image_test::coord_accessor() << R"();
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
        )";

        std::string sampler;
        if (source == sampler_source::program_scope)
        {
            s << R"(
        sampler sampler_program_scope = make_sampler<addressing_mode::none, normalized_coordinates::unnormalized, filtering_mode::nearest>();
            )";
            sampler = "sampler_program_scope";
        }
        else if (source == sampler_source::param)
        {
            sampler = "sampler_param";
        }

        s << R"(
        kernel void test(
            const )" << image_test::image_type_name() << R"(<element_type, image_access::sample> img,
            const global_ptr<int4[]> coords,
            global_ptr<element_type[]> output,
            sampler sampler_param
        ) {
            const ulong gid = get_global_linear_id();

            output[gid] = img.sample()" << sampler << ", coords[gid]." << image_test::coord_accessor() << R"();
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

        std::vector<channel_type> image_values = generate_input(
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

        cl_mem img = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &image_format, &image_desc, static_cast<void *>(image_values.data()), &error);
        RETURN_ON_CL_ERROR(error, "clCreateImage")

        cl_mem coords_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(coord_type) * count, static_cast<void *>(coords.data()), &error);
        RETURN_ON_CL_ERROR(error, "clCreateBuffer")

        cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(element_type) * count, NULL, &error);
        RETURN_ON_CL_ERROR(error, "clCreateBuffer")

        const cl_sampler_properties sampler_properties[] = {
            CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE,
            CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_NONE,
            CL_SAMPLER_FILTER_MODE, CL_FILTER_NEAREST,
            0
        };
        cl_sampler sampler = clCreateSamplerWithProperties(context, sampler_properties, &error);
        RETURN_ON_CL_ERROR(error, "clCreateSamplerWithProperties")

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 1, sizeof(coords_buffer), &coords_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 3, sizeof(sampler), &sampler);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")

        const size_t global_size = count;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

        std::vector<element_type> output(count);
        error = clEnqueueReadBuffer(
            queue, output_buffer, CL_TRUE,
            0, sizeof(element_type) * count,
            static_cast<void *>(output.data()),
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

        for (size_t i = 0; i < count; i++)
        {
            const coord_type c = coords[i];
            const element_type result = output[i];

            element_type expected;
            read_image_pixel<scalar_element_type>(static_cast<void *>(image_values.data()), &image_info,
                c.s[0], c.s[1], c.s[2],
                expected.s);

            if (!are_equal(result, expected))
            {
                RETURN_ON_ERROR_MSG(-1,
                    "Sampling from coordinates %s failed. Expected: %s, got: %s",
                    format_value(c).c_str(), format_value(expected).c_str(), format_value(result).c_str()
                );
            }
        }

        clReleaseMemObject(img);
        clReleaseMemObject(coords_buffer);
        clReleaseMemObject(output_buffer);
        clReleaseSampler(sampler);
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
    for (auto source : sampler_sources)
    {
        error = image_test<ImageType, CL_SIGNED_INT8>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_SIGNED_INT16>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_SIGNED_INT32>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)

        error = image_test<ImageType, CL_UNSIGNED_INT8>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_UNSIGNED_INT16>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
        error = image_test<ImageType, CL_UNSIGNED_INT32>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)

        error = image_test<ImageType, CL_FLOAT>(channel_order, source)
            .run(device, context, queue, num_elements);
        RETURN_ON_ERROR(error)
    }

    return error;
}


AUTO_TEST_CASE(test_images_sample_1d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE1D>(device, context, queue, num_elements);
}

AUTO_TEST_CASE(test_images_sample_2d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE2D>(device, context, queue, num_elements);
}

AUTO_TEST_CASE(test_images_sample_3d)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return run_test_cases<CL_MEM_OBJECT_IMAGE3D>(device, context, queue, num_elements);
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_IMAGES_TEST_SAMPLE_HPP

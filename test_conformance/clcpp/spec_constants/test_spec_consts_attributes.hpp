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
#ifndef TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_ATTRIBUTES_HPP
#define TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_ATTRIBUTES_HPP

#include <type_traits>

#include "common.hpp"

// In this test we check if specialization constant can be successfully used
// in kernel attribute cl::required_work_group_size(X, Y, Z).
struct spec_const_required_work_group_size_test : public spec_constants_test<cl_uint>
{
    // See generate_program() to know what set_spec_constant is for.
    spec_const_required_work_group_size_test(const bool set_spec_constant,
                                             const cl_uint work_group_size_0)
        : m_set_spec_constant(set_spec_constant),
          m_work_group_size_0(work_group_size_0)
    {

    }

    std::string str()
    {
        if(m_set_spec_constant)
            return "spec_const_in_required_work_group_size_" + std::to_string(m_work_group_size_0);
        else
            return "spec_const_in_required_work_group_size_not_set";
    }

    bool set_local_size()
    {
        return true;
    }

    size_t get_max_local_size(const std::vector<cl_kernel>& kernels,
                              cl_device_id device,
                              size_t work_group_size, // default work-group size
                              cl_int& error)
    {
        if(m_set_spec_constant)
        {
            return m_work_group_size_0;
        }
        return size_t(1);
    }

    cl_uint operator()(size_t i, size_t work_group_size)
    {
        (void) work_group_size;
        if(m_set_spec_constant)
        {
            return m_work_group_size_0;
        }
        return cl_uint(1);
    }

    // Check if query for CL_KERNEL_COMPILE_WORK_GROUP_SIZE using clGetKernelWorkGroupInfo
    // return correct values. It should return the work-group size specified by the
    // cl::required_work_group_size(X, Y, Z) qualifier.
    cl_int check_queries(const std::vector<cl_kernel>& kernels,
                         cl_device_id device,
                         cl_context context,
                         cl_command_queue queue)
    {
        (void) device;
        (void) context;
        size_t compile_wg_size[] = { 1, 1, 1 };
        cl_int error = clGetKernelWorkGroupInfo(
            kernels[0], device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
            sizeof(compile_wg_size), compile_wg_size, NULL
        );
        RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
        if(m_set_spec_constant)
        {
            if(compile_wg_size[0] != m_work_group_size_0
               || compile_wg_size[1] != 1
               || compile_wg_size[2] != 1)
            {
                error = -1;
            }
        }
        else
        {
            if(compile_wg_size[0] != 1
               || compile_wg_size[1] != 1
               || compile_wg_size[2] != 1)
            {
                error = -1;
            }
        }
        return error;
    }

    // Sets spec constant
    cl_int set_spec_constants(const cl_program& program)
    {
        cl_int error = CL_SUCCESS;
        if(m_set_spec_constant)
        {
            error = clSetProgramSpecializationConstant(
                program, cl_uint(1), sizeof(cl_uint), static_cast<void*>(&m_work_group_size_0)
            );
            RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
        }
        return error;
    }

    // Each work-item writes get_local_size(0) to output[work-item-global-id]
    std::string generate_program(bool with_attribute)
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            std::string att = " ";
            if(with_attribute)
            {
                std::string work_group_size_0 = "1";
                if(m_set_spec_constant)
                {
                    work_group_size_0 = std::to_string(m_work_group_size_0);
                }
                att = "\n__attribute__((reqd_work_group_size(" + work_group_size_0 + ",1,1)))\n";
            }
            return
                "__kernel" + att + "void " + this->get_kernel_name() + "(global uint *output)\n"
                "{\n"
                "    uint gid = get_global_id(0);\n"
                "    output[gid] = get_local_size(0);\n"
                "}\n";

        #else
            std::string att = "";
            if(with_attribute)
            {
                att = "[[cl::required_work_group_size(spec1, 1, 1)]]\n";
            }
            return
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_spec_constant>\n"
                "using namespace cl;\n"
                "spec_constant<uint, 1> spec1{1};\n"
                + att +
                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output)\n"
                "{\n"
                "    uint gid = get_global_id(0);\n"
                "    output[gid] = get_local_size(0);\n"
                "}\n";
        #endif
    }

    // Each work-item writes get_local_size(0) to output[work-item-global-id]
    std::string generate_program()
    {
        return generate_program(true);
    }

private:
    bool m_set_spec_constant;
    cl_uint m_work_group_size_0;
};

// This function return max work-group size that can be used
// for kernels defined in source
size_t get_max_wg_size(const std::string& source,
                       const std::vector<std::string>& kernel_names,
                       size_t work_group_size, // max wg size we want to have
                       cl_device_id device,
                       cl_context context,
                       cl_command_queue queue,
                       cl_int& err)
{
    cl_program program;
    std::vector<cl_kernel> kernels;
    if(kernel_names.empty())
    {
        RETURN_ON_ERROR_MSG(-1, "No kernel to run");
    }
    kernels.resize(kernel_names.size());
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    err = create_opencl_kernel(context, &program, &(kernels[0]), source, kernel_names[0], "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &err);
        RETURN_ON_CL_ERROR(err, "clCreateKernel");
    }
#else
    err = create_opencl_kernel(context, &program, &(kernels[0]), source, kernel_names[0]);
    RETURN_ON_ERROR(err)
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &err);
        RETURN_ON_CL_ERROR(err, "clCreateKernel");
    }
#endif
    size_t wg_size = work_group_size;
    for(auto& k : kernels)
    {
        size_t max_wg_size;
        err = clGetKernelWorkGroupInfo(
            k, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
        );
        RETURN_ON_CL_ERROR(err, "clGetKernelWorkGroupInfo")
        wg_size = (std::min)(wg_size, max_wg_size);
    }
    return wg_size;
}

AUTO_TEST_CASE(test_spec_constants_in_kernel_attributes)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// If ONLY_SPIRV_COMPILATION is defined we can't check the max work-group size for the
// kernel because OpenCL kernel object is never created in that mode.
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    const size_t max_wg_size = 16;
#else
    // Get max work-group size that can be used in [[cl::required_work_group_size(X, 1, 1)]]
    // We do this by building kernel without this attribute and checking what is the max
    // work-group size we can use with it.
    auto test = spec_const_required_work_group_size_test(true, 1);
    const size_t max_wg_size = get_max_wg_size(
        test.generate_program(false), test.get_kernel_names(),
        1024, // max wg size we want to test
        device, context, queue,
        error
    );
    RETURN_ON_ERROR_MSG(error, "Can't get max work-group size");
#endif

    // Run tests when specialization constant spec1 is set (kernel
    // attribute is [[cl::required_work_group_size(spec1, 1, 1)]]).
    for(size_t i = 1; i <= max_wg_size; i *=2)
    {
        RUN_SPEC_CONSTANTS_TEST_MACRO(
            spec_const_required_work_group_size_test(
                true, i
            )
        );
    }
    // This test does not set spec constant
    RUN_SPEC_CONSTANTS_TEST_MACRO(
        spec_const_required_work_group_size_test(
            false, 9999999 // This value is incorrect, but won't be set and kernel
                           // attribute should be [[cl::required_work_group_size(1, 1, 1)]]
        )
    );

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_ATTRIBUTES_HPP

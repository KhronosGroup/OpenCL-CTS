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
#ifndef TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_STORAGE_TYPES_HPP
#define TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_STORAGE_TYPES_HPP

#include <type_traits>

#include "common.hpp"

// ----------------------------
// ---------- PRIVATE
// ----------------------------

template <class T>
struct private_storage_test : public address_spaces_test<T>
{
    std::string str()
    {
        return "private_storage";
    }

    T operator()(size_t i, size_t work_group_size)
    {
        typedef typename scalar_type<T>::type SCALAR;
        (void) work_group_size;
        return detail::make_value<T>(static_cast<SCALAR>(i));
    }

    // Each work-item writes its global id to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global " + type_name<T>() + " *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = (" + type_name<T>() + ")(gid);\n"
                "}\n";

        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<" + type_name<T>() + "[]> output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    typedef " + type_name<T>() + " TYPE;\n"
                "    priv<TYPE> v = { TYPE(gid) };\n"
                "    const TYPE *v_ptr1 = &v;\n"
                "    private_ptr<TYPE> v_ptr2 = v.ptr();\n"
                "    TYPE v2 = *v_ptr2;\n"
                "    priv<array<TYPE, 1>> a;\n"
                "    *(a.begin()) = v2;\n"
                "    output[gid] = a[0];\n"
                "}\n";        
        #endif
    }
};

AUTO_TEST_CASE(test_private_storage)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // private storage
    RUN_ADDRESS_SPACES_TEST_MACRO(private_storage_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_storage_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_storage_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_storage_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_storage_test<cl_uint16>());

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

// ----------------------------
// ---------- LOCAL
// ----------------------------

template <class T>
struct local_storage_test : public address_spaces_test<T>
{
    std::string str()
    {
        return "local_storage";
    }

    T operator()(size_t i, size_t work_group_size)
    {
        typedef typename scalar_type<T>::type SCALAR;
        size_t r = i / work_group_size;
        return detail::make_value<T>(static_cast<SCALAR>(r));
    }

    bool set_local_size()
    {
        return true;
    }

    // Every work-item writes id of its work-group to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global " + type_name<T>() + " *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = (" + type_name<T>() + ")(get_group_id(0));\n"
                "}\n";

        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_synchronization>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                // Using program scope local variable
                "local<" + type_name<T>() + "> program_scope_var;"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<" + type_name<T>() + "[]> output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    size_t lid = get_local_id(0);\n"
                "    typedef " + type_name<T>() + " TYPE;\n"
                // 1st work-item in work-group writes get_group_id() to var
                "    local<TYPE> var;\n"
                "    if(lid == 0) { var = TYPE(get_group_id(0)); }\n"
                "    work_group_barrier(mem_fence::local);\n"
                // last work-item in work-group writes var to 1st element of a
                "    local_ptr<TYPE> var_ptr = var.ptr();\n"
                "    TYPE var2 = *var_ptr;\n"
                "    local<array<TYPE, 1>> a;\n"
                "    if(lid == (get_local_size(0) - 1)) { *(a.begin()) = var2; }\n"
                "    work_group_barrier(mem_fence::local);\n"
                // 1st work-item in work-group writes a[0] to program_scope_var
                "    if(lid == 0) { program_scope_var = a[0]; }\n"
                "    work_group_barrier(mem_fence::local);\n"
                "    const TYPE *program_scope_var_ptr = &program_scope_var;\n"
                "    output[gid] = *program_scope_var_ptr;\n"
                "}\n";        
        #endif
    }
};

AUTO_TEST_CASE(test_local_storage)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // local storage
    RUN_ADDRESS_SPACES_TEST_MACRO(local_storage_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_storage_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_storage_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_storage_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_storage_test<cl_int16>());

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

// ----------------------------
// ---------- GLOBAL
// ----------------------------

template <class T>
struct global_storage_test : public address_spaces_test<T>
{
    // m_test_value is just a random value we use in this test.
    // m_test_value should not be zero.
    global_storage_test() : m_test_value(0xdeaddeadU)
    {

    }

    std::string str()
    {
        return "global_storage";
    }

    T operator()(size_t i, size_t work_group_size)
    {
        typedef typename scalar_type<T>::type SCALAR;
        return detail::make_value<T>(static_cast<SCALAR>(m_test_value));
    }

    std::vector<std::string> get_kernel_names()
    {
        return 
        {
            this->get_kernel_name() + "1",
            this->get_kernel_name() + "2"
        };
    }

    // Every work-item writes m_test_value to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_names()[0] + "(global " + type_name<T>() + " *output, "
                                                                  "uint test_value)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = (" + type_name<T>() + ")(test_value);\n"
                "}\n"
                "__kernel void " + this->get_kernel_names()[1] + "(global " + type_name<T>() + " *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = output[gid];\n"
                "}\n";
        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                "typedef " + type_name<T>() + " TYPE;\n"
                // Using program scope global variable
                "global<array<TYPE, 1>> program_scope_global_array;"
                "__kernel void " + this->get_kernel_names()[0] + "(global_ptr<" + type_name<T>() + "[]> output, "
                                                                  "uint test_value)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                // 1st work-item writes test_value to program_scope_global_array[0]
                "    if(gid == 0) { program_scope_global_array[0] = test_value; }\n"
                "}\n" 
                "__kernel void " + this->get_kernel_names()[1] + "(global_ptr<" + type_name<T>() + "[]> output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    static global<uint> func_scope_global_var { 0 };\n"
                // if (func_scope_global_var == 1) is true then
                // each work-item saves program_scope_global_array[0] to output[work-item-global-id]
                "    if(func_scope_global_var == uint(1))\n"
                "    {\n"
                "        output[gid] = program_scope_global_array[0];\n"
                "        return;\n"
                "    }\n"
                // 1st work-item writes 1 to func_scope_global_var
                "    if(gid == 0) { func_scope_global_var = uint(1); }\n"
                "}\n";         
        #endif
    }

    // In this test execution is quite complicated. We have two kernels.
    // 1st kernel tests program scope global variable, and 2nd kernel tests 
    // function scope global variable (that's why it is run twice).
    cl_int execute(const std::vector<cl_kernel>& kernels,
                   cl_mem& output_buffer,
                   cl_command_queue& queue,
                   size_t work_size,
                   size_t wg_size)
    {           
        cl_int err;
        err = clSetKernelArg(kernels[0], 0, sizeof(output_buffer), &output_buffer);
        err |= clSetKernelArg(kernels[0], 1, sizeof(cl_uint), &m_test_value);
        RETURN_ON_CL_ERROR(err, "clSetKernelArg");

        // Run first kernel, once.
        // This kernel saves m_test_value to program scope global variable called program_scope_global_var
        err = clEnqueueNDRangeKernel(
            queue, kernels[0], 1, NULL, &work_size, this->set_local_size() ? &wg_size : NULL, 0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");
        err = clFinish(queue);
        RETURN_ON_CL_ERROR(err, "clFinish")

        err = clSetKernelArg(kernels[1], 0, sizeof(output_buffer), &output_buffer);
        // Run 2nd kernel, twice.
        // 1st run: program_scope_global_var is saved to function scope global array called func_scope_global_array
        // 2nd run: each work-item saves func_scope_global_array[0] to ouput[work-item-global-id]
        for(size_t i = 0; i < 2; i++)
        {
            err = clEnqueueNDRangeKernel(
                queue, kernels[1], 1, NULL, &work_size, this->set_local_size() ? &wg_size : NULL, 0, NULL, NULL
            );
            RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");
            err = clFinish(queue);
            RETURN_ON_CL_ERROR(err, "clFinish")
        }
        return err;
    }

private:
    cl_uint m_test_value;
};

AUTO_TEST_CASE(test_global_storage)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    RUN_ADDRESS_SPACES_TEST_MACRO(global_storage_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_storage_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_storage_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_storage_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_storage_test<cl_int16>());

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

// ----------------------------
// ---------- CONSTANT
// ----------------------------

template <class T>
struct constant_storage_test : public address_spaces_test<T>
{
    // m_test_value is just a random value we use in this test.
    constant_storage_test() : m_test_value(0xdeaddeadU)
    {

    }

    std::string str()
    {
        return "constant_storage";
    }

    T operator()(size_t i, size_t work_group_size)
    {
        typedef typename scalar_type<T>::type SCALAR;
        return detail::make_value<T>(static_cast<SCALAR>(m_test_value));
    }

    // Every work-item writes m_test_value to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global " + type_name<T>() + " *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = (" + type_name<T>() + ")(" + std::to_string(m_test_value) + ");\n"
                "}\n";

        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                // Program scope constant variable, program_scope_var == (m_test_value - 1)
                "constant<uint> program_scope_const{ (" + std::to_string(m_test_value) + " - 1) };"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<" + type_name<T>() + "[]> output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    typedef " + type_name<T>() + " TYPE;\n"
                "    static constant<uint> func_scope_const{ 1 };\n"
                "    constant_ptr<uint> ps_const_ptr = program_scope_const.ptr();\n"
                // "    constant_ptr<array<uint, 1>> fs_const_ptr = &func_scope_const;\n"
                "    output[gid] = TYPE(*ps_const_ptr + func_scope_const);\n"
                "}\n";        
        #endif
    }
private:
    cl_uint m_test_value;
};

AUTO_TEST_CASE(test_constant_storage)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    RUN_ADDRESS_SPACES_TEST_MACRO(constant_storage_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_storage_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_storage_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_storage_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_storage_test<cl_int16>());

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_STORAGE_TYPES_HPP

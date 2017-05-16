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
#ifndef TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_IF_HPP
#define TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_IF_HPP

#include <type_traits>

#include "common.hpp"

// This class tests using specialization constant in if statement
template <class T /* spec constant type*/>
struct spec_const_in_if_test : public spec_constants_test<cl_uint>
{
    // See generate_program() to know what set_spec_constant is for.
    spec_const_in_if_test(const bool set_spec_constant)
        : m_set_spec_constant(set_spec_constant)
    {
        static_assert(
            is_vector_type<T>::value == false,
            "Specialization constant can be only scalar int or float type"
        );
        switch (sizeof(T))
        {
            case 1:
                m_test_value = T(127);
                break;
            case 2:
                m_test_value = T(0xdeadU);
                break;
            // 4 and 8
            default:
                m_test_value = T(0xdeaddeadU);
                break;
        }
    }

    std::string str()
    {
        return "spec_const_in_if_" + type_name<T>();
    }

    cl_uint operator()(size_t i, size_t work_group_size)
    {
        (void) work_group_size;
        if(m_set_spec_constant)
        {
            return m_test_value;
        }
        return static_cast<cl_uint>(i);
    }

    // Sets spec constant
    cl_int set_spec_constants(const cl_program& program)
    {
        cl_int error = CL_SUCCESS;
        if(m_set_spec_constant)
        {
            T spec1 = static_cast<T>(m_test_value);
            error = clSetProgramSpecializationConstant(
                program, cl_uint(1), sizeof(T), static_cast<void*>(&spec1)
            );
            RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
        }
        return error;
    }

    // IF set_spec_constant == true:
    // each work-item writes T(m_test_value) to output[work-item-global-id]
    // Otherwise:
    // each work-item writes T(get_global_id(0)) to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            std::string result = "gid";
            if(m_set_spec_constant)
                result = std::to_string(m_test_value);
            return 
                "__kernel void " + this->get_kernel_name() + "(global uint *output)\n"
                "{\n"
                "    uint gid = get_global_id(0);\n"
                "    output[gid] = " + result + ";\n"
                "}\n";

        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_spec_constant>\n"
                "using namespace cl;\n"
                "typedef " + type_name<T>() + " TYPE;\n"
                "spec_constant<TYPE,  1> spec1{TYPE(0)};\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output)\n"
                "{\n"
                "    uint gid = get_global_id(0);\n"
                "    if(get(spec1) == TYPE(" + std::to_string(m_test_value) +"))\n"
                "    {\n"
                "        output[gid] = " + std::to_string(m_test_value) +";\n"
                "    }\n"
                "    else\n"
                "    {\n"
                "        output[gid] = gid;\n"
                "    }\n"
                "}\n";        
        #endif
    }

private:
    bool m_set_spec_constant;
    cl_uint m_test_value;
};

AUTO_TEST_CASE(test_spec_constants_in_if_statement)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    const std::vector<bool> set_spec_const_options { true, false };
    for(auto option : set_spec_const_options)
    {        
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_char>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_uchar>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_int>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_uint>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_long>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_ulong>(option));
        RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_float>(option));
        if(is_extension_available(device, "cl_khr_fp16"))
        {
            RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_half>(option));
        }
        if(is_extension_available(device, "cl_khr_fp64"))
        {
            RUN_SPEC_CONSTANTS_TEST_MACRO(spec_const_in_if_test<cl_double>(option));
        }
    }

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_TEST_SPEC_CONSTS_IF_HPP

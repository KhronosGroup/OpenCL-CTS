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
#ifndef TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_TEST_CTORS_DTORS_HPP
#define TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_TEST_CTORS_DTORS_HPP

#include "common.hpp"

// Test for program scope variable with non-trivial ctor
struct ps_ctor_test : public ps_ctors_dtors_test_base
{
    ps_ctor_test(const cl_uint test_value)
        : ps_ctors_dtors_test_base(true, false),
          m_test_value(test_value)
    {

    }
    
    std::string str()
    {
        return "ps_ctor_test";
    }

    std::vector<std::string> get_kernel_names()
    {
        return { 
            this->str() + "_set",
            this->str() + "_read"
        };
    }

    // Returns value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i)
    {
        if(i % 2 == 0)
            return m_test_value;
        return cl_uint(0xbeefbeef);
    }

    // In 1st kernel 0th work-tem sets member m_x of program scope variable global_var to
    // m_test_value and m_y to uint(0xbeefbeef),
    // In 2nd kernel:
    // 1) if global id is even, then work-item reads global_var.m_x and writes it to output[its-global-id];
    // 2) otherwise, work-item reads global_var.m_y and writes it to output[its-global-id].
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_names()[0] + "(global uint *output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                "   output[gid] = 0xbeefbeef;\n"
                "}\n"
                "__kernel void " + this->get_kernel_names()[1] + "(global uint *output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                "   if(gid % 2 == 0)\n"
                "      output[gid] = " + std::to_string(m_test_value) + ";\n"
                "}\n";
        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                // struct template
                "template<class T>\n"    
                "struct ctor_test_class_base {\n"
                // non-trivial ctor
                "   ctor_test_class_base(T x) { m_x = x;};\n"
                "   T m_x;\n"
                "};\n"
                // struct template
                "template<class T>\n"    
                "struct ctor_test_class : public ctor_test_class_base<T> {\n"
                // non-trivial ctor
                "   ctor_test_class(T x, T y) : ctor_test_class_base<T>(x), m_y(y) { };\n"
                "   T m_y;\n"
                "};\n"
                // global scope program variables
                "ctor_test_class<uint> global_var(uint(0), uint(0));\n"

                "__kernel void " + this->get_kernel_names()[0] + "(global_ptr<uint[]> output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                "   if(gid == 0) {\n"
                "       global_var.m_x = " + std::to_string(m_test_value) + ";\n"  
                "       global_var.m_y = 0xbeefbeef;\n"  
                "   }\n"
                "}\n"

                "__kernel void " + this->get_kernel_names()[1] + "(global_ptr<uint[]> output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                "   if(gid % 2 == 0)\n"
                "      output[gid] = global_var.m_x;\n"
                "   else\n"
                "      output[gid] = global_var.m_y;\n"
                "}\n";        
        #endif
    }

private:
    cl_uint m_test_value;
};

// Test for program scope variable with non-trivial dtor
struct ps_dtor_test : public ps_ctors_dtors_test_base
{
    ps_dtor_test(const cl_uint test_value)
        : ps_ctors_dtors_test_base(false, true),
          m_test_value(test_value)
    {

    }
    
    std::string str()
    {
        return "ps_dtor_test";
    }

    // Returns value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i)
    {
        if(i % 2 == 0)
            return m_test_value;
        return 1;
    }

    // In 1st kernel 0th work-item saves pointer to output buffer and its size in program scope
    // variable global_var, it also sets counter to 1;
    // After global_var is destroyed all even elements of output buffer should equal m_test_value, 
    // and all odd should equal 1.
    // If odd elements of output buffer are >1 it means dtor was executed more than once.
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global uint *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    if(gid % 2 == 0)\n"
                "        output[gid] = " + std::to_string(m_test_value) + ";\n"
                "    else\n"
                "        output[gid] = 1;\n"
                "}\n";
        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                // struct template
                "template<class T>\n"
                "struct dtor_test_class_base {\n"
                // non-trivial dtor
                // set all odd elements in buffer to counter
                "   ~dtor_test_class_base() {\n"
                "       for(size_t i = 1; i < this->size; i+=2)\n"
                "       {\n"
                "           this->buffer[i] = counter;\n"
                "       }\n"
                "       counter++;\n"
                "   };\n"
                "   global_ptr<uint[]> buffer;\n"
                "   size_t size;\n"
                "   T counter;\n"
                "};\n" 
                // struct   
                "struct dtor_test_class : public dtor_test_class_base<uint> {\n"
                // non-trivial dtor
                // set all values in buffer to m_test_value
                "   ~dtor_test_class() {\n"
                "       for(size_t i = 0; i < this->size; i+=2)\n"
                "           this->buffer[i] = " + std::to_string(m_test_value) + ";\n"
                "   };\n"
                "};\n" 
                // global scope program variable
                "dtor_test_class global_var;\n"

                // When global_var is being destroyed, first dtor ~dtor_test_class is called,
                // and then ~dtor_test_class_base is called.

                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                // set buffer and size in global var
                "   if(gid == 0){\n"
                "       global_var.buffer = output;\n"
                "       global_var.size = get_global_size(0);\n"
                "       global_var.counter = 1;\n"
                "   }\n"
                "}\n";
        #endif
    }

private:
    cl_uint m_test_value;
};

// Test for program scope variable with both non-trivial ctor
// and non-trivial dtor
struct ps_ctor_dtor_test : public ps_ctors_dtors_test_base
{
    ps_ctor_dtor_test(const cl_uint test_value)
        : ps_ctors_dtors_test_base(false, true),
          m_test_value(test_value)
    {

    }
    
    std::string str()
    {
        return "ps_ctor_dtor_test";
    }

    // Returns value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i)
    {
        return m_test_value;
    }

    // In 1st kernel 0th work-item saves pointer to output buffer and its size in program scope
    // variable global_var.
    // After global_var is destroyed all even elements of output buffer should equal m_test_value, 
    // and all odd should equal 1.
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global uint *output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = " + std::to_string(m_test_value) + ";\n"
                "}\n";
        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                // struct template
                "template<class T>\n"    
                "struct ctor_test_class {\n"
                // non-trivial ctor
                "   ctor_test_class(T value) : m_value(value) { };\n"
                "   T m_value;\n"
                "};\n\n"
                // struct   
                "struct ctor_dtor_test_class {\n"
                // non-trivial ctor
                "   ctor_dtor_test_class(uint value) : ctor_test(value) { } \n"
                // non-trivial dtor
                // set all values in buffer to m_test_value
                "   ~ctor_dtor_test_class() {\n"
                "       for(size_t i = 0; i < this->size; i++)\n"
                "       {\n"
                "          this->buffer[i] = ctor_test.m_value;\n"            
                "       }\n"
                "   };\n"
                "   ctor_test_class<uint> ctor_test;\n"
                "   global_ptr<uint[]> buffer;\n"
                "   size_t size;\n"
                "};\n" 
                // global scope program variable
                "ctor_dtor_test_class global_var(" + std::to_string(m_test_value) + ");\n"

                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output)\n"
                "{\n"
                "   size_t gid = get_global_id(0);\n"
                // set buffer and size in global var
                "   if(gid == 0){\n"
                "       global_var.buffer = output;\n"
                "       global_var.size = get_global_size(0);\n"
                "   }\n"
                "}\n";
        #endif
    }

private:
    cl_uint m_test_value;
};

// This contains tests for program scope (global) constructors and destructors, more
// detailed tests are also in clcpp/api.
AUTO_TEST_CASE(test_program_scope_ctors_dtors)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    RUN_PS_CTORS_DTORS_TEST_MACRO(ps_ctor_test(0xdeadbeefU))
    RUN_PS_CTORS_DTORS_TEST_MACRO(ps_dtor_test(0xbeefdeadU))
    RUN_PS_CTORS_DTORS_TEST_MACRO(ps_ctor_dtor_test(0xdeaddeadU))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_TEST_CTORS_DTORS_HPP

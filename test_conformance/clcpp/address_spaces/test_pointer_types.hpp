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
#ifndef TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_POINTER_TYPES_HPP
#define TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_POINTER_TYPES_HPP

#include <type_traits>

#include "common.hpp"

// ----------------------------
// ---------- PRIVATE
// ----------------------------

template <class T>
struct private_pointer_test : public address_spaces_test<T>
{
    std::string str()
    {
        return "private_pointer";
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
                "    TYPE v = TYPE(gid);\n"
                "    private_ptr<TYPE> v_ptr1(dynamic_asptr_cast<private_ptr<TYPE>>(&v));\n"
                "    private_ptr<TYPE> v_ptr2(v_ptr1);\n"
                "    TYPE a[] = { TYPE(0), TYPE(1) };\n"
                "    private_ptr<TYPE> a_ptr = dynamic_asptr_cast<private_ptr<TYPE>>(a);\n"
                "    a_ptr++;\n"
                "    TYPE * a_ptr2 = a_ptr.get();\n"
                "    *a_ptr2 = *v_ptr2;\n"
                "    output[gid] = a[1];\n"
                "}\n";        
        #endif
    }
};

AUTO_TEST_CASE(test_private_pointer)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // private pointer
    RUN_ADDRESS_SPACES_TEST_MACRO(private_pointer_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_pointer_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_pointer_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_pointer_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(private_pointer_test<cl_uint16>());

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
struct local_pointer_test : public address_spaces_test<T>
{
    std::string str()
    {
        return "local_pointer";
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

    size_t get_max_local_size(const std::vector<cl_kernel>& kernels, 
                              cl_device_id device,
                              size_t work_group_size, // default work-group size
                              cl_int& error)
    {
        // Set size of the local memory, we need to to this to correctly calculate
        // max possible work-group size.
        // Additionally this already set 2nd argument of the test kernel, so we don't
        // have to modify execute() method.
        error = clSetKernelArg(kernels[0], 1, sizeof(cl_uint), NULL);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg");

        size_t wg_size;
        error = clGetKernelWorkGroupInfo(
            kernels[0], device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wg_size, NULL
        );
        RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
        wg_size = wg_size <= work_group_size ? wg_size : work_group_size;        
        return wg_size;
    }

    // Every work-item writes id of its work-group to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global " + type_name<T>() + " *output, "
                                                              "local uint * local_mem_ptr)\n"
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
                "__kernel void " + this->get_kernel_name() + "(global_ptr<" + type_name<T>() + "[]> output, "
                                                              "local_ptr<uint[]> local_mem_ptr)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    size_t lid = get_local_id(0);\n"
                "    typedef " + type_name<T>() + " TYPE;\n"
                // 1st work-item in work-group writes get_group_id() to var
                "    local<uint> var;\n"
                "    local_ptr<uint> var_ptr = var.ptr();\n"
                "    if(lid == 0) { *var_ptr = get_group_id(0); }\n"
                "    work_group_barrier(mem_fence::local);\n"
                // last work-item in work-group writes var to 1st element of local_mem
                "    local_ptr<uint[]> local_mem_ptr2(local_mem_ptr);\n"
                "    auto local_mem_ptr3 = local_mem_ptr2.release();\n"
                "    if(lid == (get_local_size(0) - 1)) { *(local_mem_ptr3) = var; }\n"
                "    work_group_barrier(mem_fence::local);\n"
                // each work-item in work-group writes local_mem_ptr[0] to output[work-item-global-id]
                "    output[gid] = local_mem_ptr[0];\n"
                "}\n";        
        #endif
    }
};

AUTO_TEST_CASE(test_local_pointer)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // local pointer
    RUN_ADDRESS_SPACES_TEST_MACRO(local_pointer_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_pointer_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_pointer_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_pointer_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(local_pointer_test<cl_uint16>());

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
struct global_pointer_test : public address_spaces_test<T>
{
    std::string str()
    {
        return "global_pointer";
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
                "typedef " + type_name<T>() + " TYPE;\n"
                "void set_to_gid(global_ptr<TYPE> ptr)\n"
                "{\n"
                "    *ptr = TYPE(get_global_id(0));"
                "}\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<TYPE[]> output)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    auto ptr = output.get();\n"
                "    global_ptr<TYPE> ptr2(ptr);\n"
                "    ptr2 += ptrdiff_t(gid);\n"
                "    set_to_gid(ptr2);\n"
                "}\n";        
        #endif
    }
};

AUTO_TEST_CASE(test_global_pointer)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // global pointer
    RUN_ADDRESS_SPACES_TEST_MACRO(global_pointer_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_pointer_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_pointer_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_pointer_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(global_pointer_test<cl_uint16>());

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
struct constant_pointer_test : public address_spaces_test<T>
{
    // m_test_value is just a random value we use in this test.
    constant_pointer_test() : m_test_value(0xdeaddeadU)
    {

    }

    std::string str()
    {
        return "constant_pointer";
    }

    T operator()(size_t i, size_t work_group_size)
    {
        typedef typename scalar_type<T>::type SCALAR;
        (void) work_group_size;
        return detail::make_value<T>(static_cast<SCALAR>(m_test_value));
    }

    // Each work-item writes m_test_value to output[work-item-global-id]
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS) 
            return 
                "__kernel void " + this->get_kernel_name() + "(global " + type_name<T>() + " *output, "
                                                              "constant uint * const_ptr)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    output[gid] = (" + type_name<T>() + ")(const_ptr[0]);\n"
                "}\n";

        #else
            return         
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_array>\n"
                "using namespace cl;\n"
                "typedef " + type_name<T>() + " TYPE;\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<TYPE[]> output, "
                                                              "constant_ptr<uint[]> const_ptr)\n"
                "{\n"
                "    size_t gid = get_global_id(0);\n"
                "    constant_ptr<uint[]> const_ptr2 = const_ptr;\n"
                "    auto const_ptr3 = const_ptr2.get();\n"
                "    output[gid] = *const_ptr3;\n"
                "}\n";        
        #endif
    }

    // execute() method needs to be modified, to create additional buffer
    // and set it in 2nd arg (constant_ptr<uint[]> const_ptr)
    cl_int execute(const std::vector<cl_kernel>& kernels,
                   cl_mem& output_buffer,
                   cl_command_queue& queue,
                   size_t work_size,
                   size_t work_group_size)
    {           
        cl_int err;

        // Get context from queue
        cl_context context;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
        RETURN_ON_CL_ERROR(err, "clGetCommandQueueInfo");

        // Create constant buffer
        auto const_buff = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_ONLY), sizeof(cl_uint), NULL, &err);
        RETURN_ON_CL_ERROR(err, "clCreateBuffer");

        // Write m_test_value to const_buff
        err = clEnqueueWriteBuffer(
            queue, const_buff, CL_TRUE, 0, sizeof(cl_uint),
            static_cast<void *>(&m_test_value), 0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer");

        err = clSetKernelArg(kernels[0], 0, sizeof(output_buffer), &output_buffer);
        err |= clSetKernelArg(kernels[0], 1, sizeof(const_buff), &const_buff);
        RETURN_ON_CL_ERROR(err, "clSetKernelArg");

        err = clEnqueueNDRangeKernel(
            queue, kernels[0], 1, NULL, &work_size, this->set_local_size() ? &work_group_size : NULL, 0, NULL, NULL
        );      
        RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

        err = clFinish(queue);
        RETURN_ON_CL_ERROR(err, "clFinish");

        err = clReleaseMemObject(const_buff);
        RETURN_ON_CL_ERROR(err, "clReleaseMemObject");
        return err;
    }

private:
    cl_uint m_test_value;
};

AUTO_TEST_CASE(test_constant_pointer)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // constant pointer
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_pointer_test<cl_uint>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_pointer_test<cl_float2>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_pointer_test<cl_float4>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_pointer_test<cl_float8>());
    RUN_ADDRESS_SPACES_TEST_MACRO(constant_pointer_test<cl_uint16>());

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_ADDRESS_SPACES_TEST_POINTER_TYPES_HPP

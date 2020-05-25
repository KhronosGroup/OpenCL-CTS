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
#include <vector>


const char* macro_supported_source =
    "kernel void enabled(global int * buf) { \r\n"
    "int n = get_global_id(0); \r\n"
    "buf[n] = 0; \r\n "
    "#ifndef %s \r\n"
    "ERROR; \r\n"
    "#endif \r\n"
    "\r\n } \r\n";


const char* macro_not_supported_source =
    "kernel void not_enabled(global int * buf) { \r\n"
    "int n = get_global_id(0); \r\n"
    "buf[n] = 0; \r\n "
    "#ifdef %s \r\n"
    "ERROR; \r\n"
    "#endif \r\n"
    "\r\n } \r\n";


template <typename T>
cl_int check_api_feature_info(cl_device_id deviceID, cl_context context,
                              bool& status, cl_device_info check_property,
                              cl_device_atomic_capabilities check_atomic_cap,
                              cl_mem_flags mem_flags,
                              cl_mem_object_type image_type)
{
    cl_int error = CL_SUCCESS;
    T response;
    if (mem_flags)
    {
        cl_uint image_format_count;
        error = clGetSupportedImageFormats(context, mem_flags, image_type, 0,
                                           NULL, &image_format_count);
        response = image_format_count;
        test_error(error, "clGetSupportedImageFormats failed.\n");
    }
    else
    {
        error = clGetDeviceInfo(deviceID, check_property, sizeof(response),
                                &response, NULL);
        test_error(error, "clGetDeviceInfo failed.\n");
    }

    if (std::is_same<T, cl_bool>::value)
    {
        status = response;
    }
    else if (std::is_same<T, cl_device_atomic_capabilities>::value)
    {
        if (response & check_atomic_cap)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    else
    {
        if (response > 0)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    return error;
}
cl_int check_compiler_feature_info(cl_device_id deviceID, cl_context context,
                                   std::string feature_macro, bool& status)
{
    cl_int error = CL_SUCCESS;
    clProgramWrapper program_supported;
    clProgramWrapper program_not_supported;
    char kernel_supported_src[1024];
    char kernel_not_supported_src[1024];
    sprintf(kernel_supported_src, macro_supported_source,
            feature_macro.c_str());
    const char* ptr_supported = kernel_supported_src;
    error = create_single_kernel_helper_create_program(
        context, &program_supported, 1, &ptr_supported);
    test_error(error, "Failure creating program, [%d] \n");

    sprintf(kernel_not_supported_src, macro_not_supported_source,
            feature_macro.c_str());
    const char* ptr_not_supported = kernel_not_supported_src;
    error = create_single_kernel_helper_create_program(
        context, &program_not_supported, 1, &ptr_not_supported);
    test_error(error, "Failure creating program, [%d] \n");

    cl_int status_supported = CL_SUCCESS;
    cl_int status_not_supported = CL_SUCCESS;
    status_supported =
        clBuildProgram(program_supported, 1, &deviceID, NULL, NULL, NULL);
    status_not_supported =
        clBuildProgram(program_not_supported, 1, &deviceID, NULL, NULL, NULL);
    if (status_supported != status_not_supported)
    {
        if (status_not_supported == CL_SUCCESS)
        {
            // kernel which verifies not supporting return passed
            status = false;
        }
        if (status_supported == CL_SUCCESS)
        {
            // kernel which verifies supporting return passed
            status = true;
        }
    }
    else
    {
        log_error("Failure reason: The macro feature is defined and undefined "
                  "in the same time\n");
        return TEST_FAIL;
    }
    return error;
}

template <typename T>
int test_feature_macro(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements,
                       std::string feature_macro, cl_device_info check_property,
                       cl_device_atomic_capabilities check_atomic_cap = 0,
                       cl_mem_flags mem_flags = 0,
                       cl_mem_object_type image_type = 0)
{
    cl_int error = TEST_FAIL;
    bool api_status;
    bool compiler_status;

    log_info("Testing macro: %s\n", feature_macro.c_str());
    error =
        check_api_feature_info<T>(deviceID, context, api_status, check_property,
                                  check_atomic_cap, mem_flags, image_type);
    if (error != CL_SUCCESS)
    {
        return error;
    }
    error = check_compiler_feature_info(deviceID, context, feature_macro,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }
    log_info("Feature verification: API status - %s, compiler status - %s\n",
             api_status == true ? "supported" : "not supported",
             compiler_status == true ? "supported" : "not supported");
    if (api_status != compiler_status)
    {
        log_error("Failure reason: feature status reported by API and compiler "
                  "should be the same.\n",
                  feature_macro.c_str());
        return TEST_FAIL;
    }
    return error;
}

int test_feature_macro_atomic_order_acq_rel(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_order_acq_rel";
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, queue, num_elements, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_ORDER_ACQ_REL);
}

int test_feature_macro_atomic_order_seq_cst(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_order_seq_cst";
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, queue, num_elements, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_ORDER_SEQ_CST);
}

int test_feature_macro_atomic_scope_device(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_scope_device";
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, queue, num_elements, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_SCOPE_DEVICE);
}

int test_feature_macro_atomic_scope_all_devices(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_scope_all_devices";
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, queue, num_elements, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES);
}

int test_feature_macro_3d_image_writes(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_3d_image_writes";
    return test_feature_macro<cl_uint>(
        deviceID, context, queue, num_elements, test_macro_name, 0, 0,
        CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE | CL_MEM_KERNEL_READ_AND_WRITE,
        CL_MEM_OBJECT_IMAGE3D);
}

int test_feature_macro_device_enqueue(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_device_enqueue";
    return test_feature_macro<cl_bool>(deviceID, context, queue, num_elements,
                                       test_macro_name,
                                       CL_DEVICE_DEVICE_ENQUEUE_SUPPORT);
}

int test_feature_macro_generic_adress_space(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    std::string test_macro_name = "__opencl_c_generic_address_space";
    return test_feature_macro<cl_bool>(deviceID, context, queue, num_elements,
                                       test_macro_name,
                                       CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT);
}

int test_feature_macro_pipes(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_pipes";
    return test_feature_macro<cl_bool>(deviceID, context, queue, num_elements,
                                       test_macro_name, CL_DEVICE_PIPE_SUPPORT);
}

int test_feature_macro_program_scope_global_variables(cl_device_id deviceID,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements)
{
    std::string test_macro_name = "__opencl_c_program_scope_global_variables";
    return test_feature_macro<size_t>(deviceID, context, queue, num_elements,
                                       test_macro_name,
                                       CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE);
}

int test_feature_macro_read_write_images(cl_device_id deviceID,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    std::string test_macro_name = "__opencl_c_read_write_images";
    return test_feature_macro<cl_uint>(deviceID, context, queue, num_elements,
                                       test_macro_name,
                                       CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS);
}

int test_feature_macro_subgroups(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_subgroups";
    return test_feature_macro<cl_uint>(deviceID, context, queue, num_elements,
                                       test_macro_name,
                                       CL_DEVICE_MAX_NUM_SUB_GROUPS);
}

int test_feature_macro_work_group_collective_functions(cl_device_id deviceID,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    std::string test_macro_name = "__opencl_c_work_group_collective_functions";
    return test_feature_macro<cl_bool>(
        deviceID, context, queue, num_elements, test_macro_name,
        CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT);
}
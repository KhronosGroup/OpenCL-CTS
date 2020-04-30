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


cl_int check_api_feature_info(cl_device_id deviceID, cl_context context,
                              std::string feature_macro, bool& status)
{
    cl_int error = CL_SUCCESS;
    if (feature_macro == "__opencl_c_device_enqueue")
    {
        cl_bool support;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_DEVICE_ENQUEUE_SUPPORT,
                                sizeof(support), &support, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_DEVICE_ENQUEUE_SUPPORT: %d\n",
                      error);
            return error;
        }
        else
        {
            status = support;
        }
    }
    if (feature_macro == "__opencl_c_generic_address_space")
    {
        cl_bool support;
        error =
            clGetDeviceInfo(deviceID, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                            sizeof(support), &support, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: %d\n",
                      error);
            return error;
        }
        else
        {
            status = support;
        }
    }
    if (feature_macro == "__opencl_c_pipes")
    {
        cl_bool support;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_SUPPORT,
                                sizeof(support), &support, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for CL_DEVICE_PIPE_SUPPORT: %d\n",
                      error);
            return error;
        }
        else
        {
            status = support;
        }
    }
    if (feature_macro == "__opencl_c_work_group_collective_functions")
    {
        cl_bool support;
        error = clGetDeviceInfo(
            deviceID, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
            sizeof(support), &support, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: %d\n",
                      error);
            return error;
        }
        else
        {
            status = support;
        }
    }
    if (feature_macro == "__opencl_c_read_write_images")
    {
        cl_uint max_read_write_images_count = 0;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                sizeof(max_read_write_images_count),
                                &max_read_write_images_count, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: %d\n",
                      error);
            return error;
        }
        else
        {
            if (max_read_write_images_count > 0)
            {
                status = true;
            }
            else
            {
                status = false;
            }
        }
    }
    if (feature_macro == "__opencl_c_program_scope_global_variables")
    {
        cl_uint max_global_variable_size = 0;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                sizeof(max_global_variable_size),
                                &max_global_variable_size, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: %d\n",
                      error);
            return error;
        }
        else
        {
            if (max_global_variable_size > 0)
            {
                status = true;
            }
            else
            {
                status = false;
            }
        }
    }
    if (feature_macro == "__opencl_c_subgroups")
    {
        cl_uint max_num_subgroups = 0;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                sizeof(max_num_subgroups), &max_num_subgroups,
                                NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_MAX_NUM_SUB_GROUPS: %d\n",
                      error);
            return error;
        }
        else
        {
            if (max_num_subgroups > 0)
            {
                status = true;
            }
            else
            {
                status = false;
            }
        }
    }
    if (feature_macro == "__opencl_c_3d_image_writes")
    {
        cl_uint supported_formats_count;
        std::vector<cl_image_format> supported_image_formats;
        error = clGetSupportedImageFormats(context,
                                           CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE
                                               | CL_MEM_KERNEL_READ_AND_WRITE,
                                           CL_MEM_OBJECT_IMAGE3D, 0, NULL,
                                           &supported_formats_count);
        if (error != CL_SUCCESS)
        {
            log_error("clGetSupportedImageFormats error for "
                      "CL_MEM_OBJECT_IMAGE3D: %d\n",
                      error);
            return error;
        }
        if (supported_formats_count > 0)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    if (feature_macro == "__opencl_c_atomic_order_acq_rel")
    {
        cl_device_atomic_capabilities caps_atomic;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(caps_atomic), &caps_atomic, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: %d\n",
                      error);
            return error;
        }
        if (caps_atomic & CL_DEVICE_ATOMIC_ORDER_ACQ_REL)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    if (feature_macro == "__opencl_c_atomic_order_seq_cst")
    {
        cl_device_atomic_capabilities caps_atomic;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(caps_atomic), &caps_atomic, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: %d\n",
                      error);
            return error;
        }
        if (caps_atomic & CL_DEVICE_ATOMIC_ORDER_SEQ_CST)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    if (feature_macro == "__opencl_c_atomic_scope_device")
    {
        cl_device_atomic_capabilities caps_atomic;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(caps_atomic), &caps_atomic, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: %d\n",
                      error);
            return error;
        }
        if (caps_atomic & CL_DEVICE_ATOMIC_SCOPE_DEVICE)
        {
            status = true;
        }
        else
        {
            status = false;
        }
    }
    if (feature_macro == "__opencl_c_atomic_scope_all_devices")
    {
        cl_device_atomic_capabilities caps_atomic;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(caps_atomic), &caps_atomic, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo error for "
                      "CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: %d\n",
                      error);
            return error;
        }
        if (caps_atomic & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES)
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

    if (error != CL_SUCCESS)
    {
        log_error("Failure creating program, [%d] \n", error);
        return error;
    }
    sprintf(kernel_not_supported_src, macro_not_supported_source,
            feature_macro.c_str());
    const char* ptr_not_supported = kernel_not_supported_src;
    error = create_single_kernel_helper_create_program(
        context, &program_not_supported, 1, &ptr_not_supported);

    if (error != CL_SUCCESS)
    {
        log_error("Failure creating program, [%d] \n", error);
        return error;
    }
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

int test_feature_macro(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements,
                       std::string feature_macro)
{
    cl_int error = TEST_FAIL;
    bool api_status;
    bool compiler_status;

    log_info("Testing macro: %s\n", feature_macro.c_str());
    error =
        check_api_feature_info(deviceID, context, feature_macro, api_status);
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
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_atomic_order_seq_cst(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_order_seq_cst";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_atomic_scope_device(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_scope_device";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_atomic_scope_all_devices(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    std::string test_macro_name = "__opencl_c_atomic_scope_all_devices";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}

int test_feature_macro_3d_image_writes(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_3d_image_writes";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}

int test_feature_macro_device_enqueue(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_device_enqueue";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}

int test_feature_macro_generic_adress_space(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    std::string test_macro_name = "__opencl_c_generic_address_space";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}

int test_feature_macro_pipes(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_pipes";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}

int test_feature_macro_program_scope_global_variables(cl_device_id deviceID,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements)
{
    std::string test_macro_name = "__opencl_c_program_scope_global_variables";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_read_write_images(cl_device_id deviceID,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    std::string test_macro_name = "__opencl_c_read_write_images";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_subgroups(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    std::string test_macro_name = "__opencl_c_subgroups";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
int test_feature_macro_work_group_collective_functions(cl_device_id deviceID,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    std::string test_macro_name = "__opencl_c_work_group_collective_functions";
    return test_feature_macro(deviceID, context, queue, num_elements,
                              test_macro_name);
}
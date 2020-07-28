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
#include <algorithm>

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
                              cl_bitfield check_cap, cl_mem_flags mem_flags,
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

    if (std::is_same<T, bool>::value)
    {
        status = response;
    }
    else if (check_cap)
    {
        if ((response & check_cap) == check_cap)
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
        context, &program_supported, 1, &ptr_supported, "-cl-std=CL3.0");
    test_error(error, "create_single_kernel_helper_create_program failed.\n");

    sprintf(kernel_not_supported_src, macro_not_supported_source,
            feature_macro.c_str());
    const char* ptr_not_supported = kernel_not_supported_src;
    error = create_single_kernel_helper_create_program(
        context, &program_not_supported, 1, &ptr_not_supported,
        "-cl-std=CL3.0");
    test_error(error, "create_single_kernel_helper_create_program failed.\n");

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
        log_error("Error: The macro feature is defined and undefined "
                  "in the same time\n");
        return TEST_FAIL;
    }
    return error;
}

template <typename T>
int test_feature_macro(cl_device_id deviceID, cl_context context,
                       std::string feature_macro, cl_device_info check_property,
                       cl_bitfield check_cap, cl_mem_flags mem_flags,
                       cl_mem_object_type image_type, bool& supported)
{
    cl_int error = TEST_FAIL;
    bool api_status;
    bool compiler_status;

    log_info("\n%s ...\n", feature_macro.c_str());
    error =
        check_api_feature_info<T>(deviceID, context, api_status, check_property,
                                  check_cap, mem_flags, image_type);
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
    log_info("Feature status: API - %s, compiler - %s\n",
             api_status == true ? "supported" : "not supported",
             compiler_status == true ? "supported" : "not supported");
    if (api_status != compiler_status)
    {
        log_info("%s - failed\n", feature_macro.c_str());
        supported = false;
        return TEST_FAIL;
    }
    else
    {
        log_info("%s - passed\n", feature_macro.c_str());
    }
    supported = api_status;
    return error;
}

int test_feature_macro_atomic_order_acq_rel(cl_device_id deviceID,
                                            cl_context context,
                                            std::string test_macro_name,
                                            bool& supported)
{
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_ORDER_ACQ_REL, 0,
        0, supported);
}

int test_feature_macro_atomic_order_seq_cst(cl_device_id deviceID,
                                            cl_context context,
                                            std::string test_macro_name,
                                            bool& supported)
{
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_ORDER_SEQ_CST, 0,
        0, supported);
}

int test_feature_macro_atomic_scope_device(cl_device_id deviceID,
                                           cl_context context,
                                           std::string test_macro_name,
                                           bool& supported)
{
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, CL_DEVICE_ATOMIC_SCOPE_DEVICE, 0,
        0, supported);
}

int test_feature_macro_atomic_scope_all_devices(cl_device_id deviceID,
                                                cl_context context,
                                                std::string test_macro_name,
                                                bool& supported)
{
    return test_feature_macro<cl_device_atomic_capabilities>(
        deviceID, context, test_macro_name,
        CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES, 0, 0, supported);
}

int test_feature_macro_3d_image_writes(cl_device_id deviceID,
                                       cl_context context,
                                       std::string test_macro_name,
                                       bool& supported)
{
    return test_feature_macro<cl_uint>(deviceID, context, test_macro_name, 0, 0,
                                       CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE
                                           | CL_MEM_KERNEL_READ_AND_WRITE,
                                       CL_MEM_OBJECT_IMAGE3D, supported);
}

int test_feature_macro_device_enqueue(cl_device_id deviceID, cl_context context,
                                      std::string test_macro_name,
                                      bool& supported)
{
    return test_feature_macro<cl_device_device_enqueue_capabilities>(
        deviceID, context, test_macro_name,
        CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES, CL_DEVICE_QUEUE_SUPPORTED, 0, 0,
        supported);
}

int test_feature_macro_generic_address_space(cl_device_id deviceID,
                                             cl_context context,
                                             std::string test_macro_name,
                                             bool& supported)
{
    return test_feature_macro<cl_bool>(deviceID, context, test_macro_name,
                                       CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                                       0, 0, 0, supported);
}

int test_feature_macro_pipes(cl_device_id deviceID, cl_context context,
                             std::string test_macro_name, bool& supported)
{
    return test_feature_macro<cl_bool>(deviceID, context, test_macro_name,
                                       CL_DEVICE_PIPE_SUPPORT, 0, 0, 0,
                                       supported);
}

int test_feature_macro_program_scope_global_variables(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    bool& supported)
{
    return test_feature_macro<size_t>(deviceID, context, test_macro_name,
                                      CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, 0, 0,
                                      0, supported);
}

int test_feature_macro_read_write_images(cl_device_id deviceID,
                                         cl_context context,
                                         std::string test_macro_name,
                                         bool& supported)
{
    return test_feature_macro<cl_uint>(deviceID, context, test_macro_name,
                                       CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, 0,
                                       0, 0, supported);
}

int test_feature_macro_subgroups(cl_device_id deviceID, cl_context context,
                                 std::string test_macro_name, bool& supported)
{
    return test_feature_macro<cl_uint>(deviceID, context, test_macro_name,
                                       CL_DEVICE_MAX_NUM_SUB_GROUPS, 0, 0, 0,
                                       supported);
}

int test_feature_macro_work_group_collective_functions(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    bool& supported)
{
    return test_feature_macro<cl_bool>(
        deviceID, context, test_macro_name,
        CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT, 0, 0, 0, supported);
}

int test_consistency_c_features_list(cl_device_id deviceID,
                                     std::vector<std::string> vec_to_cmp)
{
    log_info("\nComparison list of features: CL_DEVICE_OPENCL_C_FEATURES vs "
             "API/compiler queries.\n");
    cl_int error;
    size_t config_size;
    std::vector<cl_name_version> vec_device_feature;
    std::vector<std::string> vec_device_feature_names;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_FEATURES, 0, NULL,
                            &config_size);

    test_error(
        error,
        "clGetDeviceInfo asking for CL_DEVICE_OPENCL_C_FEATURES failed.\n");
    if (config_size == 0)
    {
        log_info("Empty list of CL_DEVICE_OPENCL_C_FEATURES returned by "
                 "clGetDeviceInfo on this device.\n");
    }
    else
    {
        int vec_elements = config_size / sizeof(cl_name_version);
        vec_device_feature.resize(vec_elements);
        error = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_FEATURES,
                                config_size, vec_device_feature.data(), 0);
        test_error(
            error,
            "clGetDeviceInfo asking for CL_DEVICE_OPENCL_C_FEATURES failed.\n");
    }
    for (auto each_f : vec_device_feature)
    {
        vec_device_feature_names.push_back(each_f.name);
    }
    sort(vec_to_cmp.begin(), vec_to_cmp.end());
    sort(vec_device_feature_names.begin(), vec_device_feature_names.end());

    bool result =
        std::equal(vec_device_feature_names.begin(),
                   vec_device_feature_names.end(), vec_to_cmp.begin());
    if (result)
    {
        log_info("Comparison list of features - passed\n");
    }
    else
    {
        log_info("Comparison list of features - failed\n");
        error = TEST_FAIL;
    }
    log_info(
        "Supported features based on CL_DEVICE_OPENCL_C_FEATURES API query:\n");
    for (auto each_f : vec_device_feature_names)
    {
        log_info("%s\n", each_f.c_str());
    }

    log_info("\nSupported features based on queries to API/compiler :\n");
    for (auto each_f : vec_to_cmp)
    {
        log_info("%s\n", each_f.c_str());
    }

    return error;
}

#define NEW_FEATURE_MACRO_TEST(feat)                                           \
    test_macro_name = "__opencl_c_" #feat;                                     \
    error |= test_feature_macro_##feat(deviceID, context, test_macro_name,     \
                                       supported);                             \
    if (supported) supported_features_vec.push_back(test_macro_name);


int test_features_macro(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    cl_int error = CL_SUCCESS;
    bool supported = false;
    std::string test_macro_name = "";
    std::vector<std::string> supported_features_vec;
    NEW_FEATURE_MACRO_TEST(program_scope_global_variables);
    NEW_FEATURE_MACRO_TEST(3d_image_writes);
    NEW_FEATURE_MACRO_TEST(atomic_order_acq_rel);
    NEW_FEATURE_MACRO_TEST(atomic_order_seq_cst);
    NEW_FEATURE_MACRO_TEST(atomic_scope_device);
    NEW_FEATURE_MACRO_TEST(atomic_scope_all_devices);
    NEW_FEATURE_MACRO_TEST(device_enqueue);
    NEW_FEATURE_MACRO_TEST(generic_address_space);
    NEW_FEATURE_MACRO_TEST(pipes);
    NEW_FEATURE_MACRO_TEST(read_write_images);
    NEW_FEATURE_MACRO_TEST(subgroups);
    NEW_FEATURE_MACRO_TEST(work_group_collective_functions);

    error |= test_consistency_c_features_list(deviceID, supported_features_vec);

    return error;
}

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
#include "errorHelpers.h"
#include "harness/featureHelpers.h"

const char* macro_supported_source = R"(kernel void enabled(global int * buf) {
        int n = get_global_id(0);
        buf[n] = 0;
        #ifndef %s
            #error Feature macro was not defined
        #endif
})";

const char* macro_not_supported_source =
    R"(kernel void not_enabled(global int * buf) {
        int n = get_global_id(0);
        buf[n] = 0;
        #ifdef %s
            #error Feature macro was defined
        #endif
})";

template <typename T>
cl_int check_api_feature_info_capabilities(cl_device_id deviceID,
                                           cl_context context, cl_bool& status,
                                           cl_device_info check_property,
                                           cl_bitfield check_cap)
{
    cl_int error = CL_SUCCESS;
    T response;
    error = clGetDeviceInfo(deviceID, check_property, sizeof(response),
                            &response, NULL);
    test_error(error, "clGetDeviceInfo failed.\n");

    if ((response & check_cap) == check_cap)
    {
        status = CL_TRUE;
    }
    else
    {
        status = CL_FALSE;
    }
    return error;
}

cl_int check_api_feature_info_support(cl_device_id deviceID, cl_context context,
                                      cl_bool& status,
                                      cl_device_info check_property)
{
    cl_int error = CL_SUCCESS;
    cl_bool response;
    error = clGetDeviceInfo(deviceID, check_property, sizeof(response),
                            &response, NULL);
    test_error(error, "clGetDeviceInfo failed.\n");
    status = response;
    return error;
}

template <typename T>
cl_int check_api_feature_info_number(cl_device_id deviceID, cl_context context,
                                     cl_bool& status,
                                     cl_device_info check_property)
{
    cl_int error = CL_SUCCESS;
    T response;
    error = clGetDeviceInfo(deviceID, check_property, sizeof(response),
                            &response, NULL);
    test_error(error, "clGetDeviceInfo failed.\n");
    if (response > 0)
    {
        status = CL_TRUE;
    }
    else
    {
        status = CL_FALSE;
    }
    return error;
}

cl_int check_api_feature_info_supported_image_formats(cl_device_id deviceID,
                                                      cl_context context,
                                                      cl_bool& status)
{
    cl_int error = CL_SUCCESS;
    cl_uint response = 0;
    cl_uint image_format_count;
    error = clGetSupportedImageFormats(context, CL_MEM_WRITE_ONLY,
                                       CL_MEM_OBJECT_IMAGE3D, 0, NULL,
                                       &image_format_count);
    test_error(error, "clGetSupportedImageFormats failed");
    response += image_format_count;
    error = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE,
                                       CL_MEM_OBJECT_IMAGE3D, 0, NULL,
                                       &image_format_count);
    test_error(error, "clGetSupportedImageFormats failed");
    response += image_format_count;
    error = clGetSupportedImageFormats(context, CL_MEM_KERNEL_READ_AND_WRITE,
                                       CL_MEM_OBJECT_IMAGE3D, 0, NULL,
                                       &image_format_count);
    test_error(error, "clGetSupportedImageFormats failed");
    response += image_format_count;
    if (response > 0)
    {
        status = CL_TRUE;
    }
    else
    {
        status = CL_FALSE;
    }
    return error;
}

cl_int check_compiler_feature_info(cl_device_id deviceID, cl_context context,
                                   std::string feature_macro, cl_bool& status)
{
    cl_int error = CL_SUCCESS;
    clProgramWrapper program_supported;
    clProgramWrapper program_not_supported;
    char kernel_supported_src[1024];
    char kernel_not_supported_src[1024];
    sprintf(kernel_supported_src, macro_supported_source,
            feature_macro.c_str());
    const char* ptr_supported = kernel_supported_src;
    const char* build_options = "-cl-std=CL3.0";

    error = create_single_kernel_helper_create_program(
        context, &program_supported, 1, &ptr_supported, build_options);
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
    status_supported = clBuildProgram(program_supported, 1, &deviceID,
                                      build_options, NULL, NULL);
    status_not_supported = clBuildProgram(program_not_supported, 1, &deviceID,
                                          build_options, NULL, NULL);
    if (status_supported != status_not_supported)
    {
        if (status_not_supported == CL_SUCCESS)
        {
            // kernel which verifies not supporting return passed
            status = CL_FALSE;
        }
        else
        {
            // kernel which verifies supporting return passed
            status = CL_TRUE;
        }
    }
    else
    {
        log_error("Error: The feature macro is defined and undefined "
                  "at the same time\n");
        error = OutputBuildLogs(program_supported, 1, &deviceID);
        test_error(error, "OutputBuildLogs failed.\n");
        error = OutputBuildLogs(program_not_supported, 1, &deviceID);
        test_error(error, "OutputBuildLogs failed.\n");
        return TEST_FAIL;
    }
    return error;
}

int feature_macro_verify_results(std::string test_macro_name,
                                 cl_bool api_status, cl_bool compiler_status,
                                 cl_bool& supported)
{
    cl_int error = TEST_PASS;
    log_info("Feature status: API - %s, compiler - %s\n",
             api_status == CL_TRUE ? "supported" : "not supported",
             compiler_status == CL_TRUE ? "supported" : "not supported");
    if (api_status != compiler_status)
    {
        log_info("%s - failed\n", test_macro_name.c_str());
        supported = CL_FALSE;
        return TEST_FAIL;
    }
    else
    {
        log_info("%s - passed\n", test_macro_name.c_str());
    }
    supported = api_status;
    return error;
}

int test_feature_macro_atomic_order_acq_rel(cl_device_id deviceID,
                                            cl_context context,
                                            std::string test_macro_name,
                                            cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_capabilities<cl_device_atomic_capabilities>(
        deviceID, context, api_status, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_ORDER_ACQ_REL);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_atomic_order_seq_cst(cl_device_id deviceID,
                                            cl_context context,
                                            std::string test_macro_name,
                                            cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());

    error = check_api_feature_info_capabilities<cl_device_atomic_capabilities>(
        deviceID, context, api_status, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_ORDER_SEQ_CST);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_atomic_scope_device(cl_device_id deviceID,
                                           cl_context context,
                                           std::string test_macro_name,
                                           cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_capabilities<cl_device_atomic_capabilities>(
        deviceID, context, api_status, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_SCOPE_DEVICE);
    if (error != CL_SUCCESS)
    {
        return error;
    }
    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_atomic_scope_all_devices(cl_device_id deviceID,
                                                cl_context context,
                                                std::string test_macro_name,
                                                cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_capabilities<cl_device_atomic_capabilities>(
        deviceID, context, api_status, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
        CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES);
    if (error != CL_SUCCESS)
    {
        return error;
    }
    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_3d_image_writes(cl_device_id deviceID,
                                       cl_context context,
                                       std::string test_macro_name,
                                       cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_supported_image_formats(deviceID, context,
                                                           api_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_device_enqueue(cl_device_id deviceID, cl_context context,
                                      std::string test_macro_name,
                                      cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_capabilities<
        cl_device_device_enqueue_capabilities>(
        deviceID, context, api_status, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
        CL_DEVICE_QUEUE_SUPPORTED);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_generic_address_space(cl_device_id deviceID,
                                             cl_context context,
                                             std::string test_macro_name,
                                             cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_support(
        deviceID, context, api_status, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_pipes(cl_device_id deviceID, cl_context context,
                             std::string test_macro_name, cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_support(deviceID, context, api_status,
                                           CL_DEVICE_PIPE_SUPPORT);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_program_scope_global_variables(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_number<size_t>(
        deviceID, context, api_status, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_read_write_images(cl_device_id deviceID,
                                         cl_context context,
                                         std::string test_macro_name,
                                         cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_number<cl_uint>(
        deviceID, context, api_status, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_subgroups(cl_device_id deviceID, cl_context context,
                                 std::string test_macro_name,
                                 cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_number<cl_uint>(
        deviceID, context, api_status, CL_DEVICE_MAX_NUM_SUB_GROUPS);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_work_group_collective_functions(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_support(
        deviceID, context, api_status,
        CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_images(cl_device_id deviceID, cl_context context,
                              std::string test_macro_name, cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_support(deviceID, context, api_status,
                                           CL_DEVICE_IMAGE_SUPPORT);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_fp64(cl_device_id deviceID, cl_context context,
                            std::string test_macro_name, cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());
    error = check_api_feature_info_capabilities<cl_device_fp_config>(
        deviceID, context, api_status, CL_DEVICE_DOUBLE_FP_CONFIG,
        CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_DENORM);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_integer_dot_product_input_4x8bit_packed(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());

    if (!is_extension_available(deviceID, "cl_khr_integer_dot_product"))
    {
        supported = false;
        return TEST_PASS;
    }

    error = check_api_feature_info_capabilities<
        cl_device_integer_dot_product_capabilities_khr>(
        deviceID, context, api_status,
        CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
        CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_integer_dot_product_input_4x8bit(
    cl_device_id deviceID, cl_context context, std::string test_macro_name,
    cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    log_info("\n%s ...\n", test_macro_name.c_str());

    if (!is_extension_available(deviceID, "cl_khr_integer_dot_product"))
    {
        supported = false;
        return TEST_PASS;
    }

    error = check_api_feature_info_capabilities<
        cl_device_integer_dot_product_capabilities_khr>(
        deviceID, context, api_status,
        CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
        CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
}

int test_feature_macro_int64(cl_device_id deviceID, cl_context context,
                             std::string test_macro_name, cl_bool& supported)
{
    cl_int error = TEST_FAIL;
    cl_bool api_status;
    cl_bool compiler_status;
    cl_int full_profile = 0;
    log_info("\n%s ...\n", test_macro_name.c_str());
    size_t ret_len;
    char profile[32] = { 0 };
    error = clGetDeviceInfo(deviceID, CL_DEVICE_PROFILE, sizeof(profile),
                            profile, &ret_len);
    test_error(error, "clGetDeviceInfo(CL_DEVICE_PROFILE) failed");
    if (ret_len < sizeof(profile) && strcmp(profile, "FULL_PROFILE") == 0)
    {
        full_profile = 1;
    }
    else if (ret_len < sizeof(profile)
             && strcmp(profile, "EMBEDDED_PROFILE") == 0)
    {
        full_profile = 0;
    }
    else
    {
        log_error("Unknown device profile: %s\n", profile);
        return TEST_FAIL;
    }

    if (full_profile)
    {
        api_status = CL_TRUE;
    }
    else
    {
        if (is_extension_available(deviceID, "cles_khr_int64"))
        {
            api_status = CL_TRUE;
        }
        else
        {
            cl_bool double_supported = CL_FALSE;
            error = check_api_feature_info_capabilities<cl_device_fp_config>(
                deviceID, context, double_supported, CL_DEVICE_DOUBLE_FP_CONFIG,
                CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN
                    | CL_FP_DENORM);
            test_error(error, "checking CL_DEVICE_DOUBLE_FP_CONFIG failed");
            if (double_supported == CL_FALSE)
            {
                api_status = CL_FALSE;
            }
            else
            {
                log_error("FP double type is supported and cles_khr_int64 "
                          "extension not supported\n");
                return TEST_FAIL;
            }
        }
    }

    error = check_compiler_feature_info(deviceID, context, test_macro_name,
                                        compiler_status);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    return feature_macro_verify_results(test_macro_name, api_status,
                                        compiler_status, supported);
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

    for (auto each_f : vec_to_cmp)
    {
        if (find(vec_device_feature_names.begin(),
                 vec_device_feature_names.end(), each_f)
            == vec_device_feature_names.end())
        {
            log_info("Comparison list of features - failed - missing %s\n",
                     each_f.c_str());
            return TEST_FAIL;
        }
    }

    log_info("Comparison list of features - passed\n");

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

    // Note: Not checking that the feature array is empty for the compiler not
    // available case because the specification says "For devices that do not
    // support compilation from OpenCL C source, this query may return an empty
    // array."  It "may" return an empty array implies that an implementation
    // also "may not".
    check_compiler_available(deviceID);

    int error = TEST_PASS;
    cl_bool supported = CL_FALSE;
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
    NEW_FEATURE_MACRO_TEST(images);
    NEW_FEATURE_MACRO_TEST(fp64);
    NEW_FEATURE_MACRO_TEST(int64);
    NEW_FEATURE_MACRO_TEST(integer_dot_product_input_4x8bit);
    NEW_FEATURE_MACRO_TEST(integer_dot_product_input_4x8bit_packed);

    error |= test_consistency_c_features_list(deviceID, supported_features_vec);

    return error;
}

// This test checks that a supported feature comes with other required features
int test_features_macro_coupling(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    OpenCLCFeatures features;
    int error = get_device_cl_c_features(deviceID, features);
    if (error)
    {
        log_error("Couldn't query OpenCL C features for the device!\n");
        return TEST_FAIL;
    }

    if (features.supports__opencl_c_3d_image_writes
        && !features.supports__opencl_c_images)
    {
        log_error("OpenCL C compilers that define the feature macro "
                  "__opencl_c_3d_image_writes must also define the feature "
                  "macro __opencl_c_images!\n");
        return TEST_FAIL;
    }

    if (features.supports__opencl_c_device_enqueue
        && !(features.supports__opencl_c_program_scope_global_variables
             && features.supports__opencl_c_generic_address_space))
    {
        log_error("OpenCL C compilers that define the feature macro "
                  "__opencl_c_device_enqueue must also define "
                  "__opencl_c_generic_address_space and "
                  "__opencl_c_program_scope_global_variables!\n");
        return TEST_FAIL;
    }

    if (features.supports__opencl_c_pipes
        && !features.supports__opencl_c_generic_address_space)
    {
        log_error("OpenCL C compilers that define the feature macro "
                  "__opencl_c_pipes must also define the feature macro "
                  "__opencl_c_generic_address_space!\n");
        return TEST_FAIL;
    }

    if (features.supports__opencl_c_read_write_images
        && !features.supports__opencl_c_images)
    {
        log_error("OpenCL C compilers that define the feature macro "
                  "__opencl_c_read_write_images must also define the feature "
                  "macro __opencl_c_images!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

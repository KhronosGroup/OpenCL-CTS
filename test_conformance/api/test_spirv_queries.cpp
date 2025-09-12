//
// Copyright (c) 2025 The Khronos Group Inc.
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
#include <algorithm>
#include <map>
#include <vector>

#ifdef SPIRV_FILES_AVAILABLE

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

static bool is_spirv_version_supported(cl_device_id deviceID,
                                       const std::string& version)
{
    std::string ilVersions = get_device_il_version_string(deviceID);
    return ilVersions.find(version) != std::string::npos;
}

static int doQueries(cl_device_id device,
                     std::vector<const char*>& extendedInstructionSets,
                     std::vector<const char*>& extensions,
                     std::vector<cl_uint>& capabilities)
{
    cl_int error = CL_SUCCESS;

    size_t size = 0;
    error =
        clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR,
                        0, nullptr, &size);
    test_error(error,
               "clGetDeviceInfo failed for "
               "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR size\n");

    extendedInstructionSets.resize(size / sizeof(const char*));
    error =
        clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR,
                        size, extendedInstructionSets.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for "
               "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR\n");

    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, 0, nullptr,
                            &size);
    test_error(
        error,
        "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR size\n");

    extensions.resize(size / sizeof(const char*));
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, size,
                            extensions.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR\n");

    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR, 0,
                            nullptr, &size);
    test_error(
        error,
        "clGetDeviceInfo failed for CL_DEVICE_SPIRV_CAPABILITIES_KHR size\n");

    capabilities.resize(size / sizeof(cl_uint));
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR, size,
                            capabilities.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_SPIRV_CAPABILITIES_KHR\n");

    return CL_SUCCESS;
}

static int findRequirements(cl_device_id device,
                            std::vector<const char*>& extendedInstructionSets,
                            std::vector<const char*>& extensions,
                            std::vector<cl_uint>& capabilities)
{
    cl_int error = CL_SUCCESS;

    auto version = get_device_cl_version(device);
    auto ilVersions = get_device_il_version_string(device);

    // If no SPIR-V versions are supported, there are no requirements.
    if (ilVersions.find("SPIR-V") == std::string::npos)
    {
        return CL_SUCCESS;
    }

    cl_bool deviceImageSupport = CL_FALSE;
    cl_bool deviceReadWriteImageSupport = CL_FALSE;
    cl_bool deviceSubGroupsSupport = CL_FALSE;
    cl_bool deviceGenericAddressSpaceSupport = CL_FALSE;
    cl_bool deviceWorkGroupCollectiveFunctionsSupport = CL_FALSE;
    cl_bool devicePipeSupport = CL_FALSE;
    cl_bool deviceDeviceEnqueueSupport = CL_FALSE;
    cl_device_integer_dot_product_capabilities_khr
        deviceIntegerDotProductCapabilities = 0;
    cl_device_fp_atomic_capabilities_ext deviceFp32AtomicCapabilities = 0;
    cl_device_fp_atomic_capabilities_ext deviceFp16AtomicCapabilities = 0;
    cl_device_fp_atomic_capabilities_ext deviceFp64AtomicCapabilities = 0;

    error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(deviceImageSupport), &deviceImageSupport,
                            nullptr);
    test_error(error, "clGetDeviceInfo failed for CL_DEVICE_IMAGE_SUPPORT\n");

    if (version >= Version(2, 0))
    {
        cl_uint deviceMaxReadWriteImageArgs = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                sizeof(deviceMaxReadWriteImageArgs),
                                &deviceMaxReadWriteImageArgs, nullptr);
        test_error(
            error,
            "clGetDeviceInfo failed for CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS\n");

        deviceReadWriteImageSupport =
            deviceMaxReadWriteImageArgs != 0 ? CL_TRUE : CL_FALSE;
    }

    if (version >= Version(2, 1))
    {
        cl_uint deviceMaxNumSubGroups = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                sizeof(deviceMaxNumSubGroups),
                                &deviceMaxNumSubGroups, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for CL_DEVICE_MAX_NUM_SUB_GROUPS\n");

        deviceSubGroupsSupport =
            deviceMaxNumSubGroups != 0 ? CL_TRUE : CL_FALSE;
    }
    else if (is_extension_available(device, "cl_khr_subgroups"))
    {
        deviceSubGroupsSupport = CL_TRUE;
    }

    if (version >= Version(3, 0))
    {
        error = clGetDeviceInfo(device, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                                sizeof(deviceGenericAddressSpaceSupport),
                                &deviceGenericAddressSpaceSupport, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT\n");

        error = clGetDeviceInfo(
            device, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
            sizeof(deviceWorkGroupCollectiveFunctionsSupport),
            &deviceWorkGroupCollectiveFunctionsSupport, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT\n");

        error = clGetDeviceInfo(device, CL_DEVICE_PIPE_SUPPORT,
                                sizeof(devicePipeSupport), &devicePipeSupport,
                                nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for CL_DEVICE_PIPE_SUPPORT\n");

        cl_device_device_enqueue_capabilities deviceDeviceEnqueueCapabilities =
            0;
        error = clGetDeviceInfo(device, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                                sizeof(deviceDeviceEnqueueCapabilities),
                                &deviceDeviceEnqueueCapabilities, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES\n");

        deviceDeviceEnqueueSupport =
            deviceDeviceEnqueueCapabilities != 0 ? CL_TRUE : CL_FALSE;
    }
    else if (version >= Version(2, 0))
    {
        deviceGenericAddressSpaceSupport = CL_TRUE;
        deviceWorkGroupCollectiveFunctionsSupport = CL_TRUE;
        devicePipeSupport = CL_TRUE;
        deviceDeviceEnqueueSupport = CL_TRUE;
    }

    if (is_extension_available(device, "cl_khr_integer_dot_product"))
    {
        error = clGetDeviceInfo(device,
                                CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                                sizeof(deviceIntegerDotProductCapabilities),
                                &deviceIntegerDotProductCapabilities, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR\n");
    }

    if (is_extension_available(device, "cl_ext_float_atomics"))
    {
        error =
            clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
                            sizeof(deviceFp32AtomicCapabilities),
                            &deviceFp32AtomicCapabilities, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT\n");

        error =
            clGetDeviceInfo(device, CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT,
                            sizeof(deviceFp16AtomicCapabilities),
                            &deviceFp16AtomicCapabilities, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT\n");

        error =
            clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                            sizeof(deviceFp64AtomicCapabilities),
                            &deviceFp64AtomicCapabilities, nullptr);
        test_error(error,
                   "clGetDeviceInfo failed for "
                   "CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT\n");
    }

    // Required.
    extendedInstructionSets.push_back("OpenCL.std");

    capabilities.push_back(spv::CapabilityAddresses);
    capabilities.push_back(spv::CapabilityFloat16Buffer);
    capabilities.push_back(spv::CapabilityInt16);
    capabilities.push_back(spv::CapabilityInt8);
    capabilities.push_back(spv::CapabilityKernel);
    capabilities.push_back(spv::CapabilityLinkage);
    capabilities.push_back(spv::CapabilityVector16);

    // Required for FULL_PROFILE devices, or devices supporting
    // cles_khr_int64.
    if (gHasLong)
    {
        capabilities.push_back(spv::CapabilityInt64);
    }

    // Required for devices supporting images.
    if (deviceImageSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityImage1D);
        capabilities.push_back(spv::CapabilityImageBasic);
        capabilities.push_back(spv::CapabilityImageBuffer);
        capabilities.push_back(spv::CapabilityLiteralSampler);
        capabilities.push_back(spv::CapabilitySampled1D);
        capabilities.push_back(spv::CapabilitySampledBuffer);
    }

    // Required for devices supporting SPIR-V 1.6.
    if (ilVersions.find("SPIR-V_1.6") != std::string::npos)
    {
        capabilities.push_back(spv::CapabilityUniformDecoration);
    }

    // Required for devices supporting read-write images.
    if (deviceReadWriteImageSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityImageReadWrite);
    }

    // Required for devices supporting the generic address space.
    if (deviceGenericAddressSpaceSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityGenericPointer);
    }

    // Required for devices supporting sub-groups or work-group collective
    // functions.
    if (deviceSubGroupsSupport == CL_TRUE
        || deviceWorkGroupCollectiveFunctionsSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityGroups);
    }

    // Required for devices supporting pipes.
    if (devicePipeSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityPipes);
    }

    // Required for devices supporting device-side enqueue.
    if (deviceDeviceEnqueueSupport == CL_TRUE)
    {
        capabilities.push_back(spv::CapabilityDeviceEnqueue);
    }

    // Required for devices supporting SPIR-V 1.1 and OpenCL 2.2.
    if (ilVersions.find("SPIR-V_1.1") != std::string::npos
        && version == Version(2, 2))
    {
        capabilities.push_back(spv::CapabilityPipeStorage);
    }

    // Required for devices supporting SPIR-V 1.1 and either OpenCL 2.2 or
    // OpenCL 3.0 devices supporting sub-groups.
    if (ilVersions.find("SPIR-V_1.1") != std::string::npos
        && (version == Version(2, 2)
            || (version >= Version(3, 0) && deviceSubGroupsSupport == CL_TRUE)))
    {
        capabilities.push_back(spv::CapabilitySubgroupDispatch);
    }

    // Required for devices supporting cl_khr_expect_assume.
    if (is_extension_available(device, "cl_khr_expect_assume"))
    {
        extensions.push_back("SPV_KHR_expect_assume");
        capabilities.push_back(spv::CapabilityExpectAssumeKHR);
    }

    // Required for devices supporting cl_khr_extended_bit_ops.
    if (is_extension_available(device, "cl_khr_extended_bit_ops"))
    {
        extensions.push_back("SPV_KHR_bit_instructions");
        capabilities.push_back(spv::CapabilityBitInstructions);
    }

    // Required for devices supporting half-precision floating-point
    // (cl_khr_fp16).
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        capabilities.push_back(spv::CapabilityFloat16);
    }

    // Required for devices supporting double-precision floating-point
    // (cl_khr_fp64).
    if (is_extension_available(device, "cl_khr_fp64"))
    {
        capabilities.push_back(spv::CapabilityFloat64);
    }

    // Required for devices supporting 64-bit atomics
    // (cl_khr_int64_base_atomics or cl_khr_int64_extended_atomics).
    if (is_extension_available(device, "cl_khr_int64_base_atomics")
        || is_extension_available(device, "cl_khr_int64_extended_atomics"))
    {
        capabilities.push_back(spv::CapabilityInt64Atomics);
    }

    // Required for devices supporting cl_khr_integer_dot_product.
    if (is_extension_available(device, "cl_khr_integer_dot_product"))
    {
        extensions.push_back("SPV_KHR_integer_dot_product");
        capabilities.push_back(spv::CapabilityDotProduct);
        capabilities.push_back(spv::CapabilityDotProductInput4x8BitPacked);
    }

    // Required for devices supporting cl_khr_integer_dot_product and
    // CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR.
    if (is_extension_available(device, "cl_khr_integer_dot_product")
        && (deviceIntegerDotProductCapabilities
            & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR))
    {
        capabilities.push_back(spv::CapabilityDotProductInput4x8Bit);
    }

    // Required for devices supporting cl_khr_kernel_clock.
    if (is_extension_available(device, "cl_khr_kernel_clock"))
    {
        extensions.push_back("SPV_KHR_shader_clock");
        capabilities.push_back(spv::CapabilityShaderClockKHR);
    }

    // Required for devices supporting both cl_khr_mipmap_image and
    // cl_khr_mipmap_image_writes.
    if (is_extension_available(device, "cl_khr_mipmap_image")
        && is_extension_available(device, "cl_khr_mipmap_image_writes"))
    {
        capabilities.push_back(spv::CapabilityImageMipmap);
    }

    // Required for devices supporting cl_khr_spirv_extended_debug_info.
    if (is_extension_available(device, "cl_khr_spirv_extended_debug_info"))
    {
        extendedInstructionSets.push_back("OpenCL.DebugInfo.100");
    }

    // Required for devices supporting cl_khr_spirv_linkonce_odr.
    if (is_extension_available(device, "cl_khr_spirv_linkonce_odr"))
    {
        extensions.push_back("SPV_KHR_linkonce_odr");
    }

    // Required for devices supporting
    // cl_khr_spirv_no_integer_wrap_decoration.
    if (is_extension_available(device,
                               "cl_khr_spirv_no_integer_wrap_decoration"))
    {
        extensions.push_back("SPV_KHR_no_integer_wrap_decoration");
    }

    // Required for devices supporting cl_khr_subgroup_ballot.
    if (is_extension_available(device, "cl_khr_subgroup_ballot"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniformBallot);
    }

    // Required for devices supporting cl_khr_subgroup_clustered_reduce.
    if (is_extension_available(device, "cl_khr_subgroup_clustered_reduce"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniformClustered);
    }

    // Required for devices supporting cl_khr_subgroup_named_barrier.
    if (is_extension_available(device, "cl_khr_subgroup_named_barrier"))
    {
        capabilities.push_back(spv::CapabilityNamedBarrier);
    }

    // Required for devices supporting
    // cl_khr_subgroup_non_uniform_arithmetic.
    if (is_extension_available(device,
                               "cl_khr_subgroup_non_uniform_arithmetic"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniformArithmetic);
    }

    // Required for devices supporting cl_khr_subgroup_non_uniform_vote.
    if (is_extension_available(device, "cl_khr_subgroup_non_uniform_vote"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniform);
        capabilities.push_back(spv::CapabilityGroupNonUniformVote);
    }

    // Required for devices supporting cl_khr_subgroup_rotate.
    if (is_extension_available(device, "cl_khr_subgroup_rotate"))
    {
        extensions.push_back("SPV_KHR_subgroup_rotate");
        capabilities.push_back(spv::CapabilityGroupNonUniformRotateKHR);
    }

    // Required for devices supporting cl_khr_subgroup_shuffle.
    if (is_extension_available(device, "cl_khr_subgroup_shuffle"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniformShuffle);
    }

    // Required for devices supporting cl_khr_subgroup_shuffle_relative.
    if (is_extension_available(device, "cl_khr_subgroup_shuffle_relative"))
    {
        capabilities.push_back(spv::CapabilityGroupNonUniformShuffleRelative);
    }

    // Required for devices supporting cl_khr_work_group_uniform_arithmetic.
    if (is_extension_available(device, "cl_khr_work_group_uniform_arithmetic"))
    {
        extensions.push_back("SPV_KHR_uniform_group_instructions");
        capabilities.push_back(spv::CapabilityGroupUniformArithmeticKHR);
    }

    // Required for devices supporting cl_ext_float_atomics and fp32 atomic
    // adds.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp32AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)))
    {
        capabilities.push_back(spv::CapabilityAtomicFloat32AddEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp32 atomic
    // min and max.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp32AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)))
    {
        capabilities.push_back(spv::CapabilityAtomicFloat32MinMaxEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp16 atomic
    // adds.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp16AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)))
    {
        extensions.push_back("SPV_EXT_shader_atomic_float16_add");
        capabilities.push_back(spv::CapabilityAtomicFloat16AddEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp16 atomic
    // min and max.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp16AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)))
    {
        capabilities.push_back(spv::CapabilityAtomicFloat16MinMaxEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp64 atomic
    // adds.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp64AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)))
    {
        capabilities.push_back(spv::CapabilityAtomicFloat64AddEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp64 atomic
    // min and max.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && (deviceFp64AtomicCapabilities
            & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
               | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)))
    {
        capabilities.push_back(spv::CapabilityAtomicFloat64MinMaxEXT);
    }

    // Required for devices supporting cl_ext_float_atomics and fp16, fp32,
    // or fp64 atomic min or max.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && ((deviceFp32AtomicCapabilities
             & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))
            || (deviceFp16AtomicCapabilities
                & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                   | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))
            || (deviceFp64AtomicCapabilities
                & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT
                   | CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT))))
    {
        extensions.push_back("SPV_EXT_shader_atomic_float_min_max");
    }

    // Required for devices supporting cl_ext_float_atomics and fp32 or fp64
    // atomic adds.
    if (is_extension_available(device, "cl_ext_float_atomics")
        && ((deviceFp32AtomicCapabilities
             & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
                | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT))
            || (deviceFp64AtomicCapabilities
                & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT
                   | CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT))))
    {
        extensions.push_back("SPV_EXT_shader_atomic_float_add");
    }

    // Required for devices supporting cl_intel_bfloat16_conversions.
    if (is_extension_available(device, "cl_intel_bfloat16_conversions"))
    {
        extensions.push_back("SPV_INTEL_bfloat16_conversion");
        capabilities.push_back(spv::CapabilityBFloat16ConversionINTEL);
    }

    // Required for devices supporting
    // cl_intel_spirv_device_side_avc_motion_estimation.
    if (is_extension_available(
            device, "cl_intel_spirv_device_side_avc_motion_estimation"))
    {
        extensions.push_back("SPV_INTEL_device_side_avc_motion_estimation");
        capabilities.push_back(
            spv::CapabilitySubgroupAvcMotionEstimationChromaINTEL);
        capabilities.push_back(spv::CapabilitySubgroupAvcMotionEstimationINTEL);
        capabilities.push_back(
            spv::CapabilitySubgroupAvcMotionEstimationIntraINTEL);
    }

    // Required for devices supporting cl_intel_spirv_media_block_io.
    if (is_extension_available(device, "cl_intel_spirv_media_block_io"))
    {
        extensions.push_back("SPV_INTEL_media_block_io");
        capabilities.push_back(spv::CapabilitySubgroupImageMediaBlockIOINTEL);
    }

    // Required for devices supporting cl_intel_spirv_subgroups.
    if (is_extension_available(device, "cl_intel_spirv_subgroups"))
    {
        extensions.push_back("SPV_INTEL_subgroups");
        capabilities.push_back(spv::CapabilitySubgroupBufferBlockIOINTEL);
        capabilities.push_back(spv::CapabilitySubgroupImageBlockIOINTEL);
        capabilities.push_back(spv::CapabilitySubgroupShuffleINTEL);
    }

    // Required for devices supporting cl_intel_split_work_group_barrier.
    if (is_extension_available(device, "cl_intel_split_work_group_barrier"))
    {
        extensions.push_back("SPV_INTEL_split_barrier");
        capabilities.push_back(spv::CapabilitySplitBarrierINTEL);
    }

    // Required for devices supporting cl_intel_subgroup_buffer_prefetch.
    if (is_extension_available(device, "cl_intel_subgroup_buffer_prefetch"))
    {
        extensions.push_back("SPV_INTEL_subgroup_buffer_prefetch");
        capabilities.push_back(spv::CapabilitySubgroupBufferPrefetchINTEL);
    }

    return CL_SUCCESS;
}

REGISTER_TEST(spirv_query_requirements)
{
    if (!is_extension_available(device, "cl_khr_spirv_queries"))
    {
        log_info("cl_khr_spirv_queries is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error;

    std::vector<const char*> queriedExtendedInstructionSets;
    std::vector<const char*> queriedExtensions;
    std::vector<cl_uint> queriedCapabilities;

    error = doQueries(device, queriedExtendedInstructionSets, queriedExtensions,
                      queriedCapabilities);
    test_error_fail(error, "Unable to perform SPIR-V queries");

    std::vector<const char*> requiredExtendedInstructionSets;
    std::vector<const char*> requiredExtensions;
    std::vector<cl_uint> requiredCapabilities;
    error = findRequirements(device, requiredExtendedInstructionSets,
                             requiredExtensions, requiredCapabilities);
    test_error_fail(error, "Unable to find SPIR-V requirements");

    for (auto check : requiredExtendedInstructionSets)
    {
        auto cmp = [=](const char* queried) {
            return strcmp(check, queried) == 0;
        };
        auto it = std::find_if(queriedExtendedInstructionSets.begin(),
                               queriedExtendedInstructionSets.end(), cmp);
        if (it == queriedExtendedInstructionSets.end())
        {
            test_fail("Missing required extended instruction set: %s\n", check);
        }
    }

    for (auto check : requiredExtensions)
    {
        auto cmp = [=](const char* queried) {
            return strcmp(check, queried) == 0;
        };
        auto it = std::find_if(queriedExtensions.begin(),
                               queriedExtensions.end(), cmp);
        if (it == queriedExtensions.end())
        {
            test_fail("Missing required extension: %s\n", check);
        }
    }

    for (auto check : requiredCapabilities)
    {
        if (std::find(queriedCapabilities.begin(), queriedCapabilities.end(),
                      check)
            == queriedCapabilities.end())
        {
            test_fail(
                "Missing required capability: %s\n",
                spv::CapabilityToString(static_cast<spv::Capability>(check)));
        }
    }

    // Find any extraneous capabilities (informational):
    for (auto check : queriedCapabilities)
    {
        if (std::find(requiredCapabilities.begin(), requiredCapabilities.end(),
                      check)
            == requiredCapabilities.end())
        {
            log_info(
                "Found non-required capability: %s\n",
                spv::CapabilityToString(static_cast<spv::Capability>(check)));
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv_query_dependencies)
{
    if (!is_extension_available(device, "cl_khr_spirv_queries"))
    {
        log_info("cl_khr_spirv_queries is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error;

    std::vector<const char*> queriedExtendedInstructionSets;
    std::vector<const char*> queriedExtensions;
    std::vector<cl_uint> queriedCapabilities;

    error = doQueries(device, queriedExtendedInstructionSets, queriedExtensions,
                      queriedCapabilities);
    test_error_fail(error, "Unable to perform SPIR-V queries");

    struct CapabilityDependencies
    {
        std::vector<std::string> extensions;
        std::string version;
    };

    std::map<spv::Capability, CapabilityDependencies> dependencies;

#define SPIRV_CAPABILITY_VERSION_DEPENDENCY(_cap, _ver)                        \
    dependencies[spv::Capability##_cap].version = _ver;
#define SPIRV_CAPABILITY_EXTENSION_DEPENDENCY(_cap, _ext)                      \
    dependencies[spv::Capability##_cap].extensions.push_back(_ext);
#include "spirv_capability_deps.def"

    // For each queried SPIR-V capability, ensure that either that any SPIR-V
    // version dependencies or SPIR-V extension dependencies are satisfied.

    for (auto check : queriedCapabilities)
    {
        // Log and skip any unknown capabilities
        auto it = dependencies.find(static_cast<spv::Capability>(check));
        if (it == dependencies.end())
        {
            log_info(
                "No known dependencies for queried capability %s!\n",
                spv::CapabilityToString(static_cast<spv::Capability>(check)));
            continue;
        }

        // Check if a SPIR-V version dependency is satisfied
        const auto& version_dep = it->second.version;
        if (!version_dep.empty()
            && is_spirv_version_supported(device, version_dep))
        {
            continue;
        }

        // Check if a SPIR-V extension dependency is satisfied
        bool found = false;
        for (const auto& extension_dep : it->second.extensions)
        {
            if (std::find(queriedExtensions.begin(), queriedExtensions.end(),
                          extension_dep)
                != queriedExtensions.end())
            {
                found = true;
                break;
            }
        }
        if (found)
        {
            continue;
        }

        // If we get here then the capability has an unsatisfied dependency.
        log_error("Couldn't find a dependency for queried capability %s!\n",
                  spv::CapabilityToString(static_cast<spv::Capability>(check)));
        if (!version_dep.empty())
        {
            log_error("Checked for SPIR-V version %s.\n", version_dep.c_str());
        }
        for (const auto& extension_dep : it->second.extensions)
        {
            log_error("Checked for SPIR-V extension %s.n",
                      extension_dep.c_str());
        }
        return TEST_FAIL;
    }

    return TEST_PASS;
}

#else

REGISTER_TEST(spirv_query_extension_check_absence_of_files)
{
    if (is_extension_available(device, "cl_khr_spirv_queries"))
    {
        log_info("cl_khr_spirv_queries is supported; Enable proper tests!\n");
        return TEST_FAIL;
    }
    return TEST_PASS;
}
#endif

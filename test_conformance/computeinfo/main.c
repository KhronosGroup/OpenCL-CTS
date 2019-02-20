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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/kernelHelpers.h"

typedef struct
{
    cl_device_type device_type;
    const char*    device_type_name;
    unsigned       num_devices;
    cl_device_id*  devices;
    // more infos here
} device_info;

device_info device_infos[] =
{
    {CL_DEVICE_TYPE_DEFAULT,     "CL_DEVICE_TYPE_DEFAULT",     -1, NULL},
    {CL_DEVICE_TYPE_CPU,         "CL_DEVICE_TYPE_CPU",         -1, NULL},
    {CL_DEVICE_TYPE_GPU,         "CL_DEVICE_TYPE_GPU",         -1, NULL},
    {CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", -1, NULL},
    {CL_DEVICE_TYPE_ALL,         "CL_DEVICE_TYPE_ALL",         -1, NULL},
};

// config types
enum
{
    type_cl_device_type,
    type_cl_device_fp_config,
    type_cl_device_mem_cache_type,
    type_cl_local_mem_type,
    type_cl_device_exec_capabilities,
    type_cl_command_queue_properties,
    type_cl_device_id,
    type_cl_device_affinity_domain,
    type_cl_uint,
    type_size_t,
    type_size_t_arr,
    type_cl_ulong,
    type_string,
};

typedef union
{
    cl_device_type                  type;
    cl_device_fp_config             fp_config;
    cl_device_mem_cache_type        mem_cache_type;
    cl_device_local_mem_type        local_mem_type;
    cl_device_exec_capabilities     exec_capabilities;
    cl_command_queue_properties     queue_properties;
    cl_device_id                    device_id;
    cl_device_affinity_domain       affinity_domain;
    cl_int                          uint;
    size_t                          sizet;
    size_t                          sizet_arr[3];
    cl_ulong                        ull;
    char                            string[1024];
} config_data;

typedef struct
{
    cl_device_info        opcode;
    const char*           opcode_name;
    int                   config_type;
    config_data           config;
} config_info;

config_info config_infos[] =
{
    {CL_DEVICE_TYPE, "CL_DEVICE_TYPE", type_cl_device_type, {0}},
    {CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID", type_cl_uint, {0}},
    {CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS", type_cl_uint, {0}},
    {CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", type_cl_uint, {0}},
    {CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES", type_size_t_arr, {0}},
    {CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE", type_size_t, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,"CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,"CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", type_cl_uint, {0}},
    {CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,"CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,"CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE", type_cl_uint, {0}},
    {CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF", type_cl_uint, {0}},

    {CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY", type_cl_uint, {0}},
    {CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS", type_cl_uint, {0}},
    {CL_DEVICE_MAX_READ_IMAGE_ARGS, "CL_DEVICE_MAX_READ_IMAGE_ARGS", type_cl_uint, {0}},
    {CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "CL_DEVICE_MAX_WRITE_IMAGE_ARGS", type_cl_uint, {0}},
    {CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE", type_cl_ulong, {0}},
    {CL_DEVICE_IMAGE2D_MAX_WIDTH,  "CL_DEVICE_IMAGE2D_MAX_WIDTH",  type_size_t,  {0}},
    {CL_DEVICE_IMAGE2D_MAX_HEIGHT, "CL_DEVICE_IMAGE2D_MAX_HEIGHT", type_size_t,  {0}},
    {CL_DEVICE_IMAGE3D_MAX_WIDTH,  "CL_DEVICE_IMAGE3D_MAX_WIDTH",  type_size_t,  {0}},
    {CL_DEVICE_IMAGE3D_MAX_HEIGHT, "CL_DEVICE_IMAGE3D_MAX_HEIGHT", type_size_t,  {0}},
    {CL_DEVICE_IMAGE3D_MAX_DEPTH,  "CL_DEVICE_IMAGE3D_MAX_DEPTH",  type_size_t,  {0}},
    {CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE",  type_size_t,  {0}},
    {CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE",  type_size_t,  {0}},
    {CL_DEVICE_IMAGE_SUPPORT,      "CL_DEVICE_IMAGE_SUPPORT",      type_cl_uint, {0}},
    {CL_DEVICE_MAX_PARAMETER_SIZE, "CL_DEVICE_MAX_PARAMETER_SIZE", type_size_t,  {0}},
    {CL_DEVICE_MAX_SAMPLERS,       "CL_DEVICE_MAX_SAMPLERS",       type_cl_uint, {0}},

    {CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN", type_cl_uint, {0}},
    {CL_DEVICE_SINGLE_FP_CONFIG, "CL_DEVICE_SINGLE_FP_CONFIG", type_cl_device_fp_config, {0}},
    {CL_DEVICE_DOUBLE_FP_CONFIG, "CL_DEVICE_DOUBLE_FP_CONFIG", type_cl_device_fp_config, {0}},
    {CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,"CL_DEVICE_GLOBAL_MEM_CACHE_TYPE",  type_cl_device_mem_cache_type, {0}},
    {CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,"CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE",type_cl_uint, {0}},
    {CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE", type_cl_ulong, {0}},
    {CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE", type_cl_ulong, {0}},

    {CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", type_cl_ulong, {0}},
    {CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS", type_cl_uint, {0}},
    {CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE", type_cl_local_mem_type, {0}},
    {CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE", type_cl_ulong, {0}},
    {CL_DEVICE_ERROR_CORRECTION_SUPPORT,"CL_DEVICE_ERROR_CORRECTION_SUPPORT", type_cl_uint, {0}},
    {CL_DEVICE_HOST_UNIFIED_MEMORY,"CL_DEVICE_HOST_UNIFIED_MEMORY", type_cl_uint, {0}},
    {CL_DEVICE_PROFILING_TIMER_RESOLUTION, "CL_DEVICE_PROFILING_TIMER_RESOLUTION", type_size_t, {0}},
    {CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE", type_cl_uint, {0}},
    {CL_DEVICE_AVAILABLE,"CL_DEVICE_AVAILABLE",     type_cl_uint, {0}},
    {CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE", type_cl_uint, {0}},
    {CL_DEVICE_LINKER_AVAILABLE, "CL_DEVICE_LINKER_AVAILABLE", type_cl_uint, {0}},

    {CL_DEVICE_BUILT_IN_KERNELS, "CL_DEVICE_BUILT_IN_KERNELS", type_string, {0}},

    {CL_DEVICE_PRINTF_BUFFER_SIZE, "CL_DEVICE_PRINTF_BUFFER_SIZE", type_size_t, {0}},
    {CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC", type_cl_uint, {0}},

    {CL_DEVICE_PARENT_DEVICE, "CL_DEVICE_PARENT_DEVICE", type_cl_device_id, {0}},
    {CL_DEVICE_PARTITION_MAX_SUB_DEVICES, "CL_DEVICE_PARTITION_MAX_SUB_DEVICES", type_cl_uint, {0}},
    {CL_DEVICE_PARTITION_AFFINITY_DOMAIN, "CL_DEVICE_PARTITION_AFFINITY_DOMAIN", type_cl_device_affinity_domain, {0}},
    {CL_DEVICE_REFERENCE_COUNT, "CL_DEVICE_REFERENCE_COUNT", type_cl_uint, {0}},

    {CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES", type_cl_device_exec_capabilities, {0}},
    {CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES", type_cl_command_queue_properties, {0}},
    {CL_DEVICE_NAME,       "CL_DEVICE_NAME",       type_string, {0}},
    {CL_DEVICE_VENDOR,     "CL_DEVICE_VENDOR",     type_string, {0}},
    {CL_DRIVER_VERSION,    "CL_DRIVER_VERSION",    type_string, {0}},
    {CL_DEVICE_PROFILE,    "CL_DEVICE_PROFILE",    type_string, {0}},
    {CL_DEVICE_VERSION,    "CL_DEVICE_VERSION",    type_string, {0}},
    {CL_DEVICE_OPENCL_C_VERSION, "CL_DEVICE_OPENCL_C_VERSION", type_string, {0}},
    {CL_DEVICE_EXTENSIONS, "CL_DEVICE_EXTENSIONS", type_string, {0}},
};

int getConfigInfo(cl_device_id device, config_info* info)
{
    int err = CL_SUCCESS;
    int size_err = 0;
    size_t config_size_ret;
    switch(info->config_type)
    {
        case type_cl_device_type:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.type), &info->config.type, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.type);
            break;
        case type_cl_device_fp_config:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.fp_config), &info->config.fp_config, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.fp_config);
            break;
        case type_cl_device_mem_cache_type:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.mem_cache_type), &info->config.mem_cache_type, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.mem_cache_type);
            break;
        case type_cl_local_mem_type:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.local_mem_type), &info->config.local_mem_type, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.local_mem_type);
            break;
        case type_cl_device_exec_capabilities:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.exec_capabilities), &info->config.exec_capabilities, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.exec_capabilities);
            break;
        case type_cl_command_queue_properties:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.queue_properties), &info->config.queue_properties, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.queue_properties);
            break;
        case type_cl_device_id:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.device_id), &info->config.device_id, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.device_id);
            break;
        case type_cl_device_affinity_domain:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.affinity_domain), &info->config.affinity_domain, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.affinity_domain);
            break;
        case type_cl_uint:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.uint), &info->config.uint, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.uint);
            break;
        case type_size_t_arr:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.sizet_arr), &info->config.sizet_arr, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.sizet_arr);
            break;
        case type_size_t:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.sizet), &info->config.sizet, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.sizet);
            break;
        case type_cl_ulong:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.ull), &info->config.ull, &config_size_ret);
            size_err = config_size_ret != sizeof(info->config.ull);
            break;
        case type_string:
            err = clGetDeviceInfo(device, info->opcode, sizeof(info->config.string), &info->config.string, &config_size_ret);
            break;
        default:
            log_error("Unknown config type: %d\n", info->config_type);
            break;
    }
    if (err | size_err)
        log_error("\tFailed clGetDeviceInfo for %s.\n", info->opcode_name);
    if(err)
        print_error(err, "\t\tclGetDeviceInfo failed.");
    if (size_err)
        log_error("\t\tWrong size return from clGetDeviceInfo.\n");
    return err | size_err;
}

void dumpConfigInfo(cl_device_id device, config_info* info)
{
    // We should not error if we find an unknown configuration since vendors
    // may specify their own options beyond the list in the specification.
    switch(info->config_type)
    {
        case type_cl_device_type:
            log_info("\t%s == %s|%s|%s|%s\n", info->opcode_name,
                     (info->config.fp_config & CL_DEVICE_TYPE_CPU) ? "CL_DEVICE_TYPE_CPU":"",
                     (info->config.fp_config & CL_DEVICE_TYPE_GPU) ? "CL_DEVICE_TYPE_GPU":"",
                     (info->config.fp_config & CL_DEVICE_TYPE_ACCELERATOR) ? "CL_DEVICE_TYPE_ACCELERATOR":"",
                     (info->config.fp_config & CL_DEVICE_TYPE_DEFAULT) ? "CL_DEVICE_TYPE_DEFAULT":""
                     );
            {
                cl_device_type all_device_types = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_DEFAULT;
                if(info->config.fp_config & ~all_device_types)
                {
                    log_info("WARNING: %s unknown bits found 0x%08llX", info->opcode_name, (info->config.fp_config & ~all_device_types));
                }
            }
            break;
        case type_cl_device_fp_config:
            log_info("\t%s == %s|%s|%s|%s|%s|%s|%s\n", info->opcode_name,
                     (info->config.fp_config & CL_FP_DENORM) ? "CL_FP_DENORM":"",
                     (info->config.fp_config & CL_FP_INF_NAN) ? "CL_FP_INF_NAN":"",
                     (info->config.fp_config & CL_FP_ROUND_TO_NEAREST) ? "CL_FP_ROUND_TO_NEAREST":"",
                     (info->config.fp_config & CL_FP_ROUND_TO_ZERO) ? "CL_FP_ROUND_TO_ZERO":"",
                     (info->config.fp_config & CL_FP_ROUND_TO_INF) ? "CL_FP_ROUND_TO_INF":"",
                     (info->config.fp_config & CL_FP_FMA) ? "CL_FP_FMA":"",
                     (info->config.fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) ? "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT":""
                     );
            {
                cl_device_fp_config all_fp_config = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                                                    CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA |
                                                    CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
                if(info->config.fp_config & ~all_fp_config)
                    log_info("WARNING: %s unknown bits found 0x%08llX", info->opcode_name, (info->config.fp_config & ~all_fp_config));
            }
            break;
        case type_cl_device_mem_cache_type:
            switch(info->config.mem_cache_type)
            {
                case CL_NONE:
                    log_info("\t%s == CL_NONE\n", info->opcode_name);
                    break;
                case CL_READ_ONLY_CACHE:
                    log_info("\t%s == CL_READ_ONLY_CACHE\n", info->opcode_name);
                    break;
                case CL_READ_WRITE_CACHE:
                    log_info("\t%s == CL_READ_WRITE_CACHE\n", info->opcode_name);
                    break;
                default:
                    log_error("ERROR: %s out of range, %d\n", info->opcode_name, info->config.mem_cache_type);
                    break;
            }
            break;
        case type_cl_local_mem_type:
            switch(info->config.local_mem_type)
            {
                case CL_LOCAL:
                    log_info("\t%s == CL_LOCAL\n", info->opcode_name);
                    break;
                case CL_GLOBAL:
                    log_info("\t%s == CL_GLOBAL\n", info->opcode_name);
                    break;
                default:
                    log_info("WARNING: %s out of range, %d\n", info->opcode_name, info->config.local_mem_type);
                    break;
            }
            break;
        case type_cl_device_exec_capabilities:
            log_info("\t%s == %s|%s\n", info->opcode_name,
                     (info->config.exec_capabilities & CL_EXEC_KERNEL) ? "CL_EXEC_KERNEL":"",
                     (info->config.exec_capabilities & CL_EXEC_NATIVE_KERNEL) ? "CL_EXEC_NATIVE_KERNEL":"" );
            {
                cl_device_exec_capabilities all_exec_cap = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
                if(info->config.exec_capabilities & ~all_exec_cap)
                    log_info("WARNING: %s unknown bits found 0x%08llX", info->opcode_name, (info->config.exec_capabilities & ~all_exec_cap));
            }
            break;
        case type_cl_command_queue_properties:
            log_info("\t%s == %s|%s\n", info->opcode_name,
                     (info->config.queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE":"",
                     (info->config.queue_properties & CL_QUEUE_PROFILING_ENABLE) ? "CL_QUEUE_PROFILING_ENABLE":"");
            {
                cl_command_queue_properties all_queue_properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
                if(info->config.queue_properties & ~all_queue_properties)
                    log_info("WARNING: %s unknown bits found 0x%08llX", info->opcode_name, (info->config.exec_capabilities & ~all_queue_properties));
            }
            break;
        case type_cl_device_id:
            log_info("\t%s == %ld\n", info->opcode_name, info->config.device_id);
            break;
        case type_cl_device_affinity_domain:
            log_info("\t%s == %s|%s|%s|%s|%s|%s\n", info->opcode_name,
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_NUMA) ? "CL_DEVICE_AFFINITY_DOMAIN_NUMA":"",
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE) ? "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE":"",
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE) ? "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE":"",
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE) ? "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE":"",
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE) ? "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE":"",
                     (info->config.affinity_domain & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE) ? "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE":""
                     );
            {
                cl_device_affinity_domain all_affinity_domain = CL_DEVICE_AFFINITY_DOMAIN_NUMA |
                                                                CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE |
                                                                CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE |
                                                                CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE |
                                                                CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE |
                                                                CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
                if(info->config.affinity_domain & ~all_affinity_domain)
                    log_error("ERROR: %s unknown bits found 0x%08llX", info->opcode_name, (info->config.affinity_domain & ~all_affinity_domain));
            }
            break;
        case type_cl_uint:
            log_info("\t%s == %d\n", info->opcode_name, info->config.uint);
            break;
        case type_size_t_arr:
            log_info("\t%s == %d %d %d\n", info->opcode_name, info->config.sizet_arr[0],
                     info->config.sizet_arr[1], info->config.sizet_arr[2]);
            break;
        case type_size_t:
            log_info("\t%s == %ld\n", info->opcode_name, info->config.sizet);
            break;
        case type_cl_ulong:
            log_info("\t%s == %lld\n", info->opcode_name, info->config.ull);
            break;
        case type_string:
            log_info("\t%s == \"%s\"\n", info->opcode_name, info->config.string);
            break;
    }
}

void print_platform_string_selector( cl_platform_id platform, const char *selector_name, cl_platform_info selector )
{
    // Currently all the selectors are strings
    size_t size = 0;
    char *value;
    int err;

    if(( err = clGetPlatformInfo( platform, selector, 0, NULL, &size )))
    {
        log_error( "FAILURE: Unable to get platform info size for %s.\n", selector_name );
        exit( -1 );
    }

    if( size == 0 )
    {
        log_error( "FAILURE: The size of %s was returned to be zero.\n", selector_name );
        exit( -1 );
    }

    value = (char*) malloc( size );
    if( NULL == value )
    {
        log_error( "Internal test failure:  Unable to allocate %ld bytes\n", size );
        exit(-1);
    }

    memset( value, -1, size );
    if(( err = clGetPlatformInfo( platform, selector, size, value, NULL )))
    {
        log_error( "FAILURE: Unable to get platform info for %s.\n", selector_name );
        free( value );
        exit( -1 );
    }

    if( value[size-1] != '\0' )
    {
        log_error( "FAILURE: platform info for %s is either not NUL terminated, or the size is wrong.\n", selector_name );
        free( value );
        exit( -1 );
    }

    log_info( "\t%s: %s\n", selector_name, value );
    free( value );
}

int main(int argc, const char** argv)
{
    cl_platform_id platform;
    test_start();

    int err;
    int total_errors = 0;

    err = clGetPlatformIDs(1, &platform, NULL);
    test_error(err, "clGetPlatformIDs failed");
    if (err != CL_SUCCESS) {
        total_errors++;
    }

    // print platform info
    log_info( "\nclGetPlatformInfo:\n------------------\n" );
    print_platform_string_selector( platform, "CL_PLATFORM_PROFILE", CL_PLATFORM_PROFILE );
    print_platform_string_selector( platform, "CL_PLATFORM_VERSION", CL_PLATFORM_VERSION );
    print_platform_string_selector( platform, "CL_PLATFORM_NAME", CL_PLATFORM_NAME );
    print_platform_string_selector( platform, "CL_PLATFORM_VENDOR", CL_PLATFORM_VENDOR );
    print_platform_string_selector( platform, "CL_PLATFORM_EXTENSIONS", CL_PLATFORM_EXTENSIONS );
    log_info( "\n" );

    // Check to see if this test is being run on a specific device
    char* device_type_env = getenv("CL_DEVICE_TYPE");
    char* device_index_env = getenv("CL_DEVICE_INDEX");

    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    size_t device_type_idx = 0;
    size_t device_index = 0;

    // Check to see if a device type was specified.
    if (device_type_env) {
        if (!strcmp(device_type_env,"default") || !strcmp(device_type_env,"CL_DEVICE_TYPE_DEFAULT")) {
            device_type = CL_DEVICE_TYPE_CPU;
            device_type_idx = 0;
        }
        else if (!strcmp(device_type_env,"cpu") || !strcmp(device_type_env,"CL_DEVICE_TYPE_CPU")) {
            device_type = CL_DEVICE_TYPE_CPU;
            device_type_idx = 1;
        }
        else if (!strcmp(device_type_env,"gpu") || !strcmp(device_type_env,"CL_DEVICE_TYPE_GPU")) {
            device_type = CL_DEVICE_TYPE_GPU;
            device_type_idx = 2;
        }
        else if (!strcmp(device_type_env,"accelerator") || !strcmp(device_type_env,"CL_DEVICE_TYPE_ACCELERATOR")) {
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
            device_type_idx = 3;
        }
        else {
            log_error("CL_DEVICE_TYPE=\"%s\" is invalid\n",device_type_env);
            return -1;
        }
    }

    // Check to see if a device index was specified
    if (device_index_env)
        device_index = atoi(device_index_env);

    // print device info
    int onInfo;
    for(onInfo = 0; onInfo < sizeof(device_infos) / sizeof(device_infos[0]); onInfo++)
    {
        if (device_type_env && (device_type != device_infos[onInfo].device_type))
        {
            continue;
        }
        log_info("Getting device IDs for %s devices\n", device_infos[onInfo].device_type_name);
        err = clGetDeviceIDs(platform, device_infos[onInfo].device_type, 0, NULL, &device_infos[onInfo].num_devices);
        if (err == CL_DEVICE_NOT_FOUND)
        {
            log_info("No devices of type %s found.\n", device_infos[onInfo].device_type_name);
            continue;
        }
        test_error(err, "clGetDeviceIDs failed");

        log_info("Found %d %s devices:\n", device_infos[onInfo].num_devices, device_infos[onInfo].device_type_name);
        if(device_infos[onInfo].num_devices)
        {
            device_infos[onInfo].devices = (cl_device_id *)malloc(sizeof(cl_device_id) * device_infos[onInfo].num_devices);
            err = clGetDeviceIDs(platform, device_infos[onInfo].device_type, device_infos[onInfo].num_devices, device_infos[onInfo].devices, NULL);
            test_error(err, "clGetDeviceIDs failed");
        }

        int onDevice;
        for(onDevice = 0; onDevice < device_infos[onInfo].num_devices; onDevice++)
        {
            if (device_index_env && (device_index != onDevice))
            {
                continue;
            }
            log_info("%s Device %d of %d Info:\n", device_infos[onInfo].device_type_name, onDevice+1, device_infos[onInfo].num_devices);
            cl_device_id device = device_infos[onInfo].devices[onDevice];
            int onConfigInfo;
            for(onConfigInfo = 0; onConfigInfo < sizeof(config_infos) / sizeof(config_infos[0]); onConfigInfo++)
            {
                err = getConfigInfo(device, &config_infos[onConfigInfo]);
                if(!err) {
                    dumpConfigInfo(device, &config_infos[onConfigInfo]);
                } else {
                    total_errors++;
                }
            }
            log_info("\n");
        }

        if(device_infos[onInfo].num_devices)
        {
            free(device_infos[onInfo].devices);
        }
    }

    if (total_errors)
        log_error("FAILED computeinfo.\n");
    else
        log_info("PASSED computeinfo.\n");

    test_finish();
    if (total_errors)
        return -1;
    return 0;
}


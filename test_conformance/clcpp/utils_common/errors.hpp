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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_ERRORS_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_ERRORS_HPP

#include <string>

#include "../harness/errorHelpers.h"

// ------------- Check OpenCL error helpers (marcos) -----------------

std::string get_cl_error_string(cl_int error)
{
#define CASE_CL_ERROR(x) case x: return #x;
    switch (error)
    {
        CASE_CL_ERROR(CL_SUCCESS)
        CASE_CL_ERROR(CL_DEVICE_NOT_FOUND)
        CASE_CL_ERROR(CL_DEVICE_NOT_AVAILABLE)
        CASE_CL_ERROR(CL_COMPILER_NOT_AVAILABLE)
        CASE_CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CASE_CL_ERROR(CL_OUT_OF_RESOURCES)
        CASE_CL_ERROR(CL_OUT_OF_HOST_MEMORY)
        CASE_CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE)
        CASE_CL_ERROR(CL_MEM_COPY_OVERLAP)
        CASE_CL_ERROR(CL_IMAGE_FORMAT_MISMATCH)
        CASE_CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CASE_CL_ERROR(CL_BUILD_PROGRAM_FAILURE)
        CASE_CL_ERROR(CL_MAP_FAILURE)
        CASE_CL_ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CASE_CL_ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        CASE_CL_ERROR(CL_COMPILE_PROGRAM_FAILURE)
        CASE_CL_ERROR(CL_LINKER_NOT_AVAILABLE)
        CASE_CL_ERROR(CL_LINK_PROGRAM_FAILURE)
        CASE_CL_ERROR(CL_DEVICE_PARTITION_FAILED)
        CASE_CL_ERROR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)

        CASE_CL_ERROR(CL_INVALID_VALUE)
        CASE_CL_ERROR(CL_INVALID_DEVICE_TYPE)
        CASE_CL_ERROR(CL_INVALID_PLATFORM)
        CASE_CL_ERROR(CL_INVALID_DEVICE)
        CASE_CL_ERROR(CL_INVALID_CONTEXT)
        CASE_CL_ERROR(CL_INVALID_QUEUE_PROPERTIES)
        CASE_CL_ERROR(CL_INVALID_COMMAND_QUEUE)
        CASE_CL_ERROR(CL_INVALID_HOST_PTR)
        CASE_CL_ERROR(CL_INVALID_MEM_OBJECT)
        CASE_CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CASE_CL_ERROR(CL_INVALID_IMAGE_SIZE)
        CASE_CL_ERROR(CL_INVALID_SAMPLER)
        CASE_CL_ERROR(CL_INVALID_BINARY)
        CASE_CL_ERROR(CL_INVALID_BUILD_OPTIONS)
        CASE_CL_ERROR(CL_INVALID_PROGRAM)
        CASE_CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE)
        CASE_CL_ERROR(CL_INVALID_KERNEL_NAME)
        CASE_CL_ERROR(CL_INVALID_KERNEL_DEFINITION)
        CASE_CL_ERROR(CL_INVALID_KERNEL)
        CASE_CL_ERROR(CL_INVALID_ARG_INDEX)
        CASE_CL_ERROR(CL_INVALID_ARG_VALUE)
        CASE_CL_ERROR(CL_INVALID_ARG_SIZE)
        CASE_CL_ERROR(CL_INVALID_KERNEL_ARGS)
        CASE_CL_ERROR(CL_INVALID_WORK_DIMENSION)
        CASE_CL_ERROR(CL_INVALID_WORK_GROUP_SIZE)
        CASE_CL_ERROR(CL_INVALID_WORK_ITEM_SIZE)
        CASE_CL_ERROR(CL_INVALID_GLOBAL_OFFSET)
        CASE_CL_ERROR(CL_INVALID_EVENT_WAIT_LIST)
        CASE_CL_ERROR(CL_INVALID_EVENT)
        CASE_CL_ERROR(CL_INVALID_OPERATION)
        CASE_CL_ERROR(CL_INVALID_GL_OBJECT)
        CASE_CL_ERROR(CL_INVALID_BUFFER_SIZE)
        CASE_CL_ERROR(CL_INVALID_MIP_LEVEL)
        CASE_CL_ERROR(CL_INVALID_GLOBAL_WORK_SIZE)
        CASE_CL_ERROR(CL_INVALID_PROPERTY)
        CASE_CL_ERROR(CL_INVALID_IMAGE_DESCRIPTOR)
        CASE_CL_ERROR(CL_INVALID_COMPILER_OPTIONS)
        CASE_CL_ERROR(CL_INVALID_LINKER_OPTIONS)
        CASE_CL_ERROR(CL_INVALID_DEVICE_PARTITION_COUNT)
        CASE_CL_ERROR(CL_INVALID_PIPE_SIZE)
        CASE_CL_ERROR(CL_INVALID_DEVICE_QUEUE)
        CASE_CL_ERROR(CL_INVALID_SPEC_ID)
        CASE_CL_ERROR(CL_MAX_SIZE_RESTRICTION_EXCEEDED)
        default: return "(unknown error code)";
    }
#undef CASE_CL_ERROR
}

#define CHECK_ERROR(x) \
    if(x != CL_SUCCESS) \
    { \
        log_error("ERROR: %d, file: %s, line: %d\n", x, __FILE__, __LINE__);\
    }
#define CHECK_ERROR_MSG(x, ...) \
    if(x != CL_SUCCESS) \
    { \
        log_error("ERROR: " __VA_ARGS__);\
        log_error("\n");\
        log_error("ERROR: %d, file: %s, line: %d\n", x, __FILE__, __LINE__);\
    }
#define RETURN_ON_ERROR(x) \
    if(x != CL_SUCCESS) \
    { \
        log_error("ERROR: %d, file: %s, line: %d\n", x, __FILE__, __LINE__);\
        return x;\
    }
#define RETURN_ON_ERROR_MSG(x, ...) \
    if(x != CL_SUCCESS) \
    { \
        log_error("ERROR: " __VA_ARGS__);\
        log_error("\n");\
        log_error("ERROR: %d, file: %s, line: %d\n", x, __FILE__, __LINE__);\
        return x;\
    }

#define RETURN_ON_CL_ERROR(x, cl_func_name) \
    if(x != CL_SUCCESS) \
    { \
        log_error("ERROR: %s failed: %s (%d)\n", cl_func_name, get_cl_error_string(x).c_str(), x);\
        log_error("ERROR: %d, file: %s, line: %d\n", x, __FILE__, __LINE__);\
        return x;\
    }

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_ERRORS_HPP

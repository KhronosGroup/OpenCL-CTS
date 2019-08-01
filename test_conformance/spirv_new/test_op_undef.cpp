/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
in whole or in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include "testBase.h"
#include "types.hpp"



template<typename T>
int test_undef(cl_device_id deviceID, cl_context context,
               cl_command_queue queue, const char *name)
{
    if(std::string(name).find("double") != std::string::npos) {
        if(!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return 0;
        }
    }
    int num = (int)(1 << 10);
    cl_int err = CL_SUCCESS;

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    size_t bytes = num * sizeof(T);
    clMemWrapper mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    std::vector<T> host(num);
    err = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes, &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    return 0;
}

#define TEST_UNDEF(NAME, TYPE)                              \
    TEST_SPIRV_FUNC(op_undef_##NAME##_simple)               \
    {                                                       \
        return test_undef<TYPE>(deviceID, context, queue,   \
                                "undef_" #NAME "_simple");  \
    }                                                       \

// Boolean tests
TEST_UNDEF(true  , cl_int  )
TEST_UNDEF(false , cl_int  )

// Integer tests
TEST_UNDEF(int   , cl_int  )
TEST_UNDEF(uint  , cl_uint )
TEST_UNDEF(char  , cl_char )
TEST_UNDEF(uchar , cl_uchar)
TEST_UNDEF(ushort, cl_ushort)
TEST_UNDEF(long  , cl_long )
TEST_UNDEF(ulong , cl_ulong)

#ifdef __GNUC__
// std::vector<cl_short> is causing compilation errors on GCC 5.3 (works on gcc 4.8)
// Needs further investigation
TEST_UNDEF(short , int16_t )
#else
TEST_UNDEF(short , cl_short)
#endif

// Float tests
TEST_UNDEF(float , cl_float)
TEST_UNDEF(double, cl_double)
TEST_UNDEF(int4  , cl_int4)
TEST_UNDEF(int3  , cl_int3)


TEST_SPIRV_FUNC(op_undef_struct_int_float_simple)
{
    typedef AbstractStruct2<cl_int, cl_float> CustomType;
    return test_undef<CustomType>(deviceID, context, queue, "undef_struct_int_float_simple");
}

TEST_SPIRV_FUNC(op_undef_struct_int_char_simple)
{
    typedef AbstractStruct2<cl_int, cl_char> CustomType;
    return test_undef<CustomType>(deviceID, context, queue, "undef_struct_int_char_simple");
}

TEST_SPIRV_FUNC(op_undef_struct_struct_simple)
{
    typedef AbstractStruct2<cl_int, cl_char> CustomType1;
    typedef AbstractStruct2<cl_int2, CustomType1> CustomType2;
    return test_undef<CustomType2>(deviceID, context, queue, "undef_struct_struct_simple");
}

TEST_SPIRV_FUNC(op_undef_half_simple)
{
    PASSIVE_REQUIRE_FP16_SUPPORT(deviceID);
    return test_undef<cl_float>(deviceID, context, queue,
                                "undef_half_simple");
}

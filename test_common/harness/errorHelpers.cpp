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
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "errorHelpers.h"

#include "parseParameters.h"
#include "testHarness.h"

#include <CL/cl_half.h>

const char *IGetErrorString(int clErrorCode)
{
    switch (clErrorCode)
    {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
        case CL_INVALID_SPEC_ID: return "CL_INVALID_SPEC_ID";
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
            return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        default: return "(unknown)";
    }
}

const char *GetChannelOrderName(cl_channel_order order)
{
    switch (order)
    {
        case CL_R: return "CL_R";
        case CL_A: return "CL_A";
        case CL_Rx: return "CL_Rx";
        case CL_RG: return "CL_RG";
        case CL_RA: return "CL_RA";
        case CL_RGx: return "CL_RGx";
        case CL_RGB: return "CL_RGB";
        case CL_RGBx: return "CL_RGBx";
        case CL_RGBA: return "CL_RGBA";
        case CL_ARGB: return "CL_ARGB";
        case CL_BGRA: return "CL_BGRA";
        case CL_INTENSITY: return "CL_INTENSITY";
        case CL_LUMINANCE: return "CL_LUMINANCE";
#if defined CL_1RGB_APPLE
        case CL_1RGB_APPLE: return "CL_1RGB_APPLE";
#endif
#if defined CL_BGR1_APPLE
        case CL_BGR1_APPLE: return "CL_BGR1_APPLE";
#endif
#if defined CL_ABGR_APPLE
        case CL_ABGR_APPLE: return "CL_ABGR_APPLE";
#endif
        case CL_DEPTH: return "CL_DEPTH";
        case CL_DEPTH_STENCIL: return "CL_DEPTH_STENCIL";
        case CL_sRGB: return "CL_sRGB";
        case CL_sRGBA: return "CL_sRGBA";
        case CL_sRGBx: return "CL_sRGBx";
        case CL_sBGRA: return "CL_sBGRA";
        case CL_ABGR: return "CL_ABGR";
        default: return NULL;
    }
}

int IsChannelOrderSupported(cl_channel_order order)
{
    switch (order)
    {
        case CL_R:
        case CL_A:
        case CL_Rx:
        case CL_RG:
        case CL_RA:
        case CL_RGx:
        case CL_RGB:
        case CL_RGBx:
        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
        case CL_INTENSITY:
        case CL_LUMINANCE:
        case CL_ABGR:
        case CL_sRGB:
        case CL_sRGBx:
        case CL_sBGRA:
        case CL_sRGBA:
        case CL_DEPTH: return 1;
#if defined CL_1RGB_APPLE
        case CL_1RGB_APPLE: return 1;
#endif
#if defined CL_BGR1_APPLE
        case CL_BGR1_APPLE: return 1;
#endif
        default: return 0;
    }
}

const char *GetChannelTypeName(cl_channel_type type)
{
    switch (type)
    {
        case CL_SNORM_INT8: return "CL_SNORM_INT8";
        case CL_SNORM_INT16: return "CL_SNORM_INT16";
        case CL_UNORM_INT8: return "CL_UNORM_INT8";
        case CL_UNORM_INT16: return "CL_UNORM_INT16";
        case CL_UNORM_SHORT_565: return "CL_UNORM_SHORT_565";
        case CL_UNORM_SHORT_555: return "CL_UNORM_SHORT_555";
        case CL_UNORM_INT_101010: return "CL_UNORM_INT_101010";
        case CL_SIGNED_INT8: return "CL_SIGNED_INT8";
        case CL_SIGNED_INT16: return "CL_SIGNED_INT16";
        case CL_SIGNED_INT32: return "CL_SIGNED_INT32";
        case CL_UNSIGNED_INT8: return "CL_UNSIGNED_INT8";
        case CL_UNSIGNED_INT16: return "CL_UNSIGNED_INT16";
        case CL_UNSIGNED_INT32: return "CL_UNSIGNED_INT32";
        case CL_HALF_FLOAT: return "CL_HALF_FLOAT";
        case CL_FLOAT: return "CL_FLOAT";
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: return "CL_SFIXED14_APPLE";
#endif
        case CL_UNORM_INT24: return "CL_UNORM_INT24";
        default: return NULL;
    }
}

int IsChannelTypeSupported(cl_channel_type type)
{
    switch (type)
    {
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_UNORM_INT24:
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_UNORM_INT_101010:
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
        case CL_HALF_FLOAT:
        case CL_FLOAT: return 1;
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: return 1;
#endif
        default: return 0;
    }
}

const char *GetAddressModeName(cl_addressing_mode mode)
{
    switch (mode)
    {
        case CL_ADDRESS_NONE: return "CL_ADDRESS_NONE";
        case CL_ADDRESS_CLAMP_TO_EDGE: return "CL_ADDRESS_CLAMP_TO_EDGE";
        case CL_ADDRESS_CLAMP: return "CL_ADDRESS_CLAMP";
        case CL_ADDRESS_REPEAT: return "CL_ADDRESS_REPEAT";
        case CL_ADDRESS_MIRRORED_REPEAT: return "CL_ADDRESS_MIRRORED_REPEAT";
        default: return NULL;
    }
}

const char *GetDeviceTypeName(cl_device_type type)
{
    switch (type)
    {
        case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_GPU";
        case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
        case CL_DEVICE_TYPE_ALL: return "CL_DEVICE_TYPE_ALL";
        default: return NULL;
    }
}

const char *GetDataVectorString(void *dataBuffer, size_t typeSize,
                                size_t vecSize, char *buffer)
{
    static char scratch[1024];
    size_t i, j;

    if (buffer == NULL) buffer = scratch;

    unsigned char *p = (unsigned char *)dataBuffer;
    char *bPtr;

    buffer[0] = 0;
    bPtr = buffer;
    for (i = 0; i < vecSize; i++)
    {
        if (i > 0)
        {
            bPtr[0] = ' ';
            bPtr++;
        }
        for (j = 0; j < typeSize; j++)
        {
            sprintf(bPtr, "%02x", (unsigned int)p[typeSize - j - 1]);
            bPtr += 2;
        }
        p += typeSize;
    }
    bPtr[0] = 0;

    return buffer;
}

const char *GetQueuePropertyName(cl_command_queue_properties property)
{
    switch (property)
    {
        case CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:
            return "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
        case CL_QUEUE_PROFILING_ENABLE: return "CL_QUEUE_PROFILING_ENABLE";
        case CL_QUEUE_ON_DEVICE: return "CL_QUEUE_ON_DEVICE";
        case CL_QUEUE_ON_DEVICE_DEFAULT: return "CL_QUEUE_ON_DEVICE_DEFAULT";
        default: return "(unknown)";
    }
}

#if defined(_MSC_VER)
#define scalbnf(_a, _i) ldexpf(_a, _i)
#define scalbn(_a, _i) ldexp(_a, _i)
#define scalbnl(_a, _i) ldexpl(_a, _i)
#endif

// taken from math tests
#define HALF_MIN_EXP -13
#define HALF_MANT_DIG 11
static float Ulp_Error_Half_Float(float test, double reference)
{
    union {
        double d;
        uint64_t u;
    } u;
    u.d = reference;

    // Note: This function presumes that someone has already tested whether the
    // result is correctly, rounded before calling this function.  That test:
    //
    //    if( (float) reference == test )
    //        return 0.0f;
    //
    // would ensure that cases like fabs(reference) > FLT_MAX are weeded out
    // before we get here. Otherwise, we'll return inf ulp error here, for what
    // are otherwise correctly rounded results.

    double testVal = test;

    if (isinf(reference))
    {
        if (testVal == reference) return 0.0f;

        return (float)(testVal - reference);
    }

    if (isinf(testVal))
    {
        // Allow overflow within the limit of the allowed ulp error. Towards
        // that end we pretend the test value is actually 2**16, the next value
        // that would appear in the number line if half had sufficient range.
        testVal = copysign(65536.0, testVal);
    }


    if (u.u & 0x000fffffffffffffULL)
    { // Non-power of two and NaN
        if (isnan(reference) && isnan(test))
            return 0.0f; // if we are expecting a NaN, any NaN is fine

        // The unbiased exponent of the ulp unit place
        int ulp_exp =
            HALF_MANT_DIG - 1 - std::max(ilogb(reference), HALF_MIN_EXP - 1);

        // Scale the exponent of the error
        return (float)scalbn(testVal - reference, ulp_exp);
    }

    // reference is a normal power of two or a zero
    int ulp_exp =
        HALF_MANT_DIG - 1 - std::max(ilogb(reference) - 1, HALF_MIN_EXP - 1);

    // Scale the exponent of the error
    return (float)scalbn(testVal - reference, ulp_exp);
}

float Ulp_Error_Half(cl_half test, float reference)
{
    return Ulp_Error_Half_Float(cl_half_to_float(test), reference);
}


float Ulp_Error(float test, double reference)
{
    union {
        double d;
        uint64_t u;
    } u;
    u.d = reference;
    double testVal = test;

    // Note: This function presumes that someone has already tested whether the
    // result is correctly, rounded before calling this function.  That test:
    //
    //    if( (float) reference == test )
    //        return 0.0f;
    //
    // would ensure that cases like fabs(reference) > FLT_MAX are weeded out
    // before we get here. Otherwise, we'll return inf ulp error here, for what
    // are otherwise correctly rounded results.


    if (isinf(reference))
    {
        if (testVal == reference) return 0.0f;

        return (float)(testVal - reference);
    }

    if (isinf(testVal))
    { // infinite test value, but finite (but possibly overflowing in float)
      // reference.
      //
      // The function probably overflowed prematurely here. Formally, the spec
      // says this is an infinite ulp error and should not be tolerated.
      // Unfortunately, this would mean that the internal precision of some
      // half_pow implementations would have to be 29+ bits at half_powr(
      // 0x1.fffffep+31, 4) to correctly determine that 4*log2( 0x1.fffffep+31 )
      // is not exactly 128.0. You might represent this for example as 4*(32 -
      // ~2**-24), which after rounding to single is 4*32 = 128, which will
      // ultimately result in premature overflow, even though a good faith
      // representation would be correct to within 2**-29 interally.

        // In the interest of not requiring the implementation go to
        // extraordinary lengths to deliver a half precision function, we allow
        // premature overflow within the limit of the allowed ulp error.
        // Towards, that end, we "pretend" the test value is actually 2**128,
        // the next value that would appear in the number line if float had
        // sufficient range.
        testVal = copysign(MAKE_HEX_DOUBLE(0x1.0p128, 0x1LL, 128), testVal);

        // Note that the same hack may not work in long double, which is not
        // guaranteed to have more range than double.  It is not clear that
        // premature overflow should be tolerated for double.
    }

    if (u.u & 0x000fffffffffffffULL)
    { // Non-power of two and NaN
        if (isnan(reference) && isnan(test))
            return 0.0f; // if we are expecting a NaN, any NaN is fine

        // The unbiased exponent of the ulp unit place
        int ulp_exp =
            FLT_MANT_DIG - 1 - std::max(ilogb(reference), FLT_MIN_EXP - 1);

        // Scale the exponent of the error
        return (float)scalbn(testVal - reference, ulp_exp);
    }

    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp =
        FLT_MANT_DIG - 1 - std::max(ilogb(reference) - 1, FLT_MIN_EXP - 1);

    // Scale the exponent of the error
    return (float)scalbn(testVal - reference, ulp_exp);
}

float Ulp_Error_Double(double test, long double reference)
{
    // Deal with long double = double
    // On most systems long double is a higher precision type than double. They
    // provide either a 80-bit or greater floating point type, or they provide a
    // head-tail double double format. That is sufficient to represent the
    // accuracy of a floating point result to many more bits than double and we
    // can calculate sub-ulp errors. This is the standard system for which this
    // test suite is designed.
    //
    // On some systems double and long double are the same thing. Then we run
    // into a problem, because our representation of the infinitely precise
    // result (passed in as reference above) can be off by as much as a half
    // double precision ulp itself.  In this case, we inflate the reported error
    // by half an ulp to take this into account.  A more correct and permanent
    // fix would be to undertake refactoring the reference code to return
    // results in this format:
    //
    //    typedef struct DoubleReference
    //    {
    //        // true value = correctlyRoundedResult + ulps *
    //        //    ulp(correctlyRoundedResult)  (infinitely precise)
    //        // as best we can:
    //        double correctlyRoundedResult;
    //        // plus a fractional amount to account for the difference
    //        // between infinitely precise result and correctlyRoundedResult,
    //        // in units of ulps:
    //        double ulps;
    //    } DoubleReference;
    //
    // This would provide a useful higher-than-double precision format for
    // everyone that we can use, and would solve a few problems with
    // representing absolute errors below DBL_MIN and over DBL_MAX for systems
    // that use a head to tail double double for long double.

    // Note: This function presumes that someone has already tested whether the
    // result is correctly, rounded before calling this function.  That test:
    //
    //    if( (float) reference == test )
    //        return 0.0f;
    //
    // would ensure that cases like fabs(reference) > FLT_MAX are weeded out
    // before we get here. Otherwise, we'll return inf ulp error here, for what
    // are otherwise correctly rounded results.


    int x;
    long double testVal = test;
    if (0.5L != frexpl(reference, &x))
    { // Non-power of two and NaN
        if (isinf(reference))
        {
            if (testVal == reference) return 0.0f;

            return (float)(testVal - reference);
        }

        if (isnan(reference) && isnan(test))
            return 0.0f; // if we are expecting a NaN, any NaN is fine

        // The unbiased exponent of the ulp unit place
        int ulp_exp =
            DBL_MANT_DIG - 1 - std::max(ilogbl(reference), DBL_MIN_EXP - 1);

        // Scale the exponent of the error
        float result = (float)scalbnl(testVal - reference, ulp_exp);

        // account for rounding error in reference result on systems that do not
        // have a higher precision floating point type (see above)
        if (sizeof(long double) == sizeof(double))
            result += copysignf(0.5f, result);

        return result;
    }

    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp =
        DBL_MANT_DIG - 1 - std::max(ilogbl(reference) - 1, DBL_MIN_EXP - 1);

    // Scale the exponent of the error
    float result = (float)scalbnl(testVal - reference, ulp_exp);

    // account for rounding error in reference result on systems that do not
    // have a higher precision floating point type (see above)
    if (sizeof(long double) == sizeof(double))
        result += copysignf(0.5f, result);

    return result;
}

cl_int OutputBuildLogs(cl_program program, cl_uint num_devices,
                       cl_device_id *device_list)
{
    int error;
    size_t size_ret;

    // Does the program object exist?
    if (program != NULL)
    {

        // Was the number of devices given
        if (num_devices == 0)
        {

            // If zero devices were specified then allocate and query the device
            // list from the context
            cl_context context;
            error = clGetProgramInfo(program, CL_PROGRAM_CONTEXT,
                                     sizeof(context), &context, NULL);
            test_error(error, "Unable to query program's context");
            error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
                                     &size_ret);
            test_error(error, "Unable to query context's device size");
            num_devices = static_cast<cl_uint>(size_ret / sizeof(cl_device_id));
            device_list = (cl_device_id *)malloc(size_ret);
            if (device_list == NULL)
            {
                print_error(error, "malloc failed");
                return CL_OUT_OF_HOST_MEMORY;
            }
            error = clGetContextInfo(context, CL_CONTEXT_DEVICES, size_ret,
                                     device_list, NULL);
            test_error(error, "Unable to query context's devices");
        }

        // For each device in the device_list
        unsigned int i;
        for (i = 0; i < num_devices; i++)
        {

            // Get the build status
            cl_build_status build_status;
            error = clGetProgramBuildInfo(
                program, device_list[i], CL_PROGRAM_BUILD_STATUS,
                sizeof(build_status), &build_status, &size_ret);
            test_error(error, "Unable to query build status");

            // If the build failed then log the status, and allocate the build
            // log, log it and free it
            if (build_status != CL_BUILD_SUCCESS)
            {

                log_error("ERROR: CL_PROGRAM_BUILD_STATUS=%d\n",
                          (int)build_status);
                error = clGetProgramBuildInfo(program, device_list[i],
                                              CL_PROGRAM_BUILD_LOG, 0, NULL,
                                              &size_ret);
                test_error(error, "Unable to query build log size");
                char *build_log = (char *)malloc(size_ret);
                error = clGetProgramBuildInfo(program, device_list[i],
                                              CL_PROGRAM_BUILD_LOG, size_ret,
                                              build_log, &size_ret);
                test_error(error, "Unable to query build log");
                log_error("ERROR: CL_PROGRAM_BUILD_LOG:\n%s\n", build_log);
                free(build_log);
            }
        }

        // Was the number of devices given
        if (num_devices == 0)
        {

            // If zero devices were specified then free the device list
            free(device_list);
        }
    }

    return CL_SUCCESS;
}

const char *subtests_to_skip_with_offline_compiler[] = {
    "get_kernel_arg_info",
    "binary_create",
    "load_program_source",
    "load_multistring_source",
    "load_two_kernel_source",
    "load_null_terminated_source",
    "load_null_terminated_multi_line_source",
    "load_null_terminated_partial_multi_line_source",
    "load_discreet_length_source",
    "get_program_source",
    "get_program_build_info",
    "options_build_optimizations",
    "options_build_macro",
    "options_build_macro_existence",
    "options_include_directory",
    "options_denorm_cache",
    "preprocessor_define_udef",
    "preprocessor_include",
    "preprocessor_line_error",
    "preprocessor_pragma",
    "compiler_defines_for_extensions",
    "image_macro",
    "simple_extern_compile_only",
    "simple_embedded_header_compile",
    "two_file_regular_variable_access",
    "two_file_regular_struct_access",
    "two_file_regular_function_access",
    "simple_embedded_header_link",
    "execute_after_simple_compile_and_link_with_defines",
    "execute_after_simple_compile_and_link_with_callbacks",
    "execute_after_embedded_header_link",
    "execute_after_included_header_link",
    "multi_file_libraries",
    "multiple_files",
    "multiple_libraries",
    "multiple_files_multiple_libraries",
    "multiple_embedded_headers",
    "program_binary_type",
    "compile_and_link_status_options_log",
    "kernel_preprocessor_macros",
    "execute_after_serialize_reload_library",
    "execute_after_serialize_reload_object",
    "execute_after_simple_compile_and_link",
    "execute_after_simple_compile_and_link_no_device_info",
    "execute_after_simple_library_with_link",
    "execute_after_two_file_link",
    "simple_compile_only",
    "simple_compile_with_callback",
    "simple_library_only",
    "simple_library_with_callback",
    "simple_library_with_link",
    "simple_link_only",
    "simple_link_with_callback",
    "simple_static_compile_only",
    "two_file_link",
    "async_build",
    "unload_repeated",
    "unload_compile_unload_link",
    "unload_build_unload_create_kernel",
    "unload_link_different",
    "unload_build_threaded",
    "unload_build_info",
    "unload_program_binaries",
    "features_macro",
    "progvar_prog_scope_misc",
    "library_function"
};

bool check_functions_for_offline_compiler(const char *subtestname)
{
    if (gCompilationMode != kOnline)
    {
        size_t nNotRequiredWithOfflineCompiler =
            ARRAY_SIZE(subtests_to_skip_with_offline_compiler);
        for (size_t i = 0; i < nNotRequiredWithOfflineCompiler; ++i)
        {
            if (!strcmp(subtestname, subtests_to_skip_with_offline_compiler[i]))
            {
                return false;
            }
        }
    }
    return true;
}

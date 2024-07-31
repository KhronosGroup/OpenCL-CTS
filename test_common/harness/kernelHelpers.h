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
#ifndef _kernelHelpers_h
#define _kernelHelpers_h

#include "compat.h"
#include "testHarness.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(__MINGW32__)
#include <malloc.h>
#endif

#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "deviceInfo.h"
#include "harness/alloc.h"

#include <functional>

/*
 *  The below code is intended to be used at the top of kernels that appear
 * inline in files to set line and file info for the kernel:
 *
 *  const char *source = {
 *      INIT_OPENCL_DEBUG_INFO
 *      "__kernel void foo( int x )\n"
 *      "{\n"
 *      "   ...\n"
 *      "}\n"
 *  };
 */
#define INIT_OPENCL_DEBUG_INFO SET_OPENCL_LINE_INFO(__LINE__, __FILE__)
#define SET_OPENCL_LINE_INFO(_line, _file)                                     \
    "#line " STRINGIFY(_line) " " STRINGIFY(_file) "\n"
#ifndef STRINGIFY_VALUE
#define STRINGIFY_VALUE(_x) STRINGIFY(_x)
#endif
#ifndef STRINGIFY
#define STRINGIFY(_x) #_x
#endif

const int MAX_LEN_FOR_KERNEL_LIST = 20;

/* Helper that creates a single program and kernel from a single-kernel program
 * source */
extern int
create_single_kernel_helper(cl_context context, cl_program *outProgram,
                            cl_kernel *outKernel, unsigned int numKernelLines,
                            const char **kernelProgram, const char *kernelName,
                            const char *buildOptions = NULL);

extern int create_single_kernel_helper_with_build_options(
    cl_context context, cl_program *outProgram, cl_kernel *outKernel,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *kernelName, const char *buildOptions);

extern int create_single_kernel_helper_create_program(
    cl_context context, cl_program *outProgram, unsigned int numKernelLines,
    const char **kernelProgram, const char *buildOptions = NULL);

extern int create_single_kernel_helper_create_program_for_device(
    cl_context context, cl_device_id device, cl_program *outProgram,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *buildOptions = NULL);

/* Creates OpenCL C++ program. This one must be used for creating OpenCL C++
 * program. */
extern int create_openclcpp_program(cl_context context, cl_program *outProgram,
                                    unsigned int numKernelLines,
                                    const char **kernelProgram,
                                    const char *buildOptions = NULL);

/* Builds program (outProgram) and creates one kernel */
int build_program_create_kernel_helper(
    cl_context context, cl_program *outProgram, cl_kernel *outKernel,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *kernelName, const char *buildOptions = NULL);

/* Helper to obtain the biggest fit work group size for all the devices in a
 * given group and for the given global thread size */
extern int get_max_common_work_group_size(cl_context context, cl_kernel kernel,
                                          size_t globalThreadSize,
                                          size_t *outSize);

/* Helper to obtain the biggest fit work group size for all the devices in a
 * given group and for the given global thread size */
extern int get_max_common_2D_work_group_size(cl_context context,
                                             cl_kernel kernel,
                                             size_t *globalThreadSize,
                                             size_t *outSizes);

/* Helper to obtain the biggest fit work group size for all the devices in a
 * given group and for the given global thread size */
extern int get_max_common_3D_work_group_size(cl_context context,
                                             cl_kernel kernel,
                                             size_t *globalThreadSize,
                                             size_t *outSizes);

/* Helper to obtain the biggest allowed work group size for all the devices in a
 * given group */
extern int get_max_allowed_work_group_size(cl_context context, cl_kernel kernel,
                                           size_t *outSize, size_t *outLimits);

/* Helper to obtain the biggest allowed 1D work group size on a given device */
extern int get_max_allowed_1d_work_group_size_on_device(cl_device_id device,
                                                        cl_kernel kernel,
                                                        size_t *outSize);

/* Helper to determine if a device supports an image format */
extern int is_image_format_supported(cl_context context, cl_mem_flags flags,
                                     cl_mem_object_type image_type,
                                     const cl_image_format *fmt);

/* Helper to get pixel size for a pixel format */
size_t get_pixel_bytes(const cl_image_format *fmt);

/* Verify the given device supports images. */
extern test_status verifyImageSupport(cl_device_id device);

/* Checks that the given device supports images. Same as verify, but doesn't
 * print an error */
extern int checkForImageSupport(cl_device_id device);
extern int checkFor3DImageSupport(cl_device_id device);
extern int checkForReadWriteImageSupport(cl_device_id device);

/* Checks that a given queue property is supported on the specified device.
 * Returns 1 if supported, 0 if not or an error. */
extern int checkDeviceForQueueSupport(cl_device_id device,
                                      cl_command_queue_properties prop);

/* Helper to obtain the min alignment for a given context, i.e the max of all
 * min alignments for devices attached to the context*/
size_t get_min_alignment(cl_context context);

/* Helper to obtain the default rounding mode for single precision computation.
 * (Double is always CL_FP_ROUND_TO_NEAREST.) Returns 0 on error. */
cl_device_fp_config
get_default_rounding_mode(cl_device_id device,
                          const cl_uint &param = CL_DEVICE_SINGLE_FP_CONFIG);

#define PASSIVE_REQUIRE_IMAGE_SUPPORT(device)                                  \
    if (checkForImageSupport(device))                                          \
    {                                                                          \
        log_info(                                                              \
            "\n\tNote: device does not support images. Skipping test...\n");   \
        return TEST_SKIPPED_ITSELF;                                            \
    }

#define PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)                               \
    if (checkFor3DImageSupport(device))                                        \
    {                                                                          \
        log_info("\n\tNote: device does not support 3D images. Skipping "      \
                 "test...\n");                                                 \
        return TEST_SKIPPED_ITSELF;                                            \
    }

#define PASSIVE_REQUIRE_FP16_SUPPORT(device)                                   \
    if (!device_supports_half(device))                                         \
    {                                                                          \
        log_info(                                                              \
            "\n\tNote: device does not support fp16. Skipping test...\n");     \
        return TEST_SKIPPED_ITSELF;                                            \
    }

/* Prints out the standard device header for all tests given the device to print
 * for */
extern int printDeviceHeader(cl_device_id device);

// Execute the CL_DEVICE_OPENCL_C_VERSION query and return the OpenCL C version
// is supported by the device.
Version get_device_cl_c_version(cl_device_id device);

// Gets the latest (potentially non-backward compatible) OpenCL C version
// supported by the device.
Version get_device_latest_cl_c_version(cl_device_id device);

// Gets the maximum universally supported OpenCL C version in a context, i.e.
// the OpenCL C version supported by all devices in a context.
Version get_max_OpenCL_C_for_context(cl_context context);

// Checks whether a particular OpenCL C version is supported by the device.
bool device_supports_cl_c_version(cl_device_id device, Version version);

// Poll fn every interval_ms until timeout_ms or it returns true
bool poll_until(unsigned timeout_ms, unsigned interval_ms,
                std::function<bool()> fn);

// Checks whether the device supports double data types
bool device_supports_double(cl_device_id device);

// Checks whether the device supports half data types
bool device_supports_half(cl_device_id device);

#endif // _kernelHelpers_h

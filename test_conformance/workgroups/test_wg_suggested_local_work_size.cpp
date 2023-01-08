//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include <CL/cl_ext.h>

/** @brief Gets the number of elements of type s in a fixed length array of s */
#define NELEMS(s) (sizeof(s) / sizeof((s)[0]))
#define test_error_ret_and_free(errCode, msg, retValue, ptr)                   \
    {                                                                          \
        auto errCodeResult = errCode;                                          \
        if (errCodeResult != CL_SUCCESS)                                       \
        {                                                                      \
            print_error(errCodeResult, msg);                                   \
            free(ptr);                                                         \
            return retValue;                                                   \
        }                                                                      \
    }

const char* wg_scan_local_work_group_size = R"(
    bool is_zero_linear_id()
    {
        size_t linear_id;
#if __OPENCL_VERSION__ < CL_VERSION_2_0
        linear_id = ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) +
                    ((get_global_id(1) - get_global_offset(1)) * get_global_size(0)) +
                    (get_global_id(0) - get_global_offset(0));
#else
        linear_id = get_global_linear_id();
#endif
        return linear_id == 0;
    }

    uint get_l_size(size_t dim)
    {
#if __OPENCL_VERSION__ < CL_VERSION_2_0
        return get_local_size(dim);
#else
        return get_enqueued_local_size(dim);
#endif
    }

    __kernel void test_wg_scan_local_work_group_size(global uint *output)
    {
        if(!is_zero_linear_id()) return;
        for (uint i = 0; i < 3; i++)
        {
            output[i] = get_l_size(i);
        }
    }
    __kernel void test_wg_scan_local_work_group_size_static_local(
                                            global uint *output)
    {
        __local char c[LOCAL_MEM_SIZE];
    
        if(!is_zero_linear_id()) return;
        for (uint i = 0; i < 3; i++)
        {
            output[i] = get_l_size(i);
        }
    }
    __kernel void test_wg_scan_local_work_group_size_dynlocal(
                                        global uint *output,
                                        __local char * c)
    {
        if(!is_zero_linear_id()) return;
        for (uint i = 0; i < 3; i++)
        {
            output[i] = get_l_size(i);
        }
    };)";

bool is_prime(size_t a)
{
    size_t c;

    for (c = 2; c < a; c++)
    {
        if (a % c == 0) return false;
    }
    return true;
}

bool is_not_prime(size_t a) { return !is_prime(a); }

bool is_not_even(size_t a) { return (is_prime(a) || (a % 2 == 1)); }

bool is_not_odd(size_t a) { return (is_prime(a) || (a % 2 == 0)); }

#define NELEMS(s) (sizeof(s) / sizeof((s)[0]))
/* The value_range_nD contains numbers to be used for the experiments with 2D
   and 3D global work sizes. This is because we need smaller numbers so that the
   resulting number of work items is meaningful and does not become too large.
   The cases here are: 64 that is a power of 2, 3 is an odd and small prime
   number, 12 is a multiple of 4 but not a power of 2, 113 is a large prime
   number
   and 1 is to test the lack of this dimension if the others are present */
const size_t value_range_nD[] = { 64, 3, 12, 113, 1 };
const size_t basic_increment = 16;
const size_t primes_increment = 1;
enum num_dims
{
    _1D = 1,
    _2D = 2,
    _3D = 3
};

int do_test(cl_device_id device, cl_context context, cl_command_queue queue,
            cl_kernel scan_kernel, int work_dim, size_t global_work_offset[3],
            size_t test_values[3], size_t dyn_mem_size)
{
    size_t local_work_size[] = { 1, 1, 1 };
    size_t suggested_total_size;
    size_t workgroupinfo_size;
    cl_uint kernel_work_size[3] = { 0 };
    clMemWrapper buffer;
    cl_platform_id platform;

    int err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                              &platform, NULL);
    test_error_ret(err, "clGetDeviceInfo failed", -1);
    clGetKernelSuggestedLocalWorkSizeKHR_fn
        clGetKernelSuggestedLocalWorkSizeKHR =
            (clGetKernelSuggestedLocalWorkSizeKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clGetKernelSuggestedLocalWorkSizeKHR");

    if (clGetKernelSuggestedLocalWorkSizeKHR == NULL)
    {
        log_info("Extension 'cl_khr_suggested_local_work_size' could not be "
                 "found.\n");
        return TEST_FAIL;
    }

    /* Create the actual buffer, using local_buffer as the host pointer, and ask
     * to copy that into the buffer */
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(kernel_work_size), NULL, &err);
    test_error_ret(err, "clCreateBuffer failed", -1);
    err = clSetKernelArg(scan_kernel, 0, sizeof(buffer), &buffer);
    test_error_ret(err, "clSetKernelArg failed", -1);
    if (dyn_mem_size)
    {
        err = clSetKernelArg(scan_kernel, 1, dyn_mem_size, NULL);
        test_error_ret(err, "clSetKernelArg failed", -1);
    }
    err = clGetKernelSuggestedLocalWorkSizeKHR(queue, scan_kernel, work_dim,
                                               global_work_offset, test_values,
                                               local_work_size);
    test_error_ret(err, "clGetKernelSuggestedLocalWorkSizeKHR failed", -1);
    suggested_total_size =
        local_work_size[0] * local_work_size[1] * local_work_size[2];
    err = clGetKernelWorkGroupInfo(
        scan_kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(workgroupinfo_size), &workgroupinfo_size, NULL);
    test_error_ret(err, "clGetKernelWorkGroupInfo failed", -1);
    if (suggested_total_size > workgroupinfo_size)
    {
        std::cout << "The suggested work group size consist of "
                  << suggested_total_size << " work items.\n"
                  << "Work items are limited by " << workgroupinfo_size
                  << std::endl;
        std::cout << "Size from clGetKernelWorkGroupInfo: "
                  << workgroupinfo_size;
        std::cout << "\nSize from clGetKernelSuggestedLocalWorkSizeKHR: "
                  << local_work_size[0] * local_work_size[1]
                * local_work_size[2]
                  << std::endl;
        return -1;
    }

    err =
        clEnqueueNDRangeKernel(queue, scan_kernel, work_dim, global_work_offset,
                               test_values, // global work size
                               NULL, 0, NULL, NULL);
    test_error_ret(err, "clEnqueueNDRangeKernel failed", -1);
    err = clEnqueueReadBuffer(queue, buffer, CL_NON_BLOCKING, 0,
                              sizeof(kernel_work_size), kernel_work_size, 0,
                              NULL, NULL);
    test_error_ret(err, "clEnqueueReadBuffer failed", -1);
    err = clFinish(queue);
    test_error_ret(err, "clFinish failed", -1);

    if (kernel_work_size[0] != local_work_size[0]
        || kernel_work_size[1] != local_work_size[1]
        || kernel_work_size[2] != local_work_size[2])
    {
        std::cout
            << "Kernel work size differs from local work size suggested:\n"
            << "Kernel work size: (" << kernel_work_size[0] << ", "
            << kernel_work_size[1] << ", " << kernel_work_size[2] << ")"
            << "Local work size: (" << local_work_size[0] << ", "
            << local_work_size[1] << ", " << local_work_size[2] << ")\n";
        return -1;
    }
    return err;
}

int do_test_work_group_suggested_local_size(
    cl_device_id device, cl_context context, cl_command_queue queue,
    bool (*skip_cond)(size_t), size_t start, size_t end, size_t incr,
    cl_long max_local_mem_size, size_t global_work_offset[], num_dims dim)
{
    clProgramWrapper scan_program;
    clKernelWrapper scan_kernel;
    int err;
    size_t test_values[] = { 1, 1, 1 };
    std::string kernel_names[6] = {
        "test_wg_scan_local_work_group_size",
        "test_wg_scan_local_work_group_size_static_local",
        "test_wg_scan_local_work_group_size_static_local",
        "test_wg_scan_local_work_group_size_static_local",
        "test_wg_scan_local_work_group_size_static_local",
        "test_wg_scan_local_work_group_size_dynlocal"
    };
    std::string str_local_mem_size[6] = {
        "-DLOCAL_MEM_SIZE=1",     "-DLOCAL_MEM_SIZE=1024",
        "-DLOCAL_MEM_SIZE=4096",  "-DLOCAL_MEM_SIZE=16384",
        "-DLOCAL_MEM_SIZE=32768", "-DLOCAL_MEM_SIZE=1"
    };
    size_t local_mem_size[6] = { 1, 1024, 4096, 16384, 32768, 1 };
    size_t dyn_mem_size[6] = { 0, 0, 0, 0, 0, 1024 };
    cl_ulong kernel_local_mem_size;
    for (int kernel_num = 0; kernel_num < 6; kernel_num++)
    {
        if (max_local_mem_size < local_mem_size[kernel_num]) continue;
        // Create the kernel
        err = create_single_kernel_helper(
            context, &scan_program, &scan_kernel, 1,
            &wg_scan_local_work_group_size, (kernel_names[kernel_num]).c_str(),
            (str_local_mem_size[kernel_num]).c_str());
        test_error_ret(err,
                       ("create_single_kernel_helper failed for kernel "
                        + kernel_names[kernel_num])
                           .c_str(),
                       -1);

        // Check if the local memory used by the kernel is going to exceed the
        // max_local_mem_size
        err = clGetKernelWorkGroupInfo(
            scan_kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
            sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL);
        test_error_ret(err, "clGetKernelWorkGroupInfo failed", -1);
        if (kernel_local_mem_size > max_local_mem_size) continue;
        // return error if no number is found due to the skip condition
        err = -1;
        unsigned int j = 0;
        size_t num_elems = NELEMS(value_range_nD);
        for (size_t i = start; i < end; i += incr)
        {
            if (skip_cond(i)) continue;
            err = 0;
            test_values[0] = i;
            if (dim == _2D) test_values[1] = value_range_nD[j++ % num_elems];
            if (dim == _3D)
            {
                test_values[1] = value_range_nD[j++ % num_elems];
                test_values[2] = value_range_nD[rand() % num_elems];
            }
            err |= do_test(device, context, queue, scan_kernel, dim,
                           global_work_offset, test_values,
                           dyn_mem_size[kernel_num]);
            test_error_ret(
                err,
                ("do_test failed for kernel " + kernel_names[kernel_num])
                    .c_str(),
                -1);
        }
    }
    return err;
}

int test_work_group_suggested_local_size_1D(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue, int n_elems)
{
    if (!is_extension_available(device, "cl_khr_suggested_local_work_size"))
    {
        log_info("Device does not support 'cl_khr_suggested_local_work_size'. "
                 "Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    cl_long max_local_mem_size;
    cl_int err =
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error_ret(err, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.",
                   -1);

    size_t start, end, incr;
    size_t global_work_offset[] = { 0, 0, 0 };
    size_t max_work_items = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_work_items), &max_work_items, NULL);

    // odds
    start = 1;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_1D for odds failed.", -1);
    log_info("test_work_group_suggested_local_size_1D odds passed\n");

    // evens
    start = 2;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_1D for evens failed.", -1);
    log_info("test_work_group_suggested_local_size_1D evens passed\n");

    // primes
    start = max_work_items + 1;
    end = 2 * max_work_items;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_1D for primes failed.", -1);
    log_info("test_work_group_suggested_local_size_1D primes passed\n");

    global_work_offset[0] = 10;
    global_work_offset[1] = 10;
    global_work_offset[2] = 10;
    // odds
    start = 1;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_1D for odds with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_1D odds with "
             "global_work_offset passed\n");

    // evens
    start = 2;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_1D for evens with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_1D evens with "
             "global_work_offset passed\n");

    // primes
    start = max_work_items + 1;
    end = 2 * max_work_items;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _1D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_1D for primes with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_1D primes with "
             "global_work_offset passed\n");

    return err;
}

int test_work_group_suggested_local_size_2D(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue, int n_elems)
{
    if (!is_extension_available(device, "cl_khr_suggested_local_work_size"))
    {
        log_info("Device does not support 'cl_khr_suggested_local_work_size'. "
                 "Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    cl_long max_local_mem_size;
    cl_int err =
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error_ret(err, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.",
                   -1);

    size_t start, end, incr;
    size_t global_work_offset[] = { 0, 0, 0 };
    size_t max_work_items = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_work_items), &max_work_items, NULL);

    // odds
    start = 1;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_2D for odds failed.", -1);
    log_info("test_work_group_suggested_local_size_2D odds passed\n");

    // evens
    start = 2;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_2D for evens failed.", -1);
    log_info("test_work_group_suggested_local_size_2D evens passed\n");

    // primes
    start = max_work_items + 1;
    end = max_work_items + max_work_items / 4;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_2D for primes failed.", -1);
    log_info("test_work_group_suggested_local_size_2D primes passed\n");

    global_work_offset[0] = 10;
    global_work_offset[1] = 10;
    global_work_offset[2] = 10;

    // odds
    start = 1;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_2D for odds with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_2D odds with "
             "global_work_offset passed\n");

    // evens
    start = 2;
    end = max_work_items;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_2D for evens with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_2D evens with "
             "global_work_offset passed\n");

    // primes
    start = max_work_items + 1;
    end = max_work_items + max_work_items / 4;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _2D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_2D for primes with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_2D primes with "
             "global_work_offset passed\n");

    return err;
}

int test_work_group_suggested_local_size_3D(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue, int n_elems)
{
    if (!is_extension_available(device, "cl_khr_suggested_local_work_size"))
    {
        log_info("Device does not support 'cl_khr_suggested_local_work_size'. "
                 "Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    cl_long max_local_mem_size;
    cl_int err =
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error_ret(err, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.",
                   -1);

    size_t start, end, incr;
    size_t global_work_offset[] = { 0, 0, 0 };
    size_t max_work_items = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_work_items), &max_work_items, NULL);

    // odds
    start = 1;
    end = max_work_items / 2;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_3D for odds failed.", -1);
    log_info("test_work_group_suggested_local_size_3D odds passed\n");

    // evens
    start = 2;
    end = max_work_items / 2;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_3D for evens failed.", -1);
    log_info("test_work_group_suggested_local_size_3D evens passed\n");

    // primes
    start = max_work_items + 1;
    end = max_work_items + max_work_items / 4;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(
        err, "test_work_group_suggested_local_size_3D for primes failed.", -1);
    log_info("test_work_group_suggested_local_size_3D primes passed\n");

    global_work_offset[0] = 10;
    global_work_offset[1] = 10;
    global_work_offset[2] = 10;

    // odds
    start = 1;
    end = max_work_items / 2;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_odd, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_3D for odds with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_3D odds with "
             "global_work_offset passed\n");

    // evens
    start = 2;
    end = max_work_items / 2;
    incr = basic_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_even, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_3D for evens with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_3D evens with "
             "global_work_offset passed\n");

    // primes
    start = max_work_items + 1;
    end = max_work_items + max_work_items / 4;
    incr = primes_increment;
    err = do_test_work_group_suggested_local_size(
        device, context, queue, is_not_prime, start, end, incr,
        max_local_mem_size, global_work_offset, _3D);
    test_error_ret(err,
                   "test_work_group_suggested_local_size_3D for primes with "
                   "global_work_offset failed.",
                   -1);
    log_info("test_work_group_suggested_local_size_3D primes with "
             "global_work_offset passed\n");

    return err;
}

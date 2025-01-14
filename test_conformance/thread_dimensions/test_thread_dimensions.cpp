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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cinttypes>

#include "procs.h"

#define ITERATIONS 4
#define DEBUG 0

// If the environment variable DO_NOT_LIMIT_THREAD_SIZE is not set, the test
// will limit the maximum total global dimensions tested to this value.
#define MAX_TOTAL_GLOBAL_THREADS_FOR_TEST (1 << 24)
int limit_size = 0;

extern cl_uint maxThreadDimension;
extern cl_uint bufferSize;
extern cl_uint bufferStep;

static int get_maximums(cl_kernel kernel, cl_context context,
                        size_t *max_workgroup_size_result,
                        cl_ulong *max_allcoation_result,
                        cl_ulong *max_physical_result)
{
    int err = 0;
    cl_uint i;
    cl_device_id *devices;

    // Get all the devices in the device group
    size_t num_devices_returned;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
                           &num_devices_returned);
    if (err != CL_SUCCESS)
    {
        log_error("clGetContextInfo() failed (%d).\n", err);
        return -10;
    }
    devices = (cl_device_id *)malloc(num_devices_returned);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, num_devices_returned,
                           devices, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clGetContextInfo() failed (%d).\n", err);
        return -10;
    }
    num_devices_returned /= sizeof(cl_device_id);
    if (num_devices_returned > 1)
        log_info("%d devices in device group.\n", (int)num_devices_returned);
    if (num_devices_returned < 1)
    {
        log_error("0 devices found for this kernel.\n");
        return -1;
    }

    // Iterate over them and find the maximum local workgroup size
    size_t max_workgroup_size = 0;
    size_t current_workgroup_size = 0;
    cl_ulong max_allocation = 0;
    cl_ulong current_allocation = 0;
    cl_ulong max_physical = 0;
    cl_ulong current_physical = 0;

    for (i = 0; i < num_devices_returned; i++)
    {
        // Max workgroup size for this kernel on this device
        err = clGetKernelWorkGroupInfo(
            kernel, devices[i], CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(current_workgroup_size), &current_workgroup_size, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clGetKernelWorkGroupInfo() failed (%d) for device %d.\n",
                      err, i);
            return -10;
        }
        if (max_workgroup_size == 0)
            max_workgroup_size = current_workgroup_size;
        else if (current_workgroup_size < max_workgroup_size)
            max_workgroup_size = current_workgroup_size;

        // Get the maximum allocation size
        err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                              sizeof(current_allocation), &current_allocation,
                              NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clGetDeviceConfigInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE) "
                      "failed (%d) for device %d.\n",
                      err, i);
            return -10;
        }
        if (max_allocation == 0)
            max_allocation = current_allocation;
        else if (current_allocation < max_allocation)
            max_allocation = current_allocation;

        // Get the maximum physical size
        err =
            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(current_physical), &current_physical, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clGetDeviceConfigInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed "
                      "(%d) for device %d.\n",
                      err, i);
            return -10;
        }
        if (max_physical == 0)
            max_physical = current_physical;
        else if (current_physical < max_allocation)
            max_physical = current_physical;
    }
    free(devices);

    log_info("Device maximums: max local workgroup size:%d, max allocation "
             "size: %g MB, max physical memory %gMB\n",
             (int)max_workgroup_size,
             (double)(max_allocation / 1024.0 / 1024.0),
             (double)(max_physical / 1024.0 / 1024.0));
    *max_workgroup_size_result = max_workgroup_size;
    *max_allcoation_result = max_allocation;
    *max_physical_result = max_physical;
    return 0;
}

static const char *thread_dimension_kernel_code_atomic_long =
    "\n"
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n"
    "__kernel void test_thread_dimension_atomic(__global uint *dst, \n"
    "          uint final_x_size,   uint final_y_size,   uint final_z_size,\n"
    "          ulong start_address,  ulong end_address)\n"
    "{\n"
    "    uint error = 0;\n"
    "            if (get_global_id(0) >= final_x_size)\n"
    "                error = 64;\n"
    "            if (get_global_id(1) >= final_y_size)\n"
    "                error = 128;\n"
    "            if (get_global_id(2) >= final_z_size)\n"
    "                error = 256;\n"
    "\n"
    "        unsigned long t_address = (unsigned "
    "long)get_global_id(2)*(unsigned long)final_y_size*(unsigned "
    "long)final_x_size + \n"
    "                (unsigned long)get_global_id(1)*(unsigned "
    "long)final_x_size + (unsigned long)get_global_id(0);\n"
    "        if ((t_address >= start_address) && (t_address < end_address))\n"
    "                atom_add(&dst[t_address-start_address], 1u);\n"
    "        if (error)\n"
    "                atom_or(&dst[t_address-start_address], error);\n"
    "\n"
    "}\n";

static const char *thread_dimension_kernel_code_not_atomic_long =
    "\n"
    "__kernel void test_thread_dimension_not_atomic(__global uint *dst, \n"
    "          uint final_x_size,   uint final_y_size,   uint final_z_size,\n"
    "          ulong start_address,  ulong end_address)\n"
    "{\n"
    "    uint error = 0;\n"
    "            if (get_global_id(0) >= final_x_size)\n"
    "                error = 64;\n"
    "            if (get_global_id(1) >= final_y_size)\n"
    "                error = 128;\n"
    "            if (get_global_id(2) >= final_z_size)\n"
    "                error = 256;\n"
    "\n"
    "        unsigned long t_address = (unsigned "
    "long)get_global_id(2)*(unsigned long)final_y_size*(unsigned "
    "long)final_x_size + \n"
    "                (unsigned long)get_global_id(1)*(unsigned "
    "long)final_x_size + (unsigned long)get_global_id(0);\n"
    "        if ((t_address >= start_address) && (t_address < end_address))\n"
    "                dst[t_address-start_address]++;\n"
    "        if (error)\n"
    "                dst[t_address-start_address]|=error;\n"
    "\n"
    "}\n";

static const char *thread_dimension_kernel_code_atomic_not_long =
    "\n"
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n"
    "__kernel void test_thread_dimension_atomic(__global uint *dst, \n"
    "         uint final_x_size,   uint final_y_size,   uint final_z_size,\n"
    "         uint start_address,  uint end_address)\n"
    "{\n"
    "    uint error = 0;\n"
    "           if (get_global_id(0) >= final_x_size)\n"
    "               error = 64;\n"
    "           if (get_global_id(1) >= final_y_size)\n"
    "               error = 128;\n"
    "           if (get_global_id(2) >= final_z_size)\n"
    "               error = 256;\n"
    "\n"
    "       unsigned int t_address = (unsigned int)get_global_id(2)*(unsigned "
    "int)final_y_size*(unsigned int)final_x_size + \n"
    "               (unsigned int)get_global_id(1)*(unsigned int)final_x_size "
    "+ (unsigned int)get_global_id(0);\n"
    "       if ((t_address >= start_address) && (t_address < end_address))\n"
    "               atom_add(&dst[t_address-start_address], 1u);\n"
    "       if (error)\n"
    "               atom_or(&dst[t_address-start_address], error);\n"
    "\n"
    "}\n";

static const char *thread_dimension_kernel_code_not_atomic_not_long =
    "\n"
    "__kernel void test_thread_dimension_not_atomic(__global uint *dst, \n"
    "         uint final_x_size,   uint final_y_size,   uint final_z_size,\n"
    "         uint start_address,  uint end_address)\n"
    "{\n"
    "    uint error = 0;\n"
    "           if (get_global_id(0) >= final_x_size)\n"
    "               error = 64;\n"
    "           if (get_global_id(1) >= final_y_size)\n"
    "               error = 128;\n"
    "           if (get_global_id(2) >= final_z_size)\n"
    "               error = 256;\n"
    "\n"
    "       unsigned int t_address = (unsigned int)get_global_id(2)*(unsigned "
    "int)final_y_size*(unsigned int)final_x_size + \n"
    "               (unsigned int)get_global_id(1)*(unsigned int)final_x_size "
    "+ (unsigned int)get_global_id(0);\n"
    "       if ((t_address >= start_address) && (t_address < end_address))\n"
    "               dst[t_address-start_address]++;\n"
    "       if (error)\n"
    "               dst[t_address-start_address]|=error;\n"
    "\n"
    "}\n";

char *print_dimensions(char *dim_str, size_t x, size_t y, size_t z, cl_uint dim)
{
    if (dim == 1)
    {
        snprintf(dim_str, 128, "[%d]", (int)x);
    }
    else if (dim == 2)
    {
        snprintf(dim_str, 128, "[%d x %d]", (int)x, (int)y);
    }
    else if (dim == 3)
    {
        snprintf(dim_str, 128, "[%d x %d x %d]", (int)x, (int)y, (int)z);
    }
    else
    {
        snprintf(dim_str, 128, "INVALID DIM: %d", dim);
    }
    return dim_str;
}

char *print_dimensions2(char *dim_str2, size_t x, size_t y, size_t z,
                        cl_uint dim)
{
    if (dim == 1)
    {
        snprintf(dim_str2, 128, "[%d]", (int)x);
    }
    else if (dim == 2)
    {
        snprintf(dim_str2, 128, "[%d x %d]", (int)x, (int)y);
    }
    else if (dim == 3)
    {
        snprintf(dim_str2, 128, "[%d x %d x %d]", (int)x, (int)y, (int)z);
    }
    else
    {
        snprintf(dim_str2, 128, "INVALID DIM: %d", dim);
    }
    return dim_str2;
}


/*
 This tests thread dimensions by executing a kernel across a range of
 dimensions. Each kernel instance does an atomic write into a specific location
 in a buffer to ensure that the correct dimensions are run. To handle large
 dimensions, the kernel masks its execution region internally. This allows a
 small (128MB) buffer to be used for very large executions by running the kernel
 multiple times.
 */
int run_test(cl_context context, cl_command_queue queue, cl_kernel kernel,
             cl_mem array, cl_uint memory_size, cl_uint dimensions,
             cl_uint final_x_size, cl_uint final_y_size, cl_uint final_z_size,
             cl_uint local_x_size, cl_uint local_y_size, cl_uint local_z_size,
             int explict_local)
{
    cl_uint errors = 0;
    size_t global_size[3], local_size[3];
    global_size[0] = final_x_size;
    local_size[0] = local_x_size;
    global_size[1] = final_y_size;
    local_size[1] = local_y_size;
    global_size[2] = final_z_size;
    local_size[2] = local_z_size;

    char dim_str[128];
    char dim_str2[128];

    cl_ulong start_valid_memory_address = 0;
    cl_ulong end_valid_memory_address = memory_size;
    cl_ulong last_memory_address = (cl_ulong)final_x_size
        * (cl_ulong)final_y_size * (cl_ulong)final_z_size * sizeof(cl_uint);
    if (end_valid_memory_address > last_memory_address)
        end_valid_memory_address = last_memory_address;

    int number_of_iterations_required =
        (int)ceil((double)last_memory_address / (double)memory_size);
    log_info("\t\tTest requires %gMB (%d test iterations using an allocation "
             "of %gMB).\n",
             (double)last_memory_address / (1024.0 * 1024.0),
             number_of_iterations_required,
             (double)memory_size / (1024.0 * 1024.0));
    // log_info("Last memory address: %llu, memory_size: %llu\n",
    // last_memory_address, memory_size);

    while (end_valid_memory_address <= last_memory_address)
    {
        int err;
        const int fill_pattern = 0x0;
        err = clEnqueueFillBuffer(queue, array, (void *)&fill_pattern,
                                  sizeof(fill_pattern), 0, memory_size, 0, NULL,
                                  NULL);
        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to set fill buffer.");
            return -3;
        }

        cl_ulong start_valid_index =
            start_valid_memory_address / sizeof(cl_uint);
        cl_ulong end_valid_index = end_valid_memory_address / sizeof(cl_uint);

        cl_uint start_valid_index_int = (cl_uint)start_valid_index;
        cl_uint end_valid_index_int = (cl_uint)end_valid_index;

        // Set the arguments
        err = clSetKernelArg(kernel, 0, sizeof(array), &array);
        err |= clSetKernelArg(kernel, 1, sizeof(final_x_size), &final_x_size);
        err |= clSetKernelArg(kernel, 2, sizeof(final_y_size), &final_y_size);
        err |= clSetKernelArg(kernel, 3, sizeof(final_z_size), &final_z_size);
        if (gHasLong)
        {
            err |= clSetKernelArg(kernel, 4, sizeof(start_valid_index),
                                  &start_valid_index);
            err |= clSetKernelArg(kernel, 5, sizeof(end_valid_index),
                                  &end_valid_index);
        }
        else
        {
            err |= clSetKernelArg(kernel, 4, sizeof(start_valid_index_int),
                                  &start_valid_index_int);
            err |= clSetKernelArg(kernel, 5, sizeof(end_valid_index_int),
                                  &end_valid_index_int);
        }

        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to set arguments.");
            return -3;
        }


        // Execute the kernel
        if (explict_local == 0)
        {
            err = clEnqueueNDRangeKernel(queue, kernel, dimensions, NULL,
                                         global_size, NULL, 0, NULL, NULL);
            if (DEBUG)
                log_info("\t\t\tExecuting kernel with global %s, NULL local, "
                         "%d dim, start address %" PRIu64
                         ", end address %" PRIu64 ".\n",
                         print_dimensions(dim_str, global_size[0],
                                          global_size[1], global_size[2],
                                          dimensions),
                         dimensions, start_valid_memory_address,
                         end_valid_memory_address);
        }
        else
        {
            err =
                clEnqueueNDRangeKernel(queue, kernel, dimensions, NULL,
                                       global_size, local_size, 0, NULL, NULL);
            if (DEBUG)
                log_info(
                    "\t\t\tExecuting kernel with global %s, local %s, %d "
                    "dim, start address %" PRIu64 ", end address %" PRIu64
                    ".\n",
                    print_dimensions(dim_str, global_size[0], global_size[1],
                                     global_size[2], dimensions),
                    print_dimensions2(dim_str2, local_size[0], local_size[1],
                                      local_size[2], dimensions),
                    dimensions, start_valid_memory_address,
                    end_valid_memory_address);
        }
        if (err == CL_OUT_OF_RESOURCES)
        {
            log_info(
                "WARNING: kernel reported CL_OUT_OF_RESOURCES, indicating the "
                "global dimensions are too large. Skipping this size.\n");
            return 0;
        }
        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to execute kernel\n");
            return -3;
        }

        void *mapped = clEnqueueMapBuffer(queue, array, CL_TRUE, CL_MAP_READ, 0,
                                          memory_size, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to map results\n");
            return -4;
        }
        cl_uint *data = (cl_uint *)mapped;

        // Verify the data
        cl_uint i;
        cl_uint last_address =
            (cl_uint)(end_valid_memory_address - start_valid_memory_address)
            / (cl_uint)sizeof(cl_uint);
        for (i = 0; i < last_address; i++)
        {
            if (i < last_address)
            {
                if (data[i] != 1)
                {
                    errors++;
                    //        log_info("%d expected 1 got %d\n", i, data[i]);
                }
            }
            else
            {
                if (data[i] != 0)
                {
                    errors++;
                    log_info("%d expected 0 got %d\n", i, data[i]);
                }
            }
        }

        err = clEnqueueUnmapMemObject(queue, array, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to unmap results\n");
            return -4;
        }

        err = clFlush(queue);
        if (err != CL_SUCCESS)
        {
            print_error(err, "Failed to flush\n");
            return -4;
        }

        // Increment the addresses
        if (end_valid_memory_address == last_memory_address) break;
        start_valid_memory_address +=
            memory_size * (bufferStep ? bufferStep : 1);
        end_valid_memory_address += memory_size * (bufferStep ? bufferStep : 1);
        if (end_valid_memory_address > last_memory_address)
            end_valid_memory_address = last_memory_address;
    }

    if (errors) log_error("%d errors.\n", errors);
    return errors;
}


#define set_min(x, y, z)                                                       \
    {                                                                          \
        if (x < min_x_size) x = min_x_size;                                    \
        if (y < min_y_size) y = min_y_size;                                    \
        if (z < min_z_size) z = min_z_size;                                    \
        if (x > max_x_size) x = max_x_size;                                    \
        if (y > max_y_size) y = max_y_size;                                    \
        if (z > max_z_size) z = max_z_size;                                    \
    }


int test_thread_dimensions(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_uint dimensions,
                           cl_uint min_dim, cl_uint max_dim, cl_uint quick_test,
                           cl_uint size_increase_per_iteration,
                           int explicit_local)
{
    cl_mem array;
    cl_program program;
    cl_kernel kernel;
    int err;
    cl_uint memory_size, max_memory_size;
    size_t max_local_workgroup_size[3];
    cl_uint device_max_dimensions;
    int use_atomics = 1;
    MTdata d;

    char dim_str[128];
    char dim_str2[128];

    cl_uint max_x_size = 1, min_x_size = 1, max_y_size = 1, min_y_size = 1,
            max_z_size = 1, min_z_size = 1;

    if (getenv("CL_WIMPY_MODE") && !quick_test)
    {
        log_info("CL_WIMPY_MODE enabled, skipping test\n");
        return 0;
    }

    // Unconditionally test larger sizes for CL 1.1
    log_info("Testing large global dimensions.\n");
    limit_size = 0;

    /* Check if atomics are supported. */
    if (!is_extension_available(device, "cl_khr_global_int32_base_atomics"))
    {
        log_info("WARNING: Base atomics not supported "
                 "(cl_khr_global_int32_base_atomics). Test will not be "
                 "guaranteed to catch overlaping thread dimensions.\n");
        use_atomics = 0;
    }

    if (quick_test)
        log_info("WARNING: Running quick test. This will only test the base "
                 "dimensions (power of two) and base-1 with all local threads "
                 "fixed in one dim.\n");

    // Verify that we can test this many dimensions
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                          sizeof(device_max_dimensions), &device_max_dimensions,
                          NULL);
    test_error(err,
               "clGetDeviceInfo for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS failed");

    if (dimensions > device_max_dimensions)
    {
        log_info("Can not test %d dimensions when device only supports %d.\n",
                 dimensions, device_max_dimensions);
        return 0;
    }

    log_info("Setting random seed to 0.\n");

    if (gHasLong)
    {
        if (use_atomics)
        {
            err = create_single_kernel_helper(
                context, &program, &kernel, 1,
                &thread_dimension_kernel_code_atomic_long,
                "test_thread_dimension_atomic");
        }
        else
        {
            err = create_single_kernel_helper(
                context, &program, &kernel, 1,
                &thread_dimension_kernel_code_not_atomic_long,
                "test_thread_dimension_not_atomic");
        }
    }
    else
    {
        if (use_atomics)
        {
            err = create_single_kernel_helper(
                context, &program, &kernel, 1,
                &thread_dimension_kernel_code_atomic_not_long,
                "test_thread_dimension_atomic");
        }
        else
        {
            err = create_single_kernel_helper(
                context, &program, &kernel, 1,
                &thread_dimension_kernel_code_not_atomic_not_long,
                "test_thread_dimension_not_atomic");
        }
    }
    test_error(err, "Unable to create testing kernel");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(max_local_workgroup_size),
                          max_local_workgroup_size, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Get the maximum sizes supported by this device
    size_t max_workgroup_size = 0;
    cl_ulong max_allocation = 0;
    cl_ulong max_physical = 0;
    int found_size = 0;

    err = get_maximums(kernel, context, &max_workgroup_size, &max_allocation,
                       &max_physical);

    // Make sure we don't try to allocate more than half the physical memory
    // present.
    if (max_allocation > (max_physical / 2))
    {
        log_info("Limiting max allocation to half of the maximum physical "
                 "memory (%gMB of %gMB physical).\n",
                 (max_physical / 2 / (1024.0 * 1024.0)),
                 (max_physical / (1024.0 * 1024.0)));
        max_allocation = max_physical / 2;
    }

    // Limit the maximum we'll allocate for this test to 512 to be reasonable.
    if (max_allocation > 1024 * 1024 * 512)
    {
        log_info("Limiting max allocation to 512MB from device maximum "
                 "allocation of %gMB.\n",
                 (max_allocation / 1024.0 / 1024.0));
        max_allocation = 1024 * 1024 * 512;
    }

    max_memory_size = bufferSize ? bufferSize : (cl_uint)(max_allocation);
    if (max_memory_size > 512 * 1024 * 1024)
        max_memory_size = 512 * 1024 * 1024;
    memory_size = max_memory_size;

    log_info(
        "Memory allocation size to use is %gMB, max workgroup size is %d.\n",
        max_memory_size / (1024.0 * 1024.0), (int)max_workgroup_size);

    while (!found_size && memory_size >= max_memory_size / 8)
    {
        array =
            clCreateBuffer(context, CL_MEM_READ_WRITE, memory_size, NULL, &err);
        if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE
            || err == CL_OUT_OF_HOST_MEMORY)
        {
            memory_size -= max_memory_size / 16;
            continue;
        }
        if (err)
        {
            print_error(err, "clCreateBuffer failed");
            return -1;
        }
        found_size = 1;
    }

    if (!found_size)
    {
        log_error("Failed to find a working size greater than 1/8th of the "
                  "reported allocation size.\n");
        return -1;
    }

    if (memory_size < max_memory_size)
    {
        log_info("Note: failed to allocate %gMB, using %gMB instead.\n",
                 max_memory_size / (1024.0 * 1024.0),
                 memory_size / (1024.0 * 1024.0));
    }

    int errors = 0;
    // Each dimension's size is multiplied by this amount on each iteration.
    //  uint size_increase_per_iteration = 4;
    // 1 test at the specified size
    // 2 tests with each dimensions +/- 1
    // 2 tests with all dimensions +/- 1
    // 2 random tests
    cl_uint tests_per_size = 1 + 2 * dimensions + 2 + 2;

    // 1 test with 1 as the local threads in each dimensions
    // 1 test with all the local threads in each dimension
    // 2 random tests
    cl_uint local_tests_per_size = 1 + dimensions + 2;
    if (explicit_local == 0) local_tests_per_size = 1;

    if (dimensions > 3)
    {
        log_error("Invalid dimensions: %d\n", dimensions);
        return -1;
    }
    max_x_size = max_dim;
    min_x_size = min_dim;
    if (dimensions > 1)
    {
        max_y_size = max_dim;
        min_y_size = min_dim;
    }
    if (dimensions > 2)
    {
        max_z_size = max_dim;
        min_z_size = min_dim;
    }

    log_info("Testing with dimensions up to %s.\n",
             print_dimensions(dim_str, max_x_size, max_y_size, max_z_size,
                              dimensions));
    if (bufferSize)
    {
        log_info("Testing with buffer size %d.\n", bufferSize);
    }
    if (bufferStep)
    {
        log_info("Testing with buffer step %d.\n", bufferStep);
    }
    cl_uint x_size, y_size, z_size;

    d = init_genrand(gRandomSeed);
    z_size = min_z_size;
    while (z_size <= max_z_size)
    {
        y_size = min_y_size;
        while (y_size <= max_y_size)
        {
            x_size = min_x_size;
            while (x_size <= max_x_size)
            {

                log_info("Base test size %s:\n",
                         print_dimensions(dim_str, x_size, y_size, z_size,
                                          dimensions));

                cl_uint sub_test;
                cl_uint final_x_size, final_y_size, final_z_size;
                for (sub_test = 0; sub_test < tests_per_size; sub_test++)
                {
                    final_x_size = x_size;
                    final_y_size = y_size;
                    final_z_size = z_size;

                    if (sub_test == 0)
                    {
                        if (DEBUG)
                            log_info("\tTesting with base dimensions %s.\n",
                                     print_dimensions(
                                         dim_str, final_x_size, final_y_size,
                                         final_z_size, dimensions));
                    }
                    else if (quick_test)
                    {
                        // If we are in quick mode just do 1 run with x-1, y-1,
                        // and z-1.
                        if (sub_test > 1) break;
                        final_x_size--;
                        final_y_size--;
                        final_z_size--;
                        set_min(final_x_size, final_y_size, final_z_size);
                        if (DEBUG)
                            log_info(
                                "\tTesting with all base dimensions - 1 %s.\n",
                                print_dimensions(dim_str, final_x_size,
                                                 final_y_size, final_z_size,
                                                 dimensions));
                    }
                    else if (sub_test <= dimensions * 2)
                    {
                        int dim_to_change = (sub_test - 1) % dimensions;
                        // log_info ("dim_to_change: %d (sub_test:%d) dimensions
                        // %d\n", dim_to_change,sub_test, dimensions);
                        int up_down = (sub_test > dimensions) ? 0 : 1;

                        if (dim_to_change == 0)
                        {
                            final_x_size += (up_down) ? -1 : +1;
                        }
                        else if (dim_to_change == 1)
                        {
                            final_y_size += (up_down) ? -1 : +1;
                        }
                        else if (dim_to_change == 2)
                        {
                            final_z_size += (up_down) ? -1 : +1;
                        }
                        else
                        {
                            log_error("Invalid dim_to_change: %d\n",
                                      dim_to_change);
                            return -1;
                        }
                        set_min(final_x_size, final_y_size, final_z_size);
                        if (DEBUG)
                            log_info(
                                "\tTesting with one base dimension +/- 1 %s.\n",
                                print_dimensions(dim_str, final_x_size,
                                                 final_y_size, final_z_size,
                                                 dimensions));
                    }
                    else if (sub_test == (dimensions * 2 + 1))
                    {
                        if (dimensions == 1) continue;
                        final_x_size--;
                        final_y_size--;
                        final_z_size--;
                        set_min(final_x_size, final_y_size, final_z_size);
                        if (DEBUG)
                            log_info(
                                "\tTesting with all base dimensions - 1 %s.\n",
                                print_dimensions(dim_str, final_x_size,
                                                 final_y_size, final_z_size,
                                                 dimensions));
                    }
                    else if (sub_test == (dimensions * 2 + 2))
                    {
                        if (dimensions == 1) continue;
                        final_x_size++;
                        final_y_size++;
                        final_z_size++;
                        set_min(final_x_size, final_y_size, final_z_size);
                        if (DEBUG)
                            log_info(
                                "\tTesting with all base dimensions + 1 %s.\n",
                                print_dimensions(dim_str, final_x_size,
                                                 final_y_size, final_z_size,
                                                 dimensions));
                    }
                    else
                    {
                        final_x_size =
                            (int)get_random_float(
                                0, (x_size / size_increase_per_iteration), d)
                            + x_size / size_increase_per_iteration;
                        final_y_size =
                            (int)get_random_float(
                                0, (y_size / size_increase_per_iteration), d)
                            + y_size / size_increase_per_iteration;
                        final_z_size =
                            (int)get_random_float(
                                0, (z_size / size_increase_per_iteration), d)
                            + z_size / size_increase_per_iteration;
                        set_min(final_x_size, final_y_size, final_z_size);
                        if (DEBUG)
                            log_info("\tTesting with random dimensions %s.\n",
                                     print_dimensions(
                                         dim_str, final_x_size, final_y_size,
                                         final_z_size, dimensions));
                    }

                    if (limit_size
                        && final_x_size * final_y_size * final_z_size
                            >= MAX_TOTAL_GLOBAL_THREADS_FOR_TEST)
                    {
                        log_info("Skipping size %s as it exceeds max test "
                                 "threads of %d.\n",
                                 print_dimensions(dim_str, final_x_size,
                                                  final_y_size, final_z_size,
                                                  dimensions),
                                 MAX_TOTAL_GLOBAL_THREADS_FOR_TEST);
                        continue;
                    }

                    cl_uint local_test;
                    cl_uint local_x_size, local_y_size, local_z_size;
                    cl_uint previous_local_x_size = 0,
                            previous_local_y_size = 0,
                            previous_local_z_size = 0;
                    for (local_test = 0; local_test < local_tests_per_size;
                         local_test++)
                    {

                        local_x_size = 1;
                        local_y_size = 1;
                        local_z_size = 1;

                        if (local_test == 0)
                        {
                        }
                        else if (local_test <= dimensions)
                        {
                            int dim_to_change = (local_test - 1) % dimensions;
                            if (dim_to_change == 0)
                            {
                                local_x_size = (cl_uint)max_workgroup_size;
                            }
                            else if (dim_to_change == 1)
                            {
                                local_y_size = (cl_uint)max_workgroup_size;
                            }
                            else if (dim_to_change == 2)
                            {
                                local_z_size = (cl_uint)max_workgroup_size;
                            }
                            else
                            {
                                log_error("Invalid dim_to_change: %d\n",
                                          dim_to_change);
                                free_mtdata(d);
                                return -1;
                            }
                        }
                        else
                        {
                            local_x_size = (int)get_random_float(
                                1, (int)max_workgroup_size, d);
                            while ((local_x_size > 1)
                                   && (final_x_size % local_x_size != 0))
                                local_x_size--;
                            int remainder = (int)floor(
                                (double)max_workgroup_size / local_x_size);
                            // Evenly prefer dimensions 2 and 1 first
                            if (local_test % 2)
                            {
                                if (dimensions > 1)
                                {
                                    local_y_size = (int)get_random_float(
                                        1, (int)remainder, d);
                                    while (
                                        (local_y_size > 1)
                                        && (final_y_size % local_y_size != 0))
                                        local_y_size--;
                                    remainder = (int)floor((double)remainder
                                                           / local_y_size);
                                }
                                if (dimensions > 2)
                                {
                                    local_z_size = (int)get_random_float(
                                        1, (int)remainder, d);
                                    while (
                                        (local_z_size > 1)
                                        && (final_z_size % local_z_size != 0))
                                        local_z_size--;
                                }
                            }
                            else
                            {
                                if (dimensions > 2)
                                {
                                    local_z_size = (int)get_random_float(
                                        1, (int)remainder, d);
                                    while (
                                        (local_z_size > 1)
                                        && (final_z_size % local_z_size != 0))
                                        local_z_size--;
                                    remainder = (int)floor((double)remainder
                                                           / local_z_size);
                                }
                                if (dimensions > 1)
                                {
                                    local_y_size = (int)get_random_float(
                                        1, (int)remainder, d);
                                    while (
                                        (local_y_size > 1)
                                        && (final_y_size % local_y_size != 0))
                                        local_y_size--;
                                }
                            }
                        }

                        // Put all the threads in one dimension to speed up the
                        // test in quick mode.
                        if (quick_test)
                        {
                            local_y_size = 1;
                            local_z_size = 1;
                            local_x_size = 1;
                            if (final_z_size > final_y_size
                                && final_z_size > final_x_size)
                                local_z_size = (cl_uint)max_workgroup_size;
                            else if (final_y_size > final_x_size)
                                local_y_size = (cl_uint)max_workgroup_size;
                            else
                                local_x_size = (cl_uint)max_workgroup_size;
                        }

                        if (local_x_size > max_local_workgroup_size[0])
                            local_x_size = (int)max_local_workgroup_size[0];
                        if (dimensions > 1
                            && local_y_size > max_local_workgroup_size[1])
                            local_y_size = (int)max_local_workgroup_size[1];
                        if (dimensions > 2
                            && local_z_size > max_local_workgroup_size[2])
                            local_z_size = (int)max_local_workgroup_size[2];

                        // Cleanup the local dimensions
                        while ((local_x_size > 1)
                               && (final_x_size % local_x_size != 0))
                            local_x_size--;
                        while ((local_y_size > 1)
                               && (final_y_size % local_y_size != 0))
                            local_y_size--;
                        while ((local_z_size > 1)
                               && (final_z_size % local_z_size != 0))
                            local_z_size--;
                        if ((previous_local_x_size == local_x_size)
                            && (previous_local_y_size == local_y_size)
                            && (previous_local_z_size == local_z_size))
                            continue;

                        if (explicit_local == 0)
                        {
                            local_x_size = 0;
                            local_y_size = 0;
                            local_z_size = 0;
                        }

                        if (DEBUG)
                            log_info("\t\tTesting local size %s.\n",
                                     print_dimensions(
                                         dim_str, local_x_size, local_y_size,
                                         local_z_size, dimensions));

                        if (explicit_local == 0)
                        {
                            log_info("\tTesting global %s local [NULL]...\n",
                                     print_dimensions(
                                         dim_str, final_x_size, final_y_size,
                                         final_z_size, dimensions));
                        }
                        else
                        {
                            log_info("\tTesting global %s local %s...\n",
                                     print_dimensions(dim_str, final_x_size,
                                                      final_y_size,
                                                      final_z_size, dimensions),
                                     print_dimensions2(
                                         dim_str2, local_x_size, local_y_size,
                                         local_z_size, dimensions));
                        }

                        // Avoid running with very small local sizes on very
                        // large global sizes
                        cl_uint total_local_size =
                            local_x_size * local_y_size * local_z_size;
                        long total_global_size = final_x_size * final_y_size * final_z_size;
                        if (total_local_size < max_workgroup_size) {
                            if (((total_global_size > 16384 * 16384)
                                 && (total_local_size < 64))
                                || ((total_global_size > 8192 * 8192)
                                    && (total_local_size < 16)))
                            {
                                log_info("Skipping test as local_size is small "
                                         "and it will take a long time.\n");
                                continue;
                            }
                        }

                        err =
                            run_test(context, queue, kernel, array, memory_size,
                                     dimensions, final_x_size, final_y_size,
                                     final_z_size, local_x_size, local_y_size,
                                     local_z_size, explicit_local);

                        // If we failed to execute, then return so we don't
                        // crash.
                        if (err < 0)
                        {
                            clReleaseMemObject(array);
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                            free_mtdata(d);
                            return -1;
                        }

                        // Otherwise, if we had errors add them up.
                        if (err)
                        {
                            log_error("Test global %s local %s failed.\n",
                                      print_dimensions(
                                          dim_str, final_x_size, final_y_size,
                                          final_z_size, dimensions),
                                      print_dimensions2(
                                          dim_str2, local_x_size, local_y_size,
                                          local_z_size, dimensions));
                            errors++;
                            clReleaseMemObject(array);
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                            free_mtdata(d);
                            return -1;
                        }


                        previous_local_x_size = local_x_size;
                        previous_local_y_size = local_y_size;
                        previous_local_z_size = local_z_size;

                        // Only test one config in quick mode.
                        if (quick_test) break;
                    } // local_test size
                } // sub_test
                  // Increment the x_size
                if (x_size == max_x_size) break;
                x_size *= size_increase_per_iteration;
                if (x_size > max_x_size) x_size = max_x_size;
            } // x_size
              // Increment the y_size
            if (y_size == max_y_size) break;
            y_size *= size_increase_per_iteration;
            if (y_size > max_y_size) y_size = max_y_size;
        } // y_size
          // Increment the z_size
        if (z_size == max_z_size) break;
        z_size *= size_increase_per_iteration;
        if (z_size > max_z_size) z_size = max_z_size;
    } // z_size


    free_mtdata(d);
    clReleaseMemObject(array);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    if (errors) log_error("%d total errors.\n", errors);
    return errors;
}

#define QUICK 1
#define FULL 0

int test_quick_1d_explicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 1, 1,
        maxThreadDimension ? maxThreadDimension : 65536 * 512, QUICK, 4, 1);
}

int test_quick_2d_explicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 2, 1,
        maxThreadDimension ? maxThreadDimension : 65536 / 4, QUICK, 16, 1);
}

int test_quick_3d_explicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 3, 1,
        maxThreadDimension ? maxThreadDimension : 1024, QUICK, 32, 1);
}


int test_quick_1d_implicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 1, 1,
        maxThreadDimension ? maxThreadDimension : 65536 * 256, QUICK, 4, 0);
}

int test_quick_2d_implicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 2, 1,
        maxThreadDimension ? maxThreadDimension : 65536 / 4, QUICK, 16, 0);
}

int test_quick_3d_implicit_local(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 3, 1,
        maxThreadDimension ? maxThreadDimension : 1024, QUICK, 32, 0);
}


int test_full_1d_explicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 1, 1,
        maxThreadDimension ? maxThreadDimension : 65536 * 512, FULL, 4, 1);
}

int test_full_2d_explicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 2, 1,
        maxThreadDimension ? maxThreadDimension : 65536 / 4, FULL, 16, 1);
}

int test_full_3d_explicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 3, 1,
        maxThreadDimension ? maxThreadDimension : 1024, FULL, 32, 1);
}


int test_full_1d_implicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 1, 1,
        maxThreadDimension ? maxThreadDimension : 65536 * 256, FULL, 4, 0);
}

int test_full_2d_implicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 2, 1,
        maxThreadDimension ? maxThreadDimension : 65536 / 4, FULL, 16, 0);
}

int test_full_3d_implicit_local(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_thread_dimensions(
        deviceID, context, queue, 3, 1,
        maxThreadDimension ? maxThreadDimension : 1024, FULL, 32, 0);
}

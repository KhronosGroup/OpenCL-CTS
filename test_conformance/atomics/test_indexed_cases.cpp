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
#include "testBase.h"
#include "harness/conversions.h"

// clang-format off
const char *atomic_index_source =
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    "// Counter keeps track of which index in counts we are using.\n"
    "// We get that value, increment it, and then set that index in counts to our thread ID.\n"
    "// At the end of this we should have all thread IDs in some random location in counts\n"
    "// exactly once. If atom_add failed then we will write over various thread IDs and we\n"
    "// will be missing some.\n"
    "\n"
    "__kernel void add_index_test(__global int *counter, __global int *counts) {\n"
    "    int tid = get_global_id(0);\n"
    "    \n"
    "    int counter_to_use = atom_add(counter, 1);\n"
    "    counts[counter_to_use] = tid;\n"
    "}";
// clang-format on

int test_atomic_add_index(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper counter, counters;
    size_t numGlobalThreads, numLocalThreads;
    int fail = 0, err;

    /* Check if atomics are supported. */
    if (!is_extension_available(deviceID, "cl_khr_global_int32_base_atomics"))
    {
        log_info("Base atomics not supported "
                 "(cl_khr_global_int32_base_atomics). Skipping test.\n");
        return 0;
    }

    //===== add_index test
    // The index test replicates what particles does.
    // It uses one memory location to keep track of the current index and then
    // each thread does an atomic add to it to get its new location. The threads
    // then write to their assigned location. At the end we check to make sure
    // that each thread's ID shows up exactly once in the output.

    numGlobalThreads = 2048;

    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    &atomic_index_source, "add_index_test"))
        return -1;

    if (get_max_common_work_group_size(context, kernel, numGlobalThreads,
                                       &numLocalThreads))
        return -1;

    log_info("Execute global_threads:%d local_threads:%d\n",
             (int)numGlobalThreads, (int)numLocalThreads);

    // Create the counter that will keep track of where each thread writes.
    counter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * 1,
                             NULL, NULL);
    // Create the counters that will hold the results of each thread writing
    // its ID into a (hopefully) unique location.
    counters = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(cl_int) * numGlobalThreads, NULL, NULL);

    // Reset all those locations to -1 to indciate they have not been used.
    cl_int *values = (cl_int *)malloc(sizeof(cl_int) * numGlobalThreads);
    if (values == NULL)
    {
        log_error(
            "add_index_test FAILED to allocate memory for initial values.\n");
        fail = 1;
    }
    else
    {
        memset(values, -1, numLocalThreads);
        unsigned int i = 0;
        for (i = 0; i < numGlobalThreads; i++) values[i] = -1;
        int init = 0;
        err = clEnqueueWriteBuffer(queue, counters, true, 0,
                                   numGlobalThreads * sizeof(cl_int), values, 0,
                                   NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, counter, true, 0, 1 * sizeof(cl_int),
                                    &init, 0, NULL, NULL);
        if (err)
        {
            log_error(
                "add_index_test FAILED to write initial values to arrays: %d\n",
                err);
            fail = 1;
        }
        else
        {
            err = clSetKernelArg(kernel, 0, sizeof(counter), &counter);
            err |= clSetKernelArg(kernel, 1, sizeof(counters), &counters);
            if (err)
            {
                log_error("add_index_test FAILED to set kernel arguments: %d\n",
                          err);
                fail = 1;
            }
            else
            {
                err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                             &numGlobalThreads,
                                             &numLocalThreads, 0, NULL, NULL);
                if (err)
                {
                    log_error("add_index_test FAILED to execute kernel: %d\n",
                              err);
                    fail = 1;
                }
                else
                {
                    err = clEnqueueReadBuffer(queue, counters, true, 0,
                                              sizeof(cl_int) * numGlobalThreads,
                                              values, 0, NULL, NULL);
                    if (err)
                    {
                        log_error(
                            "add_index_test FAILED to read back results: %d\n",
                            err);
                        fail = 1;
                    }
                    else
                    {
                        unsigned int looking_for, index;
                        for (looking_for = 0; looking_for < numGlobalThreads;
                             looking_for++)
                        {
                            int instances_found = 0;
                            for (index = 0; index < numGlobalThreads; index++)
                            {
                                if (values[index] == (int)looking_for)
                                    instances_found++;
                            }
                            if (instances_found != 1)
                            {
                                log_error(
                                    "add_index_test FAILED: wrong number of "
                                    "instances (%d!=1) for counter %d.\n",
                                    instances_found, looking_for);
                                fail = 1;
                            }
                        }
                    }
                }
            }
        }
        if (!fail)
        {
            log_info(
                "add_index_test passed. Each thread used exactly one index.\n");
        }
        free(values);
    }
    return fail;
}

// clang-format off
const char *add_index_bin_kernel[] = {
    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
    "// This test assigns a bunch of values to bins and then tries to put them in the bins in parallel\n"
    "// using an atomic add to keep track of the current location to write into in each bin.\n"
    "// This is the same as the memory update for the particles demo.\n"
    "\n"
    "__kernel void add_index_bin_test(__global int *bin_counters, __global int *bins, __global int *bin_assignments, int max_counts_per_bin) {\n"
    "    int tid = get_global_id(0);\n"
    "\n"
    "    int location = bin_assignments[tid];\n"
    "    int counter = atom_add(&bin_counters[location], 1);\n"
    "    bins[location*max_counts_per_bin + counter] = tid;\n"
    "}" };
// clang-format on

// This test assigns a bunch of values to bins and then tries to put them in the
// bins in parallel using an atomic add to keep track of the current location to
// write into in each bin. This is the same as the memory update for the
// particles demo.
int add_index_bin_test(size_t *global_threads, cl_command_queue queue,
                       cl_context context, MTdata d)
{
    int number_of_items = (int)global_threads[0];
    size_t local_threads[1];
    int divisor = 12;
    int number_of_bins = number_of_items / divisor;
    int max_counts_per_bin = divisor * 2;

    int fail = 0;
    int err;

    clProgramWrapper program;
    clKernelWrapper kernel;

    //  log_info("add_index_bin_test: %d items, into %d bins, with a max of %d
    //  items per bin (bins is %d long).\n",
    //           number_of_items, number_of_bins, max_counts_per_bin,
    //           number_of_bins*max_counts_per_bin);

    //===== add_index_bin test
    // The index test replicates what particles does.
    err =
        create_single_kernel_helper(context, &program, &kernel, 1,
                                    add_index_bin_kernel, "add_index_bin_test");
    test_error(err, "Unable to create testing kernel");

    if (get_max_common_work_group_size(context, kernel, global_threads[0],
                                       &local_threads[0]))
        return -1;

    log_info("Execute global_threads:%d local_threads:%d\n",
             (int)global_threads[0], (int)local_threads[0]);

    // Allocate our storage
    cl_mem bin_counters =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_int) * number_of_bins, NULL, NULL);
    cl_mem bins = clCreateBuffer(
        context, CL_MEM_READ_WRITE,
        sizeof(cl_int) * number_of_bins * max_counts_per_bin, NULL, NULL);
    cl_mem bin_assignments =
        clCreateBuffer(context, CL_MEM_READ_ONLY,
                       sizeof(cl_int) * number_of_items, NULL, NULL);

    if (bin_counters == NULL)
    {
        log_error("add_index_bin_test FAILED to allocate bin_counters.\n");
        return -1;
    }
    if (bins == NULL)
    {
        log_error("add_index_bin_test FAILED to allocate bins.\n");
        return -1;
    }
    if (bin_assignments == NULL)
    {
        log_error("add_index_bin_test FAILED to allocate bin_assignments.\n");
        return -1;
    }

    // Initialize our storage
    cl_int *l_bin_counts = (cl_int *)malloc(sizeof(cl_int) * number_of_bins);
    if (!l_bin_counts)
    {
        log_error("add_index_bin_test FAILED to allocate initial values for "
                  "bin_counters.\n");
        return -1;
    }
    int i;
    for (i = 0; i < number_of_bins; i++) l_bin_counts[i] = 0;
    err = clEnqueueWriteBuffer(queue, bin_counters, true, 0,
                               sizeof(cl_int) * number_of_bins, l_bin_counts, 0,
                               NULL, NULL);
    if (err)
    {
        log_error("add_index_bin_test FAILED to set initial values for "
                  "bin_counters: %d\n",
                  err);
        return -1;
    }

    cl_int *values =
        (cl_int *)malloc(sizeof(cl_int) * number_of_bins * max_counts_per_bin);
    if (!values)
    {
        log_error(
            "add_index_bin_test FAILED to allocate initial values for bins.\n");
        return -1;
    }
    for (i = 0; i < number_of_bins * max_counts_per_bin; i++) values[i] = -1;
    err = clEnqueueWriteBuffer(queue, bins, true, 0,
                               sizeof(cl_int) * number_of_bins
                                   * max_counts_per_bin,
                               values, 0, NULL, NULL);
    if (err)
    {
        log_error(
            "add_index_bin_test FAILED to set initial values for bins: %d\n",
            err);
        return -1;
    }
    free(values);

    cl_int *l_bin_assignments =
        (cl_int *)malloc(sizeof(cl_int) * number_of_items);
    if (!l_bin_assignments)
    {
        log_error("add_index_bin_test FAILED to allocate initial values for "
                  "l_bin_assignments.\n");
        return -1;
    }
    for (i = 0; i < number_of_items; i++)
    {
        int bin = random_in_range(0, number_of_bins - 1, d);
        while (l_bin_counts[bin] >= max_counts_per_bin)
        {
            bin = random_in_range(0, number_of_bins - 1, d);
        }
        if (bin >= number_of_bins)
            log_error("add_index_bin_test internal error generating bin "
                      "assignments: bin %d >= number_of_bins %d.\n",
                      bin, number_of_bins);
        if (l_bin_counts[bin] + 1 > max_counts_per_bin)
            log_error(
                "add_index_bin_test internal error generating bin assignments: "
                "bin %d has more entries (%d) than max_counts_per_bin (%d).\n",
                bin, l_bin_counts[bin], max_counts_per_bin);
        l_bin_counts[bin]++;
        l_bin_assignments[i] = bin;
        //     log_info("item %d assigned to bin %d (%d items)\n", i, bin,
        //     l_bin_counts[bin]);
    }
    err = clEnqueueWriteBuffer(queue, bin_assignments, true, 0,
                               sizeof(cl_int) * number_of_items,
                               l_bin_assignments, 0, NULL, NULL);
    if (err)
    {
        log_error("add_index_bin_test FAILED to set initial values for "
                  "bin_assignments: %d\n",
                  err);
        return -1;
    }
    // Setup the kernel
    err = clSetKernelArg(kernel, 0, sizeof(bin_counters), &bin_counters);
    err |= clSetKernelArg(kernel, 1, sizeof(bins), &bins);
    err |= clSetKernelArg(kernel, 2, sizeof(bin_assignments), &bin_assignments);
    err |= clSetKernelArg(kernel, 3, sizeof(max_counts_per_bin),
                          &max_counts_per_bin);
    if (err)
    {
        log_error("add_index_bin_test FAILED to set kernel arguments: %d\n",
                  err);
        fail = 1;
        return -1;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_threads,
                                 local_threads, 0, NULL, NULL);
    if (err)
    {
        log_error("add_index_bin_test FAILED to execute kernel: %d\n", err);
        fail = 1;
    }

    cl_int *final_bin_assignments =
        (cl_int *)malloc(sizeof(cl_int) * number_of_bins * max_counts_per_bin);
    if (!final_bin_assignments)
    {
        log_error("add_index_bin_test FAILED to allocate initial values for "
                  "final_bin_assignments.\n");
        return -1;
    }
    err = clEnqueueReadBuffer(queue, bins, true, 0,
                              sizeof(cl_int) * number_of_bins
                                  * max_counts_per_bin,
                              final_bin_assignments, 0, NULL, NULL);
    if (err)
    {
        log_error("add_index_bin_test FAILED to read back bins: %d\n", err);
        fail = 1;
    }

    cl_int *final_bin_counts =
        (cl_int *)malloc(sizeof(cl_int) * number_of_bins);
    if (!final_bin_counts)
    {
        log_error("add_index_bin_test FAILED to allocate initial values for "
                  "final_bin_counts.\n");
        return -1;
    }
    err = clEnqueueReadBuffer(queue, bin_counters, true, 0,
                              sizeof(cl_int) * number_of_bins, final_bin_counts,
                              0, NULL, NULL);
    if (err)
    {
        log_error("add_index_bin_test FAILED to read back bin_counters: %d\n",
                  err);
        fail = 1;
    }

    // Verification.
    int errors = 0;
    int current_bin;
    int search;
    //  Print out all the contents of the bins.
    //  for (current_bin=0; current_bin<number_of_bins; current_bin++)
    //        for (search=0; search<max_counts_per_bin; search++)
    //      log_info("[bin %d, entry %d] = %d\n", current_bin, search,
    //      final_bin_assignments[current_bin*max_counts_per_bin+search]);

    // First verify that there are the correct number in each bin.
    for (current_bin = 0; current_bin < number_of_bins; current_bin++)
    {
        int expected_number = l_bin_counts[current_bin];
        int actual_number = final_bin_counts[current_bin];
        if (expected_number != actual_number)
        {
            log_error("add_index_bin_test FAILED: bin %d reported %d entries "
                      "when %d were expected.\n",
                      current_bin, actual_number, expected_number);
            errors++;
        }
        for (search = 0; search < expected_number; search++)
        {
            if (final_bin_assignments[current_bin * max_counts_per_bin + search]
                == -1)
            {
                log_error("add_index_bin_test FAILED: bin %d had no entry at "
                          "position %d when it should have had %d entries.\n",
                          current_bin, search, expected_number);
                errors++;
            }
        }
        for (search = expected_number; search < max_counts_per_bin; search++)
        {
            if (final_bin_assignments[current_bin * max_counts_per_bin + search]
                != -1)
            {
                log_error(
                    "add_index_bin_test FAILED: bin %d had an extra entry at "
                    "position %d when it should have had only %d entries.\n",
                    current_bin, search, expected_number);
                errors++;
            }
        }
    }
    // Now verify that the correct ones are in each bin
    int index;
    for (index = 0; index < number_of_items; index++)
    {
        int expected_bin = l_bin_assignments[index];
        int found_it = 0;
        for (search = 0; search < l_bin_counts[expected_bin]; search++)
        {
            if (final_bin_assignments[expected_bin * max_counts_per_bin
                                      + search]
                == index)
            {
                found_it = 1;
            }
        }
        if (found_it == 0)
        {
            log_error(
                "add_index_bin_test FAILED: did not find item %d in bin %d.\n",
                index, expected_bin);
            errors++;
        }
    }
    free(l_bin_counts);
    free(l_bin_assignments);
    free(final_bin_assignments);
    free(final_bin_counts);
    clReleaseMemObject(bin_counters);
    clReleaseMemObject(bins);
    clReleaseMemObject(bin_assignments);
    if (errors == 0)
    {
        log_info("add_index_bin_test passed. Each item was put in the correct "
                 "bin in parallel.\n");
        return 0;
    }
    else
    {
        log_error("add_index_bin_test FAILED: %d errors.\n", errors);
        return -1;
    }
}

int test_atomic_add_index_bin(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    //===== add_index_bin test
    size_t numGlobalThreads = 2048;
    int iteration = 0;
    int err, failed = 0;
    MTdata d = init_genrand(gRandomSeed);

    /* Check if atomics are supported. */
    if (!is_extension_available(deviceID, "cl_khr_global_int32_base_atomics"))
    {
        log_info("Base atomics not supported "
                 "(cl_khr_global_int32_base_atomics). Skipping test.\n");
        free_mtdata(d);
        return 0;
    }

    for (iteration = 0; iteration < 10; iteration++)
    {
        log_info("add_index_bin_test with %d elements:\n",
                 (int)numGlobalThreads);
        err = add_index_bin_test(&numGlobalThreads, queue, context, d);
        if (err)
        {
            failed++;
            break;
        }
        numGlobalThreads *= 2;
    }
    free_mtdata(d);
    return failed;
}

//
// Copyright (c) 2026 The Khronos Group Inc.
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

// Verify that clEnqueueNDRangeKernel returns CL_INVALID_WORK_ITEM_SIZE when
// local_work_size is greater than CL_DEVICE_MAX_WORK_GROUP_SIZES in any
// dimension. When the total number of work-items in the work-group is also
// greater than CL_KERNEL_WORK_GROUP_SIZE, CL_INVALID_WORK_GROUP_SIZE is
// accepted as well.
REGISTER_TEST(negative_enqueue_work_item_size_exceeded)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    int maxThreadDim = 0;

    const char* kernel_test = "__kernel void test(){}\n";

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_test, "test");
    test_error(error, "create_single_kernel_helper failed");

    size_t maxWorkGroupSize = 0;
    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(maxWorkGroupSize),
                                     &maxWorkGroupSize, NULL);
    test_error(error, "Unable to get max work group size from kernel");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(maxThreadDim), &maxThreadDim, NULL);
    test_error(error, "Unable to get max work item dimensions from device");

    std::vector<size_t> work_group_max_sizes(maxThreadDim, 0);
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZES,
                            sizeof(size_t) * work_group_max_sizes.size(),
                            work_group_max_sizes.data(), NULL);
    test_error(error, "Unable to get max work group sizes from device");

    // A dimension limited to SIZE_MAX cannot be exceeded.
    for (int d = 0; d < maxThreadDim; d++)
    {
        if (work_group_max_sizes[d] == SIZE_MAX)
        {
            log_info("Device reports no work-group size limit in dimension %d, "
                     "skipping test\n",
                     d);
            return TEST_SKIPPED_ITSELF;
        }
    }

    std::vector<size_t> local_work_size(maxThreadDim, 0);
    std::vector<size_t> global_work_size(maxThreadDim, 1);

    auto set_local_WG_sizes = [&](int dims, int dim) {
        size_t total = 1;
        for (int i = 0; i < dims; i++)
        {
            local_work_size[i] = i == dim ? work_group_max_sizes[i] + 1 : 1;
            total *= local_work_size[i];
        }
        global_work_size.assign(local_work_size.begin(),
                                local_work_size.begin() + dims);
        return total;
    };

    // Exceed exactly one dimension j for each work_dim i; the others are set
    // to 1.
    for (int i = 1; i <= maxThreadDim; i++)
    {
        for (int j = 0; j < i; j++)
        {
            size_t total = set_local_WG_sizes(i, j);
            error = clEnqueueNDRangeKernel(
                queue, kernel, i, NULL, global_work_size.data(),
                local_work_size.data(), 0, NULL, NULL);

            bool groupSizeExceeded = total > maxWorkGroupSize;
            if (!((groupSizeExceeded && error == CL_INVALID_WORK_GROUP_SIZE)
                  || error == CL_INVALID_WORK_ITEM_SIZE))
            {
                std::string sizes;
                std::string maxSizes;
                for (int d = 0; d < i; d++)
                {
                    sizes +=
                        (d ? ", " : "") + std::to_string(local_work_size[d]);
                    maxSizes += (d ? ", " : "")
                        + std::to_string(work_group_max_sizes[d]);
                }
                log_error("clEnqueueNDRangeKernel returned %s instead of %s "
                          "for work_dim %d with local_work_size {%s} "
                          "(CL_DEVICE_MAX_WORK_GROUP_SIZES {%s}, "
                          "CL_KERNEL_WORK_GROUP_SIZE %s)\n",
                          IGetErrorString(error),
                          groupSizeExceeded ? "CL_INVALID_WORK_ITEM_SIZE or "
                                              "CL_INVALID_WORK_GROUP_SIZE"
                                            : "CL_INVALID_WORK_ITEM_SIZE",
                          i, sizes.c_str(), maxSizes.c_str(),
                          std::to_string(maxWorkGroupSize).c_str());
                return TEST_FAIL;
            }
        }
    }
    return TEST_PASS;
}

// Verify that clEnqueueNDRangeKernel returns CL_INVALID_WORK_GROUP_SIZE when
// the total number of work-items in the work-group is greater than
// CL_KERNEL_WORK_GROUP_SIZE, while each local_work_size stays within
// CL_DEVICE_MAX_WORK_GROUP_SIZES.
REGISTER_TEST(negative_enqueue_work_group_size_exceeded)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    int maxThreadDim = 0;

    const char* kernel_test = "__kernel void test(){}\n";

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_test, "test");
    test_error(error, "create_single_kernel_helper failed");

    size_t maxWorkGroupSize = 0;
    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(maxWorkGroupSize),
                                     &maxWorkGroupSize, NULL);
    test_error(error, "Unable to get max work group size from kernel");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(maxThreadDim), &maxThreadDim, NULL);
    test_error(error, "Unable to get max work item dimensions from device");

    std::vector<size_t> work_group_max_sizes(maxThreadDim, 0);
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZES,
                            sizeof(size_t) * work_group_max_sizes.size(),
                            work_group_max_sizes.data(), NULL);
    test_error(error, "Unable to get max work group sizes from device");

    // A dimension limited to SIZE_MAX would overflow the computed number of
    // work-items in the work-group.
    for (int d = 0; d < maxThreadDim; d++)
    {
        if (work_group_max_sizes[d] == SIZE_MAX)
        {
            log_info("Device reports no work-group size limit in dimension %d, "
                     "skipping test\n",
                     d);
            return TEST_SKIPPED_ITSELF;
        }
    }

    std::vector<size_t> local_work_size(maxThreadDim, 0);
    std::vector<size_t> global_work_size(maxThreadDim, 1);

    // Fill nd-ranges with their max sizes until the number of work-items in the
    // work-group exceeds CL_KERNEL_WORK_GROUP_SIZE
    auto set_local_WG_sizes = [&](int dims) {
        size_t total = 1;
        for (int i = 0; i < dims; i++)
        {
            local_work_size[i] =
                total > maxWorkGroupSize ? 1 : work_group_max_sizes[i];
            total *= local_work_size[i];
        }
        global_work_size.assign(local_work_size.begin(),
                                local_work_size.begin() + dims);
        return total > maxWorkGroupSize ? total : 0;
    };

    bool testSkipped = true;
    for (int i = 1; i <= maxThreadDim; i++)
    {
        size_t total = set_local_WG_sizes(i);

        // workgroup size can't exceed CL_KERNEL_WORK_GROUP_SIZE for
        // this work_dim without exceeding CL_DEVICE_MAX_WORK_GROUP_SIZES.
        if (!total) continue;
        testSkipped = false;

        error = clEnqueueNDRangeKernel(queue, kernel, i, NULL,
                                       global_work_size.data(),
                                       local_work_size.data(), 0, NULL, NULL);

        if (error != CL_INVALID_WORK_GROUP_SIZE)
        {
            std::string sizes;
            std::string maxSizes;
            for (int d = 0; d < i; d++)
            {
                sizes += (d ? ", " : "") + std::to_string(local_work_size[d]);
                maxSizes +=
                    (d ? ", " : "") + std::to_string(work_group_max_sizes[d]);
            }
            log_error("clEnqueueNDRangeKernel returned %s instead of "
                      "CL_INVALID_WORK_GROUP_SIZE for work_dim %d with "
                      "local_work_size {%s}, %s work-items in total "
                      "(CL_DEVICE_MAX_WORK_GROUP_SIZES {%s}, "
                      "CL_KERNEL_WORK_GROUP_SIZE %s)\n",
                      IGetErrorString(error), i, sizes.c_str(),
                      std::to_string(total).c_str(), maxSizes.c_str(),
                      std::to_string(maxWorkGroupSize).c_str());
            return TEST_FAIL;
        }
    }

    if (testSkipped)
    {
        log_info("No work_dim allows a work-group larger than "
                 "CL_KERNEL_WORK_GROUP_SIZE within "
                 "CL_DEVICE_MAX_WORK_GROUP_SIZES, skipping test\n");
        return TEST_SKIPPED_ITSELF;
    }

    return TEST_PASS;
}

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
#include "harness/parseParameters.h"

// Verify that clEnqueueNDRangeKernel returns:
// -CL_INVALID_WORK_ITEM_SIZE when local_work_size is greater than
// CL_DEVICE_MAX_WORK_GROUP_SIZES in any dimension. When the total number of
// work-items in the work-group is also greater than CL_KERNEL_WORK_GROUP_SIZE,
// CL_INVALID_WORK_GROUP_SIZE is accepted as well
// -CL_INVALID_WORK_GROUP_SIZE when the total number of work-items in the
// work-group is greater than CL_KERNEL_WORK_GROUP_SIZE, while each
// local_work_size stays within CL_DEVICE_MAX_WORK_GROUP_SIZES
REGISTER_TEST(negative_enqueue_work_group_size_exceeded)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_uint maxThreadDim = 0;

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

    std::vector<size_t> local_work_size(maxThreadDim, 0);

    bool testSkipped = true;
    bool testFailed = false;

    auto set_exceeded_dimension = [&](cl_uint dims, cl_uint dim) {
        size_t total = 1;
        for (cl_uint i = 0; i < dims; i++)
        {
            local_work_size[i] = i == dim ? work_group_max_sizes[i] + 1 : 1;
            total *= local_work_size[i];
        }
        return total;
    };

    // Exceed exactly one dimension j for each work_dim i; the others are set
    // to 1.
    for (cl_uint i = 1; i <= maxThreadDim; i++)
    {
        for (cl_uint j = 0; j < i; j++)
        {
            if (work_group_max_sizes[j] == SIZE_MAX) continue;
            testSkipped = false;

            size_t total = set_exceeded_dimension(i, j);
            error = clEnqueueNDRangeKernel(
                queue, kernel, i, NULL, local_work_size.data(),
                local_work_size.data(), 0, NULL, NULL);

            bool groupSizeExceeded = total > maxWorkGroupSize;
            if (!((groupSizeExceeded && error == CL_INVALID_WORK_GROUP_SIZE)
                  || error == CL_INVALID_WORK_ITEM_SIZE))
            {
                std::string sizes;
                std::string maxSizes;
                for (cl_uint d = 0; d < i; d++)
                {
                    sizes +=
                        (d ? ", " : "") + std::to_string(local_work_size[d]);
                    maxSizes += (d ? ", " : "")
                        + std::to_string(work_group_max_sizes[d]);
                }
                log_error("clEnqueueNDRangeKernel returned %s instead of %s "
                          "for work_dim %u with local_work_size {%s} "
                          "(CL_DEVICE_MAX_WORK_GROUP_SIZES {%s}, "
                          "CL_KERNEL_WORK_GROUP_SIZE %zu)\n",
                          IGetErrorString(error),
                          groupSizeExceeded ? "CL_INVALID_WORK_ITEM_SIZE or "
                                              "CL_INVALID_WORK_GROUP_SIZE"
                                            : "CL_INVALID_WORK_ITEM_SIZE",
                          i, sizes.c_str(), maxSizes.c_str(), maxWorkGroupSize);
                testFailed = true;
            }
        }
    }

    // Grow each dimension until the work-group exceeds
    // CL_KERNEL_WORK_GROUP_SIZE
    auto set_exceeded_total = [&](cl_uint dims) {
        size_t total = 1;
        for (cl_uint i = 0; i < dims; i++)
        {
            local_work_size[i] =
                std::min(work_group_max_sizes[i], maxWorkGroupSize / total + 1);
            total *= local_work_size[i];
        }
        return total > maxWorkGroupSize ? total : 0;
    };

    for (cl_uint i = 1; i <= maxThreadDim; i++)
    {
        size_t total = set_exceeded_total(i);

        // workgroup size can't exceed CL_KERNEL_WORK_GROUP_SIZE for
        // this work_dim without exceeding CL_DEVICE_MAX_WORK_GROUP_SIZES.
        if (!total) continue;
        testSkipped = false;

        error = clEnqueueNDRangeKernel(queue, kernel, i, NULL,
                                       local_work_size.data(),
                                       local_work_size.data(), 0, NULL, NULL);

        if (error != CL_INVALID_WORK_GROUP_SIZE)
        {
            std::string sizes;
            std::string maxSizes;
            for (cl_uint d = 0; d < i; d++)
            {
                sizes += (d ? ", " : "") + std::to_string(local_work_size[d]);
                maxSizes +=
                    (d ? ", " : "") + std::to_string(work_group_max_sizes[d]);
            }
            log_error("clEnqueueNDRangeKernel returned %s instead of "
                      "CL_INVALID_WORK_GROUP_SIZE for work_dim %u with "
                      "local_work_size {%s}, %zu work-items in total "
                      "(CL_DEVICE_MAX_WORK_GROUP_SIZES {%s}, "
                      "CL_KERNEL_WORK_GROUP_SIZE %zu)\n",
                      IGetErrorString(error), i, sizes.c_str(), total,
                      maxSizes.c_str(), maxWorkGroupSize);
            testFailed = true;
        }
    }

    if (testFailed)
    {
        return TEST_FAIL;
    }

    if (testSkipped)
    {
        log_info("Device limits do not allow exceeding "
                 "CL_DEVICE_MAX_WORK_GROUP_SIZES or CL_KERNEL_WORK_GROUP_SIZE, "
                 "skipping test\n");
        return TEST_SKIPPED_ITSELF;
    }

    return TEST_PASS;
}

// Verify that clEnqueueNDRangeKernel returns CL_INVALID_WORK_DIMENSION when
// work_dim is zero or greater than CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
REGISTER_TEST(negative_enqueue_invalid_work_dim)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_uint maxThreadDim = 0;

    const char* kernel_test = "__kernel void test(){}\n";

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_test, "test");
    test_error(error, "create_single_kernel_helper failed");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(maxThreadDim), &maxThreadDim, NULL);
    test_error(error, "Unable to get max work item dimensions from device");

    // Valid global and local sizes, with one more element than the device
    // supports, so work_dim is the only error condition.
    std::vector<size_t> work_size(maxThreadDim + 1, 1);

    error = clEnqueueNDRangeKernel(queue, kernel, maxThreadDim + 1, NULL,
                                   work_size.data(), work_size.data(), 0, NULL,
                                   NULL);
    test_failure_error_ret(error, CL_INVALID_WORK_DIMENSION,
                           "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS exceeded but "
                           "clEnqueueNDRangeKernel did not return "
                           "CL_INVALID_WORK_DIMENSION as expected",
                           TEST_FAIL);

    error = clEnqueueNDRangeKernel(queue, kernel, 0, NULL, work_size.data(),
                                   work_size.data(), 0, NULL, NULL);
    test_failure_error_ret(error, CL_INVALID_WORK_DIMENSION,
                           "work_dim is zero but clEnqueueNDRangeKernel did "
                           "not return CL_INVALID_WORK_DIMENSION as expected",
                           TEST_FAIL);

    return TEST_PASS;
}

// Verify that clEnqueueNDRangeKernel returns CL_INVALID_WORK_GROUP_SIZE
// when local_work_size is not NULL and the total number of work-items in
// the work-group is zero, see OpenCL-Docs#1542.
REGISTER_TEST(negative_enqueue_zero_work_group)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    int maxThreadDim = 0;

    const char* kernel_test = "__kernel void test(){}\n";

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_test, "test");
    test_error(error, "create_single_kernel_helper failed");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(maxThreadDim), &maxThreadDim, NULL);
    test_error(error, "Unable to get max work item dimensions from device");

    const int maxCheckedDim = 16;
    if (maxThreadDim > maxCheckedDim)
    {
        log_info("Device reports %d work-item dimensions, checking only the "
                 "first %d\n",
                 maxThreadDim, maxCheckedDim);
        maxThreadDim = maxCheckedDim;
    }

    // A global size of 1 in every dimension is divisible by a local size of
    // 1, so a zero local size is the only error condition in each enqueue.
    std::vector<size_t> local_work_size(maxThreadDim, 0);
    std::vector<size_t> global_work_size(maxThreadDim, 1);

    // Bit d of pattern selects local_work_size[d]: 0 for a clear bit and 1
    // for a set bit.
    auto set_bit_pattern = [&local_work_size](int dims, int pattern) {
        for (int i = 0; i < dims; i++)
            local_work_size[i] = size_t((pattern >> i) & 0x1);
    };

    for (int i = 1; i <= maxThreadDim; i++)
    {
        // Check every combination of zero and one local sizes for work_dim
        // i, skipping the last pattern (all ones), which is a valid size.
        int ncases = int(1) << i;
        for (int j = 0; j < ncases - 1; j++)
        {
            set_bit_pattern(i, j);
            error = clEnqueueNDRangeKernel(
                queue, kernel, i, NULL, global_work_size.data(),
                local_work_size.data(), 0, NULL, NULL);
            if (error != CL_INVALID_WORK_GROUP_SIZE)
            {
                std::string sizes;
                for (int d = 0; d < i; d++)
                {
                    sizes +=
                        (d ? ", " : "") + std::to_string(local_work_size[d]);
                }
                log_error("clEnqueueNDRangeKernel returned %s instead of "
                          "CL_INVALID_WORK_GROUP_SIZE for work_dim %d with "
                          "local_work_size {%s}\n",
                          IGetErrorString(error), i, sizes.c_str());
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

// Verify that clEnqueueNDRangeKernel returns CL_INVALID_WORK_GROUP_SIZE when
// the global size is not evenly divisible by the local size and the kernel
// comes from a program built from source without -cl-std. Such a program is
// compiled for OpenCL C 1.x, which requires uniform work-groups
REGISTER_TEST(negative_enqueue_non_uniform_work_group_size)
{
    cl_int error = CL_SUCCESS;

    // Offline compilation rules do not apply
    if (gCompilationMode != kOnline)
    {
        log_info("Test requires online compilation, skipping test\n");
        return TEST_SKIPPED_ITSELF;
    }

    // Build without options (including -cl-std), so the program is compiled for
    // the highest OpenCL C 1.x version supported by the device. Thus,
    // create_single_kernel_helper is not used - it adds -cl-std automatically
    const char* kernel_test = "__kernel void test(){}\n";

    clProgramWrapper program =
        clCreateProgramWithSource(context, 1, &kernel_test, NULL, &error);
    test_error(error, "clCreateProgramWithSource failed");

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    test_error(error, "clBuildProgram failed");

    clKernelWrapper kernel = clCreateKernel(program, "test", &error);
    test_error(error, "clCreateKernel failed");

    size_t maxWorkGroupSize = 0;
    error = get_max_allowed_1d_work_group_size_on_device(device, kernel,
                                                         &maxWorkGroupSize);
    test_error(error, "get_max_allowed_1d_work_group_size_on_device failed");

    // A local size of 2 with a global size of 3 gives a non-uniform
    // work-group, while the local size stays within all limits.
    const size_t localWorkSize = 2;
    const size_t globalWorkSize = 3;

    if (maxWorkGroupSize < localWorkSize)
    {
        log_info("Device does not allow a work-group size of %zu, skipping "
                 "test\n",
                 localWorkSize);
        return TEST_SKIPPED_ITSELF;
    }

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize,
                                   &localWorkSize, 0, NULL, NULL);
    test_failure_error_ret(error, CL_INVALID_WORK_GROUP_SIZE,
                           "Program built without -cl-std requires uniform "
                           "work-groups but clEnqueueNDRangeKernel did not "
                           "return CL_INVALID_WORK_GROUP_SIZE as expected",
                           TEST_FAIL);

    return TEST_PASS;
}

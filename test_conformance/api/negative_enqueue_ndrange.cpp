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

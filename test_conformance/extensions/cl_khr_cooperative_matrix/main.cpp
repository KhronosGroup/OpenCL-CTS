// Copyright (c) 2024-2026 The Khronos Group Inc.
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

#include <cstring>

#include "harness/deviceInfo.h"
#include "harness/testHarness.h"

#include "cooperative_matrix.hpp"

namespace {

const char *helpString = R"(
cooperative_matrix specific options:
    --variant <str>
        Only run variant described by 'str'
    -l
        Run in link check only mode (only build kernels, skip execution)
)";

TestContext writableTestContext;

// Query device and set up global state that does not change between tests.
test_status InitCL(cl_device_id device)
{
    cl_uint addressBits;
    cl_uint err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
                                  sizeof(cl_uint), &addressBits, NULL);
    test_error_fail(err,
                    "clGetDeviceInfo for CL_DEVICE_ADDRESS_BITS failed.\n");
    std::ostringstream oss;
    oss << addressBits;
    writableTestContext.addrWidth = oss.str();

    cl_platform_id platform;
    err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                          &platform, nullptr);
    test_error_fail(err, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed\n");

    err = clGetDeviceInfo(
        device, CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR,
        sizeof(cl_uint), &writableTestContext.devicePointerAlignment, nullptr);
    test_error_fail(
        err,
        "clGetDeviceInfo for "
        "CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR failed\n");
    log_info("Pointer alignment is %d bytes.\n",
             writableTestContext.devicePointerAlignment);

    err = clGetDeviceInfo(
        device, CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR,
        sizeof(cl_uint), &writableTestContext.deviceStrideMultiple, nullptr);
    test_error_fail(
        err,
        "clGetDeviceInfo for CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR "
        "failed\n");
    log_info("Stride multiple is %d bytes.\n",
             writableTestContext.deviceStrideMultiple);

    clGetDeviceCooperativeMatrixInfoKHR_fn clGetDeviceCooperativeMatrixInfoKHR =
        reinterpret_cast<clGetDeviceCooperativeMatrixInfoKHR_fn>(
            clGetExtensionFunctionAddressForPlatform(
                platform, "clGetDeviceCooperativeMatrixInfoKHR"));
    if (clGetDeviceCooperativeMatrixInfoKHR == nullptr)
    {
        log_error("clGetExtensionFunctionAddressForPlatform failed with "
                  "clGetDeviceCooperativeMatrixInfoKHR\n");
        return TEST_FAIL;
    }

    // First find out how much to allocate.
    size_t size = 0;
    err = clGetDeviceCooperativeMatrixInfoKHR(
        device, CL_DEVICE_COOPERATIVE_MATRIX_DEFAULT_SUB_GROUP_VARIANTS_KHR, 0,
        nullptr, 0, nullptr, &size);
    test_error_fail(err,
                    "clGetDeviceCooperativeMatrixInfoKHR failed to get size "
                    "needed for supported "
                    "cooperative matrix variants (default subgroup size).");
    size_t numVariants =
        size / sizeof(cl_device_cooperative_matrix_variant_khr);

    // Then perform the real query.
    std::vector<cl_device_cooperative_matrix_variant_khr> &supported_variants =
        writableTestContext.variants;
    supported_variants.resize(numVariants);
    err = clGetDeviceCooperativeMatrixInfoKHR(
        device, CL_DEVICE_COOPERATIVE_MATRIX_DEFAULT_SUB_GROUP_VARIANTS_KHR, 0,
        nullptr, size, supported_variants.data(), nullptr);
    test_error_fail(
        err,
        "clGetDeviceCooperativeMatrixInfoKHR failed to get supported "
        "cooperative matrix variants (default subgroup size).");

    // Check for fp64 support.
    writableTestContext.supportFP64 =
        is_extension_available(device, "cl_khr_fp64");

    // Take all valid matrix types reported by the device query and store
    // in a set, to get a deduplicated set of types that must be tested.
    for (const auto &v : supported_variants)
    {
        writableTestContext.types.emplace(v.a_type, v.m_size, v.k_size,
                                          MatrixType::Use::A);
        writableTestContext.types.emplace(v.b_type, v.k_size, v.n_size,
                                          MatrixType::Use::B);
        writableTestContext.types.emplace(v.c_type, v.m_size, v.n_size,
                                          MatrixType::Use::Acc);
    }

    return TEST_PASS;
}

// Parse test specific arguments and remove those from the argument list
// before invoking the harness argument parser.
int parseTestArgs(int *argc, const char *argv[])
{
    for (int i = 0; i < *argc; ++i)
    {
        int argsToRemove = 0;
        if (strcmp(argv[i], "--variant") == 0)
        {
            if (i + 1 == *argc)
            {
                log_error("Missing value for '--variant' argument.\n");
                return TEST_FAIL;
            }
            else if (!writableTestContext.runSingleVariant.empty())
            {
                log_error("--variant can only be specified once.\n");
                return TEST_FAIL;
            }
            else
            {
                writableTestContext.runSingleVariant = std::string(argv[i + 1]);
                argsToRemove += 2;
            }
        }
        else if (strcmp(argv[i], "-l") == 0)
        {
            writableTestContext.linkCheckOnly = true;
            argsToRemove++;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            // Don't consume this argument so that the harness can print generic
            // help text too.
            log_info("%s\n", helpString);
        }

        if (argsToRemove > 0)
        {
            // Shift remaining arguments down, overwriting the arguments that
            // were just successfully parsed.
            for (int j = i; j < (*argc - argsToRemove); ++j)
            {
                argv[j] = argv[j + argsToRemove];
            }
            *argc -= argsToRemove;
            --i;
        }
    }
    return TEST_PASS;
}

} // anonymous namespace

// Export a const pointer to the test context so it cannot be modified
// during testing.
const TestContext *gTestContext = &writableTestContext;

int main(int argc, const char *argv[])
{
    if (parseTestArgs(&argc, argv) != TEST_PASS)
    {
        return TEST_FAIL;
    }

    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

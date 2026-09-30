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

#include <climits>
#include <cstring>
#include <limits>
#include <numeric>

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

    REQUIRE_EXTENSION("cl_khr_cooperative_matrix");

    err = clGetDeviceInfo(
        device, CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR,
        sizeof(cl_uint), &writableTestContext.devicePointerAlignment, nullptr);
    test_error_fail(
        err,
        "clGetDeviceInfo for "
        "CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR failed\n");
    log_info("Pointer alignment is %u bytes.\n",
             writableTestContext.devicePointerAlignment);

    err = clGetDeviceInfo(
        device, CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR,
        sizeof(cl_uint), &writableTestContext.deviceStrideMultiple, nullptr);
    test_error_fail(
        err,
        "clGetDeviceInfo for CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR "
        "failed\n");
    log_info("Stride multiple is %u bytes.\n",
             writableTestContext.deviceStrideMultiple);

    cl_uint memBaseAddrAlignmentBits;
    err = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                          sizeof(cl_uint), &memBaseAddrAlignmentBits, nullptr);
    test_error_fail(err,
                    "clGetDeviceInfo for CL_DEVICE_MEM_BASE_ADDR_ALIGN "
                    "failed\n");
    // Convert the bit alignment requirement to the minimum byte-aligned
    // origin. This is lcm(alignment, CHAR_BIT) / CHAR_BIT, simplified to
    // alignment / gcd(alignment, CHAR_BIT) to avoid overflowing size_t.
    writableTestContext.deviceMemBaseAddrAlignment = static_cast<uint32_t>(
        size_t{ memBaseAddrAlignmentBits }
        / std::gcd(size_t{ memBaseAddrAlignmentBits }, size_t{ CHAR_BIT }));
    log_info("Sub-buffer origin alignment is %u bytes.\n",
             writableTestContext.deviceMemBaseAddrAlignment);

    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                          sizeof(writableTestContext.deviceLocalMemSize),
                          &writableTestContext.deviceLocalMemSize, nullptr);
    test_error_fail(err,
                    "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed\n");
    log_info("Local memory size is %llu bytes.\n",
             static_cast<unsigned long long>(
                 writableTestContext.deviceLocalMemSize));

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
    if (size % sizeof(cl_device_cooperative_matrix_variant_khr) != 0)
    {
        log_error("clGetDeviceCooperativeMatrixInfoKHR returned an invalid "
                  "variant data size.\n");
        return TEST_FAIL;
    }
    const size_t numVariants =
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

    // Skip very large matrices, to avoid unnecessary complexity in the suite.
    const uint64_t maxTestBufferSize = std::numeric_limits<uint32_t>::max();
    const BufferElementType maxBufferElementType = IndexedBufferElementType<16>(
        CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_UINT64_KHR);
    const uint64_t maxBufferElementSize =
        bufferElementTypeSizeOf(maxBufferElementType);
    // Check the largest buffer descriptor the suite can create for a matrix.
    const auto fitsTestLimits =
        [maxTestBufferSize, maxBufferElementSize, maxBufferElementType](
            cl_device_cooperative_matrix_component_type_khr type, cl_uint rows,
            cl_uint cols) {
            const auto layoutFits =
                [maxTestBufferSize, maxBufferElementSize, maxBufferElementType,
                 type](uint32_t stride, uint32_t strideCount) {
                    const auto layout = calculateBufferLayout(
                        type, stride, strideCount, maxBufferElementType,
                        writableTestContext.deviceStrideMultiple);
                    if (!layout.has_value()
                        || layout->strideSize / maxBufferElementSize
                            > std::numeric_limits<uint32_t>::max())
                        return false;
                    return layout->totalSize <= maxTestBufferSize;
                };
            // Rows and columns are respectively the stride count and pointer
            // stride for row-major layout, and vice versa for column-major.
            if (layoutFits(cols, rows) && layoutFits(rows, cols)) return true;

            log_info("Skipping %ux%u %s cooperative matrix: exceeds the test "
                     "size limit.\n",
                     rows, cols, spirvScalarTypeName(type));
            return false;
        };

    // Store individually supported matrix types in a set, to deduplicate the
    // types tested. Retain ternary variants only when all of their matrices
    // fit the test limits.
    std::vector<cl_device_cooperative_matrix_variant_khr> filtered_variants;
    filtered_variants.reserve(supported_variants.size());
    for (const auto &v : supported_variants)
    {
        const bool aFits = fitsTestLimits(v.a_type, v.m_size, v.k_size);
        const bool bFits = fitsTestLimits(v.b_type, v.k_size, v.n_size);
        const bool cFits = fitsTestLimits(v.c_type, v.m_size, v.n_size);
        const bool resultFits =
            fitsTestLimits(v.result_type, v.m_size, v.n_size);

        if (aFits)
            writableTestContext.types.emplace(v.a_type, v.m_size, v.k_size,
                                              MatrixType::Use::A);
        if (bFits)
            writableTestContext.types.emplace(v.b_type, v.k_size, v.n_size,
                                              MatrixType::Use::B);
        if (cFits)
            writableTestContext.types.emplace(v.c_type, v.m_size, v.n_size,
                                              MatrixType::Use::Acc);
        if (aFits && bFits && cFits && resultFits)
        {
            filtered_variants.push_back(v);
        }
    }
    supported_variants.swap(filtered_variants);

    return TEST_PASS;
}

// Parse test-specific arguments and remove them from the argument list before
// invoking the harness argument parser.
test_status parseTestArgs(int &argc, const char *argv[],
                          std::vector<std::string> &removedArgs,
                          std::string &help)
{
    help = helpString;
    std::vector<const char *> keptArgs{ argv[0] };
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--variant") == 0)
        {
            if (i + 1 == argc)
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
                removedArgs.emplace_back(argv[i]);
                removedArgs.emplace_back(argv[++i]);
            }
        }
        else if (strcmp(argv[i], "-l") == 0)
        {
            writableTestContext.linkCheckOnly = true;
            removedArgs.emplace_back(argv[i]);
        }
        else
        {
            keptArgs.push_back(argv[i]);
        }
    }
    update_argc_argv_from_args_list(keptArgs, argc, argv);
    return TEST_PASS;
}

} // anonymous namespace

// Export a const pointer to the test context so it cannot be modified
// during testing.
const TestContext *gTestContext = &writableTestContext;

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL,
        parseTestArgs);
}

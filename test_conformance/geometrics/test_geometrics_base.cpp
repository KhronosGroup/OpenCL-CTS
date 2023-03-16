//
// Copyright (c) 2022 The Khronos Group Inc.
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
#include "test_geometrics_base.h"

#include "harness/testHarness.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include <float.h>
#include <limits>
#include <cmath>

#include <CL/cl_half.h>

//--------------------------------------------------------------------------/

cl_half_rounding_mode GeomTestBase::halfRoundingMode = CL_HALF_RTE;
char GeomTestBase::ftype[32] = { 0 };
char GeomTestBase::extension[128] = { 0 };
char GeomTestBase::load_store[128] = { 0 };

cl_ulong GeometricsFPTest::maxAllocSize = 0;
cl_ulong GeometricsFPTest::maxGlobalMemSize = 0;

//--------------------------------------------------------------------------

GeometricsFPTest::GeometricsFPTest(cl_device_id device, cl_context context,
                                   cl_command_queue queue)
    : context(context), device(device), queue(queue), floatRoundingMode(0),
      halfRoundingMode(0), floatHasInfNan(false), halfHasInfNan(false),
      halfFlushDenormsToZero(0), num_elements(0)
{}

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::SetUp(int elements)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            GeomTestBase::halfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            GeomTestBase::halfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }

        halfRoundingMode =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);

        cl_device_fp_config config = 0;
        cl_int error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                       sizeof(config), &config, NULL);
        test_error(error, "Unable to get device CL_DEVICE_HALF_FP_CONFIG");
        halfHasInfNan = (0 != (config & CL_FP_INF_NAN));

        halfFlushDenormsToZero = (0 == (config & CL_FP_DENORM));
        log_info("Supports half precision denormals: %s\n",
                 halfFlushDenormsToZero ? "NO" : "YES");
    }

    floatRoundingMode =
        get_default_rounding_mode(device, CL_DEVICE_SINGLE_FP_CONFIG);

    cl_device_fp_config config = 0;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                                   sizeof(config), &config, NULL);
    test_error(error, "Unable to get device CL_DEVICE_SINGLE_FP_CONFIG");
    floatHasInfNan = (0 != (config & CL_FP_INF_NAN));

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::Run()
{
    cl_int error = CL_SUCCESS;
    for (auto &&p : params)
    {
        std::snprintf(GeomTestBase::ftype, sizeof(GeomTestBase::ftype), "%s",
                      get_explicit_type_name(p->dataType));

        if (p->dataType == kDouble)
            strcpy(GeomTestBase::extension,
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        else if (p->dataType == kHalf)
            strcpy(GeomTestBase::extension,
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
        else
            GeomTestBase::extension[0] = '\0';

        error = RunSingleTest(p.get());
        test_error(error, "GeometricsFPTest::Run: test_relational failed");
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::VerifyTestSize(size_t &test_size,
                                        const size_t &max_alloc,
                                        const size_t &total_buf_size)
{
    if (maxAllocSize == 0 || maxGlobalMemSize == 0)
    {
        cl_int error = CL_SUCCESS;

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                sizeof(maxAllocSize), &maxAllocSize, NULL);
        error |=
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
        test_error(error, "Unable to get device config");

        log_info("Device supports:\nCL_DEVICE_MAX_MEM_ALLOC_SIZE: "
                 "%gMB\nCL_DEVICE_GLOBAL_MEM_SIZE: %gMB\n",
                 maxGlobalMemSize / (1024.0 * 1024.0),
                 maxAllocSize / (1024.0 * 1024.0));

        if (maxGlobalMemSize > (cl_ulong)SIZE_MAX)
        {
            maxGlobalMemSize = (cl_ulong)SIZE_MAX;
        }
    }

    /* Try to allocate a bit less than the limits */
    unsigned int adjustment = 32 * 1024 * 1024;

    while ((max_alloc > (maxAllocSize - adjustment))
           || (total_buf_size > (maxGlobalMemSize - adjustment)))
    {
        test_size /= 2;
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------/

std::string GeometricsFPTest::concat_kernel(const char *sstr[], int num)
{
    std::string res;
    for (int i = 0; i < num; i++) res += std::string(sstr[i]);
    return res;
}

//--------------------------------------------------------------------------/

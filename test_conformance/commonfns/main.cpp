//
// Copyright (c) 2023 The Khronos Group Inc.
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

#include <stdio.h>
#include <string.h>
#include "test_base.h"
#include "harness/kernelHelpers.h"

std::map<size_t, std::string> BaseFunctionTest::type2name;
cl_half_rounding_mode BaseFunctionTest::halfRoundingMode = CL_HALF_RTE;

int g_arrVecSizes[kVectorSizeCount + kStrangeVectorSizeCount];
int g_arrStrangeVectorSizes[kStrangeVectorSizeCount] = {3};

static void initVecSizes() {
    int i;
    for(i = 0; i < kVectorSizeCount; ++i) {
        g_arrVecSizes[i] = (1<<i);
    }
    for(; i < kVectorSizeCount + kStrangeVectorSizeCount; ++i) {
        g_arrVecSizes[i] = g_arrStrangeVectorSizes[i-kVectorSizeCount];
    }
}

test_status InitCL(cl_device_id device)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            BaseFunctionTest::halfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            BaseFunctionTest::halfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    initVecSizes();

    BaseFunctionTest::type2name[sizeof(half)] = "half";
    BaseFunctionTest::type2name[sizeof(float)] = "float";
    BaseFunctionTest::type2name[sizeof(double)] = "double";

    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

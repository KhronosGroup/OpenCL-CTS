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
#include "harness/compat.h"
#include "harness/deviceInfo.h"
#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"

#include <CL/cl_half.h>

cl_half_rounding_mode halfRoundingMode = CL_HALF_RTE;

test_status InitCL(cl_device_id device)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            halfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            halfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode\n");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

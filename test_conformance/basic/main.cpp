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
#include "harness/parseParameters.h"

#include <CL/cl_half.h>
#include "testBase.h"

cl_half_rounding_mode halfRoundingMode = CL_HALF_RTE;
int gUseDataType = 0;
unsigned int gUseVectorSize = 0;
unsigned int gUseStride = 0;
std::vector<ExplicitType> gVecType = {};

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

static test_status parseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help_description)
{
    std::vector<const char *> kept_args;

    help_description =
        R"(        Additional options for the following test cases:
          async_copy_global_to_local
          async_copy_local_to_global
          prefetch
          async_strided_copy_global_to_local
          async_strided_copy_local_to_global

        [v1|v2|v3|v4|v8|v16] - specify vector size used in test.
        [char|uchar|short|ushort|int|uint|long|ulong|float|half|double]
                             - specify data type used in test.

        In addition, for async_strided_copy_global_to_local and
        async_strided_copy_local_to_global:
        [s1|s3|s4|s5]        - specify stride size used in test.

        Examples of usage:
          test_basic async_copy_local_to_global float v4
          test_basic prefetch int uint v2
          test_basic async_strided_copy_global_to_local uchar s4 v1
          test_basic async_strided_copy_local_to_global char s3 v4

        Notes:
          Choose exactly one vector size: v1|v2|v3|v4|v8|v16.
          You can include multiple data types (e.g., char int).
          For strided variants, include exactly one stride: s1|s3|s4|s5.
)";

    kept_args.push_back(argv[0]);
    // Parse arguments, consuming the data type / vector size / stride options
    // and leaving the rest (test names, device type, etc.) for the harness.
    for (int i = 1; i < argc; i++)
    {
        removed_args.push_back(argv[i]);
        if (strcmp(argv[i], "char") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kChar);
        }
        else if (strcmp(argv[i], "uchar") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUChar);
        }
        else if (strcmp(argv[i], "short") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kShort);
        }
        else if (strcmp(argv[i], "ushort") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUShort);
        }
        else if (strcmp(argv[i], "int") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kInt);
        }
        else if (strcmp(argv[i], "uint") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUInt);
        }
        else if (strcmp(argv[i], "long") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kLong);
        }
        else if (strcmp(argv[i], "ulong") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kULong);
        }
        else if (strcmp(argv[i], "float") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kFloat);
        }
        else if (strcmp(argv[i], "half") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kHalf);
        }
        else if (strcmp(argv[i], "double") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kDouble);
        }
        else if (strcmp(argv[i], "v1") == 0)
        {
            gUseVectorSize = 1;
        }
        else if (strcmp(argv[i], "v2") == 0)
        {
            gUseVectorSize = 2;
        }
        else if (strcmp(argv[i], "v3") == 0)
        {
            gUseVectorSize = 3;
        }
        else if (strcmp(argv[i], "v4") == 0)
        {
            gUseVectorSize = 4;
        }
        else if (strcmp(argv[i], "v8") == 0)
        {
            gUseVectorSize = 8;
        }
        else if (strcmp(argv[i], "v16") == 0)
        {
            gUseVectorSize = 16;
        }
        else if (strcmp(argv[i], "s1") == 0)
        {
            gUseStride = 1;
        }
        else if (strcmp(argv[i], "s3") == 0)
        {
            gUseStride = 3;
        }
        else if (strcmp(argv[i], "s4") == 0)
        {
            gUseStride = 4;
        }
        else if (strcmp(argv[i], "s5") == 0)
        {
            gUseStride = 5;
        }
        else
        {
            removed_args.pop_back();
            kept_args.push_back(argv[i]);
        }
    }
    update_argc_argv_from_args_list(kept_args, argc, argv);
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL,
        parseArgs);
}

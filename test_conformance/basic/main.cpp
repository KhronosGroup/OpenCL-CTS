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

static void PrintUsage(void)
{
    vlog("additional options for test cases:\n");
    vlog("\t  async_copy_global_to_local\n\t  async_copy_local_to_global\n\t  "
         "prefetch\n\t  async_strided_copy_global_to_local\n\t  "
         "async_strided_copy_local_to_global\n\n");
    vlog("\t   [v1|v2|v3|v4|v8|v16] - specify vector size used in test.\n");
    vlog("\t   [char|uchar|short|ushort|int|uint|long|ulong|float|half|double] "
         "- specify data type used in test.\n");
    vlog("\n\t   In addition for async_strided_copy_global_to_local and "
         "async_strided_copy_local_to_global:\n");
    vlog("\t   [s1|s3|s4|s5] - specify stride size used in test.\n");
    vlog("\n\t   Examples of usage:\n");
    vlog("\t     test_basic async_copy_local_to_global float v4\n");
    vlog("\t     test_basic prefetch int uint v2\n");
    vlog("\t     test_basic async_strided_copy_global_to_local uchar s4 v1\n");
    vlog("\t     test_basic async_strided_copy_local_to_global char s3 v4\n\n");
    vlog("\t   Notes:\n");
    vlog("\t     Choose exactly one vector size: v1|v2|v3|v4|v8|v16.\n");
    vlog("\t     You can include multiple data types (e.g., char int).\n");
    vlog("\t     For strided variants, include exactly one stride: "
         "s1|s3|s4|s5.\n\n");
    vlog("\n");
}

int main(int argc, const char *argv[])
{
    argc = parseCustomParam(argc, argv);

    // Parse arguments
    int argsRemoveNum = 0;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            PrintUsage();
            return -1;
        }
        if (strcmp(argv[i], "char") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kChar);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "uchar") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUChar);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "short") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kShort);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "ushort") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUShort);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "int") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kInt);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "uint") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kUInt);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "long") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kLong);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "ulong") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kULong);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "float") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kFloat);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "half") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kHalf);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "double") == 0)
        {
            gUseDataType = 1;
            gVecType.push_back(kDouble);
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v1") == 0)
        {
            gUseVectorSize = 1;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v2") == 0)
        {
            gUseVectorSize = 2;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v3") == 0)
        {
            gUseVectorSize = 3;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v4") == 0)
        {
            gUseVectorSize = 4;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v8") == 0)
        {
            gUseVectorSize = 8;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "v16") == 0)
        {
            gUseVectorSize = 16;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "s1") == 0)
        {
            gUseStride = 1;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "s3") == 0)
        {
            gUseStride = 3;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "s4") == 0)
        {
            gUseStride = 4;
            argsRemoveNum += 1;
        }
        else if (strcmp(argv[i], "s5") == 0)
        {
            gUseStride = 5;
            argsRemoveNum += 1;
        }
    }
    // remove additionally parsed args from argv
    if (argsRemoveNum > 0)
    {
        for (int j = argc; j < argc - argsRemoveNum; j++)
        {
            argv[j] = argv[j + argsRemoveNum];
        }
        argc -= argsRemoveNum;
    }
    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}

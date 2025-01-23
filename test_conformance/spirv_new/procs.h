//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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

#pragma once

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"
#include "harness/compat.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"

#include <vector>

#define SPIRV_CHECK_ERROR(err, fmt, ...)                                       \
    do                                                                         \
    {                                                                          \
        if (err == CL_SUCCESS) break;                                          \
        log_error("%s(%d): Error %d\n" fmt "\n", __FILE__, __LINE__, err,      \
                  ##__VA_ARGS__);                                              \
        return -1;                                                             \
    } while (0)

struct spec_const
{
    spec_const(cl_int id = 0, size_t sizet = 0, const void *value = NULL)
        : spec_id(id), spec_size(sizet), spec_value(value){};
    cl_int spec_id;
    size_t spec_size;
    const void *spec_value;
};

int get_program_with_il(clProgramWrapper &prog, const cl_device_id deviceID,
                        const cl_context context, const char *prog_name,
                        spec_const spec_const_def = spec_const());
std::vector<unsigned char> readSPIRV(const char *file_name);

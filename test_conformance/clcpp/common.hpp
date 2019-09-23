//
// Copyright (c) 2017 The Khronos Group Inc.
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
#ifndef TEST_CONFORMANCE_CLCPP_COMMON_INC_HPP
#define TEST_CONFORMANCE_CLCPP_COMMON_INC_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>

// harness framework
#include "harness/compat.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"

// autotest
#include "autotest/autotest.hpp"

// utils_common
#include "utils_common/is_vector_type.hpp"
#include "utils_common/scalar_type.hpp"
#include "utils_common/make_vector_type.hpp"
#include "utils_common/type_name.hpp"
#include "utils_common/type_supported.hpp"
#include "utils_common/vector_size.hpp"
#include "utils_common/kernel_helpers.hpp"
#include "utils_common/errors.hpp"
#include "utils_common/string.hpp"

size_t get_uniform_global_size(size_t global_size, size_t local_size)
{
    return static_cast<size_t>(std::ceil(static_cast<double>(global_size) / local_size)) * local_size;
}

#endif // TEST_CONFORMANCE_CLCPP_COMMON_INC_HPP

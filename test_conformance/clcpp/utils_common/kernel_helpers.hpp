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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_KERNEL_HELPERS_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_KERNEL_HELPERS_HPP

#include "../common.hpp"

// Creates a OpenCL C++/C program out_program and kernel out_kernel.
int create_opencl_kernel(cl_context context,
                         cl_program *out_program,
                         cl_kernel *out_kernel,
                         const char *source,
                         const std::string& kernel_name,
                         const std::string& build_options = "",
                         const bool openclCXX = true)
{
    return create_single_kernel_helper(
        context, out_program, out_kernel, 1, &source,
        kernel_name.c_str(), build_options.c_str(), openclCXX
    );
}

int create_opencl_kernel(cl_context context,
                         cl_program *out_program,
                         cl_kernel *out_kernel,
                         const std::string& source,
                         const std::string& kernel_name,
                         const std::string& build_options = "",
                         const bool openclCXX = true)
{
    return create_opencl_kernel(
        context, out_program, out_kernel,
        source.c_str(), kernel_name, build_options, openclCXX
    );
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_KERNEL_HELPERS_HPP

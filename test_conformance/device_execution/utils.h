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
#ifndef _utils_h_
#define _utils_h_

#include "harness/testHarness.h"
#include "harness/mt19937.h"

#include <string>

#ifndef CL_VERSION_2_0
#define CL_VERSION_2_0
#endif

#define MAX_QUEUES    1000 // Max number of queues to test
#define MAX_GWS       256 // Global Work Size (must be multiple of 16)


#define NL "\n"
#define arr_size(a) (sizeof(a)/sizeof(a[0]))
#define check_error(errCode,msg,...) ((errCode != CL_SUCCESS) ? (log_error("ERROR: " msg "! (%s:%d)\n", ## __VA_ARGS__, __FILE__, __LINE__), 1) : 0)

#define KERNEL(name) { arr_size(name), name, #name }

extern std::string gKernelName;

typedef struct
{
    unsigned int num_lines;
    const char** lines;
    const char*  kernel_name;
} kernel_src;

typedef int (*fn_check)(cl_int*, cl_int, cl_int);

typedef struct
{
    kernel_src src;
    fn_check   check;
} kernel_src_check;

typedef struct
{
    size_t size;
    const void*  ptr;
} kernel_arg;

typedef struct
{
  kernel_src src;
  cl_int dim;
  cl_bool localSize;
  cl_bool offset;
} kernel_src_dim_check;

int run_single_kernel(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, void* results, size_t res_size);
int run_single_kernel_args(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, void* results, size_t res_size, cl_uint num_args, kernel_arg* args);
int run_n_kernel_args(cl_context context, cl_command_queue queue, const char** source, unsigned int num_lines, const char* kernel_name, size_t local, size_t global, void* results, size_t res_size, cl_uint num_args, kernel_arg* args);

#endif

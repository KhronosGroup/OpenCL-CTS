//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "testBase.h"
#include "harness/typeWrappers.h"

#include <vector>

struct test_query_pipe_properties_data
{
    std::vector<cl_pipe_properties> properties;
    std::string description;
};

static int create_pipe_and_check_array_properties(
    cl_context context, const test_query_pipe_properties_data& test_case)
{
    log_info("TC description: %s\n", test_case.description.c_str());

    cl_int error = CL_SUCCESS;

    clMemWrapper test_pipe;

    if (test_case.properties.size() > 0)
    {
        test_pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, 4, 4,
                                 test_case.properties.data(), &error);
        test_error(error, "clCreatePipe failed");
    }
    else
    {
        test_pipe =
            clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, 4, 4, NULL, &error);
        test_error(error, "clCreatePipe failed");
    }

    std::vector<cl_pipe_properties> check_properties;
    size_t set_size = 0;

    error = clGetPipeInfo(test_pipe, CL_PIPE_PROPERTIES, 0, NULL, &set_size);
    test_error(error,
               "clGetPipeInfo failed asking for "
               "CL_PIPE_PROPERTIES size.");

    if (set_size == 0 && test_case.properties.size() == 0)
    {
        return TEST_PASS;
    }
    if (set_size != test_case.properties.size() * sizeof(cl_pipe_properties))
    {
        log_error("ERROR: CL_PIPE_PROPERTIES size is %zu, expected %zu.\n",
                  set_size,
                  test_case.properties.size() * sizeof(cl_pipe_properties));
        return TEST_FAIL;
    }

    log_error("Unexpected test case size.  This test needs to be updated to "
              "compare pipe properties.\n");
    return TEST_FAIL;
}

REGISTER_TEST_VERSION(pipe_properties_queries, Version(3, 0))
{
    cl_int error = CL_SUCCESS;

    cl_bool pipeSupport = CL_FALSE;
    error = clGetDeviceInfo(device, CL_DEVICE_PIPE_SUPPORT, sizeof(pipeSupport),
                            &pipeSupport, NULL);
    test_error(error, "Unable to query CL_DEVICE_PIPE_SUPPORT");

    if (pipeSupport == CL_FALSE)
    {
        return TEST_SKIPPED_ITSELF;
    }

    int result = TEST_PASS;

    std::vector<test_query_pipe_properties_data> test_cases;
    test_cases.push_back({ {}, "NULL properties" });

    for (auto test_case : test_cases)
    {
        result |= create_pipe_and_check_array_properties(context, test_case);
    }

    return result;
}

//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "harness/os_helpers.h"
#include "testBase.h"

#if defined(_WIN32)
#include <io.h>
#define streamDup(fd1) _dup(fd1)
#define streamDup2(fd1, fd2) _dup2(fd1, fd2)
#else
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#define streamDup(fd1) dup(fd1)
#define streamDup2(fd1, fd2) dup2(fd1, fd2)
#endif

#include <fstream>
#include <vector>

// TODO: Unify with test_printf.
struct StreamGrabber
{
    StreamGrabber()
    {
        char* tmp = get_temp_filename();
        tempFileName = tmp;
        free(tmp);
    }
    ~StreamGrabber()
    {
        if (acquired)
        {
            release();
        }
    }

    int acquire(void)
    {
        if (acquired == false)
        {
            old_fd = streamDup(fileno(stdout));
            if (!freopen(tempFileName.c_str(), "w", stdout))
            {
                release();
                return -1;
            }
            acquired = true;
        }
        return 0;
    }

    int release(void)
    {
        if (acquired == true)
        {
            fflush(stdout);
            streamDup2(old_fd, fileno(stdout));
            close(old_fd);
            acquired = false;
        }
        return 0;
    }

    int get_results(std::string& results)
    {
        if (acquired == false)
        {
            std::ifstream is(tempFileName, std::ios::binary);
            if (is.good())
            {
                size_t filesize = 0;
                is.seekg(0, std::ios::end);
                filesize = (size_t)is.tellg();
                is.seekg(0, std::ios::beg);

                results.clear();
                results.resize(filesize);
                is.read(&results[0], filesize);

                return 0;
            }
        }
        return -1;
    }

    std::string tempFileName;
    int old_fd = 0;
    bool acquired = false;
};

template <typename T>
static int printf_operands_helper(cl_context context, cl_device_id device,
                                  cl_command_queue queue,
                                  const char* spirvFileName,
                                  const char* kernelName,
                                  const char* expectedResults, T value)
{
    StreamGrabber grabber;
    cl_int error;

    clProgramWrapper program;
    error = get_program_with_il(program, device, context, spirvFileName);
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel = clCreateKernel(program, kernelName, &error);
    test_error(error, "Unable to create SPIR-V kernel");

    error = clSetKernelArg(kernel, 0, sizeof(value), &value);
    test_error(error, "Unable to set kernel arguments");

    size_t global = 1;
    grabber.acquire();
    error |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                    NULL, NULL);
    error |= clFinish(queue);
    grabber.release();
    test_error(error, "unable to enqueue kernel");

    std::string results;
    grabber.get_results(results);

    if (results != std::string(expectedResults))
    {
        log_error("Results do not match.\n");
        log_error("Expected: \n---\n%s---\n", expectedResults);
        log_error("Got: \n---\n%s---\n", results.c_str());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST(extinst_printf_operands_scalar_int32)
{
    static const char* expected = R"(d = 1
i = 1
o = 1
u = 1
x = 1
X = 1
hd = 1
hi = 1
ho = 1
hu = 1
hx = 1
hX = 1
hhd = 1
hhi = 1
hho = 1
hhu = 1
hhx = 1
hhX = 1
)";

    return printf_operands_helper(context, device, queue,
                                  "printf_operands_scalar_int32",
                                  "printf_operands_scalar_int32", expected, 1);
}

REGISTER_TEST(extinst_printf_operands_scalar_fp32)
{
    static const char* expected = R"(a = 0x1.0p+1
A = 0X1.0P+1
e = 2.0e+00
E = 2.0E+00
f = 2.0
F = 2.0
g = 2
G = 2
)";

    return printf_operands_helper(
        context, device, queue, "printf_operands_scalar_fp32",
        "printf_operands_scalar_fp32", expected, 2.0f);
}

REGISTER_TEST(extinst_printf_operands_scalar_int64)
{
    static const char* expected = R"(ld = 4
li = 4
lo = 4
lu = 4
lx = 4
lX = 4
)";

    if (!gHasLong)
    {
        log_info("Device does not support 64-bit integers. Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return printf_operands_helper(context, device, queue,
                                  "printf_operands_scalar_int64",
                                  "printf_operands_scalar_int64", expected, 4L);
}

REGISTER_TEST(extinst_printf_operands_scalar_fp64)
{
    static const char* expected = R"(a = 0x1.0p+3
A = 0X1.0P+3
e = 8.0e+00
E = 8.0E+00
f = 8.0
F = 8.0
g = 8
G = 8
)";

    if (!is_extension_available(device, "cl_khr_fp64"))
    {
        log_info("Device does not support fp64. Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return printf_operands_helper(context, device, queue,
                                  "printf_operands_scalar_fp64",
                                  "printf_operands_scalar_fp64", expected, 8.0);
}

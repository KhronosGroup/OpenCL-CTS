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

#include <functional>
#include <string>
#include <vector>

#include "procs.h"

template <typename T> struct TestDef
{
    const char *name;
    const char *kernel_code;
    std::function<T(T, T, T)> ref;
};

template <typename T, unsigned N>
int test_intmath(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements, std::string typestr)
{
    TestDef<T> tests[] = {
        // Test addition
        {
            "test_add",
            R"(
  __kernel void test_add(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *srcC,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] + srcB[tid];
  };
)",
            [](T a, T b, T c) { return a + b; },
        },

        // Test subtraction
        {
            "test_sub",
            R"(
  __kernel void test_sub(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *srcC,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] - srcB[tid];
  };
)",
            [](T a, T b, T c) { return a - b; },
        },

        // Test multiplication
        {
            "test_mul",
            R"(
  __kernel void test_mul(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *srcC,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] * srcB[tid];
  };
)",
            [](T a, T b, T c) { return a * b; },
        },

        // Test multiply-accumulate
        {
            "test_mad",
            R"(
  __kernel void test_mad(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *srcC,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] * srcB[tid] + srcC[tid];
  };
)",
            [](T a, T b, T c) { return a * b + c; },
        },
    };

    clMemWrapper streams[4];
    cl_int err;

    if (std::is_same<T, cl_ulong>::value && !gHasLong)
    {
        log_info("64-bit integers are not supported on this device. Skipping "
                 "test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    // Create host buffers and fill with random data.
    std::vector<T> inputA(num_elements * N);
    std::vector<T> inputB(num_elements * N);
    std::vector<T> inputC(num_elements * N);
    std::vector<T> output(num_elements * N);
    MTdataHolder d(gRandomSeed);
    for (int i = 0; i < num_elements; i++)
    {
        inputA[i] = (T)genrand_int64(d);
        inputB[i] = (T)genrand_int64(d);
        inputC[i] = (T)genrand_int64(d);
    }

    size_t datasize = sizeof(T) * num_elements * N;

    // Create device buffers.
    for (int i = 0; i < ARRAY_SIZE(streams); i++)
    {
        streams[i] =
            clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &err);
        test_error(err, "clCreateBuffer failed");
    }

    // Copy input data to device.
    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, datasize,
                               inputA.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueWriteBuffer failed\n");
    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, datasize,
                               inputB.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueWriteBuffer failed\n");
    err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, datasize,
                               inputC.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueWriteBuffer failed\n");

    std::string build_options = "-DTYPE=";
    build_options += typestr;

    // Run test for each operation
    for (auto test : tests)
    {
        log_info("%s... ", test.name);

        // Create kernel and set args
        clProgramWrapper program;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &program, &kernel, 1,
                                          &test.kernel_code, test.name,
                                          build_options.c_str());
        test_error(err, "create_single_kernel_helper failed\n");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &streams[0]);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &streams[1]);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &streams[2]);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &streams[3]);
        test_error(err, "clSetKernelArgs failed\n");

        // Run kernel
        size_t threads[1] = { static_cast<size_t>(num_elements) };
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0,
                                     NULL, NULL);
        test_error(err, "clEnqueueNDRangeKernel failed\n");

        // Read results
        err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, datasize,
                                  output.data(), 0, NULL, NULL);
        test_error(err, "clEnqueueReadBuffer failed\n");

        // Verify results
        for (int i = 0; i < num_elements * N; i++)
        {
            T r = test.ref(inputA[i], inputB[i], inputC[i]);
            if (r != output[i])
            {
                log_error("\n\nverification failed at index %d\n", i);
                log_error("-> inputs: %llu, %llu, %llu\n",
                          static_cast<cl_uint>(inputA[i]),
                          static_cast<cl_uint>(inputB[i]),
                          static_cast<cl_uint>(inputC[i]));
                log_error("-> expected %llu, got %llu\n\n",
                          static_cast<cl_uint>(r),
                          static_cast<cl_uint>(output[i]));
                return TEST_FAIL;
            }
        }
        log_info("passed\n");
    }

    return TEST_PASS;
}

int test_intmath_int(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_uint, 1>(device, context, queue, num_elements,
                                    "uint");
}

int test_intmath_int2(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_uint, 2>(device, context, queue, num_elements,
                                    "uint2");
}

int test_intmath_int4(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_uint, 4>(device, context, queue, num_elements,
                                    "uint4");
}

int test_intmath_long(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_ulong, 1>(device, context, queue, num_elements,
                                     "ulong");
}

int test_intmath_long2(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_ulong, 2>(device, context, queue, num_elements,
                                     "ulong2");
}

int test_intmath_long4(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return test_intmath<cl_ulong, 4>(device, context, queue, num_elements,
                                     "ulong4");
}

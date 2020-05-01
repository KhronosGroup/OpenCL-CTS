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

#include <string>
#include <vector>

#include "procs.h"

const char *add_kernel_code = R"(
  __kernel void test_add(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] + srcB[tid];
  };
)";

const char *sub_kernel_code = R"(
  __kernel void test_sub(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] - srcB[tid];
  };
)";

const char *mul_kernel_code = R"(
  __kernel void test_mul(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] * srcB[tid];
  };
)";

const char *mad_kernel_code = R"(
  __kernel void test_mad(__global TYPE *srcA,
                         __global TYPE *srcB,
                         __global TYPE *srcC,
                         __global TYPE *dst)
  {
      int  tid = get_global_id(0);
      dst[tid] = srcA[tid] * srcB[tid] + srcC[tid];
  };
)";

template <typename T>
int verify_add(const std::vector<T> &inputA, const std::vector<T> &inputB,
               const std::vector<T> &output)
{
    for (int i = 0; i < output.size(); i++)
    {
        T r = inputA[i] + inputB[i];
        if (r != output[i])
        {
            log_error("ADD test failed\n");
            return -1;
        }
    }

    log_info("ADD test passed\n");
    return 0;
}

template <typename T>
int verify_sub(const std::vector<T> &inputA, const std::vector<T> &inputB,
               const std::vector<T> &output)
{
    for (int i = 0; i < output.size(); i++)
    {
        T r = inputA[i] - inputB[i];
        if (r != output[i])
        {
            log_error("SUB test failed\n");
            return -1;
        }
    }

    log_info("SUB test passed\n");
    return 0;
}

template <typename T>
int verify_mul(const std::vector<T> &inputA, const std::vector<T> &inputB,
               const std::vector<T> &output)
{
    for (int i = 0; i < output.size(); i++)
    {
        T r = inputA[i] * inputB[i];
        if (r != output[i])
        {
            log_error("MUL test failed\n");
            return -1;
        }
    }

    log_info("MUL test passed\n");
    return 0;
}

template <typename T>
int verify_mad(const std::vector<T> &inputA, const std::vector<T> &inputB,
               const std::vector<T> &inputC, const std::vector<T> &output)
{
    for (T i = 0; i < output.size(); i++)
    {
        T r = inputA[i] * inputB[i] + inputC[i];
        if (r != output[i])
        {
            log_error("MAD test failed\n");
            return -1;
        }
    }

    log_info("MAD test passed\n");
    return 0;
}

template <typename T, unsigned N>
int test_intmath(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements, std::string typestr)
{
    clMemWrapper streams[4];
    clProgramWrapper program[4];
    clKernelWrapper kernel[4];
    cl_int err;

    if (std::is_same<T, cl_ulong>::value && !gHasLong)
    {
        log_info("64-bit integers are not supported on this device. Skipping "
                 "test.\n");
        return 0;
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
    streams[0] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &err);
    test_error(err, "clCreateBuffer failed");
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &err);
    test_error(err, "clCreateBuffer failed");
    streams[2] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &err);
    test_error(err, "clCreateBuffer failed");
    streams[3] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &err);
    test_error(err, "clCreateBuffer failed");

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

    // Build kernels.
    std::string build_options = "-DTYPE=";
    build_options += typestr;
    err = create_single_kernel_helper(context, &program[0], &kernel[0], 1,
                                      &add_kernel_code, "test_add",
                                      build_options.c_str());
    test_error(err, "create_single_kernel_helper failed\n");
    err = create_single_kernel_helper(context, &program[1], &kernel[1], 1,
                                      &sub_kernel_code, "test_sub",
                                      build_options.c_str());
    test_error(err, "create_single_kernel_helper failed\n");
    err = create_single_kernel_helper(context, &program[2], &kernel[2], 1,
                                      &mul_kernel_code, "test_mul",
                                      build_options.c_str());
    test_error(err, "create_single_kernel_helper failed\n");
    err = create_single_kernel_helper(context, &program[3], &kernel[3], 1,
                                      &mad_kernel_code, "test_mad",
                                      build_options.c_str());
    test_error(err, "create_single_kernel_helper failed\n");

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &streams[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &streams[1]);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &streams[3]);
    test_error(err, "clSetKernelArgs failed\n");

    err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &streams[0]);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &streams[1]);
    err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &streams[3]);
    test_error(err, "clSetKernelArgs failed\n");

    err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &streams[0]);
    err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &streams[1]);
    err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &streams[3]);
    test_error(err, "clSetKernelArgs failed\n");

    err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &streams[0]);
    err |= clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &streams[1]);
    err |= clSetKernelArg(kernel[3], 2, sizeof(cl_mem), &streams[2]);
    err |= clSetKernelArg(kernel[3], 3, sizeof(cl_mem), &streams[3]);
    test_error(err, "clSetKernelArgs failed\n");

    size_t threads[1] = { static_cast<size_t>(num_elements) };
    for (int i = 0; i < 4; i++)
    {
        err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL,
                                     0, NULL, NULL);
        test_error(err, "clEnqueueNDRangeKernel failed\n");

        err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, datasize,
                                  output.data(), 0, NULL, NULL);
        test_error(err, "clEnqueueReadBuffer failed\n");

        switch (i)
        {
            case 0: err = verify_add(inputA, inputB, output); break;
            case 1: err = verify_sub(inputA, inputB, output); break;
            case 2: err = verify_mul(inputA, inputB, output); break;
            case 3: err = verify_mad(inputA, inputB, inputC, output); break;
        }
    }

    return err;
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

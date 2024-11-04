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
#include "testBase.h"

#include <vector>

const char *pragma_unroll_kernels[] = {
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" __attribute__((opencl_unroll_hint))\n"
" for(size_t i = 0; i < 100; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" for(size_t i = 0; i < 100; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" for(size_t i = 0; i < 100; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" for(size_t i = 0; i < 100; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" __attribute__((opencl_unroll_hint))\n"
" for(size_t i = 0; i < n; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" for(size_t i = 0; i < n; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" for(size_t i = 0; i < n; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" for(size_t i = 0; i < n; ++i)\n"
"   dst[i] = i;\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint))\n"
" while(i < 100) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" while(i < 100) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" while(i < 100) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" while(i < 100) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint))\n"
" while(i < n) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" while(i < n) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" while(i < n) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" while(i < n) {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" }\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < 100);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < 100);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < 100);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < 100);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < n);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(1)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < n);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(10)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < n);\n"
"}\n",
"__kernel void pragma_unroll(__global uint *dst)\n"
"{\n"
" size_t tid = get_global_id(0);\n"
" size_t n = (tid + 1) * 100;\n"
" size_t i = 0;\n"
" __attribute__((opencl_unroll_hint(100)))\n"
" do {\n"
"   dst[i] = i;\n"
"   ++i;\n"
" } while(i < n);\n"
"}\n",
};

int test_pragma_unroll(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
  const size_t ELEMENT_NUM = 100;
  const size_t KERNEL_NUM = 24;

  cl_int error;

  //execute all kernels and check if the results are as expected
  for (size_t kernelIdx = 0; kernelIdx < KERNEL_NUM; ++kernelIdx) {
    clProgramWrapper program;
    clKernelWrapper kernel;
    if (create_single_kernel_helper(
            context, &program, &kernel, 1,
            (const char **)&pragma_unroll_kernels[kernelIdx], "pragma_unroll"))
    {
        log_error("The program we attempted to compile was: \n%s\n",
                  pragma_unroll_kernels[kernelIdx]);
        return -1;
    }

    clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ELEMENT_NUM * sizeof(cl_uint), NULL, &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
    test_error(error, "clSetKernelArg failed");

    //only one thread should be enough to verify if kernel is fully functional
    size_t workSize = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &workSize, NULL, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    std::vector<cl_uint> results(ELEMENT_NUM, 0);
    error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, ELEMENT_NUM * sizeof(cl_uint), &results[0], 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < ELEMENT_NUM; ++i) {
      if (results[i] != i) {
          log_error(
              "Kernel %zu returned invalid result. Test: %d, expected: %zu\n",
              kernelIdx + 1, results[i], i);
          return -1;
      }
    }
  }

  return 0;
}

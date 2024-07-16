//
// Copyright (c) 2017-2021 The Khronos Group Inc.
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
#include "harness/conversions.h"
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

const char *clone_kernel_test_img[] =
{
    "__kernel void img_read_kernel(read_only image2d_t img, sampler_t sampler, __global int* outbuf)\n"
    "{\n"
    "    uint4 color;\n"
    "\n"
    "    color = read_imageui(img, sampler, (int2)(0,0));\n"
    "    \n"
    "    // 7, 8, 9, 10th DWORD\n"
    "    outbuf[7] = color.x;\n"
    "    outbuf[8] = color.y;\n"
    "    outbuf[9] = color.z;\n"
    "    outbuf[10] = color.w;\n"
    "}\n"
    "\n"
    "__kernel void img_write_kernel(write_only image2d_t img, uint4 color)\n"
    "{\n"
    "    write_imageui (img, (int2)(0, 0), color);\n"
    "}\n"

};

const char *clone_kernel_test_double[] =
{
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void clone_kernel_test1(double d, __global double* outbuf)\n"
    "{\n"
    "    // use the same outbuf as rest of the tests\n"
    "    outbuf[2] = d;\n"
    "}\n"
};

const char* clone_kernel_test_kernel[] = {
    "typedef struct\n"
    "{\n"
    "    int i;\n"
    "    float f;\n"
    "} structArg;\n"
    "\n"
    "// value type test\n"
    "__kernel void clone_kernel_test0(int iarg, float farg, structArg sarg, "
    "__local int* localbuf, __global int* outbuf)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    outbuf[0] = iarg;\n"
    "    outbuf[1] = sarg.i;\n"
    "    \n"
    "    ((__global float*)outbuf)[2] = farg;\n"
    "    ((__global float*)outbuf)[3] = sarg.f;\n"
    "}\n"
    "\n"
    "__kernel void buf_read_kernel(__global int* buf, __global int* outbuf)\n"
    "{\n"
    "    // 6th DWORD\n"
    "    outbuf[6] = buf[0];\n"
    "}\n"
    "\n"
    "__kernel void buf_write_kernel(__global int* buf, int write_val)\n"
    "{\n"
    "    buf[0] = write_val;\n"
    "}\n"

};

const int BUF_SIZE = 128;

struct structArg
{
    int i;
    float f;
};

int test_image_arg_shallow_clone(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, void* pbufRes, clMemWrapper& bufOut)
{
    int error;
    cl_image_format    img_format;
    clSamplerWrapper sampler;
    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_image_desc imageDesc;
    memset(&imageDesc, 0x0, sizeof(cl_image_desc));
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = 512;
    imageDesc.image_height = 512;

    cl_uint color[4] = {1,3,5,7};

    clProgramWrapper program_read;
    clProgramWrapper program_write;
    clKernelWrapper kernel_read;
    clKernelWrapper kernel_write;
    clKernelWrapper kernel_cloned;
    size_t    ndrange1 = 1;

    clMemWrapper img;

    if (create_single_kernel_helper(context, &program_read, &kernel_read, 1,
                                    clone_kernel_test_img, "img_read_kernel")
        != 0)
    {
        return -1;
    }

    if (create_single_kernel_helper(context, &program_write, &kernel_write, 1,
                                    clone_kernel_test_img, "img_write_kernel")
        != 0)
    {
        return -1;
    }

    img = clCreateImage(context, CL_MEM_READ_WRITE, &img_format, &imageDesc, NULL, &error);
    test_error( error, "clCreateImage failed." );

    cl_sampler_properties properties[] = {
        CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE,
        CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP_TO_EDGE,
        CL_SAMPLER_FILTER_MODE, CL_FILTER_NEAREST,
        0 };
    sampler = clCreateSamplerWithProperties(context, properties, &error);
    test_error( error, "clCreateSamplerWithProperties failed." );

    error = clSetKernelArg(kernel_write, 1, sizeof(int) * 4, color);
    error += clSetKernelArg(kernel_write, 0, sizeof(cl_mem), &img);
    test_error( error, "clSetKernelArg failed." );

    error = clEnqueueNDRangeKernel(queue, kernel_write, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    error = clSetKernelArg(kernel_read, 0, sizeof(cl_mem), &img);
    error += clSetKernelArg(kernel_read, 1, sizeof(cl_sampler), &sampler);
    error += clSetKernelArg(kernel_read, 2, sizeof(cl_mem), &bufOut);

    test_error( error, "clSetKernelArg failed." );

    // clone the kernel
    kernel_cloned = clCloneKernel(kernel_read, &error);
    test_error( error, "clCloneKernel failed." );
    error = clEnqueueNDRangeKernel(queue, kernel_cloned, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    // read result back
    error = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, 128, pbufRes, 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed." );

    test_assert_error(((cl_uint*)pbufRes)[7] == color[0],
                      "clCloneKernel test failed.");

    test_assert_error(((cl_uint*)pbufRes)[8] == color[1],
                      "clCloneKernel test failed.");

    test_assert_error(((cl_uint*)pbufRes)[9] == color[2],
                      "clCloneKernel test failed.");

    test_assert_error(((cl_uint*)pbufRes)[10] == color[3],
                      "clCloneKernel test failed.");

    return TEST_PASS;
}

int test_double_arg_clone(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, void* pbufRes, clMemWrapper& bufOut)
{
    int error = 0;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clKernelWrapper kernel_cloned;
    size_t    ndrange1 = 1;

    if( create_single_kernel_helper( context, &program, &kernel, 1, clone_kernel_test_double, "clone_kernel_test1" ) != 0 )
    {
        return -1;
    }

    cl_double d = 1.23;
    error = clSetKernelArg(kernel, 0, sizeof(double), &d);
    error += clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufOut);
    test_error( error, "clSetKernelArg failed." );

    kernel_cloned = clCloneKernel(kernel, &error);
    test_error( error, "clCloneKernel failed." );

    error = clEnqueueNDRangeKernel(queue, kernel_cloned, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    // read result back
    error = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, BUF_SIZE, pbufRes, 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed." );

    test_assert_error(abs(((cl_double*)pbufRes)[2] - d) <= 0.0000001,
                      "clCloneKernel test failed.");

    return TEST_PASS;
}

int test_args_enqueue_helper(cl_context context, cl_command_queue queue,
                             cl_kernel* srcKernel, cl_int* value,
                             cl_mem* bufOut)
{
    cl_int error;
    size_t ndrange1 = 1;
    cl_int bufRes;

    // enqueue - srcKernel
    error = clEnqueueNDRangeKernel(queue, *srcKernel, 1, NULL, &ndrange1, NULL,
                                   0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");
    error = clEnqueueReadBuffer(queue, *bufOut, CL_TRUE, 0, sizeof(cl_int),
                                &bufRes, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    test_assert_error(bufRes == *value,
                      "clCloneKernel test failed to verify integer value.\n");

    return TEST_PASS;
}

int test_cloned_kernel_args(cl_device_id deviceID, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    cl_int error;
    clProgramWrapper program;
    clKernelWrapper srcKernel;
    cl_int intargs[] = { 1, 2, 3, 4 };
    clMemWrapper bufOut;

    // Create srcKernel to test with
    error = create_single_kernel_helper(context, &program, &srcKernel, 1,
                                        clone_kernel_test_kernel,
                                        "buf_write_kernel");
    test_error(error, "Unable to create srcKernel for test_cloned_kernel_args");

    bufOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL,
                            &error);
    test_error(error, "clCreateBuffer failed.");

    // srcKernel, set args
    error = clSetKernelArg(srcKernel, 1, sizeof(cl_int), &intargs[0]);
    error |= clSetKernelArg(srcKernel, 0, sizeof(cl_mem), &bufOut);
    test_error(error, "clSetKernelArg failed for srcKernel");

    // clone the srcKernel and set different arg
    clKernelWrapper cloneKernel_1 = clCloneKernel(srcKernel, &error);
    test_error(error, "clCloneKernel failed for cloneKernel_1");
    error = clSetKernelArg(cloneKernel_1, 1, sizeof(cl_int), &intargs[1]);
    test_error(error, "clSetKernelArg failed for cloneKernel_1");

    // clone the cloneKernel_1 and set different arg
    clKernelWrapper cloneKernel_2 = clCloneKernel(cloneKernel_1, &error);
    test_error(error, "clCloneKernel failed for cloneKernel_2");
    error = clSetKernelArg(cloneKernel_2, 1, sizeof(cl_int), &intargs[2]);
    test_error(error, "clSetKernelArg failed for cloneKernel_2");

    // enqueue - srcKernel
    if (test_args_enqueue_helper(context, queue, &srcKernel, &intargs[0],
                                 &bufOut)
        != 0)
    {
        test_fail("test_args_enqueue_helper failed for srcKernel.\n");
    }

    // enqueue - cloneKernel_1
    if (test_args_enqueue_helper(context, queue, &cloneKernel_1, &intargs[1],
                                 &bufOut)
        != 0)
    {
        test_fail("test_args_enqueue_helper failed for cloneKernel_1.\n");
    }

    // enqueue - cloneKernel_2
    if (test_args_enqueue_helper(context, queue, &cloneKernel_2, &intargs[2],
                                 &bufOut)
        != 0)
    {
        test_fail("test_args_enqueue_helper failed for cloneKernel_2.\n");
    }

    // srcKernel, set different arg and enqueue
    error = clSetKernelArg(srcKernel, 1, sizeof(cl_int), &intargs[3]);
    test_error(error,
               "clSetKernelArg failed for srcKernel with different value");
    if (test_args_enqueue_helper(context, queue, &srcKernel, &intargs[3],
                                 &bufOut)
        != 0)
    {
        test_fail("test_args_enqueue_helper failed for srcKernel on retry.\n");
    }

    // enqueue - cloneKernel_1 again, to check if the args were not modified
    if (test_args_enqueue_helper(context, queue, &cloneKernel_1, &intargs[1],
                                 &bufOut)
        != 0)
    {
        test_fail(
            "test_args_enqueue_helper failed for cloneKernel_1 on retry.\n");
    }

    // enqueue - cloneKernel_2 again, to check if the args were not modified
    if (test_args_enqueue_helper(context, queue, &cloneKernel_2, &intargs[2],
                                 &bufOut)
        != 0)
    {
        test_fail(
            "test_args_enqueue_helper failed for cloneKernel_2 on retry.\n");
    }

    return TEST_PASS;
}

int test_buff_image_multiple_args(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clProgramWrapper program_buf_read;
    clProgramWrapper program_buf_write;
    clKernelWrapper kernel;
    clKernelWrapper kernel_pipe_read;
    clKernelWrapper kernel_buf_read;
    clKernelWrapper kernel_pipe_write;
    clKernelWrapper kernel_buf_write;

    clKernelWrapper kernel_pipe_read_cloned;
    clKernelWrapper kernel_buf_read_cloned;
    size_t    ndrange1 = 1;

    int write_val = 123;


    cl_bool bimg = CL_FALSE;
    cl_bool bdouble = CL_FALSE;
    // test image support
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &bimg, NULL);
    test_error( error, "clGetDeviceInfo failed." );

    // test double support
    if (is_extension_available(deviceID, "cl_khr_fp64"))
    {
        bdouble = CL_TRUE;
    }

    /* Create kernels to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, clone_kernel_test_kernel, "clone_kernel_test0" ) != 0 )
    {
        return -1;
    }

    if (create_single_kernel_helper(context, &program_buf_read,
                                    &kernel_buf_read, 1,
                                    clone_kernel_test_kernel, "buf_read_kernel")
        != 0)
    {
        return -1;
    }

    if (create_single_kernel_helper(
            context, &program_buf_write, &kernel_buf_write, 1,
            clone_kernel_test_kernel, "buf_write_kernel")
        != 0)
    {
        return -1;
    }

    // Kernel args
    // Value type
    int intarg = 0;
    float farg = 1.0;
    structArg sa = { 1, 1.0f };

    // cl_mem
    clMemWrapper buf, bufOut;

    char* pbuf = new char[BUF_SIZE];
    char* pbufRes = new char[BUF_SIZE];
    buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BUF_SIZE, pbuf, &error);
    test_error( error, "clCreateBuffer failed." );

    bufOut = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BUF_SIZE, NULL, &error);
    test_error( error, "clCreateBuffer failed." );

    error = clSetKernelArg(kernel, 0, sizeof(int), &intarg);
    error += clSetKernelArg(kernel, 1, sizeof(float), &farg);
    error += clSetKernelArg(kernel, 2, sizeof(structArg), &sa);
    error += clSetKernelArg(kernel, 3, 128, NULL);    // local mem

    test_error( error, "clSetKernelArg failed." );

    // clone the kernel
    clKernelWrapper clonek = clCloneKernel(kernel, &error);
    test_error( error, "clCloneKernel failed." );

    // enqueue the kernel before the last arg is set
    error = clEnqueueNDRangeKernel(queue, clonek, 1, NULL, &ndrange1, NULL, 0,
                                   NULL, NULL);
    test_failure_error(error, CL_INVALID_KERNEL_ARGS,
                       "A kernel cloned before all args are set should return "
                       "CL_INVALID_KERNEL_ARGS if enqueued before the "
                       "remaining args are set");

    // set the last arg and enqueue
    error = clSetKernelArg(clonek, 4, sizeof(cl_mem), &bufOut);
    test_error( error, "clSetKernelArg failed." );
    error = clEnqueueNDRangeKernel(queue, clonek, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    // shallow clone tests for buffer
    error = clSetKernelArg(kernel_buf_write, 0, sizeof(cl_mem), &buf);
    error += clSetKernelArg(kernel_buf_write, 1, sizeof(int), &write_val);
    test_error( error, "clSetKernelArg failed." );
    error = clEnqueueNDRangeKernel(queue, kernel_buf_write, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    error = clSetKernelArg(kernel_buf_read, 0, sizeof(cl_mem), &buf);
    error += clSetKernelArg(kernel_buf_read, 1, sizeof(cl_mem), &bufOut);
    test_error( error, "clSetKernelArg failed." );

    // clone the kernel
    kernel_buf_read_cloned = clCloneKernel(kernel_buf_read, &error);
    test_error( error, "clCloneKernel API call failed." );
    error = clEnqueueNDRangeKernel(queue, kernel_buf_read_cloned, 1, NULL, &ndrange1, NULL, 0, NULL, NULL);
    test_error( error, "clEnqueueNDRangeKernel failed." );

    // read result back
    error = clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, BUF_SIZE, pbufRes, 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed." );

    // Compare the results
    test_assert_error(((int*)pbufRes)[0] == intarg,
                      "clCloneKernel test failed. Failed to clone integer type "
                      "argument.");

    test_assert_error(
        ((int*)pbufRes)[1] == sa.i,
        "clCloneKernel test failed. Failed to clone structure type "
        "argument.");

    test_assert_error(
        ((float*)pbufRes)[2] == farg,
        "clCloneKernel test failed. Failed to clone float type argument.");

    test_assert_error(
        ((float*)pbufRes)[3] == sa.f,
        "clCloneKernel test failed. Failed to clone structure type "
        "argument.");

    test_assert_error(
        ((int*)pbufRes)[6] == write_val,
        "clCloneKernel test failed.  Failed to clone cl_mem argument.");

    if (bimg)
    {
        error = test_image_arg_shallow_clone(deviceID, context, queue,
                                             num_elements, pbufRes, bufOut);
        test_error(error, "image arg shallow clone test failed.");
    }

    if (bdouble)
    {
        error = test_double_arg_clone(deviceID, context, queue, num_elements,
                                      pbufRes, bufOut);
        test_error(error, "double arg clone test failed.");
    }

    delete[] pbuf;
    delete[] pbufRes;

    return TEST_PASS;
}

int test_clone_kernel(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    if (test_buff_image_multiple_args(deviceID, context, queue, num_elements)
        != 0)
    {
        test_fail("clCloneKernel test_buff_image_multiple_args failed.\n");
    }

    if (test_cloned_kernel_args(deviceID, context, queue, num_elements) != 0)
    {
        test_fail("clCloneKernel test_cloned_kernel_args failed.\n");
    }

    return TEST_PASS;
}

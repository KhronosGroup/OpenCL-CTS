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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

#include "checker_image_mem_host_read_only.hpp"
#include "checker_image_mem_host_no_access.hpp"
#include "checker_image_mem_host_write_only.hpp"

//======================================
static cl_int test_mem_host_read_only_RW_Image(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info("%s  ... \n ", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_read_only<int> checker(deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    err = checker.verify_RW_Image();
    test_error(err, __FUNCTION__);
    clFinish(queue);
    return err;
}

static cl_int test_mem_host_read_only_RW_Image_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info("%s  ... \n ", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_read_only<int> checker(deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    err = checker.verify_RW_Image_Mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);
    return err;
}

int test_mem_host_read_only_image(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flags[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_READ_ONLY,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY
    };

    cl_int err = CL_SUCCESS;

    cl_bool image_support;
    err = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT,
                          sizeof image_support, &image_support, NULL);
    if (err)
    {
        test_error(err, __FUNCTION__);
        return err;
    }
    if (!image_support)
    {
        log_info("Images are not supported by the device, skipping test...\n");
        return 0;
    }


    cl_mem_object_type img_type[5] = {
        CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    size_t img_dims[5][3] = { { 200, 1, 1 },
                              { 200, 80, 1 },
                              { 200, 80, 5 },
                              { 200, 1, 1 },
                              { 200, 80, 10 } }; // in elements

    size_t array_size[5] = { 1, 10, 1, 10, 1 };

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int flag = 0; flag < 2; flag++)
        for (int i = 0; i < 2; i++) // blocking
        {
            for (int p = 0; p < 3; p++)
            {
                err = test_mem_host_read_only_RW_Image(
                    deviceID, context, queue, blocking[i],
                    buffer_mem_flags[flag], img_type[p], array_size[p],
                    img_dims[p]);

                test_error(err, __FUNCTION__);

                err = test_mem_host_read_only_RW_Image_Mapping(
                    deviceID, context, queue, blocking[i],
                    buffer_mem_flags[flag], img_type[p], array_size[p],
                    img_dims[p]);

                test_error(err, __FUNCTION__);
            }
        }

    return err;
}

//----------------------------
static cl_int test_MEM_HOST_WRITE_ONLY_Image_RW(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info(" %s  ... \n ", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_write_only<int> checker(deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    checker.Setup_Test_Environment();

    err = checker.verify_RW_Image();
    clFinish(queue);
    test_error(err, __FUNCTION__);

    return err;
}

static cl_int test_MEM_HOST_WRITE_ONLY_Image_RW_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info("%s  ... \n ", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_write_only<int> checker(deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    checker.Setup_Test_Environment();

    err = checker.verify_RW_Image_Mapping();
    clFinish(queue);
    test_error(err, __FUNCTION__);

    return err;
}

int test_mem_host_write_only_image(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flags[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_WRITE_ONLY,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY
    };

    cl_int err = CL_SUCCESS;

    cl_bool image_support;
    err = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT,
                          sizeof image_support, &image_support, NULL);
    if (err)
    {
        test_error(err, __FUNCTION__);
        return err;
    }
    if (!image_support)
    {
        log_info("Images are not supported by the device, skipping test...\n");
        return 0;
    }

    cl_mem_object_type img_type[5] = {
        CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    size_t img_dims[5][3] = { { 200, 1, 1 },
                              { 200, 80, 1 },
                              { 200, 80, 5 },
                              { 200, 1, 1 },
                              { 200, 80, 1 } }; // in elements

    size_t array_size[5] = { 1, 10, 1, 10, 1 };

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 2; i++) // blocking
        {
            for (int p = 0; p < 3; p++)
            {
                err = test_MEM_HOST_WRITE_ONLY_Image_RW(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    img_type[p], array_size[p], img_dims[p]);
                test_error(err, __FUNCTION__);

                err = test_MEM_HOST_WRITE_ONLY_Image_RW_Mapping(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    img_type[p], array_size[p], img_dims[p]);
                test_error(err, __FUNCTION__);
            }
        }

    return err;
}

//--------

static cl_int test_mem_host_no_access_Image_RW(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info("%s  ... \n", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_no_access<int> checker(deviceID, context, queue);

    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Image();
    test_error(err, __FUNCTION__);
    clFinish(queue);
    return err;
}

static cl_int test_mem_host_no_access_Image_RW_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_object_type image_type_in, size_t array_size, size_t *img_dim)
{
    log_info("%s  ... \n ", __FUNCTION__);
    cl_int err = CL_SUCCESS;

    cImage_check_mem_host_no_access<int> checker(deviceID, context, queue);

    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    checker.m_cl_Image_desc.image_type = image_type_in;
    checker.m_cl_Image_desc.image_width = img_dim[0];
    checker.m_cl_Image_desc.image_height = img_dim[1];
    checker.m_cl_Image_desc.image_depth = img_dim[2];
    checker.m_cl_Image_desc.image_array_size = array_size;
    checker.m_cl_Image_desc.image_row_pitch = 0;
    checker.m_cl_Image_desc.image_slice_pitch = 0;
    checker.m_cl_Image_desc.num_mip_levels = 0;
    checker.m_cl_Image_desc.num_samples = 0;

    checker.SetupImage();
    checker.Init_rect();
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Image_Mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);
    return err;
}

int test_mem_host_no_access_image(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flags[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS
    };

    cl_int err = CL_SUCCESS;

    cl_bool image_support;
    err = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT,
                          sizeof image_support, &image_support, NULL);
    if (err)
    {
        test_error(err, __FUNCTION__);
        return err;
    }
    if (!image_support)
    {
        log_info("Images are not supported by the device, skipping test...\n");
        return 0;
    }

    cl_mem_object_type img_type[5] = {
        CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY
    };

    size_t img_dims[5][3] = { { 200, 1, 1 },
                              { 200, 80, 1 },
                              { 100, 80, 5 },
                              { 200, 1, 1 },
                              { 200, 80, 1 } }; // in elements

    size_t array_size[5] = { 1, 1, 1, 10, 10 };

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 2; i++) // blocking
        {
            for (int p = 0; p < 3; p++)
            {
                err += test_mem_host_no_access_Image_RW(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    img_type[p], array_size[p], img_dims[p]);

                err += test_mem_host_no_access_Image_RW_Mapping(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    img_type[p], array_size[p], img_dims[p]);
            }
        }

    return err;
}

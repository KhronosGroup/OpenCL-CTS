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
#ifndef test_conformance_checker_Image_MEM_HOST_NO_ACCESS_h
#define test_conformance_checker_Image_MEM_HOST_NO_ACCESS_h

#include "checker_image_mem_host_write_only.hpp"

template <class T>
class cImage_check_mem_host_no_access
    : public cImage_check_mem_host_write_only<T> {
public:
    cImage_check_mem_host_no_access(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue)
        : cImage_check_mem_host_write_only<T>(deviceID, context, queue)
    {}

    ~cImage_check_mem_host_no_access(){};

    cl_int verify_RW_Image();
    cl_int verify_RW_Image_Mapping();
};

template <class T> cl_int cImage_check_mem_host_no_access<T>::verify_RW_Image()
{
    this->Init_rect();

    cl_event event;
    size_t img_orig[3] = { 0, 0, 0 };
    size_t img_region[3] = { 0, 0, 0 };
    img_region[0] = this->m_cl_Image_desc.image_width;
    img_region[1] = this->m_cl_Image_desc.image_height;
    img_region[2] = this->m_cl_Image_desc.image_depth;

    int color[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    cl_int err = CL_SUCCESS;
    err = clEnqueueFillImage(this->m_queue, this->m_Image, &color, img_orig,
                             img_region, 0, NULL, &event);
    test_error(err, "clEnqueueFillImage error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    this->update_host_mem_2();

    int total = (int)(this->region[0] * this->region[1] * this->region[2]);

    T v = 0xFFFFFFFF;
    int tot = (int)(this->host_m_2.Count(v));
    if (tot != total)
    {
        log_error("Buffer data content difference found\n");
        return FAILURE;
    }

    err = clEnqueueWriteImage(
        this->m_queue, this->m_Image, this->m_blocking, this->buffer_origin,
        this->region, this->buffer_row_pitch_bytes,
        this->buffer_slice_pitch_bytes, this->host_m_1.pData, 0, NULL, &event);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteImage on a memory object created with the "
            "CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return err;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    v = 0;
    this->host_m_2.Set_to(v);
    err = clEnqueueReadImage(
        this->m_queue, this->m_Image, this->m_blocking, this->buffer_origin,
        this->region, this->buffer_row_pitch_bytes,
        this->buffer_slice_pitch_bytes, this->host_m_2.pData, 0, NULL, &event);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueReadImage on a memory object created with the "
            "CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return err;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    return err;
}

template <class T>
cl_int cImage_check_mem_host_no_access<T>::verify_RW_Image_Mapping()
{
    this->Init_rect();

    cl_event event;
    cl_int err = CL_SUCCESS;

    T* dataPtr = (T*)clEnqueueMapImage(
        this->m_queue, this->m_Image, this->m_blocking, CL_MAP_WRITE,
        this->buffer_origin, this->region, &(this->buffer_row_pitch_bytes),
        &(this->buffer_slice_pitch_bytes), 0, NULL, &event, &err);

    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapImage (CL_MAP_WRITE) on a memory object "
                  "created with the CL_MEM_HOST_NO_ACCESS flag should not "
                  "return CL_SUCCESS\n");
        err = FAILURE;
        return err;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    dataPtr = (T*)clEnqueueMapImage(
        this->m_queue, this->m_Image, this->m_blocking, CL_MAP_READ,
        this->buffer_origin, this->region, &(this->buffer_row_pitch_bytes),
        &(this->buffer_slice_pitch_bytes), 0, NULL, &event, &err);

    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapImage (CL_MAP_READ) on a memory object "
                  "created with the CL_MEM_HOST_NO_ACCESS flag should not "
                  "return CL_SUCCESS\n");
        err = FAILURE;
        return err;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    return err;
}

#endif

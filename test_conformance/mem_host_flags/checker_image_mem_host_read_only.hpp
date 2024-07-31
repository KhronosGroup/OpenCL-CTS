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
#ifndef test_conformance_checker_Image_MEM_HOST_READ_ONLY_h
#define test_conformance_checker_Image_MEM_HOST_READ_ONLY_h

#include "checker.h"

template <class T>
class cImage_check_mem_host_read_only : public cBuffer_checker<T> {
public:
    cImage_check_mem_host_read_only(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue)
        : cBuffer_checker<T>(deviceID, context, queue)
    {
        m_cl_image_format.image_channel_order = CL_RGBA;
        m_cl_image_format.image_channel_data_type = CL_UNSIGNED_INT8;

        m_cl_Image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        m_cl_Image_desc.image_width = 0;
        m_cl_Image_desc.image_height = 0;
        m_cl_Image_desc.image_depth = 0;
        m_cl_Image_desc.image_array_size = 0;
        m_cl_Image_desc.image_row_pitch = 0;
        m_cl_Image_desc.image_slice_pitch = 0;
        m_cl_Image_desc.num_mip_levels = 0;
        m_cl_Image_desc.num_samples = 0;
        m_cl_Image_desc.mem_object = NULL;

        m_Image = NULL;
    };

    ~cImage_check_mem_host_read_only(){};

    cl_int get_image_elements();

    cl_image_format m_cl_image_format;
    cl_image_desc m_cl_Image_desc;
    clMemWrapper m_Image;

    virtual cl_int SetupImage();
    virtual cl_int SetupBuffer();
    virtual cl_int verify_RW_Image();

    virtual cl_int verify_RW_Image_Mapping();
    virtual cl_int verify_data(T *pdtaIn);
    virtual cl_int verify_data_with_offset(T *pdtaIn, size_t *offset);

    cl_int get_image_content_size();
    cl_int get_image_data_size();

    virtual cl_int verify_RW_Buffer();
    virtual cl_int verify_RW_Buffer_rect();
    virtual cl_int verify_RW_Buffer_mapping();
    cl_int verify_mapping_ptr(T *ptr);
};

template <class T>
cl_int cImage_check_mem_host_read_only<T>::verify_mapping_ptr(T *dataPtr)
{
    int offset_pixel = (int)(this->buffer_origin[0]
                             + this->buffer_origin[1]
                                 * this->buffer_row_pitch_bytes / sizeof(T)
                             + this->buffer_origin[2]
                                 * this->buffer_slice_pitch_bytes / sizeof(T));

    dataPtr = dataPtr - offset_pixel;

    cl_int err = CL_SUCCESS;

    if (this->buffer_mem_flag & CL_MEM_USE_HOST_PTR)
    {
        if (this->pHost_ptr != this->host_m_1.pData)
        {
            log_error("Host memory pointer difference found\n");
            return FAILURE;
        }

        if (dataPtr != this->host_m_1.pData)
        {
            log_error("Mapped host pointer difference found\n");
            return FAILURE;
        }
    }

    return err;
}

template <class T> cl_int cImage_check_mem_host_read_only<T>::verify_RW_Buffer()
{
    return CL_SUCCESS;
};

template <class T>
cl_int cImage_check_mem_host_read_only<T>::verify_RW_Buffer_rect()
{
    return CL_SUCCESS;
};

template <class T>
cl_int cImage_check_mem_host_read_only<T>::verify_RW_Buffer_mapping()
{
    return CL_SUCCESS;
};

template <class T> cl_int cImage_check_mem_host_read_only<T>::SetupBuffer()
{
    return cBuffer_checker<T>::SetupBuffer();
}

template <class T>
cl_int cImage_check_mem_host_read_only<T>::get_image_content_size()
{
    return ((cl_int)(m_cl_Image_desc.image_width * m_cl_Image_desc.image_height
                     * m_cl_Image_desc.image_depth
                     * m_cl_Image_desc.image_array_size));
}

template <class T>
cl_int cImage_check_mem_host_read_only<T>::get_image_data_size()
{
    size_t slice_pitch = m_cl_Image_desc.image_slice_pitch
        ? m_cl_Image_desc.image_slice_pitch
        : (m_cl_Image_desc.image_height * m_cl_Image_desc.image_width);
    return (slice_pitch * m_cl_Image_desc.image_depth
            * m_cl_Image_desc.image_array_size);
}

template <class T>
cl_int cImage_check_mem_host_read_only<T>::get_image_elements()
{
    return ((cl_int)(m_cl_Image_desc.image_width * m_cl_Image_desc.image_height
                     * m_cl_Image_desc.image_depth
                     * m_cl_Image_desc.image_array_size));
}

template <class T> cl_int cImage_check_mem_host_read_only<T>::SetupImage()
{
    int all =
        (int)(m_cl_Image_desc.image_width * m_cl_Image_desc.image_height
              * m_cl_Image_desc.image_depth * m_cl_Image_desc.image_array_size);

    T v = TEST_VALUE;
    this->host_m_1.Init(all, v);

    cl_int err = CL_SUCCESS;
    this->m_Image = clCreateImage(
        this->m_context, this->buffer_mem_flag, &(this->m_cl_image_format),
        &(this->m_cl_Image_desc), this->host_m_1.pData, &err);
    test_error(err, "clCreateImage error");

    this->pHost_ptr = (void *)(this->host_m_1.pData);

    return err;
}

template <class T>
cl_int cImage_check_mem_host_read_only<T>::verify_data(T *pDataIN)
{
    cl_int err = CL_SUCCESS;
    if (!this->host_m_1.Equal_rect_from_orig(pDataIN, this->buffer_origin,
                                             this->region, this->host_row_pitch,
                                             this->host_slice_pitch))
    {
        log_error("Buffer data difference found\n");
        return FAILURE;
    }

    return err;
}

template <class T>
cl_int
cImage_check_mem_host_read_only<T>::verify_data_with_offset(T *pDataIN,
                                                            size_t *offset)
{
    cl_int err = CL_SUCCESS;
    if (!this->host_m_2.Equal_rect_from_orig(pDataIN, offset, this->region,
                                             this->host_row_pitch,
                                             this->host_slice_pitch))
    {
        log_error("Buffer data difference found\n");
        return FAILURE;
    }

    return err;
}

template <class T> cl_int cImage_check_mem_host_read_only<T>::verify_RW_Image()
{
    this->Init_rect();

    int imge_content_size = this->get_image_content_size();
    T v = 0;
    this->host_m_2.Init(imge_content_size, v);

    cl_event event;
    cl_int err = CL_SUCCESS;
    err = clEnqueueReadImage(
        this->m_queue, this->m_Image, this->m_blocking, this->buffer_origin,
        this->region, this->buffer_row_pitch_bytes,
        this->buffer_slice_pitch_bytes, this->host_m_2.pData, 0, NULL, &event);

    test_error(err, "clEnqueueReadImage error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    err = this->verify_data(this->host_m_2.pData);
    test_error(err, "verify_data error");

    err = clEnqueueWriteImage(
        this->m_queue, this->m_Image, this->m_blocking, this->buffer_origin,
        this->region, this->buffer_row_pitch_bytes,
        this->buffer_slice_pitch_bytes, this->host_m_2.pData, 0, NULL, &event);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteImage on a memory object created with the "
            "CL_MEM_HOST_READ_ONLY flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    return err;
}

template <class T>
cl_int cImage_check_mem_host_read_only<T>::verify_RW_Image_Mapping()
{
    cl_event event;
    cl_int err = CL_SUCCESS;

    T *dataPtr = (T *)clEnqueueMapImage(
        this->m_queue, this->m_Image, this->m_blocking, CL_MAP_READ,
        this->buffer_origin, this->region, &(this->buffer_row_pitch_bytes),
        &(this->buffer_slice_pitch_bytes), 0, NULL, &event, &err);

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    err = this->verify_mapping_ptr(dataPtr);
    test_error(err, "clEnqueueMapImage error");

    err = this->verify_data(dataPtr);
    test_error(err, "verify_data error");

    err = clEnqueueUnmapMemObject(this->m_queue, this->m_Image, dataPtr, 0,
                                  NULL, &event);
    test_error(err, "clEnqueueUnmapMemObject error");

    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    dataPtr = (T *)clEnqueueMapImage(
        this->m_queue, this->m_Image, this->m_blocking, CL_MAP_WRITE,
        this->buffer_origin, this->region, &(this->buffer_row_pitch_bytes),
        &(this->buffer_slice_pitch_bytes), 0, NULL, &event, &err);

    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapImage (CL_MAP_WRITE) on a memory object "
                  "created with the CL_MEM_HOST_READ_ONLY flag should not "
                  "return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    return err;
}

#endif

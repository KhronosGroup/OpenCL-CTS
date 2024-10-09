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
#ifndef test_conformance_check_mem_host_read_only_h
#define test_conformance_check_mem_host_read_only_h

#include "checker.h"

template <class T>
class cBuffer_check_mem_host_read_only : public cBuffer_checker<T> {
public:
    cBuffer_check_mem_host_read_only(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue)
        : cBuffer_checker<T>(deviceID, context, queue){};

    ~cBuffer_check_mem_host_read_only(){};

    virtual cl_int Check_GetMemObjectInfo(cl_mem_flags buffer_mem_flag);
    virtual cl_int SetupBuffer();
    virtual cl_int SetupASSubBuffer(cl_mem_flags flag_p);
    virtual cl_int Setup_Test_Environment();

    cl_int verifyData(cl_int err, cl_event &event);
    cl_int verify_RW_Buffer();
    cl_int verify_RW_Buffer_rect();
    cl_int verify_RW_Buffer_mapping();
};

template <class T> cl_int cBuffer_check_mem_host_read_only<T>::SetupBuffer()
{
    this->m_buffer_type = _BUFFER;

    this->m_nNumber_elements = 888;
    T vv1 = TEST_VALUE;
    this->host_m_1.Init(this->m_nNumber_elements, vv1);
    this->host_m_0.Init(this->m_nNumber_elements, vv1);

    cl_int err = CL_SUCCESS;
    int block_size_in_byte = (int)(this->m_nNumber_elements * sizeof(T));
    this->m_buffer =
        clCreateBuffer(this->m_context, this->buffer_mem_flag,
                       block_size_in_byte, this->host_m_1.pData, &err);
    test_error(err, "clCreateBuffer error");

    if (this->buffer_mem_flag & CL_MEM_USE_HOST_PTR)
    {
        this->pHost_ptr = (void *)this->host_m_1.pData;
    }

    return err;
}

template <class T>
cl_int
cBuffer_check_mem_host_read_only<T>::SetupASSubBuffer(cl_mem_flags flag_p)
{
    return cBuffer_checker<T>::SetupASSubBuffer(flag_p);
}

template <class T>
cl_int cBuffer_check_mem_host_read_only<T>::Setup_Test_Environment()
{
    cBuffer_checker<T>::Setup_Test_Environment();
    T vv2 = 0;
    this->host_m_2.Init(this->m_nNumber_elements, vv2);

    return CL_SUCCESS;
}

template <class T>
cl_int cBuffer_check_mem_host_read_only<T>::Check_GetMemObjectInfo(
    cl_mem_flags buffer_mem_flag)
{
    cl_int err = CL_SUCCESS;
    cBuffer_checker<T>::Check_GetMemObjectInfo(buffer_mem_flag);

    if (buffer_mem_flag & CL_MEM_ALLOC_HOST_PTR)
    {
        size_t size = 0;
        err = clGetMemObjectInfo(this->m_buffer, CL_MEM_SIZE, sizeof(size),
                                 &size, NULL);
        void *pp = NULL;
        err = clGetMemObjectInfo(this->m_buffer, CL_MEM_HOST_PTR, sizeof(pp),
                                 &pp, NULL);

        if (!this->host_m_1.Equal((T *)(this->pData), this->m_nNumber_elements))
        {
            log_error("Buffer data difference found\n");
            return FAILURE;
        }
    }

    return err;
}

template <class T>
cl_int cBuffer_check_mem_host_read_only<T>::verifyData(cl_int err,
                                                       cl_event &event)
{
    if (err != CL_SUCCESS)
    {
        err = this->m_nERROR_RETURN_CODE;
        test_error(err, "clEnqueueReadBuffer error");
    }

    if (!this->host_m_1.Equal(this->host_m_2))
    {
        err = this->m_nERROR_RETURN_CODE;
        test_error(err, "clEnqueueReadBuffer data difference found");
    }

    return err;
}

template <class T>
cl_int cBuffer_check_mem_host_read_only<T>::verify_RW_Buffer()
{
    cl_event event;
    cl_int err = CL_SUCCESS;

    err = clEnqueueReadBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                              0, this->get_block_size_bytes(),
                              this->host_m_2.pData, 0, NULL, &event);
    test_error(err, "clEnqueueReadBuffer error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    if (!this->host_m_1.Equal(this->host_m_2))
    {
        log_error("Buffer data difference found\n");
        return FAILURE;
    }
    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    // test write
    err = clEnqueueWriteBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                               0, this->get_block_size_bytes(),
                               this->host_m_2.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteBuffer on a memory object created with the "
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
cl_int cBuffer_check_mem_host_read_only<T>::verify_RW_Buffer_rect()
{
    this->Init_rect();

    T vv2 = 0;
    this->host_m_2.Set_to(vv2);
    cl_event event;
    cl_int err = CL_SUCCESS;

    err = clEnqueueReadBufferRect(
        this->m_queue, this->m_buffer, this->m_blocking,
        this->buffer_origin_bytes, this->host_origin_bytes, this->region_bytes,
        this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
        this->host_row_pitch_bytes, this->host_slice_pitch_bytes,
        this->host_m_2.pData, 0, NULL, &event);
    test_error(err, "clEnqueueReadBufferRect error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    if (!this->host_m_1.Equal_rect(this->host_m_2, this->host_origin,
                                   this->region, this->host_row_pitch,
                                   this->host_slice_pitch))
    {
        log_error("Buffer data diffeence found\n");
        return FAILURE;
    }
    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    // test blocking write rect
    err = clEnqueueWriteBufferRect(
        this->m_queue, this->m_buffer, this->m_blocking,
        this->buffer_origin_bytes, this->host_origin_bytes, this->region_bytes,
        this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
        this->host_row_pitch_bytes, this->host_slice_pitch_bytes,
        this->host_m_2.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteBufferRect on a memory object created with "
            "the CL_MEM_HOST_READ_ONLY flag should not return CL_SUCCESS\n");
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
cl_int cBuffer_check_mem_host_read_only<T>::verify_RW_Buffer_mapping()
{
    cl_int err = CL_SUCCESS;
    cl_event event;
    void *dataPtr;
    dataPtr = clEnqueueMapBuffer(
        this->m_queue, this->m_buffer, this->m_blocking, CL_MAP_READ, 0,
        this->get_block_size_bytes(), 0, NULL, &event, &err);
    test_error(err, "clEnqueueMapBuffer error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    if ((this->buffer_mem_flag & CL_MEM_USE_HOST_PTR)
        && dataPtr != this->pHost_ptr)
    {
        log_error("Mapped host pointer difference found\n");
        return FAILURE;
    }

    if (!this->host_m_1.Equal((T *)dataPtr, this->m_nNumber_elements))
    {
        log_error("Buffer content difference found\n");
        return FAILURE;
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    err = clEnqueueUnmapMemObject(this->m_queue, this->m_buffer, dataPtr, 0,
                                  nullptr, nullptr);
    test_error(err, "clEnqueueUnmapMemObject error");

    //  test blocking map read
    clEnqueueMapBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                       CL_MAP_WRITE, 0, this->get_block_size_bytes(), 0, NULL,
                       NULL, &err);

    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapBuffer (CL_MAP_WRITE) on a memory "
                  "object created with the CL_MEM_HOST_READ_ONLY flag should "
                  "not return CL_SUCCESS\n");
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

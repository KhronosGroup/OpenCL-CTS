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
#ifndef test_conformance_check_mem_host_write_only__h
#define test_conformance_check_mem_host_write_only__h

#include "checker.h"

template <class T>
class cBuffer_check_mem_host_write_only : public cBuffer_checker<T> {
public:
    cBuffer_check_mem_host_write_only(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue)
        : cBuffer_checker<T>(deviceID, context, queue)
    {
        this->m_nNumber_elements = 1000;
    };

    ~cBuffer_check_mem_host_write_only(){};

    cl_program program;
    cl_kernel kernel;

    clMemWrapper m_buffer2;

    cl_int Setup_Test_Environment();

    cl_int SetupBuffer();
    cl_int SetupASSubBuffer(cl_mem_flags flag_p);

    cl_int verifyData(cl_int err, cl_event &event);
    cl_int update_host_mem_2();

    cl_int verify_RW_Buffer();
    cl_int verify_RW_Buffer_rect();
    cl_int verify_RW_Buffer_mapping();

    C_host_memory_block<T> tmp_host_m;

    virtual cl_int verify_Buffer_initialization();
};

template <class T> cl_int cBuffer_check_mem_host_write_only<T>::SetupBuffer()
{
    T vv1 = 0;
    this->host_m_1.Init(this->m_nNumber_elements, vv1); // zero out buffer

    // init buffer to 0
    cl_int err;
    int block_size_in_byte = this->get_block_size_bytes();

    this->m_buffer =
        clCreateBuffer(this->m_context, this->buffer_mem_flag,
                       block_size_in_byte, this->host_m_1.pData, &err);
    test_error(err, "clCreateBuffer error");

    err = this->Check_GetMemObjectInfo(this->buffer_mem_flag);

    if (this->buffer_mem_flag | CL_MEM_USE_HOST_PTR)
    {
        this->pHost_ptr = (void *)this->host_m_1.pData;
    }

    return err;
}

template <class T>
cl_int
cBuffer_check_mem_host_write_only<T>::SetupASSubBuffer(cl_mem_flags flag_p)
{
    return cBuffer_checker<T>::SetupASSubBuffer(flag_p);
}

template <class T>
cl_int cBuffer_check_mem_host_write_only<T>::Setup_Test_Environment()
{
    cl_int err;
    T vv2 = 0;
    this->host_m_2.Init(this->m_nNumber_elements, vv2);

    // init buffer2 to 0
    cl_mem_flags buffer_mem_flag2 =
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY;
    this->m_buffer2 = clCreateBuffer(this->m_context, buffer_mem_flag2,
                                     this->get_block_size_bytes(),
                                     this->host_m_2.pData, &err);
    test_error(err, "clCreateBuffer error\n");

    return err;
}

template <class T>
cl_int cBuffer_check_mem_host_write_only<T>::verify_Buffer_initialization()
{
    cl_int err = CL_SUCCESS;

    if (this->host_m_1.pData == NULL || this->host_m_2.pData == NULL)
    {
        log_error("Data not ready\n");
        return FAILURE;
    }

    update_host_mem_2();

    if (!this->host_m_1.Equal(this->host_m_2))
    {
        log_error("Buffer content difference found\n");
        return FAILURE;
    }

    return err;
}

template <class T>
cl_int cBuffer_check_mem_host_write_only<T>::verify_RW_Buffer()
{
    T vv1 = TEST_VALUE;
    T vv2 = 0;
    this->host_m_2.Set_to(vv2);

    tmp_host_m.Init(this->host_m_1.num_elements, vv1);

    cl_event event;
    cl_int err = CL_SUCCESS;
    err = clEnqueueWriteBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                               0, this->get_block_size_bytes(),
                               tmp_host_m.pData, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        test_error(err, "clEnqueueWriteBuffer error");
    }

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error")
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    if (tmp_host_m.Equal(this->host_m_2))
    {
        log_error("Test data should be different\n");
        return FAILURE;
    }

    update_host_mem_2();

    if (!tmp_host_m.Equal(this->host_m_2))
    {
        log_error("Buffer content difference found\n");
        return FAILURE;
    }

    err = clEnqueueReadBuffer(this->m_queue, this->m_buffer, CL_TRUE, 0,
                              this->get_block_size_bytes(),
                              this->host_m_2.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueReadBuffer on a memory object created with the "
            "CL_MEM_HOST_WRITE_ONLY flag should not return CL_SUCCESS\n");
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
cl_int cBuffer_check_mem_host_write_only<T>::verify_RW_Buffer_rect()
{
    this->Init_rect();

    T vv1 = TEST_VALUE;
    this->host_m_1.Set_to(vv1);

    T vv2 = 0;
    this->host_m_2.Set_to(vv2);

    cl_event event, event_1;

    cl_int err = CL_SUCCESS;

    vv1 = 0;
    C_host_memory_block<T> tmp_host_m;
    tmp_host_m.Init(this->host_m_1.num_elements, vv1); // zero out the buffer
    err = clEnqueueWriteBuffer(this->m_queue, this->m_buffer, CL_TRUE, 0,
                               this->get_block_size_bytes(), tmp_host_m.pData,
                               0, NULL, &event_1);
    test_error(err, "clEnqueueWriteBuffer error");

    vv1 = TEST_VALUE;
    tmp_host_m.Set_to(vv1);
    err = clEnqueueWriteBufferRect(
        this->m_queue, this->m_buffer, this->m_blocking,
        this->buffer_origin_bytes, this->host_origin_bytes, this->region_bytes,
        this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
        this->host_row_pitch_bytes, this->host_slice_pitch_bytes,
        tmp_host_m.pData, 1, &event_1, &event);
    test_error(err, "clEnqueueWriteBufferRect error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error")
    }

    if (tmp_host_m.Equal(this->host_m_2))
    {
        log_error("Test data should be different\n");
        return FAILURE;
    }

    err = clReleaseEvent(event_1);
    test_error(err, "clReleaseEvent error");
    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    update_host_mem_2();

    size_t tot_in_reg = this->region[0] * this->region[1] * this->region[2];
    if (!tmp_host_m.Equal_rect(this->host_m_2, this->host_origin, this->region,
                               this->host_row_pitch, this->host_slice_pitch))
    {
        log_error("Buffer rect content difference found\n");
        return FAILURE;
    }

    if (this->host_m_2.Count(vv1) != tot_in_reg)
    {
        log_error("Buffer rect content difference found\n");
        return FAILURE;
    }

    err = clEnqueueReadBufferRect(
        this->m_queue, this->m_buffer, this->m_blocking,
        this->buffer_origin_bytes, this->host_origin_bytes, this->region_bytes,
        this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
        this->host_row_pitch_bytes, this->host_slice_pitch_bytes,
        this->host_m_2.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueReadBufferRect on a memory object created with "
            "the CL_MEM_HOST_WRITE_ONLY flag should not return CL_SUCCESS\n");
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
cl_int cBuffer_check_mem_host_write_only<T>::update_host_mem_2()
{
    cl_event event, event_2;
    cl_int err = clEnqueueCopyBuffer(
        this->m_queue, this->m_buffer, this->m_buffer2, 0, 0,
        this->m_nNumber_elements * sizeof(T), 0, NULL, &event);
    test_error(err, "clEnqueueCopyBuffer error");

    this->host_m_2.Set_to_zero();
    err = clEnqueueReadBuffer(this->m_queue, this->m_buffer2, CL_TRUE, 0,
                              this->get_block_size_bytes(),
                              this->host_m_2.pData, 1, &event, &event_2);
    test_error(err, "clEnqueueReadBuffer error");

    clWaitForEvents(1, &event_2);
    test_error(err, "clWaitForEvents error");

    err = clReleaseEvent(event_2);
    test_error(err, "clReleaseEvent error");

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");
    return err;
}

template <class T>
cl_int cBuffer_check_mem_host_write_only<T>::verify_RW_Buffer_mapping()
{
    T vv2 = 0;
    this->host_m_2.Set_to(vv2);

    cl_event event;
    cl_int err = CL_SUCCESS;

    void *dataPtr;
    int size = this->get_block_size_bytes();
    dataPtr =
        clEnqueueMapBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                           CL_MAP_WRITE, 0, size, 0, NULL, &event, &err);
    test_error(err, "clEnqueueMapBuffer error");

    if (!this->m_blocking)
    {
        err = clWaitForEvents(1, &event);
        test_error(err, "clWaitForEvents error");
    }

    err = clReleaseEvent(event);
    test_error(err, "clReleaseEvent error");

    update_host_mem_2();

    if ((this->buffer_mem_flag & CL_MEM_USE_HOST_PTR)
        && dataPtr != this->pHost_ptr)
    {
        log_error("Mapped host pointer difference found\n");
        return FAILURE;
    }

    if (!this->host_m_2.Equal((T *)dataPtr, this->m_nNumber_elements))
    {
        log_error("Buffer content difference found\n");
        return FAILURE;
    }

    err = clEnqueueUnmapMemObject(this->m_queue, this->m_buffer, dataPtr, 0,
                                  nullptr, nullptr);
    test_error(err, "clEnqueueUnmapMemObject error");

    // test map read
    clEnqueueMapBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                       CL_MAP_READ, 0, this->get_block_size_bytes(), 0, NULL,
                       NULL, &err);

    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapBuffer (CL_MAP_READ) on a memory object "
                  "created with the MEM_HOST_WRITE_ONLY flag should not return "
                  "CL_SUCCESS\n");
        err = FAILURE;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    return err;
}

#endif

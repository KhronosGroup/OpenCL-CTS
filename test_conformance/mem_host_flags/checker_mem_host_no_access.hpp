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
#ifndef test_conformance_check_mem_host_no_access_h
#define test_conformance_check_mem_host_no_access_h


#include "checker_mem_host_write_only.hpp"

template <class T>
class cBuffer_check_mem_host_no_access
    : public cBuffer_check_mem_host_write_only<T> {
public:
    cBuffer_check_mem_host_no_access(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue)
        : cBuffer_check_mem_host_write_only<T>(deviceID, context, queue){};

    cBuffer_check_mem_host_no_access(){};

    virtual cl_int SetupBuffer();
    virtual cl_int SetupASSubBuffer(cl_mem_flags parent_buffer_flag);
    virtual cl_int Setup_Test_Environment();

    cl_int verify_RW_Buffer();
    cl_int verify_RW_Buffer_rect();
    cl_int verify_RW_Buffer_mapping();
};

template <class T> cl_int cBuffer_check_mem_host_no_access<T>::SetupBuffer()
{
    this->m_nNumber_elements = 1000;
    T vv1 = TEST_VALUE;
    this->host_m_1.Init(this->m_nNumber_elements, vv1);

    T vv2 = 0;
    this->host_m_2.Init(this->m_nNumber_elements, vv2);

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
cl_int cBuffer_check_mem_host_no_access<T>::SetupASSubBuffer(
    cl_mem_flags parent_buffer_flag)
{
    return cBuffer_checker<T>::SetupASSubBuffer(parent_buffer_flag);
}

template <class T>
cl_int cBuffer_check_mem_host_no_access<T>::Setup_Test_Environment()
{
    cBuffer_check_mem_host_write_only<T>::Setup_Test_Environment();

    return CL_SUCCESS;
}

template <class T>
cl_int cBuffer_check_mem_host_no_access<T>::verify_RW_Buffer()
{
    cl_int err = clEnqueueReadBuffer(
        this->m_queue, this->m_buffer, this->m_blocking, 0,
        this->get_block_size_bytes(), this->host_m_1.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteBuffer on a memory object created with the "
            "CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    err = clEnqueueWriteBuffer(this->m_queue, this->m_buffer, this->m_blocking,
                               0, this->get_block_size_bytes(),
                               this->host_m_1.pData, 0, NULL, NULL);

    if (err == CL_SUCCESS)
    {
        log_error(
            "Calling clEnqueueWriteBuffer on a memory object created with the "
            "CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
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
cl_int cBuffer_check_mem_host_no_access<T>::verify_RW_Buffer_rect()
{
    this->Init_rect();
    cl_int err = CL_SUCCESS;
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
            "the CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

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
            "the CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
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
cl_int cBuffer_check_mem_host_no_access<T>::verify_RW_Buffer_mapping()
{
    cl_int err;

    void *dataPtr;
    dataPtr = clEnqueueMapBuffer(
        this->m_queue, this->m_buffer, this->m_blocking, CL_MAP_READ, 0,
        this->get_block_size_bytes(), 0, NULL, NULL, &err);
    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapBuffer (CL_MAP_READ) on a memory object "
                  "created with the CL_MEM_HOST_NO_ACCESS flag should not "
                  "return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else if (dataPtr != nullptr)
    {
        log_error("Calling clEnqueueMapBuffer (CL_MAP_READ) on a memory object "
                  "created with the CL_MEM_HOST_NO_ACCESS flag should fail "
                  "and return NULL\n");
        err = FAILURE;
        return err;
    }
    else
    {
        log_info("Test succeeded\n\n");
        err = CL_SUCCESS;
    }

    dataPtr = clEnqueueMapBuffer(
        this->m_queue, this->m_buffer, this->m_blocking, CL_MAP_WRITE, 0,
        this->get_block_size_bytes(), 0, NULL, NULL, &err);
    if (err == CL_SUCCESS)
    {
        log_error("Calling clEnqueueMapBuffer (CL_MAP_WRITE) on a memory "
                  "object created with the CL_MEM_HOST_NO_ACCESS flag should "
                  "not return CL_SUCCESS\n");
        err = FAILURE;
        return FAILURE;
    }
    else if (dataPtr != nullptr)
    {
        log_error(
            "Calling clEnqueueMapBuffer (CL_MAP_WRITE) on a memory object "
            "created with the CL_MEM_HOST_NO_ACCESS flag should fail "
            "and return NULL\n");
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

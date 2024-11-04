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
#ifndef test_conformance_checkers_h
#define test_conformance_checkers_h

#include "harness/compat.h"

#include <stdio.h>
#include <string.h>

#include "procs.h"
#include "C_host_memory_block.h"

#define TEST_VALUE 5
typedef cl_char TEST_ELEMENT_TYPE;

enum
{
    SUCCESS,
    FAILURE = -1000
};

extern const char *buffer_write_kernel_code[];

enum BUFFER_TYPE
{
    _BUFFER,
    _Sub_BUFFER
};

template <class T> class cBuffer_checker {
public:
    cBuffer_checker(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue);
    ~cBuffer_checker();

    cl_device_id m_deviceID;
    cl_context m_context;
    cl_command_queue m_queue;

    clMemWrapper m_buffer, m_buffer_parent;
    enum BUFFER_TYPE m_buffer_type;

    cl_buffer_region m_sub_buffer_region;

    cl_int err;
    cl_bool m_blocking;
    cl_mem_flags buffer_mem_flag;

    C_host_memory_block<T> host_m_0, host_m_1, host_m_2;
    int m_nNumber_elements;

    void *pData, *pData2;

    void *pHost_ptr; // the host ptr at creation

    size_t buffer_origin[3];
    size_t host_origin[3];
    size_t region[3];
    size_t buffer_row_pitch;
    size_t buffer_slice_pitch;
    size_t host_row_pitch;
    size_t host_slice_pitch;

    size_t buffer_origin_bytes[3];
    size_t host_origin_bytes[3];
    size_t region_bytes[3];
    size_t buffer_row_pitch_bytes;
    size_t buffer_slice_pitch_bytes;
    size_t host_row_pitch_bytes;
    size_t host_slice_pitch_bytes;

    cl_int CreateBuffer(cl_mem_flags buffer_mem_flag, void *pdata);
    int get_block_size_bytes()
    {
        return (int)(m_nNumber_elements * sizeof(T));
    };
    virtual cl_int SetupBuffer() = 0;

    virtual cl_int Setup_Test_Environment();

    virtual cl_int SetupASSubBuffer(cl_mem_flags parent_buffer_flag);

    virtual cl_int verify(cl_int err, cl_event &event);

    virtual cl_int Check_GetMemObjectInfo(cl_mem_flags buffer_mem_flag);

    void Init_rect(int bufforg[3], int host_org[3], int region[3],
                   int buffer_pitch[2], int host_pitch[2]);

    void Init_rect();

    virtual cl_int verify_RW_Buffer() = 0;
    virtual cl_int verify_RW_Buffer_rect() = 0;
    virtual cl_int verify_RW_Buffer_mapping() = 0;
};

template <class T>
cBuffer_checker<T>::cBuffer_checker(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue)
{
    m_nNumber_elements = 0;

    m_deviceID = deviceID;
    m_context = context;
    m_queue = queue;

    m_blocking = false;

    buffer_mem_flag = CL_MEM_READ_WRITE;
    pData = pData2 = NULL;

    buffer_origin[0] = buffer_origin[1] = buffer_origin[2] = 0;
    host_origin[0] = host_origin[1] = host_origin[2] = 0;
    region[0] = region[1] = region[2] = 0;
    buffer_row_pitch = buffer_slice_pitch = host_row_pitch = host_slice_pitch =
        0;

    buffer_origin_bytes[0] = buffer_origin_bytes[1] = buffer_origin_bytes[2] =
        0;
    host_origin_bytes[0] = host_origin_bytes[1] = host_origin_bytes[2] = 0;
    region_bytes[0] = region_bytes[1] = region_bytes[2] = 0;
    buffer_row_pitch_bytes = buffer_slice_pitch_bytes = 0;
    host_row_pitch_bytes = host_slice_pitch_bytes = 0;

    pHost_ptr = NULL;
}

template <class T> cBuffer_checker<T>::~cBuffer_checker() {}


template <class T> cl_int cBuffer_checker<T>::SetupBuffer()
{
    m_buffer_type = _BUFFER;
    return CL_SUCCESS;
}

template <class T> cl_int cBuffer_checker<T>::Setup_Test_Environment()
{
    return CL_SUCCESS;
}

template <class T>
cl_int cBuffer_checker<T>::SetupASSubBuffer(cl_mem_flags parent_buffer_flag)
{
    m_buffer_type = _Sub_BUFFER;

    int supersize = 8000;
    this->m_nNumber_elements = 1000;
    T vv1 = TEST_VALUE;

    int block_size_in_byte = (int)(supersize * sizeof(T));

    this->host_m_0.Init(supersize);

    m_buffer_parent =
        clCreateBuffer(this->m_context, parent_buffer_flag, block_size_in_byte,
                       this->host_m_0.pData, &err);
    test_error(err, "clCreateBuffer error");

    int size = this->m_nNumber_elements; // the size of subbuffer in elements

    cl_uint base_addr_align_bits;
    err = clGetDeviceInfo(m_deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                          sizeof base_addr_align_bits, &base_addr_align_bits,
                          NULL);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_MEM_BASE_ADDR_ALIGN");

    int base_addr_align_bytes = base_addr_align_bits / 8;

    int buffer_origin[3] = { base_addr_align_bytes, 0, 0 };
    int host_origin[3] = { 0, 0, 0 };
    int region[3] = { size, 1, 1 };
    int buffer_pitch[2] = { 0, 0 };
    int host_pitch[2] = { 0, 0 };
    this->Init_rect(buffer_origin, host_origin, region, buffer_pitch,
                    host_pitch);

    this->m_nNumber_elements = size; // the size of subbuffer in elements
    this->host_m_1.Init(this->m_nNumber_elements, vv1);

    this->m_sub_buffer_region.origin = this->buffer_origin_bytes[0]; // in bytes
    this->m_sub_buffer_region.size = this->region_bytes[0];

    cl_int err = CL_SUCCESS;
    err = clEnqueueReadBufferRect(
        this->m_queue, m_buffer_parent, CL_TRUE, this->buffer_origin_bytes,
        this->host_origin_bytes, this->region_bytes,
        this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
        this->host_row_pitch_bytes, this->host_slice_pitch_bytes,
        this->host_m_1.pData, 0, NULL,
        NULL); // update the mem_1

    if (err == CL_SUCCESS
        && (parent_buffer_flag
            & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)))
    {
        log_error("Calling clEnqueueReadBufferRect on a memory object created "
                  "with the CL_MEM_HOST_WRITE_ONLY flag or the "
                  "CL_MEM_HOST_NO_ACCESS flag should not return CL_SUCCESS\n");
        err = FAILURE;
        return err;
    }
    else
    {
        err = CL_SUCCESS;
    }

    cl_mem_flags f = 0;
    if (parent_buffer_flag & CL_MEM_HOST_READ_ONLY)
        f = CL_MEM_HOST_READ_ONLY;
    else if (parent_buffer_flag & CL_MEM_HOST_WRITE_ONLY)
        f = CL_MEM_HOST_WRITE_ONLY;
    else if (parent_buffer_flag & CL_MEM_HOST_NO_ACCESS)
        f = CL_MEM_HOST_NO_ACCESS;

    m_buffer =
        clCreateSubBuffer(m_buffer_parent, f, CL_BUFFER_CREATE_TYPE_REGION,
                          &(this->m_sub_buffer_region), &err);
    test_error(err, "clCreateSubBuffer error");

    if (parent_buffer_flag | CL_MEM_USE_HOST_PTR)
    {
        this->pHost_ptr = (this->host_m_0.pData
                           + this->m_sub_buffer_region.origin / sizeof(T));
    }

    T vv2 = 0;
    this->host_m_2.Init(this->m_nNumber_elements, vv2);

    return err;
}

template <class T>
cl_int cBuffer_checker<T>::verify(cl_int err, cl_event &event)
{
    return CL_SUCCESS;
}

template <class T>
cl_int cBuffer_checker<T>::CreateBuffer(cl_mem_flags buffer_mem_flag,
                                        void *pdata)
{
    cl_int err = CL_SUCCESS;
    int block_size_in_byte = m_nNumber_elements * sizeof(T);

    m_buffer = clCreateBuffer(m_context, buffer_mem_flag, block_size_in_byte,
                              pdata, &err);

    return err;
};

template <class T>
cl_int cBuffer_checker<T>::Check_GetMemObjectInfo(cl_mem_flags buffer_mem_flag)
{
    cl_int err = CL_SUCCESS;
    cl_mem_flags buffer_mem_flag_Check;
    err = clGetMemObjectInfo(this->m_buffer, CL_MEM_FLAGS, sizeof(cl_mem_flags),
                             &buffer_mem_flag_Check, NULL);

    if (buffer_mem_flag_Check != buffer_mem_flag)
    {
        log_error(
            "clGetMemObjectInfo result differs from the specified result\n");
        return err;
    }

    cl_uint count = 0;
    err = clGetMemObjectInfo(this->m_buffer, CL_MEM_REFERENCE_COUNT,
                             sizeof(cl_uint), &count, NULL);

    if (count > 1) log_info("========= buffer count %d\n", count);

    test_error(err, "clGetMemObjectInfo failed");

    return err;
}

template <class T> void cBuffer_checker<T>::Init_rect()
{
    int buffer_origin[3] = { 10, 0, 0 };
    int host_origin[3] = { 10, 0, 0 };
    int region[3] = { 8, 1, 1 };
    int buffer_pitch[2] = { 0, 0 };
    int host_pitch[2] = { 0, 0 };

    this->Init_rect(buffer_origin, host_origin, region, buffer_pitch,
                    host_pitch);
}

template <class T>
void cBuffer_checker<T>::Init_rect(int bufforg[3], int host_org[3],
                                   int region_in[3], int buffer_pitch[2],
                                   int host_pitch[2])
{
    buffer_origin[0] = bufforg[0];
    buffer_origin[1] = bufforg[1];
    buffer_origin[2] = bufforg[2];

    host_origin[0] = host_org[0];
    host_origin[1] = host_org[1];
    host_origin[2] = host_org[2];

    region[0] = region_in[0];
    region[1] = region_in[1];
    region[2] = region_in[2];

    buffer_row_pitch = buffer_pitch[0];
    buffer_slice_pitch = buffer_pitch[1];
    host_row_pitch = host_pitch[0];
    host_slice_pitch = host_pitch[1];

    int sizeof_element = sizeof(T);
    for (int k = 0; k < 3; k++)
    {
        buffer_origin_bytes[k] = buffer_origin[k] * sizeof_element;
        host_origin_bytes[k] = host_origin[k] * sizeof_element;
    }

    region_bytes[0] = region[0] * sizeof_element;
    region_bytes[1] = region[1];
    region_bytes[2] = region[2];
    buffer_row_pitch_bytes = buffer_row_pitch * sizeof_element;
    buffer_slice_pitch_bytes = buffer_slice_pitch * sizeof_element;
    host_row_pitch_bytes = host_row_pitch * sizeof_element;
    host_slice_pitch_bytes = host_slice_pitch * sizeof_element;
}

#endif

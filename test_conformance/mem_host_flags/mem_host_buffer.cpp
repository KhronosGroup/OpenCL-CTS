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

#include "checker_mem_host_read_only.hpp"
#include "checker_mem_host_write_only.hpp"
#include "checker_mem_host_no_access.hpp"

static int test_mem_host_read_only_buffer_RW(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);
    cBuffer_check_mem_host_read_only<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static int test_mem_host_read_only_buffer_RW_Rect(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_read_only<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_rect();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static int test_mem_host_read_only_buffer_RW_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_read_only<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

int test_mem_host_read_only_buffer(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flags[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_READ_ONLY,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 2; i++)
        {

            err = test_mem_host_read_only_buffer_RW(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_read_only_buffer_RW_Rect(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_read_only_buffer_RW_Mapping(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);
        }

    return err;
}

int test_mem_host_read_only_subbuffer(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    cl_mem_flags parent_buffer_mem_flags[1] = { CL_MEM_READ_WRITE
                                                | CL_MEM_USE_HOST_PTR
                                                | CL_MEM_HOST_READ_ONLY };

    cl_mem_flags buffer_mem_flags[4] = {
        0, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };

    for (int p = 0; p < 1; p++)
    {
        for (int k = 0; k < 4; k++)
            for (int i = 0; i < 2; i++)
            {
                err = test_mem_host_read_only_buffer_RW(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);

                err = test_mem_host_read_only_buffer_RW_Rect(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);

                err = test_mem_host_read_only_buffer_RW_Mapping(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);
            }
    }

    return err;
}

//=============================== Write only

static cl_int test_mem_host_write_only_buffer_RW(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_write_only<TEST_ELEMENT_TYPE> checker(
        deviceID, context, queue);

    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static cl_int test_mem_host_write_only_buffer_RW_Rect(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_write_only<TEST_ELEMENT_TYPE> checker(
        deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_rect();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static cl_int test_mem_host_write_only_buffer_RW_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_write_only<TEST_ELEMENT_TYPE> checker(
        deviceID, context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

int test_mem_host_write_only_buffer(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flags[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_WRITE_ONLY,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_WRITE_ONLY
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 2; i++)
        {
            err = test_mem_host_write_only_buffer_RW(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_write_only_buffer_RW_Rect(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_write_only_buffer_RW_Mapping(
                deviceID, context, queue, blocking[i], buffer_mem_flags[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);
        }

    return err;
}

int test_mem_host_write_only_subbuffer(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    cl_mem_flags parent_buffer_mem_flags[1] = { CL_MEM_READ_WRITE
                                                | CL_MEM_USE_HOST_PTR
                                                | CL_MEM_HOST_WRITE_ONLY };

    cl_mem_flags buffer_mem_flags[4] = {
        0, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };

    for (int p = 0; p < 1; p++)
    {
        for (int m = 0; m < 4; m++)
        {
            for (int i = 0; i < 2; i++)
            {
                err = test_mem_host_write_only_buffer_RW(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[m],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);

                err = test_mem_host_write_only_buffer_RW_Rect(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[m],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);

                err = test_mem_host_write_only_buffer_RW_Mapping(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[m],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
                test_error(err, __FUNCTION__);
            }
        }
    }

    return err;
}

//=====================  NO ACCESS

static cl_int test_mem_host_no_access_buffer_RW(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_no_access<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;

    cl_int err = CL_SUCCESS;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static cl_int test_mem_host_no_access_buffer_RW_Rect(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_no_access<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);
    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

static cl_int test_mem_host_no_access_buffer_RW_Mapping(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    cl_bool blocking, cl_mem_flags buffer_mem_flag,
    cl_mem_flags parent_buffer_flag, enum BUFFER_TYPE buffer_type)
{
    log_info("%s\n", __FUNCTION__);

    cBuffer_check_mem_host_no_access<TEST_ELEMENT_TYPE> checker(deviceID,
                                                                context, queue);

    checker.m_blocking = blocking;
    checker.buffer_mem_flag = buffer_mem_flag;
    cl_int err;
    switch (buffer_type)
    {
        case _BUFFER: err = checker.SetupBuffer(); break;
        case _Sub_BUFFER:
            err = checker.SetupASSubBuffer(parent_buffer_flag);
            break;
    }

    test_error(err, __FUNCTION__);
    checker.Setup_Test_Environment();
    err = checker.verify_RW_Buffer_mapping();
    test_error(err, __FUNCTION__);
    clFinish(queue);

    return err;
}

int test_mem_host_no_access_buffer(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    cl_mem_flags buffer_mem_flag[2] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 2; i++)
        {
            err = test_mem_host_no_access_buffer_RW(
                deviceID, context, queue, blocking[i], buffer_mem_flag[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_no_access_buffer_RW_Rect(
                deviceID, context, queue, blocking[i], buffer_mem_flag[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);

            err = test_mem_host_no_access_buffer_RW_Mapping(
                deviceID, context, queue, blocking[i], buffer_mem_flag[k], 0,
                _BUFFER);
            test_error(err, __FUNCTION__);
        }

    return err;
}

int test_mem_host_no_access_subbuffer(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    cl_mem_flags parent_buffer_mem_flags[3] = {
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS
    };

    cl_mem_flags buffer_mem_flags[4] = {
        0, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
    };

    cl_int err = CL_SUCCESS;

    cl_bool blocking[2] = { CL_TRUE, CL_FALSE };
    for (int p = 0; p < 3; p++)
    {
        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 2; i++)
            {
                err += test_mem_host_no_access_buffer_RW(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);

                err += test_mem_host_no_access_buffer_RW_Rect(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);

                err += test_mem_host_no_access_buffer_RW_Mapping(
                    deviceID, context, queue, blocking[i], buffer_mem_flags[k],
                    parent_buffer_mem_flags[p], _Sub_BUFFER);
            }
        }
    }

    return err;
}

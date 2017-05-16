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
#include "harness.h"

#define ADD_BUFFER_PROPERTIES(w, x, y, z) \
    { w, x, y, z, #x, #y, #z, }

BufferProperties bufferProperties[] =
{
    ADD_BUFFER_PROPERTIES(     0x110, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),
    ADD_BUFFER_PROPERTIES(    0x1100, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),
    ADD_BUFFER_PROPERTIES(    0x8000, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),

    ADD_BUFFER_PROPERTIES(   0x7FFFF, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(  0x110000, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(  0x110000, D3D11_BIND_STREAM_OUTPUT,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(  0x110001, D3D11_BIND_STREAM_OUTPUT,   D3D11_USAGE_DEFAULT, 0),

    ADD_BUFFER_PROPERTIES(      0x11, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(      0x11, D3D11_BIND_INDEX_BUFFER,    D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),
    ADD_BUFFER_PROPERTIES(     0x121, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(    0x1234, D3D11_BIND_INDEX_BUFFER,    D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES(   0x12345, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),
    ADD_BUFFER_PROPERTIES(  0x123456, D3D11_BIND_INDEX_BUFFER,    D3D11_USAGE_DEFAULT, 0),
#if 0 // avoid large sizes on automation
    ADD_BUFFER_PROPERTIES( 0x1234567, D3D11_BIND_INDEX_BUFFER,    D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE),

    ADD_BUFFER_PROPERTIES( 0x4000000, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES( 0x4000004, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES( 0x4000008, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES( 0x4000011, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
    ADD_BUFFER_PROPERTIES( 0x4000014, D3D11_BIND_VERTEX_BUFFER,   D3D11_USAGE_DEFAULT, 0),
#endif
};
UINT bufferPropertyCount = sizeof(bufferProperties)/sizeof(bufferProperties[0]);

void SubTestBuffer(
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC,
    const BufferProperties* props)
{
    ID3D11Buffer* pBuffer = NULL;
    HRESULT hr = S_OK;
    cl_mem mem = NULL;
    cl_int result = CL_SUCCESS;

    HarnessD3D11_TestBegin("Buffer: Size=%d, BindFlags=%s, Usage=%s, CPUAccess=%s",
        props->ByteWidth,
        props->name_BindFlags,
        props->name_Usage,
        props->name_CPUAccess);

    // create the D3D11 resource
    {
        D3D11_BUFFER_DESC desc = {0};
        desc.ByteWidth = props->ByteWidth;
        desc.Usage = props->Usage;
        desc.CPUAccessFlags = props->CPUAccess;
        desc.BindFlags = props->BindFlags;
        desc.MiscFlags = 0;
        hr = pDevice->CreateBuffer(&desc, NULL, &pBuffer);
        TestRequire(SUCCEEDED(hr), "Creating vertex buffer failed!");
    }

    // populate the D3D11 resource with data
    {
        ID3D11Buffer* pStagingBuffer = NULL;
        char *pStagingData = NULL;
        D3D11_MAPPED_SUBRESOURCE map = {0};

        // create a staging buffer to use to copy data to the D3D buffer
        D3D11_BUFFER_DESC desc = {0};
        desc.ByteWidth      = 16;
        desc.Usage          = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE|D3D11_CPU_ACCESS_READ;
        desc.BindFlags      = 0;
        desc.MiscFlags      = 0;
        hr = pDevice->CreateBuffer(&desc, NULL, &pStagingBuffer);
        TestRequire(SUCCEEDED(hr), "Creating staging vertex buffer failed!");

        // populate the staging buffer
        hr = pDC->Map(
            pStagingBuffer,
            0,
            D3D11_MAP_READ_WRITE,
            0,
            &map);
        TestRequire(SUCCEEDED(hr), "Map failed!");
        memcpy(map.pData, "abcdXXXXxxxx1234", 16);
        pDC->Unmap(pStagingBuffer, 0);
        TestRequire(SUCCEEDED(hr), "Unmap failed!");

        // copy 'abcdXXXX' to the front of the buffer and 'xxxx1234' to the back
        D3D11_BOX box = {0};
        box.front   = 0;
        box.back    = 1;
        box.top     = 0;
        box.bottom  = 1;

        box.left    = 0;
        box.right   = 8;
        pDC->CopySubresourceRegion(
            pBuffer,
            0,
            0,
            0,
            0,
            pStagingBuffer,
            0,
            &box);
        box.left    = 8;
        box.right   = 16;
        pDC->CopySubresourceRegion(
            pBuffer,
            0,
            props->ByteWidth-8,
            0,
            0,
            pStagingBuffer,
            0,
            &box);
        pStagingBuffer->Release();
    }

    // share the resource with OpenCL
    {
        mem = clCreateFromD3D11BufferKHR(
            context,
            0,
            pBuffer,
            &result);
        TestRequire(CL_SUCCESS == result, "clCreateFromD3D11BufferKHR failed");
    }

    // validate the OpenCL mem obj's properties
    {
        ID3D11Resource* clResource = NULL;
        result = clGetMemObjectInfo(
            mem,
            CL_MEM_D3D11_RESOURCE_KHR,
            sizeof(clResource),
            &clResource,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetMemObjectInfo for CL_MEM_D3D11_RESOURCE_KHR failed.");
        TestRequire(clResource == pBuffer, "clGetMemObjectInfo for CL_MEM_D3D11_RESOURCE_KHR returned incorrect value.");
    }

    // acquire the resource from OpenCL
    {
        result = clEnqueueAcquireD3D11ObjectsKHR(
            command_queue,
            1,
            &mem,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueAcquireD3D11ObjectsKHR failed.");
    }

    // read+write data from the buffer in OpenCL
    {
        // overwrite the 'XXXX' with '1234' and the 'xxxx' with 'abcd' so we now have
        // 'abcd1234' at the beginning and end of the buffer
        result = clEnqueueCopyBuffer(
            command_queue,
            mem,
            mem,
            0,
            props->ByteWidth-8,
            4,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueCopyBuffer failed.");

        result = clEnqueueCopyBuffer(
            command_queue,
            mem,
            mem,
            props->ByteWidth-4,
            4,
            4,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueCopyBuffer failed.");
    }

    // release the resource from OpenCL
    {
        result = clEnqueueReleaseD3D11ObjectsKHR(
            command_queue,
            1,
            &mem,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueReleaseD3D11ObjectsKHR failed.");
    }

    // read data in D3D
    {
        ID3D11Buffer* pStagingBuffer = NULL;
        char *pStagingData = NULL;
        D3D11_MAPPED_SUBRESOURCE map = {0};

        // create a staging buffer to read the data back
        D3D11_BUFFER_DESC desc = {0};
        desc.ByteWidth      = 16;
        desc.Usage          = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE|D3D11_CPU_ACCESS_READ;
        desc.BindFlags      = 0;
        desc.MiscFlags      = 0;
        hr = pDevice->CreateBuffer(&desc, NULL, &pStagingBuffer);
        TestRequire(SUCCEEDED(hr), "Creating staging vertex buffer failed!");

        // make sure the staging buffer doesn't get stale data
        hr = pDC->Map(
            pStagingBuffer,
            0,
            D3D11_MAP_READ_WRITE,
            0,
            &map);
        TestRequire(SUCCEEDED(hr), "Map failed!");
        memset(map.pData, 0, 16);
        pDC->Unmap(pStagingBuffer, 0);
        TestRequire(SUCCEEDED(hr), "Unmap failed!");

        // copy the 'abcd1234' from the front and back of the buffer to the staging buffer
        D3D11_BOX box = {0};
        box.front   = 0;
        box.back    = 1;
        box.top     = 0;
        box.bottom  = 1;

        box.left    = 0;
        box.right   = 8;
        pDC->CopySubresourceRegion(
            pStagingBuffer,
            0,
            0,
            0,
            0,
            pBuffer,
            0,
            &box);
        box.left    = props->ByteWidth-8;
        box.right   = props->ByteWidth;
        pDC->CopySubresourceRegion(
            pStagingBuffer,
            0,
            8,
            0,
            0,
            pBuffer,
            0,
            &box);
        TestRequire(SUCCEEDED(hr), "CopySubresourceRegion failed!");

        // verify that we got the 'abcd1234'
        hr = pDC->Map(
            pStagingBuffer,
            0,
            D3D11_MAP_READ_WRITE,
            0,
            &map);
        TestRequire(SUCCEEDED(hr), "Map failed!");
        TestRequire(!memcmp(map.pData, "abcd1234abcd1234", 16), "Data was not accurately");
        pDC->Unmap(pStagingBuffer, 0);
        TestRequire(SUCCEEDED(hr), "Unmap failed!");

        pStagingBuffer->Release();
    }

Cleanup:

    if (pBuffer)
    {
        pBuffer->Release();
    }
    if (mem)
    {
        clReleaseMemObject(mem);
    }

    HarnessD3D11_TestEnd();
}


void TestDeviceBuffer(
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC)
{
    for (UINT i = 0; i < bufferPropertyCount; ++i)
    {
        SubTestBuffer(
            context,
            command_queue,
            pDevice,
            pDC,
            &bufferProperties[i]);
    }
}


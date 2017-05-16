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
#define _CRT_SECURE_NO_WARNINGS
#include "harness.h"

void SubTestMiscMultipleCreates(
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice)
{
    cl_mem mem[5] = {NULL, NULL, NULL, NULL, NULL};
    ID3D11Buffer* pBuffer = NULL;
    ID3D11Texture2D* pTexture = NULL;
    HRESULT hr = S_OK;

    cl_int result = CL_SUCCESS;

    HarnessD3D11_TestBegin("Misc: Multiple Creates");

    // create the D3D11 resources
    {
        D3D11_TEXTURE2D_DESC desc;
        memset(&desc, 0, sizeof(desc) );
        desc.Width      = 256;
        desc.Height     = 256;
        desc.MipLevels  = 4;
        desc.ArraySize  = 4;
        desc.Format     = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        hr = pDevice->CreateTexture2D(&desc, NULL, &pTexture);
        TestRequire(SUCCEEDED(hr), "Failed to create texture.");
    }

    // create the D3D11 buffer
    {
        D3D11_BUFFER_DESC desc = {0};
        desc.ByteWidth = 1124;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
        desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        desc.MiscFlags = 0;
        hr = pDevice->CreateBuffer(&desc, NULL, &pBuffer);
        TestRequire(SUCCEEDED(hr), "Creating vertex buffer failed!");
    }

    mem[0] = clCreateFromD3D11BufferKHR(
        context,
        0,
        pBuffer,
        &result);
    TestRequire(result == CL_SUCCESS, "clCreateFromD3D11BufferKHR");

    mem[1] = clCreateFromD3D11BufferKHR(
        context,
        0,
        pBuffer,
        &result);
    TestRequire(result == CL_INVALID_D3D11_RESOURCE_KHR, "clCreateFromD3D11BufferKHR succeeded when it shouldn't");

    mem[2] = clCreateFromD3D11Texture2DKHR(
        context,
        0,
        pTexture,
        1,
        &result);
    TestRequire(result == CL_SUCCESS, "clCreateFromD3D11Texture2DKHR failed");

    mem[3] = clCreateFromD3D11Texture2DKHR(
        context,
        0,
        pTexture,
        1,
        &result);
    TestRequire(result == CL_INVALID_D3D11_RESOURCE_KHR, "clCreateFromD3D11Texture2DKHR succeeded when it shouldn't");

    mem[4] = clCreateFromD3D11Texture2DKHR(
        context,
        0,
        pTexture,
        16,
        &result);
    TestRequire(result == CL_INVALID_VALUE, "clCreateFromD3D11Texture2DKHR succeeded when it shouldn't");


Cleanup:

    for (UINT i = 0; i < 4; ++i)
    {
        if (mem[i])
        {
            clReleaseMemObject(mem[i]);
        }
    }
    if (pBuffer)
    {
        pBuffer->Release();
    }
    if (pTexture)
    {
        pTexture->Release();
    }

    HarnessD3D11_TestEnd();
}

void SubTestMiscAcquireRelease(
    cl_device_id  device,
    cl_context context,
    ID3D11Device* pDevice)
{
    ID3D11Buffer* pBuffer = NULL;
    ID3D11Texture2D* pTexture = NULL;
    HRESULT hr = S_OK;

    cl_int result = CL_SUCCESS;
    cl_mem mem[2] = {NULL, NULL};

    HarnessD3D11_TestBegin("Misc: Acquire Release");


    // create the D3D11 resources
    {
        D3D11_TEXTURE2D_DESC desc;
        memset(&desc, 0, sizeof(desc) );
        desc.Width      = 256;
        desc.Height     = 256;
        desc.MipLevels  = 4;
        desc.ArraySize  = 4;
        desc.Format     = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        hr = pDevice->CreateTexture2D(&desc, NULL, &pTexture);
        TestRequire(SUCCEEDED(hr), "Failed to create texture.");
    }

    // create the D3D11 buffer
    {
        D3D11_BUFFER_DESC desc = {0};
        desc.ByteWidth = 1124;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
        desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        desc.MiscFlags = 0;
        hr = pDevice->CreateBuffer(&desc, NULL, &pBuffer);
        TestRequire(SUCCEEDED(hr), "Creating vertex buffer failed!");
    }

    // create cl_mem objects for the resources
    mem[0] = clCreateFromD3D11BufferKHR(
        context,
        0,
        pBuffer,
        &result);
    TestRequire(result == CL_SUCCESS, "clCreateFromD3D11BufferKHR");
    mem[1] = clCreateFromD3D11Texture2DKHR(
        context,
        0,
        pTexture,
        1,
        &result);
    TestRequire(result == CL_SUCCESS, "clCreateFromD3D11Texture2DKHR failed");

Cleanup:
    for (UINT i = 0; i < 2; ++i)
    {
        if (mem[i])
        {
            clReleaseMemObject(mem[i]);
        }
    }
    if (pBuffer)
    {
        pBuffer->Release();
    }
    if (pTexture)
    {
        pTexture->Release();
    }

    HarnessD3D11_TestEnd();
}

void TestDeviceMisc(
    cl_device_id device,
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice)
{
    SubTestMiscMultipleCreates(
        context,
        command_queue,
        pDevice);

    SubTestMiscAcquireRelease(
        device,
        context,
        pDevice);
}



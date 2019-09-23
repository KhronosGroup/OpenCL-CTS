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
#include "testBase.h"


#include "harness/conversions.h"
#include "harness/typeWrappers.h"

typedef struct _clState
{
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;

    cl_program m_program;
    cl_kernel m_kernel;
    size_t m_numThreads;
} clState;

clState * newClState(cl_device_id device, cl_context context, cl_command_queue queue);
clState * destroyClState(clState * pState);

int clStateMakeProgram(clState * pState, const char * prog,
               const char * kernelName);
void clStateDestroyProgramAndKernel(clState * pState);

int runKernel(clState * pState, size_t numThreads);

typedef struct _bufferStruct
{
    void * m_pIn;
    void * m_pOut;

    cl_mem m_outBuffer;
    cl_mem m_inBuffer;

    size_t m_bufSizeIn, m_bufSizeOut;
} bufferStruct;


bufferStruct * newBufferStruct(size_t inSize, size_t outSize, clState * pClState);

bufferStruct * destroyBufferStruct(bufferStruct * destroyMe, clState * pClState);

void initContents(bufferStruct * pBufferStruct, clState * pClState,
             size_t typeSize,
             size_t vecWidth);

int pushArgs(bufferStruct * pBufferStruct, clState * pClState);
int retrieveResults(bufferStruct * pBufferStruct, clState * pClState);

int checkCorrectness(bufferStruct * pBufferStruct, clState * pClState,
             size_t typeSize,
             size_t vecWidth);

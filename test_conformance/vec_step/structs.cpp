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
#include "structs.h"


#include "defines.h"

/** typedef struct _bufferStruct
 {
 void * m_pIn;
 void * m_pOut;

 cl_mem m_outBuffer;
 cl_mem m_inBuffer;

 size_t m_bufSize;
 } bufferStruct;
 */


clState * newClState(cl_device_id device, cl_context context, cl_command_queue queue)
{
    clState * pResult = (clState *)malloc(sizeof(clState));

    pResult->m_device = device;
    pResult->m_context = context;
    pResult->m_queue = queue;

    pResult->m_kernel = NULL; pResult->m_program = NULL;
    return pResult;
}

clState * destroyClState(clState * pState)
{
    clStateDestroyProgramAndKernel(pState);
    free(pState);
    return NULL;
}


int clStateMakeProgram(clState * pState, const char * prog,
                       const char * kernelName)
{
    const char * srcArr[1] = {NULL};
    srcArr[0] = prog;
    int err = create_single_kernel_helper(pState->m_context,
                                          &(pState->m_program),
                                          &(pState->m_kernel),
                                          1, srcArr, kernelName );
    return err;
}

int runKernel(clState * pState, size_t numThreads) {
    int err;
    pState->m_numThreads = numThreads;
    err = clEnqueueNDRangeKernel(pState->m_queue, pState->m_kernel,
                                 1, NULL, &(pState->m_numThreads),
                                 NULL, 0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel returned %d (%x)\n",
                  err, err);
        return -1;
    }
    return 0;
}


void clStateDestroyProgramAndKernel(clState * pState)
{
    if(pState->m_kernel != NULL) {
        clReleaseKernel( pState->m_kernel );
        pState->m_kernel = NULL;
    }
    if(pState->m_program != NULL) {
        clReleaseProgram( pState->m_program );
        pState->m_program = NULL;
    }
}

bufferStruct * newBufferStruct(size_t inSize, size_t outSize, clState * pClState) {
    int error;
    bufferStruct * pResult = (bufferStruct *)malloc(sizeof(bufferStruct));

    pResult->m_bufSizeIn = inSize;
    pResult->m_bufSizeOut = outSize;

    pResult->m_pIn = malloc(inSize);
    pResult->m_pOut = malloc(outSize);

    pResult->m_inBuffer = clCreateBuffer(pClState->m_context, CL_MEM_READ_ONLY,
                                         inSize, NULL, &error);
    if( pResult->m_inBuffer == NULL )
    {
        vlog_error( "clCreateArray failed for input (%d)\n", error );
        return destroyBufferStruct(pResult, pClState);
    }

    pResult->m_outBuffer = clCreateBuffer( pClState->m_context,
                                          CL_MEM_WRITE_ONLY,
                                          outSize,
                                          NULL,
                                          &error );
    if( pResult->m_outBuffer == NULL )
    {
        vlog_error( "clCreateArray failed for output (%d)\n", error );
        return destroyBufferStruct(pResult, pClState);
    }

    return pResult;
}

bufferStruct * destroyBufferStruct(bufferStruct * destroyMe, clState * pClState) {
    if(destroyMe)
    {
        if(destroyMe->m_outBuffer != NULL) {
            clReleaseMemObject(destroyMe->m_outBuffer);
            destroyMe->m_outBuffer = NULL;
        }
        if(destroyMe->m_inBuffer != NULL) {
            clReleaseMemObject(destroyMe->m_inBuffer);
            destroyMe->m_inBuffer = NULL;
        }
        if(destroyMe->m_pIn != NULL) {
            free(destroyMe->m_pIn);
            destroyMe->m_pIn = NULL;
        }
        if(destroyMe->m_pOut != NULL) {
            free(destroyMe->m_pOut);
            destroyMe->m_pOut = NULL;
        }

        free((void *)destroyMe);
        destroyMe = NULL;
    }
    return destroyMe;
}

void initContents(bufferStruct * pBufferStruct, clState * pClState,
                  size_t typeSize,
                  size_t countIn, size_t countOut )
{
    size_t i;

    uint64_t start = 0;

    switch(typeSize)
    {
        case 1: {
            uint8_t* ub = (uint8_t *)(pBufferStruct->m_pIn);
            for (i=0; i < countIn; ++i)
            {
                ub[i] = (uint8_t)start++;
            }
            break;
        }
        case 2: {
            uint16_t* us = (uint16_t *)(pBufferStruct->m_pIn);
            for (i=0; i < countIn; ++i)
            {
                us[i] = (uint16_t)start++;
            }
            break;
        }
        case 4: {
            if (!g_wimpyMode) {
                uint32_t* ui = (uint32_t *)(pBufferStruct->m_pIn);
                for (i=0; i < countIn; ++i) {
                    ui[i] = (uint32_t)start++;
                }
            }
            else {
                // The short test doesn't iterate over the entire 32 bit space so
                // we alternate between positive and negative values
                int32_t* ui = (int32_t *)(pBufferStruct->m_pIn);
                int32_t sign = 1;
                for (i=0; i < countIn; ++i, ++start) {
                    ui[i] = (int32_t)start*sign;
                    sign = sign * -1;
                }
            }
            break;
        }
        case 8: {
            // We don't iterate over the entire space of 64 bit so for the
            // selects, we want to test positive and negative values
            int64_t* ll = (int64_t *)(pBufferStruct->m_pIn);
            int64_t sign = 1;
            for (i=0; i < countIn; ++i, ++start) {
                ll[i] = start*sign;
                sign = sign * -1;
            }
            break;
        }
        default: {
            log_error("invalid type size %x\n", (int)typeSize);
        }
    }
    // pBufferStruct->m_bufSizeIn
    // pBufferStruct->m_bufSizeOut
}

int pushArgs(bufferStruct * pBufferStruct, clState * pClState)
{
    int err;
    err = clEnqueueWriteBuffer(pClState->m_queue, pBufferStruct->m_inBuffer,
                               CL_TRUE, 0, pBufferStruct->m_bufSizeIn,
                               pBufferStruct->m_pIn, 0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    err = clSetKernelArg(pClState->m_kernel, 0,
                         sizeof(pBufferStruct->m_inBuffer), // pBufferStruct->m_bufSizeIn,
                         &(pBufferStruct->m_inBuffer));
    if(err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed, first arg (0)\n");
        return -1;
    }

    err = clSetKernelArg(pClState->m_kernel, 1,
                         sizeof(pBufferStruct->m_outBuffer), // pBufferStruct->m_bufSizeOut,
                         &(pBufferStruct->m_outBuffer));
    if(err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed, second arg (1)\n");
        return -1;
    }

    return 0;
}

int retrieveResults(bufferStruct * pBufferStruct, clState * pClState)
{
    int err;
    err = clEnqueueReadBuffer(pClState->m_queue, pBufferStruct->m_outBuffer,
                              CL_TRUE, 0, pBufferStruct->m_bufSizeOut,
                              pBufferStruct->m_pOut, 0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }
    return 0;
}

int checkCorrectness(bufferStruct * pBufferStruct, clState * pClState,
                     size_t typeSize,
                     size_t vecWidth)
{
    size_t i;
    cl_int targetSize = (cl_int) vecWidth;
    cl_int * targetArr = (cl_int *)(pBufferStruct->m_pOut);
    if(targetSize == 3)
    {
        targetSize = 4; // hack for 4-aligned vec3 types
    }
    for(i = 0; i < pClState->m_numThreads; ++i)
    {
        if(targetArr[i] != targetSize)
        {
            vlog_error("Error %ld (of %ld).  Expected %d, got %d\n",
                       i, pClState->m_numThreads,
                       targetSize, targetArr[i]);
            return -1;
        }
    }
    return 0;
}

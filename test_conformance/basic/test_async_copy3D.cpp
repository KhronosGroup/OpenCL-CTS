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
#include "../../test_common/harness/compat.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../../test_common/harness/conversions.h"
#include "procs.h"

static const char *async_global_to_local_kernel3D = R"OpenCLC(
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable
%s // optional pragma string

__kernel void test_fn(const __global %s *src, __global %s *dst, __local %s *localBuffer,
                      int numElementsPerLine, int numLines, int planesCopiesPerWorkgroup,
                      int planesCopiesPerWorkItem, int srcLineStride,
                      int dstLineStride, int srcPlaneStride, int dstPlaneStride ) {
  // Zero the local storage first
  for (int i = 0; i < planesCopiesPerWorkItem; i++) {
    for (int j = 0; j < numLines; j++) {
      for (int k = 0; k < numElementsPerLine; k++) {
        const int index = (get_local_id(0) * planesCopiesPerWorkItem + i) * dstPlaneStride + j * dstLineStride + k;
        localBuffer[index] = (%s)(%s)0;
      }
    }
  }

  // Do this to verify all kernels are done zeroing the local buffer before we try the copy
  barrier(CLK_LOCAL_MEM_FENCE);

  event_t event = async_work_group_copy_3D3D(localBuffer, 0, src,
    planesCopiesPerWorkgroup * get_group_id(0) * srcPlaneStride,
    sizeof(%s), (size_t)numElementsPerLine, (size_t)numLines,
    planesCopiesPerWorkgroup, srcLineStride, srcPlaneStride, dstLineStride,
    dstPlaneStride, 0);

  // Wait for the copy to complete, then verify by manually copying to the dest
  wait_group_events(1, &event);

  for (int i = 0; i < planesCopiesPerWorkItem; i++) {
    for (int j = 0; j < numLines; j++) {
      for(int k = 0; k < numElementsPerLine; k++) {
        const int local_index = (get_local_id(0) * planesCopiesPerWorkItem + i) * dstPlaneStride + j * dstLineStride + k;
        const int global_index = (get_global_id(0) * planesCopiesPerWorkItem + i) * dstPlaneStride + j * dstLineStride + k;
        dst[global_index] = localBuffer[local_index];
      }
    }
  }
}
)OpenCLC";

static const char *async_local_to_global_kernel3D = R"OpenCLC(
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable
%s // optional pragma string

__kernel void test_fn(const __global %s *src, __global %s *dst, __local %s *localBuffer,
                      int numElementsPerLine, int numLines, int planesCopiesPerWorkgroup,
                      int planesCopiesPerWorkItem, int srcLineStride,
                      int dstLineStride, int srcPlaneStride, int dstPlaneStride) {
  // Zero the local storage first
  for (int i = 0; i < planesCopiesPerWorkItem; i++) {
    for (int j = 0; j < numLines; j++) {
      for (int k = 0; k < numElementsPerLine; k++) {
        const int index = (get_local_id(0) * planesCopiesPerWorkItem + i) * srcPlaneStride + j * srcLineStride + k;
        localBuffer[index] = (%s)(%s)0;
      }
    }
  }

  // Do this to verify all kernels are done zeroing the local buffer before we try the copy
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i=0; i < planesCopiesPerWorkItem; i++) {
    for (int j=0; j < numLines; j++) {
      for (int k=0; k < numElementsPerLine; k++) {
        const int local_index = (get_local_id(0) * planesCopiesPerWorkItem + i) * srcPlaneStride + j * srcLineStride + k;
        const int global_index = (get_global_id(0) * planesCopiesPerWorkItem + i) * srcPlaneStride + j*srcLineStride + k;
        localBuffer[local_index] = src[global_index];
      }
    }
  }

  // Do this to verify all kernels are done copying to the local buffer before we try the copy
  barrier(CLK_LOCAL_MEM_FENCE);

  event_t event = async_work_group_copy_3D3D(dst,
    planesCopiesPerWorkgroup * get_group_id(0) * dstPlaneStride, localBuffer, 0,
    sizeof(%s), (size_t)numElementsPerLine, (size_t)numLines, planesCopiesPerWorkgroup,
    srcLineStride, srcPlaneStride, dstLineStride, dstPlaneStride, 0);

  wait_group_events(1, &event);
}
)OpenCLC";

int test_copy3D(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, const char *kernelCode,
                ExplicitType vecType, int vecSize, int srcLineMargin,
                int dstLineMargin, int srcPlaneMargin, int dstPlaneMargin,
                bool localIsDst)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    size_t threads[1], localThreads[1];
    void *inBuffer, *outBuffer, *outBufferCopy;
    MTdata d;
    char vecNameString[64];
    vecNameString[0] = 0;
    if (vecSize == 1)
        sprintf(vecNameString, "%s", get_explicit_type_name(vecType));
    else
        sprintf(vecNameString, "%s%d", get_explicit_type_name(vecType),
                vecSize);

    size_t elementSize = get_explicit_type_size(vecType) * vecSize;
    log_info("Testing %s with srcLineMargin = %d, dstLineMargin = %d, "
             "srcPlaneMargin = %d, dstPlaneMargin = %d\n",
             vecNameString, srcLineMargin, dstLineMargin, srcPlaneMargin,
             dstPlaneMargin);

    cl_long max_local_mem_size;
    error =
        clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_size), &max_local_mem_size, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_LOCAL_MEM_SIZE failed.");

    cl_long max_global_mem_size;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(max_global_mem_size), &max_global_mem_size,
                            NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_GLOBAL_MEM_SIZE failed.");

    cl_long max_alloc_size;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(max_alloc_size), &max_alloc_size, NULL);
    test_error(error,
               "clGetDeviceInfo for CL_DEVICE_MAX_MEM_ALLOC_SIZE failed.");

    if (max_alloc_size > max_global_mem_size / 2)
        max_alloc_size = max_global_mem_size / 2;

    unsigned int num_of_compute_devices;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(num_of_compute_devices),
                            &num_of_compute_devices, NULL);
    test_error(error,
               "clGetDeviceInfo for CL_DEVICE_MAX_COMPUTE_UNITS failed.");

    char programSource[4096];
    programSource[0] = 0;
    char *programPtr;

    sprintf(programSource, kernelCode,
            vecType == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                               : "",
            vecNameString, vecNameString, vecNameString, vecNameString,
            get_explicit_type_name(vecType), vecNameString, vecNameString);
    // log_info("program: %s\n", programSource);
    programPtr = programSource;

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        (const char **)&programPtr, "test_fn");
    test_error(error, "Unable to create testing kernel");

    size_t max_workgroup_size;
    error = clGetKernelWorkGroupInfo(
        kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_workgroup_size),
        &max_workgroup_size, NULL);
    test_error(
        error,
        "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE.");

    size_t max_local_workgroup_size[3];
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(max_local_workgroup_size),
                            max_local_workgroup_size, NULL);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Pick the minimum of the device and the kernel
    if (max_workgroup_size > max_local_workgroup_size[0])
        max_workgroup_size = max_local_workgroup_size[0];

    const size_t numElementsPerLine = 10;
    const cl_int dstLineStride = numElementsPerLine + dstLineMargin;
    const cl_int srcLineStride = numElementsPerLine + srcLineMargin;

    const size_t numLines = 13;
    const cl_int dstPlaneStride = (numLines * dstLineStride) + dstPlaneMargin;
    const cl_int srcPlaneStride = (numLines * srcLineStride) + srcPlaneMargin;

    elementSize =
        get_explicit_type_size(vecType) * ((vecSize == 3) ? 4 : vecSize);
    const size_t planesCopiesPerWorkItem = 2;
    const size_t localStorageSpacePerWorkitem = elementSize
        * planesCopiesPerWorkItem
        * (localIsDst ? dstPlaneStride : srcPlaneStride);
    size_t maxLocalWorkgroupSize =
        (((int)max_local_mem_size / 2) / localStorageSpacePerWorkitem);

    // Calculation can return 0 on embedded devices due to 1KB local mem limit
    if (maxLocalWorkgroupSize == 0)
    {
        maxLocalWorkgroupSize = 1;
    }

    size_t localWorkgroupSize = maxLocalWorkgroupSize;
    if (maxLocalWorkgroupSize > max_workgroup_size)
        localWorkgroupSize = max_workgroup_size;

    const size_t maxTotalPlanesIn =
        ((max_alloc_size / elementSize) + srcPlaneMargin) / srcPlaneStride;
    const size_t maxTotalPlanesOut =
        ((max_alloc_size / elementSize) + dstPlaneMargin) / dstPlaneStride;
    const size_t maxTotalPlanes = std::min(maxTotalPlanesIn, maxTotalPlanesOut);
    const size_t maxLocalWorkgroups =
        maxTotalPlanes / (localWorkgroupSize * planesCopiesPerWorkItem);

    const size_t localBufferSize =
        localWorkgroupSize * localStorageSpacePerWorkitem
        - (localIsDst ? dstPlaneMargin : srcPlaneMargin);
    const size_t numberOfLocalWorkgroups =
        std::min(1111, (int)maxLocalWorkgroups);
    const size_t totalPlanes =
        numberOfLocalWorkgroups * localWorkgroupSize * planesCopiesPerWorkItem;
    const size_t inBufferSize = elementSize
        * (totalPlanes * numLines * srcLineStride
           + (totalPlanes - 1) * srcPlaneMargin);
    const size_t outBufferSize = elementSize
        * (totalPlanes * numLines * dstLineStride
           + (totalPlanes - 1) * dstPlaneMargin);
    const size_t globalWorkgroupSize =
        numberOfLocalWorkgroups * localWorkgroupSize;

    inBuffer = (void *)malloc(inBufferSize);
    outBuffer = (void *)malloc(outBufferSize);
    outBufferCopy = (void *)malloc(outBufferSize);

    const cl_int planesCopiesPerWorkItemInt =
        static_cast<cl_int>(planesCopiesPerWorkItem);
    const cl_int numElementsPerLineInt =
        static_cast<cl_int>(numElementsPerLine);
    const cl_int numLinesInt = static_cast<cl_int>(numLines);
    const cl_int planesCopiesPerWorkgroup =
        static_cast<cl_int>(planesCopiesPerWorkItem * localWorkgroupSize);

    log_info("Global: %d, local %d, local buffer %db, global in buffer %db, "
             "global out buffer %db, each work group will copy %d planes and "
             "each work item item will copy %d planes.\n",
             (int)globalWorkgroupSize, (int)localWorkgroupSize,
             (int)localBufferSize, (int)inBufferSize, (int)outBufferSize,
             planesCopiesPerWorkgroup, planesCopiesPerWorkItemInt);

    threads[0] = globalWorkgroupSize;
    localThreads[0] = localWorkgroupSize;

    d = init_genrand(gRandomSeed);
    generate_random_data(
        vecType, inBufferSize / get_explicit_type_size(vecType), d, inBuffer);
    generate_random_data(
        vecType, outBufferSize / get_explicit_type_size(vecType), d, outBuffer);
    free_mtdata(d);
    d = NULL;
    memcpy(outBufferCopy, outBuffer, outBufferSize);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, inBufferSize,
                                inBuffer, &error);
    test_error(error, "Unable to create input buffer");
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, outBufferSize,
                                outBuffer, &error);
    test_error(error, "Unable to create output buffer");

    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 2, localBufferSize, NULL);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 3, sizeof(numElementsPerLineInt),
                           &numElementsPerLineInt);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 4, sizeof(numLinesInt), &numLinesInt);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 5, sizeof(planesCopiesPerWorkgroup),
                           &planesCopiesPerWorkgroup);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 6, sizeof(planesCopiesPerWorkItemInt),
                           &planesCopiesPerWorkItemInt);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 7, sizeof(srcLineStride), &srcLineStride);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 8, sizeof(dstLineStride), &dstLineStride);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 9, sizeof(srcPlaneStride), &srcPlaneStride);
    test_error(error, "Unable to set kernel argument");
    error = clSetKernelArg(kernel, 10, sizeof(dstPlaneStride), &dstPlaneStride);
    test_error(error, "Unable to set kernel argument");

    // Enqueue
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to queue kernel");

    // Read
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, outBufferSize,
                                outBuffer, 0, NULL, NULL);
    test_error(error, "Unable to read results");

    // Verify
    int failuresPrinted = 0;
    // Verify
    size_t typeSize = get_explicit_type_size(vecType) * vecSize;
    for (int i = 0;
         i < (int)globalWorkgroupSize * planesCopiesPerWorkItem * elementSize;
         i += elementSize)
    {
        for (int j = 0; j < (int)numLines * elementSize; j += elementSize)
        {
            for (int k = 0; k < (int)numElementsPerLine * elementSize;
                 k += elementSize)
            {
                int inIdx = i * srcPlaneStride + j * srcLineStride + k;
                int outIdx = i * dstPlaneStride + j * dstLineStride + k;
                if (memcmp(((char *)inBuffer) + inIdx,
                           ((char *)outBuffer) + outIdx, typeSize)
                    != 0)
                {
                    unsigned char *inchar = (unsigned char *)inBuffer + inIdx;
                    unsigned char *outchar =
                        (unsigned char *)outBuffer + outIdx;
                    char values[4096];
                    values[0] = 0;

                    if (failuresPrinted == 0)
                    {
                        // Print first failure message
                        log_error("ERROR: Results of copy did not validate!");
                    }
                    sprintf(values + strlen(values), "%d -> [", inIdx);
                    for (int l = 0; l < (int)elementSize; l++)
                        sprintf(values + strlen(values), "%2x ", inchar[l]);
                    sprintf(values + strlen(values), "] != [");
                    for (int l = 0; l < (int)elementSize; l++)
                        sprintf(values + strlen(values), "%2x ", outchar[l]);
                    sprintf(values + strlen(values), "]");
                    log_error("%s\n", values);
                    failuresPrinted++;
                }

                if (failuresPrinted > 5)
                {
                    log_error("Not printing further failures...\n");
                    return -1;
                }
            }
            if (j < (int)numLines * elementSize)
            {
                int outIdx = i * dstPlaneStride + j * dstLineStride
                    + numElementsPerLine * elementSize;
                if (memcmp(((char *)outBuffer) + outIdx,
                           ((char *)outBufferCopy) + outIdx,
                           dstLineMargin * elementSize)
                    != 0)
                {
                    if (failuresPrinted == 0)
                    {
                        // Print first failure message
                        log_error("ERROR: Results of copy did not validate!\n");
                    }
                    log_error(
                        "3D copy corrupted data in output buffer in the line "
                        "stride offset of plane %d line %d\n",
                        i, j);
                    failuresPrinted++;
                }
                if (failuresPrinted > 5)
                {
                    log_error("Not printing further failures...\n");
                    return -1;
                }
            }
        }
        if (i < (int)(globalWorkgroupSize * planesCopiesPerWorkItem - 1)
                * elementSize)
        {
            int outIdx =
                i * dstPlaneStride + numLines * dstLineStride * elementSize;
            if (memcmp(((char *)outBuffer) + outIdx,
                       ((char *)outBufferCopy) + outIdx,
                       dstPlaneMargin * elementSize)
                != 0)
            {
                if (failuresPrinted == 0)
                {
                    // Print first failure message
                    log_error("ERROR: Results of copy did not validate!\n");
                }
                log_error("3D copy corrupted data in output buffer in the "
                          "plane stride "
                          "offset of plane %d\n",
                          i);
                failuresPrinted++;
            }
            if (failuresPrinted > 5)
            {
                log_error("Not printing further failures...\n");
                return -1;
            }
        }
    }

    free(inBuffer);
    free(outBuffer);
    free(outBufferCopy);

    return failuresPrinted ? -1 : 0;
}

int test_copy3D_all_types(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, const char *kernelCode,
                          bool localIsDst)
{
    ExplicitType vecType[] = {
        kChar,  kUChar, kShort,  kUShort,          kInt, kUInt, kLong,
        kULong, kFloat, kDouble, kNumExplicitTypes
    };
    // The margins below represent the number of elements between the end of
    // one line or plane and the start of the next. The strides are equivalent
    // to the size of the line or plane plus the chosen margin.
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int smallTypesMarginSizes[] = { 0, 10, 100 };
    unsigned int size, typeIndex, srcLineMargin, dstLineMargin, srcPlaneMargin,
        dstPlaneMargin;

    int errors = 0;

    if (!is_extension_available(deviceID, "cl_khr_extended_async_copies"))
    {
        log_info(
            "Device does not support extended async copies. Skipping test.\n");
        return 0;
    }

    for (typeIndex = 0; vecType[typeIndex] != kNumExplicitTypes; typeIndex++)
    {
        if (vecType[typeIndex] == kDouble
            && !is_extension_available(deviceID, "cl_khr_fp64"))
            continue;

        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong)
            && !gHasLong)
            continue;

        for (size = 0; vecSizes[size] != 0; size++)
        {
            if (get_explicit_type_size(vecType[typeIndex]) * vecSizes[size]
                <= 2) // small type
            {
                for (srcLineMargin = 0;
                     srcLineMargin < sizeof(smallTypesMarginSizes)
                         / sizeof(smallTypesMarginSizes[0]);
                     srcLineMargin++)
                {
                    for (dstLineMargin = 0;
                         dstLineMargin < sizeof(smallTypesMarginSizes)
                             / sizeof(smallTypesMarginSizes[0]);
                         dstLineMargin++)
                    {
                        for (srcPlaneMargin = 0;
                             srcPlaneMargin < sizeof(smallTypesMarginSizes)
                                 / sizeof(smallTypesMarginSizes[0]);
                             srcPlaneMargin++)
                        {
                            for (dstPlaneMargin = 0;
                                 dstPlaneMargin < sizeof(smallTypesMarginSizes)
                                     / sizeof(smallTypesMarginSizes[0]);
                                 dstPlaneMargin++)
                            {
                                if (test_copy3D(
                                        deviceID, context, queue, kernelCode,
                                        vecType[typeIndex], vecSizes[size],
                                        smallTypesMarginSizes[srcLineMargin],
                                        smallTypesMarginSizes[dstLineMargin],
                                        smallTypesMarginSizes[srcPlaneMargin],
                                        smallTypesMarginSizes[dstPlaneMargin],
                                        localIsDst))
                                {
                                    errors++;
                                }
                            }
                        }
                    }
                }
            }
            // not a small type, check only zero stride
            else if (test_copy3D(deviceID, context, queue, kernelCode,
                                 vecType[typeIndex], vecSizes[size], 0, 0, 0, 0,
                                 localIsDst))
            {
                errors++;
            }
        }
    }
    if (errors) return -1;
    return 0;
}

int test_async_copy_global_to_local3D(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    return test_copy3D_all_types(deviceID, context, queue,
                                 async_global_to_local_kernel3D, true);
}

int test_async_copy_local_to_global3D(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    return test_copy3D_all_types(deviceID, context, queue,
                                 async_local_to_global_kernel3D, false);
}

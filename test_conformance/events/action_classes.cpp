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
#include "action_classes.h"

#pragma mark -------------------- Base Action Class -------------------------

const cl_uint BufferSizeReductionFactor = 20;

cl_int Action::IGetPreferredImageSize2D(cl_device_id device, size_t &outWidth,
                                        size_t &outHeight)
{
    cl_ulong maxAllocSize;
    size_t maxWidth, maxHeight;
    cl_int error;


    // Get the largest possible buffer we could allocate
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                             sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                             sizeof(maxHeight), &maxHeight, NULL);
    test_error(error, "Unable to get device config");

    // Create something of a decent size
    if (maxWidth * maxHeight * 4 > maxAllocSize / BufferSizeReductionFactor)
    {
        float rootSize =
            sqrtf((float)(maxAllocSize / (BufferSizeReductionFactor * 4)));

        if ((size_t)rootSize > maxWidth)
            outWidth = maxWidth;
        else
            outWidth = (size_t)rootSize;
        outHeight = (size_t)((maxAllocSize / (BufferSizeReductionFactor * 4))
                             / outWidth);
        if (outHeight > maxHeight) outHeight = maxHeight;
    }
    else
    {
        outWidth = maxWidth;
        outHeight = maxHeight;
    }

    outWidth /= 2;
    outHeight /= 2;

    if (outWidth > 2048) outWidth = 2048;
    if (outHeight > 2048) outHeight = 2048;
    log_info("\tImage size: %d x %d (%gMB)\n", (int)outWidth, (int)outHeight,
             (double)((int)outWidth * (int)outHeight * 4) / (1024.0 * 1024.0));
    return CL_SUCCESS;
}

cl_int Action::IGetPreferredImageSize3D(cl_device_id device, size_t &outWidth,
                                        size_t &outHeight, size_t &outDepth)
{
    cl_ulong maxAllocSize;
    size_t maxWidth, maxHeight, maxDepth;
    cl_int error;


    // Get the largest possible buffer we could allocate
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                             sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                             sizeof(maxHeight), &maxHeight, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                             sizeof(maxDepth), &maxDepth, NULL);
    test_error(error, "Unable to get device config");

    // Create something of a decent size
    if ((cl_ulong)maxWidth * maxHeight * maxDepth
        > maxAllocSize / (BufferSizeReductionFactor * 4))
    {
        float rootSize =
            cbrtf((float)(maxAllocSize / (BufferSizeReductionFactor * 4)));

        if ((size_t)rootSize > maxWidth)
            outWidth = maxWidth;
        else
            outWidth = (size_t)rootSize;
        if ((size_t)rootSize > maxHeight)
            outHeight = maxHeight;
        else
            outHeight = (size_t)rootSize;
        outDepth = (size_t)((maxAllocSize / (BufferSizeReductionFactor * 4))
                            / (outWidth * outHeight));
        if (outDepth > maxDepth) outDepth = maxDepth;
    }
    else
    {
        outWidth = maxWidth;
        outHeight = maxHeight;
        outDepth = maxDepth;
    }

    outWidth /= 2;
    outHeight /= 2;
    outDepth /= 2;

    if (outWidth > 512) outWidth = 512;
    if (outHeight > 512) outHeight = 512;
    if (outDepth > 512) outDepth = 512;
    log_info("\tImage size: %d x %d x %d (%gMB)\n", (int)outWidth,
             (int)outHeight, (int)outDepth,
             (double)((int)outWidth * (int)outHeight * (int)outDepth * 4)
                 / (1024.0 * 1024.0));

    return CL_SUCCESS;
}

#pragma mark -------------------- Execution Sub-Classes -------------------------

cl_int NDRangeKernelAction::Setup(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
{
    const char *long_kernel[] = {
        "__kernel void sample_test(__global float *src, __global int *dst)\n"
        "{\n"
        "    int  tid = get_global_id(0);\n"
        "     int  i;\n"
        "\n"
        "    for( i = 0; i < 100000; i++ )\n"
        "    {\n"
        "        dst[tid] = (int)src[tid] * 3;\n"
        "    }\n"
        "\n"
        "}\n"
    };

    size_t threads[1] = { 1000 };
    int error;

    if (create_single_kernel_helper(context, &mProgram, &mKernel, 1,
                                    long_kernel, "sample_test"))
    {
        return -1;
    }

    error = get_max_common_work_group_size(context, mKernel, threads[0],
                                           &mLocalThreads[0]);
    test_error(error, "Unable to get work group size to use");

    mStreams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_float) * 1000, NULL, &error);
    test_error(error, "Creating test array failed");
    mStreams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_int) * 1000, NULL, &error);
    test_error(error, "Creating test array failed");

    /* Set the arguments */
    error = clSetKernelArg(mKernel, 0, sizeof(mStreams[0]), &mStreams[0]);
    test_error(error, "Unable to set kernel arguments");
    error = clSetKernelArg(mKernel, 1, sizeof(mStreams[1]), &mStreams[1]);
    test_error(error, "Unable to set kernel arguments");

    return CL_SUCCESS;
}

cl_int NDRangeKernelAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                    cl_event *waits, cl_event *outEvent)
{
    size_t threads[1] = { 1000 };
    cl_int error =
        clEnqueueNDRangeKernel(queue, mKernel, 1, NULL, threads, mLocalThreads,
                               numWaits, waits, outEvent);
    test_error(error, "Unable to execute kernel");

    return CL_SUCCESS;
}

#pragma mark -------------------- Buffer Sub-Classes -------------------------

cl_int BufferAction::Setup(cl_device_id device, cl_context context,
                           cl_command_queue queue, bool allocate)
{
    cl_int error;
    cl_ulong maxAllocSize;


    // Get the largest possible buffer we could allocate
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(maxAllocSize), &maxAllocSize, NULL);

    // Don't create a buffer quite that big, just so we have some space left
    // over for other work
    mSize = (size_t)(maxAllocSize / BufferSizeReductionFactor);

    // Cap at 128M so tests complete in a reasonable amount of time.
    if (mSize > 128 << 20) mSize = 128 << 20;

    mSize /= 2;

    log_info("\tBuffer size: %gMB\n", (double)mSize / (1024.0 * 1024.0));

    mBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                             mSize, NULL, &error);
    test_error(error, "Unable to create buffer to test against");

    mOutBuffer = malloc(mSize);
    if (mOutBuffer == NULL)
    {
        log_error("ERROR: Unable to allocate temp buffer (out of memory)\n");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int ReadBufferAction::Setup(cl_device_id device, cl_context context,
                               cl_command_queue queue)
{
    return BufferAction::Setup(device, context, queue, true);
}

cl_int ReadBufferAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                 cl_event *waits, cl_event *outEvent)
{
    cl_int error = clEnqueueReadBuffer(queue, mBuffer, CL_FALSE, 0, mSize,
                                       mOutBuffer, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue buffer read");

    return CL_SUCCESS;
}

cl_int WriteBufferAction::Setup(cl_device_id device, cl_context context,
                                cl_command_queue queue)
{
    return BufferAction::Setup(device, context, queue, true);
}

cl_int WriteBufferAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                  cl_event *waits, cl_event *outEvent)
{
    cl_int error = clEnqueueWriteBuffer(queue, mBuffer, CL_FALSE, 0, mSize,
                                        mOutBuffer, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue buffer write");

    return CL_SUCCESS;
}

MapBufferAction::~MapBufferAction()
{
    if (mQueue)
        clEnqueueUnmapMemObject(mQueue, mBuffer, mMappedPtr, 0, NULL, NULL);
}

cl_int MapBufferAction::Setup(cl_device_id device, cl_context context,
                              cl_command_queue queue)
{
    return BufferAction::Setup(device, context, queue, false);
}

cl_int MapBufferAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                cl_event *waits, cl_event *outEvent)
{
    cl_int error;
    mQueue = queue;
    mMappedPtr = clEnqueueMapBuffer(queue, mBuffer, CL_FALSE, CL_MAP_READ, 0,
                                    mSize, numWaits, waits, outEvent, &error);
    test_error(error, "Unable to enqueue buffer map");

    return CL_SUCCESS;
}

cl_int UnmapBufferAction::Setup(cl_device_id device, cl_context context,
                                cl_command_queue queue)
{
    cl_int error = BufferAction::Setup(device, context, queue, false);
    if (error != CL_SUCCESS) return error;

    mMappedPtr = clEnqueueMapBuffer(queue, mBuffer, CL_TRUE, CL_MAP_READ, 0,
                                    mSize, 0, NULL, NULL, &error);
    test_error(error, "Unable to enqueue buffer map");

    return CL_SUCCESS;
}

cl_int UnmapBufferAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                  cl_event *waits, cl_event *outEvent)
{
    cl_int error = clEnqueueUnmapMemObject(queue, mBuffer, mMappedPtr, numWaits,
                                           waits, outEvent);
    test_error(error, "Unable to enqueue buffer unmap");

    return CL_SUCCESS;
}


#pragma mark -------------------- Read/Write Image Classes -------------------------

cl_int ReadImage2DAction::Setup(cl_device_id device, cl_context context,
                                cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mImage = create_image_2d(context, CL_MEM_READ_ONLY, &format, mWidth,
                             mHeight, 0, NULL, &error);

    test_error(error, "Unable to create image to test against");

    mOutput = malloc(mWidth * mHeight * 4);
    if (mOutput == NULL)
    {
        log_error("ERROR: Unable to allocate buffer: out of memory\n");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int ReadImage2DAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                  cl_event *waits, cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, 1 };

    cl_int error = clEnqueueReadImage(queue, mImage, CL_FALSE, origin, region,
                                      0, 0, mOutput, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image read");

    return CL_SUCCESS;
}

cl_int ReadImage3DAction::Setup(cl_device_id device, cl_context context,
                                cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                             mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mOutput = malloc(mWidth * mHeight * mDepth * 4);
    if (mOutput == NULL)
    {
        log_error("ERROR: Unable to allocate buffer: out of memory\n");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int ReadImage3DAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                  cl_event *waits, cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, mDepth };

    cl_int error = clEnqueueReadImage(queue, mImage, CL_FALSE, origin, region,
                                      0, 0, mOutput, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image read");

    return CL_SUCCESS;
}

cl_int WriteImage2DAction::Setup(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mImage = create_image_2d(context, CL_MEM_WRITE_ONLY, &format, mWidth,
                             mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mOutput = malloc(mWidth * mHeight * 4);
    if (mOutput == NULL)
    {
        log_error("ERROR: Unable to allocate buffer: out of memory\n");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int WriteImage2DAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                   cl_event *waits, cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, 1 };

    cl_int error =
        clEnqueueWriteImage(queue, mImage, CL_FALSE, origin, region, 0, 0,
                            mOutput, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image write");

    return CL_SUCCESS;
}

cl_int WriteImage3DAction::Setup(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                             mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mOutput = malloc(mWidth * mHeight * mDepth * 4);
    if (mOutput == NULL)
    {
        log_error("ERROR: Unable to allocate buffer: out of memory\n");
        return CL_OUT_OF_RESOURCES;
    }

    return CL_SUCCESS;
}

cl_int WriteImage3DAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                   cl_event *waits, cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, mDepth };

    cl_int error =
        clEnqueueWriteImage(queue, mImage, CL_FALSE, origin, region, 0, 0,
                            mOutput, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image write");

    return CL_SUCCESS;
}

#pragma mark -------------------- Copy Image Classes -------------------------

cl_int CopyImageAction::Execute(cl_command_queue queue, cl_uint numWaits,
                                cl_event *waits, cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, mDepth };

    cl_int error =
        clEnqueueCopyImage(queue, mSrcImage, mDstImage, origin, origin, region,
                           numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image copy");

    return CL_SUCCESS;
}

cl_int CopyImage2Dto2DAction::Setup(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    mWidth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_2d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstImage = create_image_2d(context, CL_MEM_WRITE_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDepth = 1;
    return CL_SUCCESS;
}

cl_int CopyImage2Dto3DAction::Setup(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    mDepth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_2d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDepth = 1;
    return CL_SUCCESS;
}

cl_int CopyImage3Dto2DAction::Setup(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    mDepth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstImage = create_image_2d(context, CL_MEM_WRITE_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDepth = 1;
    return CL_SUCCESS;
}

cl_int CopyImage3Dto3DAction::Setup(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    mDepth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    return CL_SUCCESS;
}

#pragma mark -------------------- Copy Image/Buffer Classes -------------------------

cl_int Copy2DImageToBufferAction::Setup(cl_device_id device, cl_context context,
                                        cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    mWidth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_2d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                mWidth * mHeight * 4, NULL, &error);
    test_error(error, "Unable to create buffer to test against");

    return CL_SUCCESS;
}

cl_int Copy2DImageToBufferAction::Execute(cl_command_queue queue,
                                          cl_uint numWaits, cl_event *waits,
                                          cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, 1 };

    cl_int error =
        clEnqueueCopyImageToBuffer(queue, mSrcImage, mDstBuffer, origin, region,
                                   0, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image to buffer copy");

    return CL_SUCCESS;
}

cl_int Copy3DImageToBufferAction::Setup(cl_device_id device, cl_context context,
                                        cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    mDepth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mSrcImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    mDstBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                mWidth * mHeight * mDepth * 4, NULL, &error);
    test_error(error, "Unable to create buffer to test against");

    return CL_SUCCESS;
}

cl_int Copy3DImageToBufferAction::Execute(cl_command_queue queue,
                                          cl_uint numWaits, cl_event *waits,
                                          cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, mDepth };

    cl_int error =
        clEnqueueCopyImageToBuffer(queue, mSrcImage, mDstBuffer, origin, region,
                                   0, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue image to buffer copy");

    return CL_SUCCESS;
}

cl_int CopyBufferTo2DImageAction::Setup(cl_device_id device, cl_context context,
                                        cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    mWidth /= 2;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };

    mSrcBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, mWidth * mHeight * 4,
                                NULL, &error);
    test_error(error, "Unable to create buffer to test against");

    mDstImage = create_image_2d(context, CL_MEM_WRITE_ONLY, &format, mWidth,
                                mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    return CL_SUCCESS;
}

cl_int CopyBufferTo2DImageAction::Execute(cl_command_queue queue,
                                          cl_uint numWaits, cl_event *waits,
                                          cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, 1 };

    cl_int error =
        clEnqueueCopyBufferToImage(queue, mSrcBuffer, mDstImage, 0, origin,
                                   region, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue buffer to image copy");

    return CL_SUCCESS;
}

cl_int CopyBufferTo3DImageAction::Setup(cl_device_id device, cl_context context,
                                        cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize3D(device, mWidth, mHeight, mDepth)))
        return error;

    mDepth /= 2;

    mSrcBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                mWidth * mHeight * mDepth * 4, NULL, &error);
    test_error(error, "Unable to create buffer to test against");

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mDstImage = create_image_3d(context, CL_MEM_READ_ONLY, &format, mWidth,
                                mHeight, mDepth, 0, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    return CL_SUCCESS;
}

cl_int CopyBufferTo3DImageAction::Execute(cl_command_queue queue,
                                          cl_uint numWaits, cl_event *waits,
                                          cl_event *outEvent)
{
    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, mDepth };

    cl_int error =
        clEnqueueCopyBufferToImage(queue, mSrcBuffer, mDstImage, 0, origin,
                                   region, numWaits, waits, outEvent);
    test_error(error, "Unable to enqueue buffer to image copy");

    return CL_SUCCESS;
}

#pragma mark -------------------- Map Image Class -------------------------

MapImageAction::~MapImageAction()
{
    if (mQueue)
        clEnqueueUnmapMemObject(mQueue, mImage, mMappedPtr, 0, NULL, NULL);
}

cl_int MapImageAction::Setup(cl_device_id device, cl_context context,
                             cl_command_queue queue)
{
    cl_int error;


    if ((error = IGetPreferredImageSize2D(device, mWidth, mHeight)))
        return error;

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };
    mImage = create_image_2d(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             &format, mWidth, mHeight, 0, NULL, &error);
    test_error(error, "Unable to create image to test against");

    return CL_SUCCESS;
}

cl_int MapImageAction::Execute(cl_command_queue queue, cl_uint numWaits,
                               cl_event *waits, cl_event *outEvent)
{
    cl_int error;

    size_t origin[3] = { 0, 0, 0 }, region[3] = { mWidth, mHeight, 1 };
    size_t outPitch;

    mQueue = queue;
    mMappedPtr =
        clEnqueueMapImage(queue, mImage, CL_FALSE, CL_MAP_READ, origin, region,
                          &outPitch, NULL, numWaits, waits, outEvent, &error);
    test_error(error, "Unable to enqueue image map");

    return CL_SUCCESS;
}

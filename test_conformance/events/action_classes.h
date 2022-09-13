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
#ifndef _action_classes_h
#define _action_classes_h

#include "testBase.h"

// This is a base class from which all actions are born
// Note: No actions should actually feed I/O to each other, because then
// it would potentially be possible for an implementation to make actions
// wait on one another based on their shared I/O, not because of their
// wait lists!
class Action {
public:
    Action() {}
    virtual ~Action() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue) = 0;
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent) = 0;

    virtual const char *GetName(void) const = 0;

protected:
    cl_int IGetPreferredImageSize2D(cl_device_id device, size_t &outWidth,
                                    size_t &outHeight);
    cl_int IGetPreferredImageSize3D(cl_device_id device, size_t &outWidth,
                                    size_t &outHeight, size_t &outDepth);
};

// Simple NDRangeKernel execution that takes a noticable amount of time
class NDRangeKernelAction : public Action {
public:
    NDRangeKernelAction() {}
    virtual ~NDRangeKernelAction() {}

    size_t mLocalThreads[1];
    clMemWrapper mStreams[2];
    clProgramWrapper mProgram;
    clKernelWrapper mKernel;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "NDRangeKernel"; }
};

// Base action for buffer actions
class BufferAction : public Action {
public:
    clMemWrapper mBuffer;
    size_t mSize;
    void *mOutBuffer;

    BufferAction() { mOutBuffer = NULL; }
    virtual ~BufferAction() { free(mOutBuffer); }

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue, bool allocate);
};

class ReadBufferAction : public BufferAction {
public:
    ReadBufferAction() {}
    virtual ~ReadBufferAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "ReadBuffer"; }
};

class WriteBufferAction : public BufferAction {
public:
    WriteBufferAction() {}
    virtual ~WriteBufferAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "WriteBuffer"; }
};

class MapBufferAction : public BufferAction {
public:
    MapBufferAction(): mQueue(0) {}

    cl_command_queue mQueue;
    void *mMappedPtr;

    virtual ~MapBufferAction();
    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "MapBuffer"; }
};

class UnmapBufferAction : public BufferAction {
public:
    UnmapBufferAction() {}
    virtual ~UnmapBufferAction() {}

    void *mMappedPtr;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "UnmapBuffer"; }
};

class ReadImage2DAction : public Action {
public:
    ReadImage2DAction() { mOutput = NULL; }
    virtual ~ReadImage2DAction() { free(mOutput); }

    clMemWrapper mImage;
    size_t mWidth, mHeight;
    void *mOutput;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "ReadImage2D"; }
};

class ReadImage3DAction : public Action {
public:
    ReadImage3DAction() { mOutput = NULL; }
    virtual ~ReadImage3DAction() { free(mOutput); }

    clMemWrapper mImage;
    size_t mWidth, mHeight, mDepth;
    void *mOutput;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "ReadImage3D"; }
};

class WriteImage2DAction : public Action {
public:
    clMemWrapper mImage;
    size_t mWidth, mHeight;
    void *mOutput;

    WriteImage2DAction() { mOutput = NULL; }
    virtual ~WriteImage2DAction() { free(mOutput); }

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "WriteImage2D"; }
};

class WriteImage3DAction : public Action {
public:
    clMemWrapper mImage;
    size_t mWidth, mHeight, mDepth;
    void *mOutput;

    WriteImage3DAction() { mOutput = NULL; }
    virtual ~WriteImage3DAction() { free(mOutput); }

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "WriteImage3D"; }
};

class CopyImageAction : public Action {
public:
    CopyImageAction() {}
    virtual ~CopyImageAction() {}

    clMemWrapper mSrcImage, mDstImage;
    size_t mWidth, mHeight, mDepth;

    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);
};

class CopyImage2Dto2DAction : public CopyImageAction {
public:
    CopyImage2Dto2DAction() {}
    virtual ~CopyImage2Dto2DAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);

    virtual const char *GetName(void) const { return "CopyImage2Dto2D"; }
};

class CopyImage2Dto3DAction : public CopyImageAction {
public:
    CopyImage2Dto3DAction() {}
    virtual ~CopyImage2Dto3DAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);

    virtual const char *GetName(void) const { return "CopyImage2Dto3D"; }
};

class CopyImage3Dto2DAction : public CopyImageAction {
public:
    CopyImage3Dto2DAction() {}
    virtual ~CopyImage3Dto2DAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);

    virtual const char *GetName(void) const { return "CopyImage3Dto2D"; }
};

class CopyImage3Dto3DAction : public CopyImageAction {
public:
    CopyImage3Dto3DAction() {}
    virtual ~CopyImage3Dto3DAction() {}

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);

    virtual const char *GetName(void) const { return "CopyImage3Dto3D"; }
};

class Copy2DImageToBufferAction : public Action {
public:
    Copy2DImageToBufferAction() {}
    virtual ~Copy2DImageToBufferAction() {}

    clMemWrapper mSrcImage, mDstBuffer;
    size_t mWidth, mHeight;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "Copy2DImageToBuffer"; }
};

class Copy3DImageToBufferAction : public Action {
public:
    Copy3DImageToBufferAction() {}
    virtual ~Copy3DImageToBufferAction() {}

    clMemWrapper mSrcImage, mDstBuffer;
    size_t mWidth, mHeight, mDepth;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "Copy3DImageToBuffer"; }
};

class CopyBufferTo2DImageAction : public Action {
public:
    CopyBufferTo2DImageAction() {}
    virtual ~CopyBufferTo2DImageAction() {}

    clMemWrapper mSrcBuffer, mDstImage;
    size_t mWidth, mHeight;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "CopyBufferTo2D"; }
};

class CopyBufferTo3DImageAction : public Action {
public:
    CopyBufferTo3DImageAction() {}
    virtual ~CopyBufferTo3DImageAction() {}

    clMemWrapper mSrcBuffer, mDstImage;
    size_t mWidth, mHeight, mDepth;

    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "CopyBufferTo3D"; }
};

class MapImageAction : public Action {
public:
    MapImageAction(): mQueue(0) {}

    clMemWrapper mImage;
    size_t mWidth, mHeight;
    void *mMappedPtr;
    cl_command_queue mQueue;

    virtual ~MapImageAction();
    virtual cl_int Setup(cl_device_id device, cl_context context,
                         cl_command_queue queue);
    virtual cl_int Execute(cl_command_queue queue, cl_uint numWaits,
                           cl_event *waits, cl_event *outEvent);

    virtual const char *GetName(void) const { return "MapImage"; }
};


#endif // _action_classes_h

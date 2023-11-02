//
// Copyright (c) 2023 The Khronos Group Inc.
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

#ifndef CL_KHR_SVM_COMMAND_BASIC_H
#define CL_KHR_SVM_COMMAND_BASIC_H

#include "basic_command_buffer.h"


struct BasicSVMCommandBufferTest : BasicCommandBufferTest
{
    BasicSVMCommandBufferTest(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    virtual bool Skip() override;
    virtual cl_int SetUpKernelArgs(void) override;

protected:
    cl_int init_extension_functions();

    clCommandSVMMemFillKHR_fn clCommandSVMMemFillKHR = nullptr;
    clCommandSVMMemcpyKHR_fn clCommandSVMMemcpyKHR = nullptr;

    clSVMWrapper svm_in_mem, svm_out_mem;
};

#endif

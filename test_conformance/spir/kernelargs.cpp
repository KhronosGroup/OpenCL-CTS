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
#include "kernelargs.h"
#include "datagen.h"

KernelArg* KernelArg::clone(cl_context ctx, const WorkSizeInfo& ws, const cl_kernel kernel, const cl_device_id device) const
{
    return DataGenerator::getInstance()->generateKernelArg(ctx, m_argInfo, ws,
                                                           this, kernel, device);
}

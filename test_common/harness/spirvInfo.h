//
// Copyright (c) 2025 The Khronos Group Inc.
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
#ifndef _spirvQueries_h
#define _spirvQueries_h

#include <vector>

#include <CL/opencl.h>

int get_device_spirv_queries(cl_device_id device,
                             std::vector<const char*>& extendedInstructionSets,
                             std::vector<const char*>& extensions,
                             std::vector<cl_uint>& capabilities);

#endif // _spirvQueries_h

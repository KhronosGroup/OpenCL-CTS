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
#ifndef _testBase_h
#define _testBase_h

#include <CL/cl.h>

#include "harness/conversions.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

extern const cl_mem_flags flag_set[];
extern const char* flag_set_names[];

#define NUM_FLAGS 5

#endif // _testBase_h

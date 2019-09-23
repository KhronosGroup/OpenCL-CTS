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
#ifndef _testBase_h
#define _testBase_h

#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#else
//#include <OpenCL/cl.h>
#endif

#include "harness/imageHelpers.h"
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/threadTesting.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

#endif // _testBase_h

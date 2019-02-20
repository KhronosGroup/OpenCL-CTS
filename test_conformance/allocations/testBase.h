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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/typeWrappers.h"
#include "../../test_common/harness/testHarness.h"


#define MAX_NUMBER_TO_ALLOCATE 100

#define FAILED_CORRUPTED_QUEUE -2
#define FAILED_ABORT -1
#define FAILED_TOO_BIG 1
#define SUCCEEDED 0

#define BUFFER 1
#define IMAGE_READ 2
#define IMAGE_WRITE 4
#define BUFFER_NON_BLOCKING 8
#define IMAGE_READ_NON_BLOCKING 16
#define IMAGE_WRITE_NON_BLOCKING 32

#define test_error_abort(errCode,msg)    test_error_ret_abort(errCode,msg,errCode)
#define test_error_ret_abort(errCode,msg,retValue)    { if( errCode != CL_SUCCESS ) { print_error( errCode, msg ); return FAILED_ABORT ; } }


#endif // _testBase_h




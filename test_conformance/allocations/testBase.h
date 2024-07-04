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
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"


#define MAX_NUMBER_TO_ALLOCATE 100

#define FAILED_CORRUPTED_QUEUE -2
#define FAILED_ABORT -1
#define FAILED_TOO_BIG 1
// On Windows macro `SUCCEEDED' is defined in `WinError.h'. It causes compiler
// warnings. Let us avoid them.
#if defined(_WIN32) && defined(SUCCEEDED)
#undef SUCCEEDED
#endif
#define SUCCEEDED 0

enum AllocType
{
    BUFFER,
    IMAGE_READ,
    IMAGE_WRITE,
    BUFFER_NON_BLOCKING,
    IMAGE_READ_NON_BLOCKING,
    IMAGE_WRITE_NON_BLOCKING,
};

#define test_error_abort(errCode, msg)                                         \
    test_error_ret_abort(errCode, msg, errCode)
#define test_error_ret_abort(errCode, msg, retValue)                           \
    {                                                                          \
        if (errCode != CL_SUCCESS)                                             \
        {                                                                      \
            print_error(errCode, msg);                                         \
            return FAILED_ABORT;                                               \
        }                                                                      \
    }


#endif // _testBase_h

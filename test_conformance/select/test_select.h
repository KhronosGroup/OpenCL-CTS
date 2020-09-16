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
#ifndef TESTSELECTS_INCLUDED_H
#define TESTSELECTS_INCLUDED_H

#include "harness/compat.h"

#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// Defines the set of types we support (no support for double)
typedef enum {
    kuchar = 0,
    kchar = 1,
    kushort = 2,
    kshort = 3,
    kuint = 4,
    kint = 5,
    kfloat = 6,
    kulong = 7,
    klong = 8,
    kdouble = 9,
    kTypeCount  // always goes last
} Type;


// Support max vector size of 16
#define kVectorSizeCount    6
#define kMaxVectorSize      16


// Type names and their sizes in bytes
extern const char *type_name[kTypeCount];
extern const size_t type_size[kTypeCount];

// Associated comparison types
extern const Type ctype[kTypeCount][2];

// Reference functions for the primitive (non vector) type
typedef void (*Select)(void *dest, void *src1, void *src2, void *cmp, size_t c);
extern Select refSelects[kTypeCount][2];

// Reference functions for the primtive type but uses the vector
// definition of true and false
extern Select vrefSelects[kTypeCount][2];

// Check functions for each output type
typedef size_t (*CheckResults)(void *out1, void *out2, size_t count, size_t vectorSize);
extern CheckResults checkResults[kTypeCount];

// Helpful macros

// The next three functions check on different return values.  Returns -1
// if the check failed
#define checkErr(err, msg)                \
    if (err != CL_SUCCESS) {                \
    log_error("%s failed errcode:%d\n", msg, err);    \
    return -1;                    \
    }

#define checkZero(val, msg)                \
    if (val == 0) {                    \
    log_error("%s failed errcode:%d\n", msg, err);    \
    return -1;                    \
    }

#define checkNull(ptr, msg)            \
    if (!ptr) {                    \
    log_error("%s failed\n", msg);        \
    return -1;                \
    }

// When a helper returns a negative one, we want to return from main
// with negative one. This helper prevents me from having to write
// this multiple time
#define checkHelperErr(err)            \
    if (err == -1) {                \
    return err;                \
    }


#endif // TESTSELECTS_INCLUDED_H

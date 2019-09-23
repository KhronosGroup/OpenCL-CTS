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
#ifndef TESTPRINTF_INCLUDED_H
#define TESTPRINTF_INCLUDED_H

#include "harness/compat.h"

#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#endif

// Enable the test to be used with ATF
#if USE_ATF
// export BUILD_WITH_ATF=1
#include <ATF/ATF.h>
#define test_start() ATFTestStart()
#define log_info ATFLogInfo
#define log_error ATFLogError
#define test_finish() ATFTestFinish()
#else
#define test_start()
#define log_info printf
#define log_error printf
#define test_finish()
#endif // USE_ATF

#define ANALYSIS_BUFFER_SIZE 256

//-----------------------------------------
// Definitions and initializations
//-----------------------------------------

//-----------------------------------------
// Types
//-----------------------------------------
enum Type
 {
     TYPE_INT,
     TYPE_FLOAT,
     TYPE_OCTAL,
     TYPE_UNSIGNED,
     TYPE_HEXADEC,
     TYPE_CHAR,
     TYPE_STRING,
     TYPE_VECTOR,
     TYPE_ADDRESS_SPACE,
     TYPE_COUNT
};

struct printDataGenParameters
{
    const char* genericFormat;
    const char* dataRepresentation;
    const char* vectorFormatFlag;
    const char* vectorFormatSpecifier;
    const char* dataType;
    const char* vectorSize;
    const char* addrSpaceArgumentTypeQualifier;
    const char* addrSpaceVariableTypeQualifier;
    const char* addrSpaceParameter;
    const char* addrSpacePAdd;
};

//-----------------------------------------
//Test Case
//-----------------------------------------

struct testCase
{
    unsigned int _testNum;                           //test number
    enum Type _type;                                 //(data)type for test
    //const char** _strPrint;                          //auxiliary data to build the code for kernel source
    const char** _correctBuffer;                     //look-up table for correct results for printf
    struct printDataGenParameters* _genParameters;   //auxiliary data to build the code for kernel source
};


extern const char* strType[];
extern testCase* allTestCase[];

size_t verifyOutputBuffer(char *analysisBuffer,testCase* pTestCase,size_t testId,cl_ulong pAddr = 0);

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
    return TEST_FAIL;                \
    }

// When a helper returns a negative one, we want to return from main
// with negative one. This helper prevents me from having to write
// this multiple time
#define checkHelperErr(err)            \
    if (err == -1) {                \
    return err;                \
    }

#endif // TESTSPRINTF_INCLUDED_H

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
#include "harness/testHarness.h"
#include "harness/rounding_mode.h"

#include <stdio.h>
#include <string.h>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/opencl.h>
#include <CL/cl_platform.h>
#endif

#define ANALYSIS_BUFFER_SIZE 256

//-----------------------------------------
// Definitions and initializations
//-----------------------------------------

//-----------------------------------------
// Types
//-----------------------------------------
enum PrintfTestType
 {
     TYPE_INT,
     TYPE_FLOAT,
     TYPE_FLOAT_LIMITS,
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

// Reference results - filled out at run-time
static std::vector<std::string> correctBufferInt;
static std::vector<std::string> correctBufferFloat;
static std::vector<std::string> correctBufferOctal;
static std::vector<std::string> correctBufferUnsigned;
static std::vector<std::string> correctBufferHexadecimal;
// Reference results - Compile-time known
extern std::vector<std::string> correctBufferChar;
extern std::vector<std::string> correctBufferString;
extern std::vector<std::string> correctBufferFloatLimits;
extern std::vector<std::string> correctBufferVector;
extern std::vector<std::string> correctAddrSpace;

// Helper for generating reference results
void generateRef(const cl_device_id device);

//-----------------------------------------
//Test Case
//-----------------------------------------

struct testCase
{
    enum PrintfTestType _type;                           //(data)type for test
    std::vector<std::string>& _correctBuffer;            //look-up table for correct results for printf
    std::vector<printDataGenParameters>& _genParameters; //auxiliary data to build the code for kernel source
    void (*printFN)(printDataGenParameters&,
                    char*,
                    const size_t);                       //function pointer for generating reference results
    Type dataType;                                       //the data type that will be printed during reference result generation (used for setting rounding mode)
};

extern const char* strType[];
extern std::vector<testCase*> allTestCase;

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

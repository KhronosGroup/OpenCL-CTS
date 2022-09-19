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

#include <string.h>
#include <errno.h>
#include <memory>

#if ! defined( _WIN32)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#define streamDup(fd1) dup(fd1)
#define streamDup2(fd1,fd2) dup2(fd1,fd2)
#endif
#include <limits.h>
#include <time.h>
#include "test_printf.h"

#if defined(_WIN32)
#include <io.h>
#define streamDup(fd1) _dup(fd1)
#define streamDup2(fd1,fd2) _dup2(fd1,fd2)
#endif

#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/mt19937.h"
#include "harness/parseParameters.h"

#include <CL/cl_ext.h>

typedef  unsigned int uint32_t;


test_status InitCL( cl_device_id device );

//-----------------------------------------
// Static helper functions declaration
//-----------------------------------------

static void printUsage( void );

//Stream helper functions

//Associate stdout stream with the file(gFileName):i.e redirect stdout stream to the specific files (gFileName)
static int acquireOutputStream(int* error);

//Close the file(gFileName) associated with the stdout stream and disassociates it.
static void releaseOutputStream(int fd);

//Get analysis buffer to verify the correctess of printed data
static void getAnalysisBuffer(char* analysisBuffer);

//Kernel builder helper functions

//Check if the test case is for kernel that has argument
static int isKernelArgument(testCase* pTestCase,size_t testId);

//Check if the test case treats %p format for void*
static int isKernelPFormat(testCase* pTestCase,size_t testId);

//-----------------------------------------
// Static functions declarations
//-----------------------------------------
// Make a program that uses printf for the given type/format,
static cl_program makePrintfProgram(cl_kernel *kernel_ptr, const cl_context context,const unsigned int testId,const unsigned int testNum,bool isLongSupport = true,bool is64bAddrSpace = false);

// Creates and execute the printf test for the given device, context, type/format
static int doTest(cl_command_queue queue, cl_context context, const unsigned int testId, const unsigned int testNum, cl_device_id device);

// Check if device supports long
static bool isLongSupported(cl_device_id  device_id);

// Check if device address space is 64 bits
static bool is64bAddressSpace(cl_device_id  device_id);

//Wait until event status is CL_COMPLETE
int waitForEvent(cl_event* event);

//-----------------------------------------
// Definitions and initializations
//-----------------------------------------

// Tests are broken into the major test which is based on the
// src and cmp type and their corresponding vector types and
// sub tests which is for each individual test.  The following
// tracks the subtests
int s_test_cnt = 0;
int s_test_fail = 0;


static cl_context        gContext;
static cl_command_queue  gQueue;
static int               gFd;

static char gFileName[256];

//-----------------------------------------
// Static helper functions definition
//-----------------------------------------

//-----------------------------------------
// getTempFileName
//-----------------------------------------
static int getTempFileName()
{
    // Create a unique temporary file to allow parallel executed tests.
#if (defined(__linux__) || defined(__APPLE__)) && (!defined( __ANDROID__ ))
    sprintf(gFileName, "/tmp/tmpfile.XXXXXX");
    int fd = mkstemp(gFileName);
    if (fd == -1)
        return -1;
    close(fd);
#elif defined(_WIN32)
    UINT ret = GetTempFileName(".", "tmp", 0, gFileName);
    if (ret == 0)
        return -1;
#else
    MTdata d = init_genrand((cl_uint)time(NULL));
    sprintf(gFileName, "tmpfile.%u", genrand_int32(d));
#endif
    return 0;
}

//-----------------------------------------
// acquireOutputStream
//-----------------------------------------
static int acquireOutputStream(int* error)
{
    int fd = streamDup(fileno(stdout));
    *error = 0;
    if (!freopen(gFileName, "w", stdout))
    {
        releaseOutputStream(fd);
        *error = -1;
    }
    return fd;
}

//-----------------------------------------
// releaseOutputStream
//-----------------------------------------
static void releaseOutputStream(int fd)
{
    fflush(stdout);
    streamDup2(fd,fileno(stdout));
    close(fd);
}

//-----------------------------------------
// printfCallBack
//-----------------------------------------
static void CL_CALLBACK printfCallBack(const char *printf_data, size_t len, size_t final, void *user_data)
{
    fwrite(printf_data, 1, len, stdout);
}

//-----------------------------------------
// getAnalysisBuffer
//-----------------------------------------
static void getAnalysisBuffer(char* analysisBuffer)
{
    FILE *fp;
    memset(analysisBuffer,0,ANALYSIS_BUFFER_SIZE);

    fp = fopen(gFileName,"r");
    if(NULL == fp)
        log_error("Failed to open analysis buffer ('%s')\n", strerror(errno));
    else
        while(fgets(analysisBuffer, ANALYSIS_BUFFER_SIZE, fp) != NULL );
    fclose(fp);
}

//-----------------------------------------
// isKernelArgument
//-----------------------------------------
static int isKernelArgument(testCase* pTestCase,size_t testId)
{
    return strcmp(pTestCase->_genParameters[testId].addrSpaceArgumentTypeQualifier,"");
}
//-----------------------------------------
// isKernelPFormat
//-----------------------------------------
static int isKernelPFormat(testCase* pTestCase,size_t testId)
{
    return strcmp(pTestCase->_genParameters[testId].addrSpacePAdd,"");
}

//-----------------------------------------
// waitForEvent
//-----------------------------------------
int waitForEvent(cl_event* event)
{
    cl_int status = clWaitForEvents(1, event);
    if(status != CL_SUCCESS)
    {
        log_error("clWaitForEvents failed");
        return status;
    }

    status = clReleaseEvent(*event);
    if(status != CL_SUCCESS)
    {
        log_error("clReleaseEvent failed. (*event)");
        return status;
    }
    return CL_SUCCESS;
}

//-----------------------------------------
// Static helper functions definition
//-----------------------------------------

//-----------------------------------------
// makePrintfProgram
//-----------------------------------------
static cl_program makePrintfProgram(cl_kernel *kernel_ptr, const cl_context context,const unsigned int testId,const unsigned int testNum,bool isLongSupport,bool is64bAddrSpace)
{
    int err;
    cl_program program;
    char testname[256] = {0};
    char addrSpaceArgument[256] = {0};
    char addrSpacePAddArgument[256] = {0};

    //Program Source code for int,float,octal,hexadecimal,char,string
    const char *sourceGen[] = {
        "__kernel void ", testname,
        "(void)\n",
        "{\n"
        "   printf(\"",
        allTestCase[testId]->_genParameters[testNum].genericFormat,
        "\\n\",",
        allTestCase[testId]->_genParameters[testNum].dataRepresentation,
        ");",
        "}\n"
    };
    //Program Source code for vector
    const char *sourceVec[] = {
        "__kernel void ", testname,
        "(void)\n",
        "{\n",
        allTestCase[testId]->_genParameters[testNum].dataType,
        allTestCase[testId]->_genParameters[testNum].vectorSize,
        " tmp = (",
        allTestCase[testId]->_genParameters[testNum].dataType,
        allTestCase[testId]->_genParameters[testNum].vectorSize,
        ")",
        allTestCase[testId]->_genParameters[testNum].dataRepresentation,
        ";",
        "   printf(\"",
        allTestCase[testId]->_genParameters[testNum].vectorFormatFlag,
        "v",
        allTestCase[testId]->_genParameters[testNum].vectorSize,
        allTestCase[testId]->_genParameters[testNum].vectorFormatSpecifier,
        "\\n\",",
        "tmp);",
        "}\n"
    };
    //Program Source code for address space
    const char *sourceAddrSpace[] = {
        "__kernel void ", testname,"(",addrSpaceArgument,
        ")\n{\n",
        allTestCase[testId]->_genParameters[testNum].addrSpaceVariableTypeQualifier,
        "printf(",
        allTestCase[testId]->_genParameters[testNum].genericFormat,
        ",",
        allTestCase[testId]->_genParameters[testNum].addrSpaceParameter,
        "); ",
        addrSpacePAddArgument,
        "\n}\n"
    };

    //Update testname
    sprintf(testname,"%s%d","test",testId);

    //Update addrSpaceArgument and addrSpacePAddArgument types, based on FULL_PROFILE/EMBEDDED_PROFILE
    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        sprintf(addrSpaceArgument, "%s",allTestCase[testId]->_genParameters[testNum].addrSpaceArgumentTypeQualifier);

        sprintf(addrSpacePAddArgument, "%s", allTestCase[testId]->_genParameters[testNum].addrSpacePAdd);
    }

    if (strlen(addrSpaceArgument) == 0)
        sprintf(addrSpaceArgument,"void");

    // create program based on its type

    if(allTestCase[testId]->_type == TYPE_VECTOR)
    {
        err = create_single_kernel_helper(
            context, &program, kernel_ptr,
            sizeof(sourceVec) / sizeof(sourceVec[0]), sourceVec, testname);
    }
    else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        err = create_single_kernel_helper(context, &program, kernel_ptr,
                                          sizeof(sourceAddrSpace)
                                              / sizeof(sourceAddrSpace[0]),
                                          sourceAddrSpace, testname);
    }
    else
    {
        err = create_single_kernel_helper(
            context, &program, kernel_ptr,
            sizeof(sourceGen) / sizeof(sourceGen[0]), sourceGen, testname);
    }

    if (!program || err) {
        log_error("create_single_kernel_helper failed\n");
        return NULL;
    }

    return program;
}

//-----------------------------------------
// isLongSupported
//-----------------------------------------
static bool isLongSupported(cl_device_id device_id)
{
    size_t tempSize = 0;
    cl_int status;
    bool extSupport = true;

    // Device profile
    status = clGetDeviceInfo(
        device_id,
        CL_DEVICE_PROFILE,
        0,
        NULL,
        &tempSize);

    if(status != CL_SUCCESS)
    {
        log_error("*** clGetDeviceInfo FAILED ***\n\n");
        return false;
    }

    std::unique_ptr<char[]> profileType(new char[tempSize]);
    if(profileType == NULL)
    {
        log_error("Failed to allocate memory(profileType)");
        return false;
    }

    status = clGetDeviceInfo(
        device_id,
        CL_DEVICE_PROFILE,
        sizeof(char) * tempSize,
        profileType.get(),
        NULL);


    if(!strcmp("EMBEDDED_PROFILE",profileType.get()))
    {
        extSupport = is_extension_available(device_id, "cles_khr_int64");
    }
    return extSupport;
}
//-----------------------------------------
// is64bAddressSpace
//-----------------------------------------
static bool is64bAddressSpace(cl_device_id  device_id)
{
    cl_int status;
    cl_uint addrSpaceB;

    // Device profile
    status = clGetDeviceInfo(
        device_id,
        CL_DEVICE_ADDRESS_BITS,
        sizeof(cl_uint),
        &addrSpaceB,
        NULL);
    if(status != CL_SUCCESS)
    {
        log_error("*** clGetDeviceInfo FAILED ***\n\n");
        return false;
    }
    if(addrSpaceB == 64)
        return true;
    else
        return false;
}
//-----------------------------------------
// doTest
//-----------------------------------------
static int doTest(cl_command_queue queue, cl_context context, const unsigned int testId, const unsigned int testNum, cl_device_id device)
{
    if(allTestCase[testId]->_type == TYPE_VECTOR)
    {
        log_info("%d)testing printf(\"%sv%s%s\",%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].vectorFormatFlag,allTestCase[testId]->_genParameters[testNum].vectorSize,
                 allTestCase[testId]->_genParameters[testNum].vectorFormatSpecifier,allTestCase[testId]->_genParameters[testNum].dataRepresentation);
    }
    else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        if(isKernelArgument(allTestCase[testId], testNum))
        {
            log_info("%d)testing kernel //argument %s \n   printf(%s,%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].addrSpaceArgumentTypeQualifier,
                     allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].addrSpaceParameter);
        }
        else
        {
            log_info("%d)testing kernel //variable %s \n   printf(%s,%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].addrSpaceVariableTypeQualifier,
                     allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].addrSpaceParameter);
        }
    }
    else
    {
        log_info("%d)testing printf(\"%s\",%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].dataRepresentation);
    }

    // Long support for varible type
    if(allTestCase[testId]->_type == TYPE_VECTOR && !strcmp(allTestCase[testId]->_genParameters[testNum].dataType,"long") && !isLongSupported(device))
    {
        log_info( "Long is not supported, test not run.\n" );
        return 0;
    }

    // Long support for address in FULL_PROFILE/EMBEDDED_PROFILE
    bool isLongSupport = true;
    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE && isKernelPFormat(allTestCase[testId],testNum) && !isLongSupported(device))
    {
        isLongSupport = false;
    }

    int err;
    cl_program program;
    cl_kernel  kernel;
    cl_mem d_out = NULL;
    cl_mem d_a = NULL;
    char _analysisBuffer[ANALYSIS_BUFFER_SIZE];
    cl_uint out32 = 0;
    cl_ulong out64 = 0;
    int fd = -1;

   // Define an index space (global work size) of threads for execution.
   size_t globalWorkSize[1];

    program = makePrintfProgram(&kernel, context,testId,testNum,isLongSupport,is64bAddressSpace(device));
    if (!program || !kernel) {
        ++s_test_fail;
        ++s_test_cnt;
        return -1;
    }

    //For address space test if there is kernel argument - set it
    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE )
    {
        if(isKernelArgument(allTestCase[testId],testNum))
        {
            int a = 2;
            d_a = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                sizeof(int), &a, &err);
            if(err!= CL_SUCCESS || d_a == NULL) {
                log_error("clCreateBuffer failed\n");
                goto exit;
            }
            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
            if(err!= CL_SUCCESS) {
                log_error("clSetKernelArg failed\n");
                goto exit;
            }
        }
        //For address space test if %p is tested
        if(isKernelPFormat(allTestCase[testId],testNum))
        {
            d_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                sizeof(cl_ulong), NULL, &err);
            if(err!= CL_SUCCESS || d_out == NULL) {
                log_error("clCreateBuffer failed\n");
                goto exit;
            }
            err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
            if(err!= CL_SUCCESS) {
                log_error("clSetKernelArg failed\n");
                goto exit;
            }
        }
    }

    fd = acquireOutputStream(&err);
    if (err != 0)
    {
        log_error("Error while redirection stdout to file");
        goto exit;
    }
    globalWorkSize[0] = 1;
    cl_event ndrEvt;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL,&ndrEvt);
    if (err != CL_SUCCESS) {
        releaseOutputStream(fd);
        log_error("\n clEnqueueNDRangeKernel failed errcode:%d\n", err);
        ++s_test_fail;
        goto exit;
    }

    fflush(stdout);
    err = clFlush(queue);
    if(err != CL_SUCCESS)
    {
        releaseOutputStream(fd);
        log_error("clFlush failed\n");
        goto exit;
    }
    //Wait until kernel finishes its execution and (thus) the output printed from the kernel
    //is immediately printed
    err = waitForEvent(&ndrEvt);

    releaseOutputStream(fd);

    if(err != CL_SUCCESS)
    {
        log_error("waitforEvent failed\n");
        goto exit;
    }
    fflush(stdout);

    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE && isKernelPFormat(allTestCase[testId],testNum))
    {
        // Read the OpenCL output buffer (d_out) to the host output array (out)
        if(!is64bAddressSpace(device))//32-bit address space
        {
            clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(cl_int),&out32,
                0, NULL, NULL);
        }
        else //64-bit address space
        {
            clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(cl_ulong),&out64,
                0, NULL, NULL);
        }
    }

    //
    //Get the output printed from the kernel to _analysisBuffer
    //and verify its correctness
    getAnalysisBuffer(_analysisBuffer);
    if(!is64bAddressSpace(device)) //32-bit address space
    {
        if(0 != verifyOutputBuffer(_analysisBuffer,allTestCase[testId],testNum,(cl_ulong) out32))
            err = ++s_test_fail;
    }
    else //64-bit address space
    {
        if(0 != verifyOutputBuffer(_analysisBuffer,allTestCase[testId],testNum,out64))
            err = ++s_test_fail;
    }
exit:
    if(clReleaseKernel(kernel) != CL_SUCCESS)
        log_error("clReleaseKernel failed\n");
    if(clReleaseProgram(program) != CL_SUCCESS)
        log_error("clReleaseProgram failed\n");
    if(d_out)
        clReleaseMemObject(d_out);
    if(d_a)
        clReleaseMemObject(d_a);
    ++s_test_cnt;
    return err;
}


int test_int_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 0, deviceID);
}
int test_int_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 1, deviceID);
}
int test_int_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 2, deviceID);
}
int test_int_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 3, deviceID);
}
int test_int_4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 4, deviceID);
}
int test_int_5(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 5, deviceID);
}
int test_int_6(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 6, deviceID);
}
int test_int_7(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 7, deviceID);
}
int test_int_8(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, 8, deviceID);
}


int test_float_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 0, deviceID);
}
int test_float_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 1, deviceID);
}
int test_float_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 2, deviceID);
}
int test_float_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 3, deviceID);
}
int test_float_4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 4, deviceID);
}
int test_float_5(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 5, deviceID);
}
int test_float_6(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 6, deviceID);
}
int test_float_7(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 7, deviceID);
}
int test_float_8(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 8, deviceID);
}
int test_float_9(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 9, deviceID);
}
int test_float_10(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 10, deviceID);
}
int test_float_11(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 11, deviceID);
}
int test_float_12(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 12, deviceID);
}
int test_float_13(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 13, deviceID);
}
int test_float_14(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 14, deviceID);
}
int test_float_15(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 15, deviceID);
}
int test_float_16(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 16, deviceID);
}
int test_float_17(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, 17, deviceID);
}


int test_float_limits_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT_LIMITS, 0, deviceID);
}
int test_float_limits_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT_LIMITS, 1, deviceID);
}
int test_float_limits_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT_LIMITS, 2, deviceID);
}


int test_octal_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_OCTAL, 0, deviceID);
}
int test_octal_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_OCTAL, 1, deviceID);
}
int test_octal_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_OCTAL, 2, deviceID);
}
int test_octal_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_OCTAL, 3, deviceID);
}


int test_unsigned_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_UNSIGNED, 0, deviceID);
}
int test_unsigned_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_UNSIGNED, 1, deviceID);
}


int test_hexadecimal_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, 0, deviceID);
}
int test_hexadecimal_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, 1, deviceID);
}
int test_hexadecimal_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, 2, deviceID);
}
int test_hexadecimal_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, 3, deviceID);
}
int test_hexadecimal_4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, 4, deviceID);
}


int test_char_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_CHAR, 0, deviceID);
}
int test_char_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_CHAR, 1, deviceID);
}
int test_char_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_CHAR, 2, deviceID);
}


int test_string_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_STRING, 0, deviceID);
}
int test_string_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_STRING, 1, deviceID);
}
int test_string_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_STRING, 2, deviceID);
}


int test_vector_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, 0, deviceID);
}
int test_vector_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, 1, deviceID);
}
int test_vector_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, 2, deviceID);
}
int test_vector_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, 3, deviceID);
}
int test_vector_4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, 4, deviceID);
}


int test_address_space_0(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, 0, deviceID);
}
int test_address_space_1(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, 1, deviceID);
}
int test_address_space_2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, 2, deviceID);
}
int test_address_space_3(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, 3, deviceID);
}
int test_address_space_4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, 4, deviceID);
}

int test_buffer_size(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    size_t printf_buff_size = 0;
    const size_t printf_buff_size_req = !gIsEmbedded ? (1024 * 1024UL) : 1024UL;
    const size_t config_size = sizeof(printf_buff_size);
    cl_int err = CL_SUCCESS;

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PRINTF_BUFFER_SIZE, config_size,
                          &printf_buff_size, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("Unable to query CL_DEVICE_PRINTF_BUFFER_SIZE");
        return TEST_FAIL;
    }

    if (printf_buff_size < printf_buff_size_req)
    {
        log_error("CL_DEVICE_PRINTF_BUFFER_SIZE does not meet requirements");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

test_definition test_list[] = {
    ADD_TEST(int_0),           ADD_TEST(int_1),
    ADD_TEST(int_2),           ADD_TEST(int_3),
    ADD_TEST(int_4),           ADD_TEST(int_5),
    ADD_TEST(int_6),           ADD_TEST(int_7),
    ADD_TEST(int_8),

    ADD_TEST(float_0),         ADD_TEST(float_1),
    ADD_TEST(float_2),         ADD_TEST(float_3),
    ADD_TEST(float_4),         ADD_TEST(float_5),
    ADD_TEST(float_6),         ADD_TEST(float_7),
    ADD_TEST(float_8),         ADD_TEST(float_9),
    ADD_TEST(float_10),        ADD_TEST(float_11),
    ADD_TEST(float_12),        ADD_TEST(float_13),
    ADD_TEST(float_14),        ADD_TEST(float_15),
    ADD_TEST(float_16),        ADD_TEST(float_17),

    ADD_TEST(float_limits_0),  ADD_TEST(float_limits_1),
    ADD_TEST(float_limits_2),

    ADD_TEST(octal_0),         ADD_TEST(octal_1),
    ADD_TEST(octal_2),         ADD_TEST(octal_3),

    ADD_TEST(unsigned_0),      ADD_TEST(unsigned_1),

    ADD_TEST(hexadecimal_0),   ADD_TEST(hexadecimal_1),
    ADD_TEST(hexadecimal_2),   ADD_TEST(hexadecimal_3),
    ADD_TEST(hexadecimal_4),

    ADD_TEST(char_0),          ADD_TEST(char_1),
    ADD_TEST(char_2),

    ADD_TEST(string_0),        ADD_TEST(string_1),
    ADD_TEST(string_2),

    ADD_TEST(vector_0),        ADD_TEST(vector_1),
    ADD_TEST(vector_2),        ADD_TEST(vector_3),
    ADD_TEST(vector_4),

    ADD_TEST(address_space_0), ADD_TEST(address_space_1),
    ADD_TEST(address_space_2), ADD_TEST(address_space_3),
    ADD_TEST(address_space_4),

    ADD_TEST(buffer_size),
};

const int test_num = ARRAY_SIZE( test_list );

//-----------------------------------------
// main
//-----------------------------------------
int main(int argc, const char* argv[])
{
    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return -1;
    }

    const char ** argList = (const char **)calloc( argc, sizeof( char*) );

    if( NULL == argList )
    {
        log_error( "Failed to allocate memory for argList array.\n" );
        return 1;
    }

    argList[0] = argv[0];
    size_t argCount = 1;

    for (int i=1; i < argc; ++i) {
        const char *arg = argv[i];
        if (arg == NULL)
            break;

        if (arg[0] == '-')
        {
            arg++;
            while(*arg != '\0')
            {
                switch(*arg) {
                    case 'h':
                        printUsage();
                        return 0;
                    default:
                        log_error( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        printUsage();
                        return 0;
                }
                arg++;
            }
        }
        else {
            argList[argCount] = arg;
            argCount++;
        }
    }

    if (getTempFileName() == -1)
    {
        log_error("getTempFileName failed\n");
        return -1;
    }

    int err = runTestHarnessWithCheck( argCount, argList, test_num, test_list, true, 0, InitCL );

    if(gQueue)
    {
        int error = clFinish(gQueue);
        if (error) {
            log_error("clFinish failed: %d\n", error);
        }
    }

    if(clReleaseCommandQueue(gQueue)!=CL_SUCCESS)
        log_error("clReleaseCommandQueue\n");
    if(clReleaseContext(gContext)!= CL_SUCCESS)
        log_error("clReleaseContext\n");


    free(argList);
    remove(gFileName);
    return err;
}

//-----------------------------------------
// printUsage
//-----------------------------------------
static void printUsage( void )
{
    log_info("test_printf: <optional: testnames> \n");
    log_info("\tdefault is to run the full test on the default device\n");
    log_info("\n");
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}

test_status InitCL( cl_device_id device )
{
    uint32_t device_frequency = 0;
    uint32_t compute_devices = 0;

    int err;
    gFd = acquireOutputStream(&err);
    if (err != 0)
    {
        log_error("Error while redirection stdout to file");
        return TEST_FAIL;
    }

    size_t config_size = sizeof( device_frequency );
#if MULTITHREAD
    if( (err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, config_size, &compute_devices, NULL )) )
#endif
    compute_devices = 1;

    config_size = sizeof(device_frequency);
    if((err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, config_size, &device_frequency, NULL )))
        device_frequency = 1;

    releaseOutputStream(gFd);

    log_info( "\nCompute Device info:\n" );
    log_info( "\tProcessing with %d devices\n", compute_devices );
    log_info( "\tDevice Frequency: %d MHz\n", device_frequency );

    printDeviceHeader( device );

    PrintArch();

    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(1, 2);
    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    gFd = acquireOutputStream(&err);
    if (err != 0)
    {
        log_error("Error while redirection stdout to file");
        return TEST_FAIL;
    }
    cl_context_properties printf_properties[] = {
        CL_PRINTF_CALLBACK_ARM, (cl_context_properties)printfCallBack,
        CL_PRINTF_BUFFERSIZE_ARM, ANALYSIS_BUFFER_SIZE, 0
    };

    cl_context_properties* props = NULL;

    if(is_extension_available(device, "cl_arm_printf"))
    {
        props = printf_properties;
    }

    gContext = clCreateContext(props, 1, &device, notify_callback, NULL, NULL);
    checkNull(gContext, "clCreateContext");

    gQueue = clCreateCommandQueue(gContext, device, 0, NULL);
    checkNull(gQueue, "clCreateCommandQueue");

    releaseOutputStream(gFd);

    // Generate reference results
    generateRef(device);

    return TEST_PASS;
}

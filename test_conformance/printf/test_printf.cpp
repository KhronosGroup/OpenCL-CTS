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
#include "harness/os_helpers.h"
#include "harness/typeWrappers.h"
#include "harness/stringHelpers.h"
#include "harness/conversions.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <errno.h>
#include <memory>
#include <string.h>
#include <vector>

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
#include "harness/parseParameters.h"
#include "harness/rounding_mode.h"

#include <CL/cl_ext.h>

typedef  unsigned int uint32_t;


test_status InitCL( cl_device_id device );

namespace {

//-----------------------------------------
// helper functions declaration
//-----------------------------------------

//Stream helper functions

//Associate stdout stream with the file(gFileName):i.e redirect stdout stream to the specific files (gFileName)
int acquireOutputStream(int* error);

//Close the file(gFileName) associated with the stdout stream and disassociates it.
void releaseOutputStream(int fd);

//Get analysis buffer to verify the correctess of printed data
void getAnalysisBuffer(char* analysisBuffer);

//Kernel builder helper functions

//Check if the test case is for kernel that has argument
int isKernelArgument(testCase* pTestCase, size_t testId);

//Check if the test case treats %p format for void*
int isKernelPFormat(testCase* pTestCase, size_t testId);

//-----------------------------------------
// Static functions declarations
//-----------------------------------------
// Make a program that uses printf for the given type/format,
cl_program makePrintfProgram(cl_kernel* kernel_ptr, const cl_context context,
                             cl_device_id device, const unsigned int testId,
                             const unsigned int testNum,
                             const unsigned int formatNum);

// Creates and execute the printf test for the given device, context, type/format
int doTest(cl_command_queue queue, cl_context context,
           const unsigned int testId, cl_device_id device);

// Check if device supports long
bool isLongSupported(cl_device_id device_id);

// Check if device address space is 64 bits
bool is64bAddressSpace(cl_device_id device_id);

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
int s_test_skip = 0;

cl_context gContext;
cl_command_queue gQueue;
int gFd;

char gFileName[256];

MTdataHolder gMTdata;

// For the sake of proper logging of negative results
std::string gLatestKernelSource;

//-----------------------------------------
// helper functions definition
//-----------------------------------------

//-----------------------------------------
// acquireOutputStream
//-----------------------------------------
int acquireOutputStream(int* error)
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
void releaseOutputStream(int fd)
{
    fflush(stdout);
    streamDup2(fd,fileno(stdout));
    close(fd);
}

//-----------------------------------------
// printfCallBack
//-----------------------------------------
void CL_CALLBACK printfCallBack(const char* printf_data, size_t len,
                                size_t final, void* user_data)
{
    fwrite(printf_data, 1, len, stdout);
}

//-----------------------------------------
// getAnalysisBuffer
//-----------------------------------------
void getAnalysisBuffer(char* analysisBuffer)
{
    FILE *fp;
    memset(analysisBuffer,0,ANALYSIS_BUFFER_SIZE);

    fp = fopen(gFileName, "r");
    if (NULL == fp)
        log_error("Failed to open analysis buffer ('%s')\n", strerror(errno));
    else if (0
             == std::fread(analysisBuffer, sizeof(analysisBuffer[0]),
                           ANALYSIS_BUFFER_SIZE, fp))
        log_error("No data read from analysis buffer\n");

    fclose(fp);
}

//-----------------------------------------
// isKernelArgument
//-----------------------------------------
int isKernelArgument(testCase* pTestCase, size_t testId)
{
    return strcmp(pTestCase->_genParameters[testId].addrSpaceArgumentTypeQualifier,"");
}
//-----------------------------------------
// isKernelPFormat
//-----------------------------------------
int isKernelPFormat(testCase* pTestCase, size_t testId)
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
// makeMixedFormatPrintfProgram
// Generates in-flight printf kernel with format string including:
//     -data before conversion flags (randomly generated ascii string)
//     -randomly generated conversion flags (integer or floating point)
//     -data after conversion flags (randomly generated ascii string).
// Moreover it generates suitable arguments.
// example: printf("zH, %u, %a, D+{gy\n", -929240879, 24295.671875f)
//-----------------------------------------
cl_program makeMixedFormatPrintfProgram(cl_kernel* kernel_ptr,
                                        const cl_context context,
                                        const cl_device_id device,
                                        const unsigned int testId,
                                        const unsigned int testNum,
                                        const std::string& testname)
{
    auto gen_char = [&]() {
        static const char dict[] = {
            " \t!#$&()*+,-./"
            "123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~"
        };
        return dict[genrand_int32(gMTdata) % ((int)sizeof(dict) - 1)];
    };

    std::array<std::vector<std::string>, 2> formats = {
        { { "%f", "%e", "%g", "%a", "%F", "%E", "%G", "%A" },
          { "%d", "%i", "%u", "%x", "%o", "%X" } }
    };
    std::vector<char> data_before(2 + genrand_int32(gMTdata) % 8);
    std::vector<char> data_after(2 + genrand_int32(gMTdata) % 8);

    std::generate(data_before.begin(), data_before.end(), gen_char);
    std::generate(data_after.begin(), data_after.end(), gen_char);

    cl_uint num_args = 2 + genrand_int32(gMTdata) % 4;

    // Map device rounding to CTS rounding type
    // get_default_rounding_mode supports RNE and RTZ
    auto get_rounding = [](const cl_device_fp_config& fpConfig) {
        if (fpConfig == CL_FP_ROUND_TO_NEAREST)
        {
            return kRoundToNearestEven;
        }
        else if (fpConfig == CL_FP_ROUND_TO_ZERO)
        {
            return kRoundTowardZero;
        }
        else
        {
            assert(false && "Unreachable");
        }
        return kDefaultRoundingMode;
    };

    const RoundingMode hostRound = get_round();
    RoundingMode deviceRound = get_rounding(get_default_rounding_mode(device));

    std::ostringstream format_str;
    std::ostringstream ref_str;
    std::ostringstream source_gen;
    std::ostringstream args_str;
    source_gen << "__kernel void " << testname
               << "(void)\n"
                  "{\n"
                  "   printf(\"";
    for (auto it : data_before)
    {
        format_str << it;
        ref_str << it;
    }
    format_str << ", ";
    ref_str << ", ";


    for (cl_uint i = 0; i < num_args; i++)
    {
        std::uint8_t is_int = genrand_int32(gMTdata) % 2;

        // Set CPU rounding mode to match that of the device
        set_round(deviceRound, is_int != 0 ? kint : kfloat);

        std::string format =
            formats[is_int][genrand_int32(gMTdata) % formats[is_int].size()];
        format_str << format << ", ";

        if (is_int)
        {
            int arg = genrand_int32(gMTdata);
            args_str << str_sprintf("%d", arg) << ", ";
            ref_str << str_sprintf(format, arg) << ", ";
        }
        else
        {
            const float max_range = 100000.f;
            float arg = get_random_float(-max_range, max_range, gMTdata);
            args_str << str_sprintf("%f", arg) << "f, ";
            ref_str << str_sprintf(format, arg) << ", ";
        }
    }
    // Restore the original CPU rounding mode
    set_round(hostRound, kfloat);

    for (auto it : data_after)
    {
        format_str << it;
        ref_str << it;
    }

    {
        std::ostringstream args_cpy;
        args_cpy << args_str.str();
        args_cpy.seekp(-2, std::ios_base::end);
        args_cpy << ")\n";
        log_info("%d) testing printf(\"%s\\n\", %s", testNum,
                 format_str.str().c_str(), args_cpy.str().c_str());
    }

    args_str.seekp(-2, std::ios_base::end);
    args_str << ");\n}\n";


    source_gen << format_str.str() << "\\n\""
               << ", " << args_str.str();

    std::string kernel_source = source_gen.str();
    const char* ptr = kernel_source.c_str();

    cl_program program;
    cl_int err = create_single_kernel_helper(context, &program, kernel_ptr, 1,
                                             &ptr, testname.c_str());

    gLatestKernelSource = kernel_source.c_str();

    // Save the reference result
    allTestCase[testId]->_correctBuffer.push_back(ref_str.str());

    if (!program || err)
    {
        log_error("create_single_kernel_helper failed\n");
        return NULL;
    }

    return program;
}

//-----------------------------------------
// makePrintfProgram
//-----------------------------------------
cl_program makePrintfProgram(cl_kernel* kernel_ptr, const cl_context context,
                             const cl_device_id device,
                             const unsigned int testId,
                             const unsigned int testNum,
                             const unsigned int formatNum)
{
    int err;
    cl_program program;
    char testname[256] = {0};
    char addrSpaceArgument[256] = {0};
    char addrSpacePAddArgument[256] = {0};
    char extension[128] = { 0 };

    //Update testname
    std::snprintf(testname, sizeof(testname), "%s%d", "test", testId);

    if (allTestCase[testId]->_type == TYPE_HALF
        || allTestCase[testId]->_type == TYPE_HALF_LIMITS)
        strcpy(extension, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");


    //Update addrSpaceArgument and addrSpacePAddArgument types, based on FULL_PROFILE/EMBEDDED_PROFILE
    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        std::snprintf(addrSpaceArgument, sizeof(addrSpaceArgument), "%s",
                      allTestCase[testId]
                          ->_genParameters[testNum]
                          .addrSpaceArgumentTypeQualifier);

        std::snprintf(
            addrSpacePAddArgument, sizeof(addrSpacePAddArgument), "%s",
            allTestCase[testId]->_genParameters[testNum].addrSpacePAdd);
    }

    if (strlen(addrSpaceArgument) == 0)
        std::snprintf(addrSpaceArgument, sizeof(addrSpaceArgument), "void");

    // create program based on its type

    if(allTestCase[testId]->_type == TYPE_VECTOR)
    {
        if (strcmp(allTestCase[testId]->_genParameters[testNum].dataType,
                   "half")
            == 0)
            strcpy(extension,
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");

        // Program Source code for vector
        const char* sourceVec[] = {
            extension,
            "__kernel void ",
            testname,
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

        err = create_single_kernel_helper(
            context, &program, kernel_ptr,
            sizeof(sourceVec) / sizeof(sourceVec[0]), sourceVec, testname);

        gLatestKernelSource =
            concat_kernel(sourceVec, sizeof(sourceVec) / sizeof(sourceVec[0]));
    }
    else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        // Program Source code for address space
        const char* sourceAddrSpace[] = {
            "__kernel void ",
            testname,
            "(",
            addrSpaceArgument,
            ")\n{\n",
            allTestCase[testId]
                ->_genParameters[testNum]
                .addrSpaceVariableTypeQualifier,
            "printf(",
            allTestCase[testId]
                ->_genParameters[testNum]
                .genericFormats[formatNum]
                .c_str(),
            ",",
            allTestCase[testId]->_genParameters[testNum].addrSpaceParameter,
            "); ",
            addrSpacePAddArgument,
            "\n}\n"
        };

        err = create_single_kernel_helper(context, &program, kernel_ptr,
                                          sizeof(sourceAddrSpace)
                                              / sizeof(sourceAddrSpace[0]),
                                          sourceAddrSpace, testname);

        gLatestKernelSource =
            concat_kernel(sourceAddrSpace,
                          sizeof(sourceAddrSpace) / sizeof(sourceAddrSpace[0]));
    }
    else if (allTestCase[testId]->_type == TYPE_MIXED_FORMAT_RANDOM)
    {
        return makeMixedFormatPrintfProgram(kernel_ptr, context, device, testId,
                                            testNum, testname);
    }
    else
    {
        // Program Source code for int,float,octal,hexadecimal,char,string
        std::ostringstream sourceGen;
        sourceGen << extension << "__kernel void " << testname
                  << "(void)\n"
                     "{\n"
                     "   printf(\""
                  << allTestCase[testId]
                         ->_genParameters[testNum]
                         .genericFormats[formatNum]
                         .c_str()
                  << "\\n\"";

        if (allTestCase[testId]->_genParameters[testNum].dataRepresentation)
        {
            sourceGen << ","
                      << allTestCase[testId]
                             ->_genParameters[testNum]
                             .dataRepresentation;
        }

        sourceGen << ");\n}\n";

        std::string kernel_source = sourceGen.str();
        const char* ptr = kernel_source.c_str();

        err = create_single_kernel_helper(context, &program, kernel_ptr, 1,
                                          &ptr, testname);

        gLatestKernelSource = kernel_source.c_str();
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
bool isLongSupported(cl_device_id device_id)
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
bool is64bAddressSpace(cl_device_id device_id)
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
// subtest_fail
//-----------------------------------------
void subtest_fail(const char* msg, ...)
{
    if (msg)
    {
        va_list argptr;
        va_start(argptr, msg);
        vfprintf(stderr, msg, argptr);
        va_end(argptr);
    }
    ++s_test_fail;
    ++s_test_cnt;
}

//-----------------------------------------
// logTestType - printout test details
//-----------------------------------------

void logTestType(const unsigned testId, const unsigned testNum,
                 unsigned formatNum)
{
    if (allTestCase[testId]->_type == TYPE_VECTOR)
    {
        log_info(
            "%d)testing printf(\"%sv%s%s\",%s)\n", testNum,
            allTestCase[testId]->_genParameters[testNum].vectorFormatFlag,
            allTestCase[testId]->_genParameters[testNum].vectorSize,
            allTestCase[testId]->_genParameters[testNum].vectorFormatSpecifier,
            allTestCase[testId]->_genParameters[testNum].dataRepresentation);
    }
    else if (allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        if (isKernelArgument(allTestCase[testId], testNum))
        {
            log_info("%d)testing kernel //argument %s \n   printf(%s,%s)\n",
                     testNum,
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .addrSpaceArgumentTypeQualifier,
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .genericFormats[formatNum]
                         .c_str(),
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .addrSpaceParameter);
        }
        else
        {
            log_info("%d)testing kernel //variable %s \n   printf(%s,%s)\n",
                     testNum,
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .addrSpaceVariableTypeQualifier,
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .genericFormats[formatNum]
                         .c_str(),
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .addrSpaceParameter);
        }
    }
    else if (allTestCase[testId]->_type != TYPE_MIXED_FORMAT_RANDOM)
    {
        log_info("%d)testing printf(\"%s\"", testNum,
                 allTestCase[testId]
                     ->_genParameters[testNum]
                     .genericFormats[formatNum]
                     .c_str());
        if (allTestCase[testId]->_genParameters[testNum].dataRepresentation)
            log_info(",%s",
                     allTestCase[testId]
                         ->_genParameters[testNum]
                         .dataRepresentation);
        log_info(")\n");
    }

    fflush(stdout);
}

//-----------------------------------------
// doTest
//-----------------------------------------
int doTest(cl_command_queue queue, cl_context context,
           const unsigned int testId, cl_device_id device)
{
    int err = TEST_FAIL;

    if ((allTestCase[testId]->_type == TYPE_HALF
         || allTestCase[testId]->_type == TYPE_HALF_LIMITS)
        && !is_extension_available(device, "cl_khr_fp16"))
    {
        log_info("Skipping half because cl_khr_fp16 extension is not "
                 "supported.\n");
        return TEST_SKIPPED_ITSELF;
    }

    auto& genParams = allTestCase[testId]->_genParameters;

    auto fail_count = s_test_fail;
    auto pass_count = s_test_cnt;
    auto skip_count = s_test_skip;

    for (unsigned testNum = 0; testNum < genParams.size(); testNum++)
    {
        if (allTestCase[testId]->_type == TYPE_VECTOR)
        {
            if ((strcmp(allTestCase[testId]->_genParameters[testNum].dataType,
                        "half")
                 == 0)
                && !is_extension_available(device, "cl_khr_fp16"))
            {
                log_info("Skipping half because cl_khr_fp16 extension is not "
                         "supported.\n");

                s_test_skip++;
                s_test_cnt++;
                continue;
            }

            // Long support for varible type
            if (!strcmp(allTestCase[testId]->_genParameters[testNum].dataType,
                        "long")
                && !isLongSupported(device))
            {
                log_info("Long is not supported, test not run.\n");
                s_test_skip++;
                s_test_cnt++;
                continue;
            }
        }

        auto genParamsVec = allTestCase[testId]->_genParameters;
        auto genFormatVec = genParamsVec[testNum].genericFormats;

        for (unsigned formatNum = 0; formatNum < genFormatVec.size();
             formatNum++)
        {
            logTestType(testId, testNum, formatNum);

            clProgramWrapper program;
            clKernelWrapper kernel;
            clMemWrapper d_out;
            clMemWrapper d_a;
            char _analysisBuffer[ANALYSIS_BUFFER_SIZE];
            cl_uint out32 = 0;
            cl_ulong out64 = 0;
            int fd = -1;

            // Define an index space (global work size) of threads for
            // execution.
            size_t globalWorkSize[1];

            program = makePrintfProgram(&kernel, context, device, testId,
                                        testNum, formatNum);
            if (!program || !kernel)
            {
                subtest_fail(nullptr);
                continue;
            }

            // For address space test if there is kernel argument - set it
            if (allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
            {
                if (isKernelArgument(allTestCase[testId], testNum))
                {
                    int a = 2;
                    d_a = clCreateBuffer(
                        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(int), &a, &err);
                    if (err != CL_SUCCESS || d_a == NULL)
                    {
                        subtest_fail("clCreateBuffer failed\n");
                        continue;
                    }
                    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
                    if (err != CL_SUCCESS)
                    {
                        subtest_fail("clSetKernelArg failed\n");
                        continue;
                    }
                }
                // For address space test if %p is tested
                if (isKernelPFormat(allTestCase[testId], testNum))
                {
                    d_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(cl_ulong), NULL, &err);
                    if (err != CL_SUCCESS || d_out == NULL)
                    {
                        subtest_fail("clCreateBuffer failed\n");
                        continue;
                    }
                    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
                    if (err != CL_SUCCESS)
                    {
                        subtest_fail("clSetKernelArg failed\n");
                        continue;
                    }
                }
            }

            fd = acquireOutputStream(&err);
            if (err != 0)
            {
                subtest_fail("Error while redirection stdout to file");
                continue;
            }
            globalWorkSize[0] = 1;
            cl_event ndrEvt;
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize,
                                         NULL, 0, NULL, &ndrEvt);
            if (err != CL_SUCCESS)
            {
                releaseOutputStream(fd);
                subtest_fail("\n clEnqueueNDRangeKernel failed errcode:%d\n",
                             err);
                continue;
            }

            fflush(stdout);
            err = clFlush(queue);
            if (err != CL_SUCCESS)
            {
                releaseOutputStream(fd);
                subtest_fail("clFlush failed : %d\n", err);
                continue;
            }
            // Wait until kernel finishes its execution and (thus) the output
            // printed from the kernel is immediately printed
            err = waitForEvent(&ndrEvt);

            releaseOutputStream(fd);

            if (err != CL_SUCCESS)
            {
                subtest_fail("waitforEvent failed : %d\n", err);
                continue;
            }
            fflush(stdout);

            if (allTestCase[testId]->_type == TYPE_ADDRESS_SPACE
                && isKernelPFormat(allTestCase[testId], testNum))
            {
                // Read the OpenCL output buffer (d_out) to the host output
                // array (out)
                if (!is64bAddressSpace(device)) // 32-bit address space
                {
                    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                        sizeof(cl_int), &out32, 0, NULL, NULL);
                }
                else // 64-bit address space
                {
                    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                        sizeof(cl_ulong), &out64, 0, NULL,
                                        NULL);
                }
            }

            //
            // Get the output printed from the kernel to _analysisBuffer
            // and verify its correctness
            getAnalysisBuffer(_analysisBuffer);
            if (!is64bAddressSpace(device)) // 32-bit address space
            {
                if (0
                    != verifyOutputBuffer(_analysisBuffer, allTestCase[testId],
                                          testNum, (cl_ulong)out32))
                {
                    subtest_fail(
                        "verifyOutputBuffer failed with kernel: "
                        "\n%s\n expected: %s\n got:      %s\n",
                        gLatestKernelSource.c_str(),
                        allTestCase[testId]->_correctBuffer[testNum].c_str(),
                        _analysisBuffer);
                    continue;
                }
            }
            else // 64-bit address space
            {
                if (0
                    != verifyOutputBuffer(_analysisBuffer, allTestCase[testId],
                                          testNum, out64))
                {
                    subtest_fail(
                        "verifyOutputBuffer failed with kernel: "
                        "\n%s\n expected: %s\n got:      %s\n",
                        gLatestKernelSource.c_str(),
                        allTestCase[testId]->_correctBuffer[testNum].c_str(),
                        _analysisBuffer);
                    continue;
                }
            }
        }
        ++s_test_cnt;
    }

    // all subtests skipped ?
    if (s_test_skip - skip_count == s_test_cnt - pass_count)
        return TEST_SKIPPED_ITSELF;
    return s_test_fail - fail_count;
}

}

int test_int(cl_device_id deviceID, cl_context context, cl_command_queue queue,
             int num_elements)
{
    return doTest(gQueue, gContext, TYPE_INT, deviceID);
}

int test_half(cl_device_id deviceID, cl_context context, cl_command_queue queue,
              int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HALF, deviceID);
}

int test_half_limits(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HALF_LIMITS, deviceID);
}

int test_float(cl_device_id deviceID, cl_context context,
               cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT, deviceID);
}

int test_float_limits(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FLOAT_LIMITS, deviceID);
}

int test_octal(cl_device_id deviceID, cl_context context,
               cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_OCTAL, deviceID);
}

int test_unsigned(cl_device_id deviceID, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_UNSIGNED, deviceID);
}

int test_hexadecimal(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_HEXADEC, deviceID);
}

int test_char(cl_device_id deviceID, cl_context context, cl_command_queue queue,
              int num_elements)
{
    return doTest(gQueue, gContext, TYPE_CHAR, deviceID);
}

int test_string(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_STRING, deviceID);
}

int test_format_string(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_FORMAT_STRING, deviceID);
}

int test_vector(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_VECTOR, deviceID);
}

int test_address_space(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_ADDRESS_SPACE, deviceID);
}

int test_mixed_format_random(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    return doTest(gQueue, gContext, TYPE_MIXED_FORMAT_RANDOM, deviceID);
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
    ADD_TEST(int),
    ADD_TEST(half),
    ADD_TEST(half_limits),
    ADD_TEST(float),
    ADD_TEST(float_limits),
    ADD_TEST(octal),
    ADD_TEST(unsigned),
    ADD_TEST(hexadecimal),
    ADD_TEST(char),
    ADD_TEST(string),
    ADD_TEST(format_string),
    ADD_TEST(vector),
    ADD_TEST(address_space),
    ADD_TEST(buffer_size),
    ADD_TEST(mixed_format_random),
};

const int test_num = ARRAY_SIZE( test_list );

//-----------------------------------------
// printUsage
//-----------------------------------------
static void printUsage(void)
{
    log_info("test_printf: <optional: testnames> \n");
    log_info("\tdefault is to run the full test on the default device\n");
    log_info("\n");
    for (int i = 0; i < test_num; i++)
    {
        log_info("\t%s\n", test_list[i].name);
    }
}

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

    char* pcTempFname = get_temp_filename();
    if (pcTempFname != nullptr)
    {
        strncpy(gFileName, pcTempFname, sizeof(gFileName));
    }

    free(pcTempFname);

    if (strlen(gFileName) == 0)
    {
        log_error("get_temp_filename failed\n");
        return -1;
    }

    gMTdata = MTdataHolder(gRandomSeed);

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

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if (fpConfigHalf == CL_FP_ROUND_TO_NEAREST)
        {
            half_rounding_mode = CL_HALF_RTE;
        }
        else if (fpConfigHalf == CL_FP_ROUND_TO_ZERO)
        {
            half_rounding_mode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
        }
    }

    // Generate reference results
    generateRef(device);

    return TEST_PASS;
}

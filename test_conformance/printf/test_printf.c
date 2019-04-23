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
#include "../../test_common/harness/compat.h"
#if !defined(_WIN32)
#include <stdint.h>
#endif

#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include <memory>

#if ! defined( _WIN32)
#include <sys/sysctl.h>
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
#include "../../test_common/harness/testHarness.h"
#endif

#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/mt19937.h"

typedef  unsigned int uint32_t;


//-----------------------------------------
// Static helper functions declaration
//-----------------------------------------

//Stream helper functions

//Associate stdout stream with the file(gFileName):i.e redirect stdout stream to the specific files (gFileName)
static int acquireOutputStream();

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
static int doTest(cl_command_queue queue, cl_context context, const unsigned int testId, const unsigned int testNum, cl_device_id device,bool isLongSupport = true);

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
static int acquireOutputStream()
{
    int fd = streamDup(fileno(stdout));
    freopen(gFileName,"w",stdout);
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
// getAnalysisBuffer
//-----------------------------------------
static void getAnalysisBuffer(char* analysisBuffer)
{
    FILE *fp;
    memset(analysisBuffer,0,ANALYSIS_BUFFER_SIZE);

    fp = fopen(gFileName,"r");
    if(NULL == fp)
        log_error("Failed to open analysis buffer\n");
    else
    while(fgets(analysisBuffer,ANALYSIS_BUFFER_SIZE , fp) != NULL );
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
    cl_int status = CL_SUCCESS;
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = clGetEventInfo(
            *event,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            sizeof(cl_int),
            &eventStatus,
            NULL);
        if(status != CL_SUCCESS)
        {
            log_error("clGetEventInfo failed");
            return status;
        }
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
    int err,i;
    cl_program program;
    cl_device_id devID;
    char buildLog[ 1024 * 128 ];
    char testname[256] = {0};
    char addrSpaceArgument[256] = {0};

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
        ");",
        allTestCase[testId]->_genParameters[testNum].addrSpacePAdd,
        "\n}\n"
    };

    //Update testname
    sprintf(testname,"%s%d","test",testId);

    //Update addrSpaceArgument type,based on FULL_PROFILE/EMBEDDED_PROFILE
    if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        sprintf(addrSpaceArgument,allTestCase[testId]->_genParameters[testNum].addrSpaceArgumentTypeQualifier);
        if(isKernelPFormat(allTestCase[testId],testNum) && (!isLongSupport || !is64bAddrSpace))
        {
            char* pRep = strstr(addrSpaceArgument,"long");
            if(pRep != NULL)
                strncpy(pRep,"int ",4);
        }
    }

  if (strlen(addrSpaceArgument) == 0)
    sprintf(addrSpaceArgument,"void");

    // create program based on its type

    if(allTestCase[testId]->_type == TYPE_VECTOR)
    {
        program = clCreateProgramWithSource( context,sizeof(sourceVec)/sizeof(sourceVec[0]),sourceVec, NULL, NULL);
    }
    else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
    {
        program = clCreateProgramWithSource( context,sizeof(sourceAddrSpace)/sizeof(sourceAddrSpace[0]),sourceAddrSpace, NULL, NULL);
    }
    else
    {
        program = clCreateProgramWithSource( context,sizeof(sourceGen)/sizeof(sourceGen[0]),sourceGen, NULL, NULL);
    }

    if (!program) {
        log_error("clCreateProgramWithSource failed\n");
        return NULL;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {

        log_error("clBuildProgramExecutable failed errcode:%d\n", err);


        err = clGetProgramInfo( program, CL_PROGRAM_DEVICES, sizeof( devID ), &devID, NULL );
        if (err){
            log_error("Unable to get program's device: %d\n",err );
            return NULL;
        }
        err = clGetProgramBuildInfo( program, devID, CL_PROGRAM_BUILD_LOG, sizeof( buildLog ), buildLog, NULL );
        if (err){
            log_error("Unable to get program's build log: %d\n",err );
            return NULL;
        }
        size_t sourceLen;
        const char** source;

        if(allTestCase[testId]->_type == TYPE_VECTOR)
        {
            sourceLen = sizeof(sourceVec) / sizeof( sourceVec[0] );
            source = sourceVec;
        }
        else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
        {
            sourceLen = sizeof(sourceAddrSpace) / sizeof( sourceAddrSpace[0] );
            source = sourceAddrSpace;
        }
        else
        {
            sourceLen = sizeof(sourceGen) / sizeof( sourceGen[0] );
            source = sourceGen;
        }
        log_error( "Build log is: ------------\n" );
        log_error( "%s\n", buildLog );
        log_error( "----------\n" );
        log_error( " Source is ----------------\n");
        for(i = 0; i < sourceLen; ++i) {
            log_error("%s", source[i]);
        }

        log_error( "----------\n" );
        return NULL;
    }

    *kernel_ptr = clCreateKernel(program, testname, &err);
    if ( err ) {
        log_error("clCreateKernel failed (%d)\n", err);
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
        // Device extention
        status = clGetDeviceInfo(
            device_id,
            CL_DEVICE_EXTENSIONS,
            0,
            NULL,
            &tempSize);

        if(status != CL_SUCCESS)
        {
            log_error("*** clGetDeviceInfo FAILED ***\n\n");
            return false;
        }

        std::unique_ptr<char[]> devExt(new char[tempSize]);
        if(devExt == NULL)
        {
            log_error("Failed to allocate memory(devExt)");
            return false;
        }

        status = clGetDeviceInfo(
            device_id,
            CL_DEVICE_EXTENSIONS,
            sizeof(char) * tempSize,
            devExt.get(),
            NULL);

        extSupport  = (strstr(devExt.get(),"cles_khr_int64") != NULL);
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
static int doTest(cl_command_queue queue, cl_context context, const unsigned int testId, const unsigned int testNum, cl_device_id device,bool isLongSupport)
{
    int err;
    cl_program program;
    cl_kernel  kernel;
    cl_mem d_out, d_a;
    int has_d_out = 0;
    int has_d_a = 0;
    char _analysisBuffer[ANALYSIS_BUFFER_SIZE];

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
            has_d_a = 1;
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
            sizeof(cl_long), NULL, &err);
            if(err!= CL_SUCCESS || d_out == NULL) {
                log_error("clCreateBuffer failed\n");
                goto exit;
            }
            has_d_out = 1;
            err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
            if(err!= CL_SUCCESS) {
                log_error("clSetKernelArg failed\n");
                goto exit;
            }
        }
    }

    globalWorkSize[0] = 1;
    cl_event ndrEvt;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL,&ndrEvt);
    if (err != CL_SUCCESS) {
        log_error("\n clEnqueueNDRangeKernel failed errcode:%d\n", err);
        ++s_test_fail;
        goto exit;
    }

    fflush(stdout);
    err = clFlush(queue);
    if(err != CL_SUCCESS)
    {
        log_error("clFlush failed\n");
        goto exit;
    }
    //Wait until kernel finishes its execution and (thus) the output printed from the kernel
    //is immidatly printed
    err = waitForEvent(&ndrEvt);

    if(err != CL_SUCCESS)
    {
        log_error("waitforEvent failed\n");
        goto exit;
    }
    fflush(stdout);

    cl_uint out32;
    cl_ulong out64;
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
  if(has_d_out)
    if(clReleaseMemObject(d_out) != CL_SUCCESS)
      log_error("clReleaseMemObject failed\n");
    if(has_d_a)
      if(clReleaseMemObject(d_a) != CL_SUCCESS)
        log_error("clReleaseMemObject failed\n");
    if(clReleaseKernel(kernel) != CL_SUCCESS)
        log_error("clReleaseKernel failed\n");
    if(clReleaseProgram(program) != CL_SUCCESS)
        log_error("clReleaseProgram failed\n");
    ++s_test_cnt;
    return err;
}
//-----------------------------------------
// printUsage
//-----------------------------------------
static void printUsage( void )
{
    log_info("test_printf:  [-cghw] [start_test_num] \n");
    log_info("  default is to run the full test on the default device\n");
    log_info("  start_test_num will start running from that num\n");
}

//-----------------------------------------
// printArch
//-----------------------------------------
static void printArch( void )
{
    log_info( "sizeof( void*) = %d\n", (int) sizeof( void *) );

#if defined( __APPLE__ )

#if defined( __ppc__ )
    log_info( "ARCH:\tppc\n" );
#elif defined( __ppc64__ )
    log_info( "ARCH:\tppc64\n" );
#elif defined( __i386__ )
    log_info( "ARCH:\ti386\n" );
#elif defined( __x86_64__ )
    log_info( "ARCH:\tx86_64\n" );
#elif defined( __arm__ )
    log_info( "ARCH:\tarm\n" );
#elif defined( __aarch64__ )
    vlog( "\tARCH:\taarch64\n" );
#else
#error unknown arch
#endif

    int type = 0;
    size_t typeSize = sizeof( type );
    sysctlbyname( "hw.cputype", &type, &typeSize, NULL, 0 );
    log_info( "cpu type:\t%d\n", type );
    typeSize = sizeof( type );
    sysctlbyname( "hw.cpusubtype", &type, &typeSize, NULL, 0 );
    log_info( "cpu subtype:\t%d\n", type );

#endif
}

//-----------------------------------------
// notify_callback
//-----------------------------------------
void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    log_info( "%s\n", errinfo );
}


//-----------------------------------------
// main
//-----------------------------------------
int main(int argc, char* argv[]) {
    int i;
    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    cl_platform_id platform_id;
    long           test_start_num = 0;   // start test number
    const char*    exec_testname = NULL;
    cl_device_id      device_id;
    uint32_t      device_frequency = 0;
    uint32_t       compute_devices = 0;



    test_start();

    // Check the environmental to see if there is device preference
    char *device_env = getenv("CL_DEVICE_TYPE");
    if (device_env != NULL) {
        if( strcmp( device_env, "gpu" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_GPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_GPU;
        else if( strcmp( device_env, "cpu" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_CPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_CPU;
        else if( strcmp( device_env, "accelerator" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( device_env, "default" ) == 0 || strcmp( device_env, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            device_type = CL_DEVICE_TYPE_DEFAULT;
        else
        {
            log_error( "Unknown CL_DEVICE_TYPE environment variable: %s.\nAborting...\n", device_env );
            abort();
        }
    }

    // Determine if we want to run a particular test or if we want to
    // start running from a certain point and if we want to run on cpu/gpu
    // usage: test_printf [test_name] [start test num] [run_long]
    // default is to run all tests on the gpu and be short
    // test names are of the form printf_testId

    for (i=1; i < argc; ++i) {
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
            char* t = NULL;
            long num = strtol(argv[i], &t, 0);
            if (t != argv[i])
                test_start_num = num;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_CPU" ) )
                device_type = CL_DEVICE_TYPE_CPU;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_GPU" ) )
                device_type = CL_DEVICE_TYPE_GPU;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_ACCELERATOR" ) )
                device_type = CL_DEVICE_TYPE_ACCELERATOR;
            else if( 0 == strcmp( argv[i], "CL_DEVICE_TYPE_DEFAULT" ) )
                device_type = CL_DEVICE_TYPE_DEFAULT;
            else {
                // assume it is a test name that we want to execute
                exec_testname = argv[i];
            }
        }
    }

    if (getTempFileName() == -1)
    {
        log_error("getTempFileName failed\n");
        return -1;
    }

    int err;
    int fd = acquireOutputStream();

    // Get platform
    err = clGetPlatformIDs(1, &platform_id, NULL);
    checkErr(err,"clGetPlatformIDs failed");


    // Get Device information
    err = clGetDeviceIDs(platform_id, device_type, 1, &device_id, 0);
    checkErr(err,"clGetComputeDevices");


    err =  clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    checkErr(err,"clGetComputeConfigInfo 1");


    size_t config_size = sizeof( device_frequency );
#if MULTITHREAD
    if( (err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, config_size, &compute_devices, NULL )) )
#endif
    compute_devices = 1;

    config_size = sizeof(device_frequency);
    if((err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, config_size, &device_frequency, NULL )))
        device_frequency = 1;

    releaseOutputStream(fd);

    log_info( "\nCompute Device info:\n" );
    log_info( "\tProcessing with %d devices\n", compute_devices );
    log_info( "\tDevice Frequency: %d MHz\n", device_frequency );



    printDeviceHeader( device_id );

    printArch();

    log_info( "Test binary built %s %s\n", __DATE__, __TIME__ );

    fd = acquireOutputStream();

    cl_context context = clCreateContext(NULL, 1, &device_id, notify_callback, NULL, NULL);
    checkNull(context, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);
    checkNull(queue, "clCreateCommandQueue");

    // Forall types
    for (int testId = 0; testId < TYPE_COUNT; ++testId) {
        if ((testId + 1) < test_start_num) {
            releaseOutputStream(fd);
            log_info("\n*** Skipping printf  for %s ***\n",strType[testId]);
            fd = acquireOutputStream();
        }
        else {
            releaseOutputStream(fd);
            log_info("\n*** Testing printf for %s ***\n",strType[testId]);
            fd = acquireOutputStream();
            //For all formats
            for(unsigned int testNum = 0;testNum < allTestCase[testId]->_testNum;++testNum){
                releaseOutputStream(fd);
                if(allTestCase[testId]->_type == TYPE_VECTOR)
                    log_info("%d)testing printf(\"%sv%s%s\",%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].vectorFormatFlag,allTestCase[testId]->_genParameters[testNum].vectorSize,
                    allTestCase[testId]->_genParameters[testNum].vectorFormatSpecifier,allTestCase[testId]->_genParameters[testNum].dataRepresentation);
                else if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE)
                {
                    if(isKernelArgument)
                        log_info("%d)testing kernel //argument %s \n   printf(%s,%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].addrSpaceArgumentTypeQualifier,
                        allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].addrSpaceParameter);
                    else
                        log_info("%d)testing kernel //variable %s \n   printf(%s,%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].addrSpaceVariableTypeQualifier,
                        allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].addrSpaceParameter);
                }
                else
                    log_info("%d)testing printf(\"%s\",%s)\n",testNum,allTestCase[testId]->_genParameters[testNum].genericFormat,allTestCase[testId]->_genParameters[testNum].dataRepresentation);
                fd = acquireOutputStream();

                // Long support for varible type
                if(allTestCase[testId]->_type == TYPE_VECTOR && !strcmp(allTestCase[testId]->_genParameters[testNum].dataType,"long") && !isLongSupported(device_id))
                        continue;

                // Long support for address in FULL_PROFILE/EMBEDDED_PROFILE
                bool isLongSupport = true;
                if(allTestCase[testId]->_type == TYPE_ADDRESS_SPACE && isKernelPFormat(allTestCase[testId],testNum) && !isLongSupported(device_id))
                    isLongSupport = false;

                // Perform the test
                if (doTest(queue, context,testId,testNum,device_id,isLongSupport) != 0)
                {
                    releaseOutputStream(fd);
                    log_error("*** FAILED ***\n\n");
                    fd = acquireOutputStream();
                }
                else
                {
                    releaseOutputStream(fd);
                    log_info("Passed\n");
                    fd = acquireOutputStream();
                }
            }
        }
    }

    int error = clFinish(queue);
    if (error) {
        log_error("clFinish failed: %d\n", error);
    }

    if(clReleaseCommandQueue(queue)!=CL_SUCCESS)
        log_error("clReleaseCommandQueue\n");
    if(clReleaseContext(context)!= CL_SUCCESS)
        log_error("clReleaseContext\n");

    releaseOutputStream(fd);


    if (s_test_fail == 0) {
        if (s_test_cnt > 1)
            log_info("PASSED %d of %d tests.\n", s_test_cnt, s_test_cnt);
        else
            log_info("PASSED test.\n");
    } else if (s_test_fail > 0) {
        if (s_test_cnt > 1)
        {
            log_error("FAILED %d of %d tests.\n", s_test_fail, s_test_cnt);
            log_info("PASSED %d of %d tests.\n", s_test_cnt - s_test_fail, s_test_cnt);
        }
        else
            log_error(" FAILED test.\n");
    }

    remove(gFileName);
    test_finish();
    return s_test_fail;
}

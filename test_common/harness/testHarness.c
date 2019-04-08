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
#include "testHarness.h"
#include "compat.h"
#include <stdio.h>
#include <string.h>
#include "threadTesting.h"
#include "errorHelpers.h"
#include "kernelHelpers.h"
#include "fpcontrol.h"
#include "typeWrappers.h"
#include "parseParameters.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <time.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

int gTestsPassed = 0;
int gTestsFailed = 0;
cl_uint gRandomSeed = 0;
cl_uint gReSeed = 0;

int     gFlushDenormsToZero = 0;
int     gInfNanSupport = 1;
int     gIsEmbedded = 0;
int     gIsOpenCL_C_1_0_Device = 0;
int     gIsOpenCL_1_0_Device = 0;
int     gHasLong = 1;

#define DEFAULT_NUM_ELEMENTS        0x4000

int runTestHarness( int argc, const char *argv[], int testNum, test_definition testList[],
                    int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps )
{
    return runTestHarnessWithCheck( argc, argv, testNum, testList, imageSupportRequired, forceNoContextCreation, queueProps,
                          ( imageSupportRequired ) ? verifyImageSupport : NULL );
}

int runTestHarnessWithCheck( int argc, const char *argv[], int testNum, test_definition testList[],
                             int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps,
                             DeviceCheckFn deviceCheckFn )
{
    test_start();

    cl_device_type     device_type = CL_DEVICE_TYPE_DEFAULT;
    cl_uint            num_platforms = 0;
    cl_platform_id     *platforms;
    cl_device_id       device;
    int                num_elements = DEFAULT_NUM_ELEMENTS;
    cl_uint            num_devices = 0;
    cl_device_id       *devices = NULL;
    cl_uint            choosen_device_index = 0;
    cl_uint            choosen_platform_index = 0;

    int            err, ret;
    char *endPtr;
    int based_on_env_var = 0;


    /* Check for environment variable to set device type */
    char *env_mode = getenv( "CL_DEVICE_TYPE" );
    if( env_mode != NULL )
    {
        based_on_env_var = 1;
        if( strcmp( env_mode, "gpu" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_GPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_GPU;
        else if( strcmp( env_mode, "cpu" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_CPU" ) == 0 )
            device_type = CL_DEVICE_TYPE_CPU;
        else if( strcmp( env_mode, "accelerator" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( env_mode, "default" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            device_type = CL_DEVICE_TYPE_DEFAULT;
        else
        {
            log_error( "Unknown CL_DEVICE_TYPE env variable setting: %s.\nAborting...\n", env_mode );
            abort();
        }
    }

#if defined( __APPLE__ )
    {
        // report on any unusual library search path indirection
        char *libSearchPath = getenv( "DYLD_LIBRARY_PATH");
        if( libSearchPath )
            log_info( "*** DYLD_LIBRARY_PATH = \"%s\"\n", libSearchPath );

        // report on any unusual framework search path indirection
        char *frameworkSearchPath = getenv( "DYLD_FRAMEWORK_PATH");
        if( libSearchPath )
            log_info( "*** DYLD_FRAMEWORK_PATH = \"%s\"\n", frameworkSearchPath );
    }
#endif

    env_mode = getenv( "CL_DEVICE_INDEX" );
    if( env_mode != NULL )
    {
        choosen_device_index = atoi(env_mode);
    }

    env_mode = getenv( "CL_PLATFORM_INDEX" );
    if( env_mode != NULL )
    {
        choosen_platform_index = atoi(env_mode);
    }

    /* Process the command line arguments */

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        test_finish();
        return 0;
    }

    /* Special case: just list the tests */
    if( ( argc > 1 ) && (!strcmp( argv[ 1 ], "-list" ) || !strcmp( argv[ 1 ], "-h" ) || !strcmp( argv[ 1 ], "--help" )))
    {
        log_info( "Usage: %s [<test name>*] [pid<num>] [id<num>] [<device type>]\n", argv[0] );
        log_info( "\t<function name>\tOne or more of: (wildcard character '*') (default *)\n");
        log_info( "\tpid<num>\t\tIndicates platform at index <num> should be used (default 0).\n" );
        log_info( "\tid<num>\t\tIndicates device at index <num> should be used (default 0).\n" );
        log_info( "\t<device_type>\tcpu|gpu|accelerator|<CL_DEVICE_TYPE_*> (default CL_DEVICE_TYPE_DEFAULT)\n" );

        for( int i = 0; i < testNum; i++ )
        {
            log_info( "\t\t%s\n", testList[i].name );
        }
        test_finish();
        return 0;
    }

    /* How are we supposed to seed the random # generators? */
    if( argc > 1 && strcmp( argv[ argc - 1 ], "randomize" ) == 0 )
    {
        gRandomSeed = (cl_uint) time( NULL );
        log_info( "Random seed: %u.\n", gRandomSeed );
        gReSeed = 1;
        argc--;
    }
    else
    {
        log_info(" Initializing random seed to 0.\n");
    }

    /* Do we have an integer to specify the number of elements to pass to tests? */
    if( argc > 1 )
    {
        ret = (int)strtol( argv[ argc - 1 ], &endPtr, 10 );
        if( endPtr != argv[ argc - 1 ] && *endPtr == 0 )
        {
            /* By spec, this means the entire string was a valid integer, so we treat it as a num_elements spec */
            /* (hence why we stored the result in ret first) */
            num_elements = ret;
            log_info( "Testing with num_elements of %d\n", num_elements );
            argc--;
        }
    }

    /* Do we have a CPU/GPU specification? */
    if( argc > 1 )
    {
        if( strcmp( argv[ argc - 1 ], "gpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_GPU" ) == 0 )
        {
            device_type = CL_DEVICE_TYPE_GPU;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "cpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_CPU" ) == 0 )
        {
            device_type = CL_DEVICE_TYPE_CPU;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "accelerator" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
        {
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
        {
            device_type = CL_DEVICE_TYPE_DEFAULT;
            argc--;
        }
    }

    /* Did we choose a specific device index? */
    if( argc > 1 )
    {
        if( strlen( argv[ argc - 1 ] ) >= 3 && argv[ argc - 1 ][0] == 'i' && argv[ argc - 1 ][1] == 'd' )
        {
            choosen_device_index = atoi( &(argv[ argc - 1 ][2]) );
            argc--;
        }
    }

    /* Did we choose a specific platform index? */
    if( argc > 1 )
    {
        if( strlen( argv[ argc - 1 ] ) >= 3 && argv[ argc - 1 ][0] == 'p' && argv[ argc - 1 ][1] == 'i' && argv[ argc - 1 ][2] == 'd')
        {
            choosen_platform_index = atoi( &(argv[ argc - 1 ][3]) );
            argc--;
        }
    }

    switch( device_type )
    {
        case CL_DEVICE_TYPE_GPU:            log_info( "Requesting GPU device " ); break;
        case CL_DEVICE_TYPE_CPU:            log_info( "Requesting CPU device " ); break;
        case CL_DEVICE_TYPE_ACCELERATOR:    log_info( "Requesting Accelerator device " ); break;
        case CL_DEVICE_TYPE_DEFAULT:        log_info( "Requesting Default device " ); break;
        default:                            log_error( "Requesting unknown device "); return -1;
    }
    log_info( based_on_env_var ? "based on environment variable " : "based on command line " );
    log_info( "for platform index %d and device index %d\n", choosen_platform_index, choosen_device_index);

#if defined( __APPLE__ )
#if defined( __i386__ ) || defined( __x86_64__ )
#define    kHasSSE3                0x00000008
#define kHasSupplementalSSE3    0x00000100
#define    kHasSSE4_1              0x00000400
#define    kHasSSE4_2              0x00000800
    /* check our environment for a hint to disable SSE variants */
    {
        const char *env = getenv( "CL_MAX_SSE" );
        if( env )
        {
            extern int _cpu_capabilities;
            int mask = 0;
            if( 0 == strcasecmp( env, "SSE4.1" ) )
                mask = kHasSSE4_2;
            else if( 0 == strcasecmp( env, "SSSE3" ) )
                mask = kHasSSE4_2 | kHasSSE4_1;
            else if( 0 == strcasecmp( env, "SSE3" ) )
                mask = kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3;
            else if( 0 == strcasecmp( env, "SSE2" ) )
                mask = kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3 | kHasSSE3;
            else
            {
                log_error( "Error: Unknown CL_MAX_SSE setting: %s\n", env );
                return -2;
            }

            log_info( "*** Environment: CL_MAX_SSE = %s ***\n", env );
            _cpu_capabilities &= ~mask;
        }
    }
#endif
#endif

    /* Get the platform */
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err) {
        print_error(err, "clGetPlatformIDs failed");
        test_finish();
        return -1;
    }

    platforms = (cl_platform_id *) malloc( num_platforms * sizeof( cl_platform_id ) );
    if (!platforms || choosen_platform_index >= num_platforms) {
        log_error( "platform index out of range -- choosen_platform_index (%d) >= num_platforms (%d)\n", choosen_platform_index, num_platforms );
        test_finish();
        return -1;
    }
    BufferOwningPtr<cl_platform_id> platformsBuf(platforms);

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err) {
        print_error(err, "clGetPlatformIDs failed");
        test_finish();
        return -1;
    }

    /* Get the number of requested devices */
    err = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, 0, NULL, &num_devices );
    if (err) {
        print_error(err, "clGetDeviceIDs failed");
        test_finish();
        return -1;
    }

    devices = (cl_device_id *) malloc( num_devices * sizeof( cl_device_id ) );
    if (!devices || choosen_device_index >= num_devices) {
        log_error( "device index out of range -- choosen_device_index (%d) >= num_devices (%d)\n", choosen_device_index, num_devices );
        test_finish();
        return -1;
    }
    BufferOwningPtr<cl_device_id> devicesBuf(devices);


    /* Get the requested device */
    err = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, num_devices, devices, NULL );
    if (err) {
        print_error(err, "clGetDeviceIDs failed");
        test_finish();
        return -1;
    }

    device = devices[choosen_device_index];

    if( printDeviceHeader( device ) != CL_SUCCESS )
    {
        test_finish();
        return -1;
    }

    cl_device_fp_config fpconfig = 0;
    err = clGetDeviceInfo( device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( fpconfig ), &fpconfig, NULL );
    if (err) {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed");
        test_finish();
        return -1;
    }

    gFlushDenormsToZero = ( 0 == (fpconfig & CL_FP_DENORM));
    log_info( "Supports single precision denormals: %s\n", gFlushDenormsToZero ? "NO" : "YES" );
    log_info( "sizeof( void*) = %d  (host)\n", (int) sizeof( void* ) );

    //detect whether profile of the device is embedded
    char profile[1024] = "";
    err = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL);
    if (err)
    {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_PROFILE failed\n" );
        test_finish();
        return -1;
    }
    gIsEmbedded = NULL != strstr(profile, "EMBEDDED_PROFILE");

    //detect the floating point capabilities
    cl_device_fp_config floatCapabilities = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(floatCapabilities), &floatCapabilities, NULL);
    if (err)
    {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed\n");
        test_finish();
        return -1;
    }

    // Check for problems that only embedded will have
    if( gIsEmbedded )
    {
        //If the device is embedded, we need to detect if the device supports Infinity and NaN
        if ((floatCapabilities & CL_FP_INF_NAN) == 0)
            gInfNanSupport = 0;

        // check the extensions list to see if ulong and long are supported
        size_t extensionsStringSize = 0;
        if( (err = clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionsStringSize ) ))
        {
            print_error( err, "Unable to get extensions string size for embedded device" );
            test_finish();
            return -1;
        }
        char *extensions_string = (char*) malloc(extensionsStringSize);
        if( NULL == extensions_string )
        {
            print_error( CL_OUT_OF_HOST_MEMORY, "Unable to allocate storage for extensions string for embedded device" );
            test_finish();
            return -1;
        }
        BufferOwningPtr<char> extensions_stringBuf(extensions_string);

        if( (err = clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, extensionsStringSize, extensions_string, NULL ) ))
        {
            print_error( err, "Unable to get extensions string for embedded device" );
            test_finish();
            return -1;
        }

        if( extensions_string[extensionsStringSize-1] != '\0' )
        {
            log_error( "FAILURE: extensions string for embedded device is not NUL terminated" );
            test_finish();
            return -1;
        }

        if( NULL == strstr( extensions_string, "cles_khr_int64" ))
            gHasLong = 0;
    }

    if( getenv( "OPENCL_1_0_DEVICE" ) )
    {
        char c_version[1024];
        gIsOpenCL_1_0_Device = 1;
        memset( c_version, 0, sizeof( c_version ) );

        if( (err = clGetDeviceInfo( device, CL_DEVICE_OPENCL_C_VERSION, sizeof(c_version), c_version, NULL )) )
        {
            log_error( "FAILURE: unable to get CL_DEVICE_OPENCL_C_VERSION on 1.0 device. (%d)\n", err );
            test_finish();
            return -1;
        }

        if( 0 == strncmp( c_version, "OpenCL C 1.0 ", strlen( "OpenCL C 1.0 " ) ) )
        {
            gIsOpenCL_C_1_0_Device = 1;
            log_info( "Device is a OpenCL C 1.0 device\n" );
        }
        else
            log_info( "Device is a OpenCL 1.0 device, but supports OpenCL C 1.1\n" );
    }

    cl_uint device_address_bits = 0;
    if( (err = clGetDeviceInfo( device, CL_DEVICE_ADDRESS_BITS, sizeof( device_address_bits ), &device_address_bits, NULL ) ))
    {
        print_error( err, "Unable to obtain device address bits" );
        test_finish();
        return -1;
    }
    if( device_address_bits )
        log_info( "sizeof( void*) = %d  (device)\n", device_address_bits/8 );
    else
    {
        log_error("Invalid device address bit size returned by device.\n");
        test_finish();
        return -1;
    }


    /* If we have a device checking function, run it */
    if( ( deviceCheckFn != NULL ) )
    {
        test_status status = deviceCheckFn( device );
        switch (status)
        {
            case TEST_PASS:
                break;
            case TEST_FAIL:
                return 1;
            case TEST_SKIP:
                return 0;
        }
    }

    if (num_elements <= 0)
        num_elements = DEFAULT_NUM_ELEMENTS;

        // On most platforms which support denorm, default is FTZ off. However,
        // on some hardware where the reference is computed, default might be flush denorms to zero e.g. arm.
        // This creates issues in result verification. Since spec allows the implementation to either flush or
        // not flush denorms to zero, an implementation may choose not be flush i.e. return denorm result whereas
        // reference result may be zero (flushed denorm). Hence we need to disable denorm flushing on host side
        // where reference is being computed to make sure we get non-flushed reference result. If implementation
        // returns flushed result, we correctly take care of that in verification code.
#if defined(__APPLE__) && defined(__arm__)
        FPU_mode_type oldMode;
        DisableFTZ( &oldMode );
#endif

    int error = parseAndCallCommandLineTests( argc, argv, device, testNum, testList, forceNoContextCreation, queueProps, num_elements );

 #if defined(__APPLE__) && defined(__arm__)
     // Restore the old FP mode before leaving.
    RestoreFPState( &oldMode );
#endif

    return error;
}

static int find_wildcard_matching_functions( test_definition testList[], unsigned char selectedTestList[], int testNum,
                                             const char *wildcard )
{
    int found_tests = 0;
    size_t wildcard_length = strlen( wildcard ) - 1; /* -1 for the asterisk */

    for( int fnIndex = 0; fnIndex < testNum; fnIndex++ )
    {
        if( strncmp( testList[ fnIndex ].name, wildcard, wildcard_length ) == 0 )
        {
            if( selectedTestList[ fnIndex ] )
            {
                log_error( "ERROR: Test '%s' has already been selected.\n", testList[ fnIndex ].name );
                return EXIT_FAILURE;
            }

            selectedTestList[ fnIndex ] = 1;
            found_tests = 1;
        }
    }

    if( !found_tests )
    {
        log_error( "ERROR: The wildcard '%s' did not match any test names.\n", wildcard );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int find_argument_matching_function( test_definition testList[], unsigned char selectedTestList[], int testNum,
                                            const char *argument )
{
    int fnIndex;

    for( fnIndex = 0; fnIndex < testNum; fnIndex++ )
    {
        if( strcmp( argument, testList[fnIndex].name ) == 0 )
        {
            if( selectedTestList[ fnIndex ] )
            {
                log_error( "ERROR: Test '%s' has already been selected.\n", testList[fnIndex].name );
                return EXIT_FAILURE;
            }
            else
            {
                selectedTestList[ fnIndex ] = 1;
                break;
            }
        }
    }

    if( fnIndex == testNum )
    {
        log_error( "ERROR: The argument '%s' did not match any test names.\n", argument );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int parseAndCallCommandLineTests( int argc, const char *argv[], cl_device_id device, int testNum,
                                  test_definition testList[], int forceNoContextCreation,
                                  cl_command_queue_properties queueProps, int num_elements )
{
    int ret = EXIT_SUCCESS;

    unsigned char *selectedTestList = ( unsigned char* ) calloc( testNum, 1 );

    if( argc == 1 )
    {
        /* No actual arguments, all tests will be run. */
        memset( selectedTestList, 1, testNum );
    }
    else
    {
        for( int argIndex = 1; argIndex < argc; argIndex++ )
        {
            if( strchr( argv[ argIndex ], '*' ) != NULL )
            {
                ret = find_wildcard_matching_functions( testList, selectedTestList, testNum, argv[ argIndex ] );
            }
            else
            {
                if( strcmp( argv[ argIndex ], "all" ) == 0 )
                {
                    memset( selectedTestList, 1, testNum );
                    break;
                }
                else
                {
                    ret = find_argument_matching_function( testList, selectedTestList, testNum, argv[ argIndex ] );
                }
            }

            if( ret == EXIT_FAILURE )
            {
                break;
            }
        }
    }

    if( ret == EXIT_SUCCESS )
    {
        ret = callTestFunctions( testList, selectedTestList, testNum, device, forceNoContextCreation, num_elements, queueProps );

        if( gTestsFailed == 0 )
        {
            if( gTestsPassed > 1 )
            {
                log_info("PASSED %d of %d tests.\n", gTestsPassed, gTestsPassed);
            }
            else if( gTestsPassed > 0 )
            {
                log_info("PASSED test.\n");
            }
        }
        else if( gTestsFailed > 0 )
        {
            if( gTestsFailed+gTestsPassed > 1 )
            {
                log_error("FAILED %d of %d tests.\n", gTestsFailed, gTestsFailed+gTestsPassed);
            }
            else
            {
                log_error("FAILED test.\n");
            }
        }
    }

    test_finish();

    free(  selectedTestList );

    return ret;
}

int callTestFunctions( test_definition testList[], unsigned char selectedTestList[],
                       int testNum, cl_device_id deviceToUse, int forceNoContextCreation,
                       int numElementsToUse, cl_command_queue_properties queueProps )
{
    int numErrors = 0;

    for( int i = 0; i < testNum; ++i )
    {
        if( selectedTestList[i] )
        {
            /* Skip any unimplemented tests. */
            if( testList[i].func != NULL )
            {
                numErrors += callSingleTestFunction( testList[i], deviceToUse, forceNoContextCreation,
                                                     numElementsToUse, queueProps );
            }
            else
            {
                log_info( "%s test currently not implemented\n", testList[i].name );
            }
        }
    }

    return numErrors;
}

void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    log_info( "%s\n", errinfo );
}

// Actual function execution
int callSingleTestFunction( test_definition test, cl_device_id deviceToUse, int forceNoContextCreation,
                            int numElementsToUse, const cl_queue_properties queueProps )
{
    int numErrors = 0, ret;
    cl_int error;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    const cl_command_queue_properties cmd_queueProps = (queueProps)?CL_QUEUE_PROPERTIES:0;
    cl_command_queue_properties queueCreateProps[] = {cmd_queueProps, queueProps, 0};

    /* Create a context to work with, unless we're told not to */
    if( !forceNoContextCreation )
    {
        context = clCreateContext(NULL, 1, &deviceToUse, notify_callback, NULL, &error );
        if (!context)
        {
            print_error( error, "Unable to create testing context" );
            return 1;
        }

        queue = clCreateCommandQueueWithProperties( context, deviceToUse, &queueCreateProps[0], &error );
        if( queue == NULL )
        {
            print_error( error, "Unable to create testing command queue" );
            return 1;
        }
    }

    /* Run the test and print the result */
    log_info( "%s...\n", test.name );
    fflush( stdout );

    error = check_opencl_version_with_testname(test.name, deviceToUse);
    test_missing_feature(error, test.name);

    error = check_functions_for_offline_compiler(test.name, deviceToUse);
    test_missing_support_offline_cmpiler(error, test.name);

    ret = test.func(deviceToUse, context, queue, numElementsToUse);        //test_threaded_function( ptr_basefn_list[i], group, context, num_elements);
    if( ret == TEST_NOT_IMPLEMENTED )
    {
        /* Tests can also let us know they're not implemented yet */
        log_info("%s test currently not implemented\n\n", test.name);
    }
    else
    {
        /* Print result */
        if( ret == 0 ) {
            log_info( "%s passed\n", test.name );
            gTestsPassed++;
        }
        else
        {
            numErrors++;
            log_error( "%s FAILED\n", test.name );
            gTestsFailed++;
        }
    }

    /* Release the context */
    if( !forceNoContextCreation )
    {
        int error = clFinish(queue);
        if (error) {
            log_error("clFinish failed: %d", error);
            numErrors++;
        }
        clReleaseCommandQueue( queue );
        clReleaseContext( context );
    }

    return numErrors;
}

void checkDeviceTypeOverride( cl_device_type *inOutType )
{
    /* Check if we are forced to CPU mode */
    char *force_cpu = getenv( "CL_DEVICE_TYPE" );
    if( force_cpu != NULL )
    {
        if( strcmp( force_cpu, "gpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_GPU" ) == 0 )
            *inOutType = CL_DEVICE_TYPE_GPU;
        else if( strcmp( force_cpu, "cpu" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_CPU" ) == 0 )
            *inOutType = CL_DEVICE_TYPE_CPU;
        else if( strcmp( force_cpu, "accelerator" ) == 0 || strcmp( force_cpu, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            *inOutType = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( force_cpu, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            *inOutType = CL_DEVICE_TYPE_DEFAULT;
    }

    switch( *inOutType )
    {
        case CL_DEVICE_TYPE_GPU:            log_info( "Requesting GPU device " ); break;
        case CL_DEVICE_TYPE_CPU:            log_info( "Requesting CPU device " ); break;
        case CL_DEVICE_TYPE_ACCELERATOR:    log_info( "Requesting Accelerator device " ); break;
        case CL_DEVICE_TYPE_DEFAULT:        log_info( "Requesting Default device " ); break;
        default: break;
    }
    log_info( force_cpu != NULL ? "based on environment variable\n" : "based on command line\n" );

#if defined( __APPLE__ )
    {
        // report on any unusual library search path indirection
        char *libSearchPath = getenv( "DYLD_LIBRARY_PATH");
        if( libSearchPath )
            log_info( "*** DYLD_LIBRARY_PATH = \"%s\"\n", libSearchPath );

        // report on any unusual framework search path indirection
        char *frameworkSearchPath = getenv( "DYLD_FRAMEWORK_PATH");
        if( libSearchPath )
            log_info( "*** DYLD_FRAMEWORK_PATH = \"%s\"\n", frameworkSearchPath );
    }
#endif

}

#if ! defined( __APPLE__ )
void memset_pattern4(void *dest, const void *src_pattern, size_t bytes )
{
    uint32_t pat = ((uint32_t*) src_pattern)[0];
    size_t count = bytes / 4;
    size_t i;
    uint32_t *d = (uint32_t*)dest;

    for( i = 0; i < count; i++ )
        d[i] = pat;

    d += i;

    bytes &= 3;
    if( bytes )
        memcpy( d, src_pattern, bytes );
}
#endif

extern cl_device_type GetDeviceType( cl_device_id d )
{
    cl_device_type result = -1;
    cl_int err = clGetDeviceInfo( d, CL_DEVICE_TYPE, sizeof( result ), &result, NULL );
    if( CL_SUCCESS != err )
        log_error( "ERROR: Unable to get device type for device %p\n", d );
    return result;
}


cl_device_id GetOpposingDevice( cl_device_id device )
{
    cl_int error;
    cl_device_id *otherDevices;
    cl_uint actualCount;
    cl_platform_id plat;

    // Get the platform of the device to use for getting a list of devices
    error = clGetDeviceInfo( device, CL_DEVICE_PLATFORM, sizeof( plat ), &plat, NULL );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to get device's platform" );
        return NULL;
    }

    // Get a list of all devices
    error = clGetDeviceIDs( plat, CL_DEVICE_TYPE_ALL, 0, NULL, &actualCount );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to get list of devices size" );
        return NULL;
    }
    otherDevices = (cl_device_id *)malloc(actualCount*sizeof(cl_device_id));
    if (NULL == otherDevices) {
        print_error( error, "Unable to allocate list of other devices." );
        return NULL;
    }
    BufferOwningPtr<cl_device_id> otherDevicesBuf(otherDevices);

    error = clGetDeviceIDs( plat, CL_DEVICE_TYPE_ALL, actualCount, otherDevices, NULL );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to get list of devices" );
        return NULL;
    }

    if( actualCount == 1 )
    {
        return device;    // NULL means error, returning self means we couldn't find another one
    }

    // Loop and just find one that isn't the one we were given
    cl_uint i;
    for( i = 0; i < actualCount; i++ )
    {
        if( otherDevices[ i ] != device )
        {
            cl_device_type newType;
            error = clGetDeviceInfo( otherDevices[ i ], CL_DEVICE_TYPE, sizeof( newType ), &newType, NULL );
            if( error != CL_SUCCESS )
            {
                print_error( error, "Unable to get device type for other device" );
                return NULL;
            }
            cl_device_id result = otherDevices[ i ];
            return result;
        }
    }

    // Should never get here
    return NULL;
}



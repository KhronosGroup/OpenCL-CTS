//
// Copyright (c) 2017-2019 The Khronos Group Inc.
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
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "threadTesting.h"
#include "errorHelpers.h"
#include "kernelHelpers.h"
#include "fpcontrol.h"
#include "typeWrappers.h"
#include "imageHelpers.h"
#include "parseParameters.h"

#if !defined(_WIN32)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#if !defined(_WIN32) && !defined(__ANDROID__)
#include <sys/sysctl.h>
#endif

#include <time.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

int gTestsPassed = 0;
int gTestsFailed = 0;
int gFailCount;
int gTestCount;
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
    return runTestHarnessWithCheck( argc, argv, testNum, testList, forceNoContextCreation, queueProps,
                          ( imageSupportRequired ) ? verifyImageSupport : NULL );
}

int skip_init_info(int count) {
    log_info("Test skipped while initialization\n");
    log_info("SKIPPED %d of %d tests.\n", count, count);
    return EXIT_SUCCESS;
}

int fail_init_info(int count) {
    log_info("Test failed while initialization\n");
    log_info("FAILED %d of %d tests.\n", count, count);
    return EXIT_FAILURE;
}
void version_expected_info(const char * test_name, const char * expected_version, const char * device_version) {
    log_info("%s skipped (requires at least version %s, but the device reports version %s)\n",
        test_name, expected_version, device_version);
}
int runTestHarnessWithCheck( int argc, const char *argv[], int testNum, test_definition testList[],
                             int forceNoContextCreation, cl_command_queue_properties queueProps,
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
        return EXIT_FAILURE;
    }

    /* Special case: just list the tests */
    if( ( argc > 1 ) && (!strcmp( argv[ 1 ], "-list" ) || !strcmp( argv[ 1 ], "-h" ) || !strcmp( argv[ 1 ], "--help" )))
    {
        char *fileName = getenv("CL_CONFORMANCE_RESULTS_FILENAME");

        log_info( "Usage: %s [<test name>*] [pid<num>] [id<num>] [<device type>]\n", argv[0] );
        log_info( "\t<test name>\tOne or more of: (wildcard character '*') (default *)\n");
        log_info( "\tpid<num>\tIndicates platform at index <num> should be used (default 0).\n" );
        log_info( "\tid<num>\t\tIndicates device at index <num> should be used (default 0).\n" );
        log_info( "\t<device_type>\tcpu|gpu|accelerator|<CL_DEVICE_TYPE_*> (default CL_DEVICE_TYPE_DEFAULT)\n" );
        log_info( "\n" );
        log_info( "\tNOTE: You may pass environment variable CL_CONFORMANCE_RESULTS_FILENAME (currently '%s')\n",
                  fileName != NULL ? fileName : "<undefined>" );
        log_info( "\t      to save results to JSON file.\n" );

        log_info( "\n" );
        log_info( "Test names:\n" );
        for( int i = 0; i < testNum; i++ )
        {
            log_info( "\t%s\n", testList[i].name );
        }
        return EXIT_SUCCESS;
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



	switch (device_type)
	{
	case CL_DEVICE_TYPE_GPU:            log_info("Requesting GPU device "); break;
	case CL_DEVICE_TYPE_CPU:            log_info("Requesting CPU device "); break;
	case CL_DEVICE_TYPE_ACCELERATOR:    log_info("Requesting Accelerator device "); break;
	case CL_DEVICE_TYPE_DEFAULT:        log_info("Requesting Default device "); break;
	default:                            log_error("Requesting unknown device "); return EXIT_FAILURE;
	}
	log_info(based_on_env_var ? "based on environment variable " : "based on command line ");
	log_info("for platform index %d and device index %d\n", choosen_platform_index, choosen_device_index);

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
                return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }

    platforms = (cl_platform_id *) malloc( num_platforms * sizeof( cl_platform_id ) );
    if (!platforms || choosen_platform_index >= num_platforms) {
        log_error( "platform index out of range -- choosen_platform_index (%d) >= num_platforms (%d)\n", choosen_platform_index, num_platforms );
        return EXIT_FAILURE;
    }
    BufferOwningPtr<cl_platform_id> platformsBuf(platforms);

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err) {
        print_error(err, "clGetPlatformIDs failed");
        return EXIT_FAILURE;
    }

    /* Get the number of requested devices */
    err = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, 0, NULL, &num_devices );
    if (err) {
        print_error(err, "clGetDeviceIDs failed");
        return EXIT_FAILURE;
    }

    devices = (cl_device_id *) malloc( num_devices * sizeof( cl_device_id ) );
    if (!devices || choosen_device_index >= num_devices) {
        log_error( "device index out of range -- choosen_device_index (%d) >= num_devices (%d)\n", choosen_device_index, num_devices );
        return EXIT_FAILURE;
    }
    BufferOwningPtr<cl_device_id> devicesBuf(devices);


    /* Get the requested device */
    err = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, num_devices, devices, NULL );
    if (err) {
        print_error(err, "clGetDeviceIDs failed");
        return EXIT_FAILURE;
    }

    device = devices[choosen_device_index];

    err = clGetDeviceInfo( device, CL_DEVICE_TYPE, sizeof(gDeviceType), &gDeviceType, NULL );
    if( err )
    {
        print_error( err, "Unable to get device type" );
        return TEST_FAIL;
    }
    
    if( printDeviceHeader( device ) != CL_SUCCESS )
    {
        return EXIT_FAILURE;
    }

    cl_device_fp_config fpconfig = 0;
    err = clGetDeviceInfo( device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( fpconfig ), &fpconfig, NULL );
    if (err) {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed");
        return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }
    gIsEmbedded = NULL != strstr(profile, "EMBEDDED_PROFILE");

    //detect the floating point capabilities
    cl_device_fp_config floatCapabilities = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(floatCapabilities), &floatCapabilities, NULL);
    if (err)
    {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed\n");
        return EXIT_FAILURE;
    }

    // Check for problems that only embedded will have
    if( gIsEmbedded )
    {
        //If the device is embedded, we need to detect if the device supports Infinity and NaN
        if ((floatCapabilities & CL_FP_INF_NAN) == 0)
            gInfNanSupport = 0;

        // check the extensions list to see if ulong and long are supported
        if( !is_extension_available(device, "cles_khr_int64" ))
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
            return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }
    if( device_address_bits )
        log_info( "sizeof( void*) = %d  (device)\n", device_address_bits/8 );
    else
    {
        log_error("Invalid device address bit size returned by device.\n");
        return EXIT_FAILURE;
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
                return fail_init_info(testNum);
            case TEST_SKIP:
                return skip_init_info(testNum);
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

    return (error == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int find_matching_tests( test_definition testList[], unsigned char selectedTestList[], int testNum,
                                const char *argument, bool isWildcard )
{
    int found_tests = 0;
    size_t wildcard_length = strlen( argument ) - 1; /* -1 for the asterisk */

    for( int i = 0; i < testNum; i++ )
    {
        if( ( !isWildcard && strcmp( testList[i].name, argument ) == 0 ) ||
            ( isWildcard && strncmp( testList[i].name, argument, wildcard_length ) == 0 ) )
        {
            if( selectedTestList[i] )
            {
                log_error( "ERROR: Test '%s' has already been selected.\n", testList[i].name );
                return EXIT_FAILURE;
            }
            else if( testList[i].func == NULL )
            {
                log_error( "ERROR: Test '%s' is missing implementation.\n", testList[i].name );
                return EXIT_FAILURE;
            }
            else
            {
                selectedTestList[i] = 1;
                found_tests = 1;
                if( !isWildcard )
                {
                    break;
                }
            }
        }
    }

    if( !found_tests )
    {
        log_error( "ERROR: The argument '%s' did not match any test names.\n", argument );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int saveResultsToJson( const char *fileName, const char *suiteName, test_definition testList[],
                              unsigned char selectedTestList[], test_status resultTestList[], int testNum )
{
    FILE *file = fopen( fileName, "w" );
    if( NULL == file )
    {
        log_error( "ERROR: Failed to open '%s' for writing results.\n", fileName );
        return EXIT_FAILURE;
    }

    const char *save_map[] = { "success", "failure" };
    const char *result_map[] = { "pass", "fail", "skip" };
    const char *linebreak[] = { "", ",\n" };
    int add_linebreak = 0;

    fprintf( file, "{\n" );
    fprintf( file, "\t\"cmd\": \"%s\",\n", suiteName );
    fprintf( file, "\t\"results\": {\n" );

    for( int i = 0; i < testNum; ++i )
    {
        if( selectedTestList[i] )
        {
            fprintf( file, "%s\t\t\"%s\": \"%s\"", linebreak[add_linebreak], testList[i].name, result_map[(int)resultTestList[i]] );
            add_linebreak = 1;
        }
    }
    fprintf( file, "\n");

    fprintf( file, "\t}\n" );
    fprintf( file, "}\n" );

    int ret = fclose( file ) ? 1 : 0;

    log_info( "Saving results to %s: %s!\n", fileName, save_map[ret] );

    return ret;
}

static void print_results( int failed, int count, const char* name )
{
    if( count < failed )
    {
        count = failed;
    }

    if( failed == 0 )
    {
        if( count > 1 )
        {
            log_info( "PASSED %d of %d %ss.\n", count, count, name );
        }
        else
        {
            log_info( "PASSED %s.\n", name );
        }
    }
    else if( failed > 0 )
    {
        if( count > 1 )
        {
            log_error( "FAILED %d of %d %ss.\n", failed, count, name );
        }
        else
        {
            log_error( "FAILED %s.\n", name );
        }
    }
}

int parseAndCallCommandLineTests( int argc, const char *argv[], cl_device_id device, int testNum,
                                  test_definition testList[], int forceNoContextCreation,
                                  cl_command_queue_properties queueProps, int num_elements )
{
    int ret = EXIT_SUCCESS;

    unsigned char *selectedTestList = ( unsigned char* ) calloc( testNum, 1 );
    test_status *resultTestList = NULL;

    if( argc == 1 )
    {
        /* No actual arguments, all tests will be run. */
        memset( selectedTestList, 1, testNum );
    }
    else
    {
        for( int i = 1; i < argc; i++ )
        {
            if( strchr( argv[i], '*' ) != NULL )
            {
                ret = find_matching_tests( testList, selectedTestList, testNum, argv[i], true );
            }
            else
            {
                if( strcmp( argv[i], "all" ) == 0 )
                {
                    memset( selectedTestList, 1, testNum );
                    break;
                }
                else
                {
                    ret = find_matching_tests( testList, selectedTestList, testNum, argv[i], false );
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
        resultTestList = ( test_status* ) calloc( testNum, sizeof(*resultTestList) );

        callTestFunctions( testList, selectedTestList, resultTestList, testNum, device,
                           forceNoContextCreation, num_elements, queueProps );

        print_results( gFailCount, gTestCount, "sub-test" );
        print_results( gTestsFailed, gTestsFailed + gTestsPassed, "test" );

        char *filename = getenv( "CL_CONFORMANCE_RESULTS_FILENAME" );
        if( filename != NULL )
        {
            ret = saveResultsToJson( filename, argv[0], testList, selectedTestList, resultTestList, testNum );
        }
    }

    free( selectedTestList );
    free( resultTestList );

    return ret;
}

void callTestFunctions( test_definition testList[], unsigned char selectedTestList[], test_status resultTestList[],
                        int testNum, cl_device_id deviceToUse, int forceNoContextCreation, int numElementsToUse,
                        cl_command_queue_properties queueProps )
{
    for( int i = 0; i < testNum; ++i )
    {
        if( selectedTestList[i] )
        {
            resultTestList[i] = callSingleTestFunction( testList[i], deviceToUse, forceNoContextCreation,
                                                        numElementsToUse, queueProps );
        }
    }
}

void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    log_info( "%s\n", errinfo );
}

// Actual function execution
test_status callSingleTestFunction( test_definition test, cl_device_id deviceToUse, int forceNoContextCreation,
                                    int numElementsToUse, const cl_queue_properties queueProps )
{
    test_status status;
    cl_int error;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    log_info( "%s...\n", test.name );
    fflush( stdout );

    const Version device_version = get_device_cl_version(deviceToUse);
    if (test.min_version > device_version)
    {
        version_expected_info(test.name, test.min_version.to_string().c_str(), device_version.to_string().c_str());
        return TEST_SKIP;
    }

    /* Create a context to work with, unless we're told not to */
    if( !forceNoContextCreation )
    {
        context = clCreateContext(NULL, 1, &deviceToUse, notify_callback, NULL, &error );
        if (!context)
        {
            print_error( error, "Unable to create testing context" );
            return TEST_FAIL;
        }

        if (device_version < Version(2, 0)) {
            queue = clCreateCommandQueue(context, deviceToUse, queueProps, &error);
        } else {
            const cl_command_queue_properties cmd_queueProps = (queueProps)?CL_QUEUE_PROPERTIES:0;
            cl_command_queue_properties queueCreateProps[] = {cmd_queueProps, queueProps, 0};
            queue = clCreateCommandQueueWithProperties( context, deviceToUse, &queueCreateProps[0], &error );
        }

        if( queue == NULL )
        {
            print_error( error, "Unable to create testing command queue" );
            return TEST_FAIL;
        }
    }

    /* Run the test and print the result */
    error = check_functions_for_offline_compiler(test.name, deviceToUse);
    test_missing_support_offline_cmpiler(error, test.name);

    if( test.func == NULL )
    {
        // Skip unimplemented test, can happen when all of the tests are selected
        log_info("%s test currently not implemented\n", test.name);
        status = TEST_SKIP;
    }
    else
    {
        int ret = test.func(deviceToUse, context, queue, numElementsToUse);        //test_threaded_function( ptr_basefn_list[i], group, context, num_elements);
        if( ret == TEST_NOT_IMPLEMENTED )
        {
            /* Tests can also let us know they're not implemented yet */
            log_info("%s test currently not implemented\n", test.name);
            status = TEST_SKIP;
        }
        else if (ret == TEST_SKIPPED_ITSELF)
        {
            /* Tests can also let us know they're not supported by the implementation */
            log_info("%s test not supported\n", test.name);
            status = TEST_SKIP;
        }
        else
        {
            /* Print result */
            if( ret == 0 ) {
                log_info( "%s passed\n", test.name );
                gTestsPassed++;
                status = TEST_PASS;
            }
            else
            {
                log_error( "%s FAILED\n", test.name );
                gTestsFailed++;
                status = TEST_FAIL;
            }
        }
    }

    /* Release the context */
    if( !forceNoContextCreation )
    {
        int error = clFinish(queue);
        if (error) {
            log_error("clFinish failed: %d", error);
            status = TEST_FAIL;
        }
        clReleaseCommandQueue( queue );
        clReleaseContext( context );
    }

    return status;
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

cl_device_type GetDeviceType( cl_device_id d )
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

Version get_device_cl_version(cl_device_id device)
{
    size_t str_size;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &str_size);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    std::vector<char> str(str_size);
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, str_size, str.data(), NULL);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    if (strstr(str.data(), "OpenCL 1.0") != NULL)
        return Version(1, 0);
    else if (strstr(str.data(), "OpenCL 1.1") != NULL)
        return Version(1, 1);
    else if (strstr(str.data(), "OpenCL 1.2") != NULL)
        return Version(1, 2);
    else if (strstr(str.data(), "OpenCL 2.0") != NULL)
        return Version(2, 0);
    else if (strstr(str.data(), "OpenCL 2.1") != NULL)
        return Version(2, 1);
    else if (strstr(str.data(), "OpenCL 2.2") != NULL)
        return Version(2, 2);
    else if (strstr(str.data(), "OpenCL 3.0") != NULL)
        return Version(3, 0);

    throw std::runtime_error(std::string("Unknown OpenCL version: ") + str.data());
}

void PrintArch( void )
{
    vlog( "sizeof( void*) = %ld\n", sizeof( void *) );
#if defined( __ppc__ )
    vlog( "ARCH:\tppc\n" );
#elif defined( __ppc64__ )   
    vlog( "ARCH:\tppc64\n" );
#elif defined( __PPC__ )   
    vlog( "ARCH:\tppc\n" );
#elif defined( __i386__ )   
    vlog( "ARCH:\ti386\n" );
#elif defined( __x86_64__ )   
    vlog( "ARCH:\tx86_64\n" );
#elif defined( __arm__ )   
    vlog( "ARCH:\tarm\n" );
#elif defined(__aarch64__)
    vlog( "ARCH:\taarch64\n" );
#elif defined (_WIN32)
    vlog( "ARCH:\tWindows\n" );
#else
#error unknown arch
#endif

#if defined( __APPLE__ )

    int type = 0;
    size_t typeSize = sizeof( type );
    sysctlbyname( "hw.cputype", &type, &typeSize, NULL, 0 );
    vlog( "cpu type:\t%d\n", type );
    typeSize = sizeof( type );
    sysctlbyname( "hw.cpusubtype", &type, &typeSize, NULL, 0 );
    vlog( "cpu subtype:\t%d\n", type );

#elif defined( __linux__ )
    struct utsname buffer;
 
    if (uname(&buffer) != 0) {
       vlog("uname error");
    }
    else {
       vlog("system name = %s\n", buffer.sysname);
       vlog("node name   = %s\n", buffer.nodename);
       vlog("release     = %s\n", buffer.release);
       vlog("version     = %s\n", buffer.version);
       vlog("machine     = %s\n", buffer.machine);
    }
#endif
}


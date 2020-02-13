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
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>


#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include "threadTesting.h"
#include "errorHelpers.h"
#include "kernelHelpers.h"
#include "fpcontrol.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <time.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

#include "compat.h"

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

int runTestHarness( int argc, const char *argv[], unsigned int num_fns,
                   basefn fnList[], const char *fnNames[],
                   int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps )
{
    return runTestHarnessWithCheck( argc, argv, num_fns, fnList, fnNames, imageSupportRequired, forceNoContextCreation, queueProps,
                          ( imageSupportRequired ) ? verifyImageSupport : NULL );
}

int runTestHarnessWithCheck( int argc, const char *argv[], unsigned int num_fns,
                 basefn fnList[], const char *fnNames[],
                int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps,
                DeviceCheckFn deviceCheckFn )
{
    test_start();

    cl_device_type    device_type = CL_DEVICE_TYPE_DEFAULT;
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
    unsigned int            i;
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

    /* Special case: just list the tests */
    if( ( argc > 1 ) && (!strcmp( argv[ 1 ], "-list" ) || !strcmp( argv[ 1 ], "-h" ) || !strcmp( argv[ 1 ], "--help" )))
    {
        log_info( "Usage: %s [<function name>*] [pid<num>] [id<num>] [<device type>]\n", argv[0] );
        log_info( "\t<function name>\tOne or more of: (wildcard character '*') (default *)\n");
        log_info( "\tpid<num>\t\tIndicates platform at index <num> should be used (default 0).\n" );
        log_info( "\tid<num>\t\tIndicates device at index <num> should be used (default 0).\n" );
        log_info( "\t<device_type>\tcpu|gpu|accelerator|<CL_DEVICE_TYPE_*> (default CL_DEVICE_TYPE_DEFAULT)\n" );

        for( i = 0; i < num_fns - 1; i++ )
        {
            log_info( "\t\t%s\n", fnNames[ i ] );
        }
        test_finish();
        return 0;
    }

    /* How are we supposed to seed the random # generators? */
    if( argc > 1 && strcmp( argv[ argc - 1 ], "randomize" ) == 0 )
    {
        log_info(" Initializing random seed based on the clock.\n");
        gRandomSeed = (unsigned)clock();
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

    /* Get the requested device */
    err = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, num_devices, devices, NULL );
    if (err) {
        print_error(err, "clGetDeviceIDs failed");
        test_finish();
        return -1;
    }

    device = devices[choosen_device_index];
    free(devices);
    devices = NULL;
    free(platforms);
    platforms = NULL;

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

        free(extensions_string);
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
    if( ( deviceCheckFn != NULL ) && deviceCheckFn( device ) != CL_SUCCESS )
    {
        test_finish();
        return -1;
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

    int error = parseAndCallCommandLineTests( argc, argv, device, num_fns, fnList, fnNames, forceNoContextCreation, queueProps, num_elements );

 #if defined(__APPLE__) && defined(__arm__)
     // Restore the old FP mode before leaving.
    RestoreFPState( &oldMode );
#endif

    return error;
}

int parseAndCallCommandLineTests( int argc, const char *argv[], cl_device_id device, unsigned int num_fns,
                                 basefn *fnList, const char *fnNames[],
                                 int forceNoContextCreation, cl_command_queue_properties queueProps, int num_elements )
{
    int            ret, argIndex;
    unsigned int            i;
    int            fn_to_test = -1;    // initialized to test all.
                                       //    unsigned int threadSize;
    char        partial[512] = { 0 };


    /* Now that we have an environment, go through our arguments and run tests that match each argument */
    if( argc == 1 )
    {
        /* No actual arguments, so just run all tests */
        ret = callTestFunctions( fnList, num_fns - 1, fnNames,
                                device, forceNoContextCreation, num_elements, -1, NULL, queueProps );
    }
    else
    {
        /* Go through each argument and use it to process a list of functions to run */
        ret = 0;
        for( argIndex = 1; argIndex < argc; argIndex++ )
        {
            /* Are we a partial test? */
            fn_to_test = -1;
            if( strchr( argv[argIndex], '*' ) != NULL )
            {
                /* Yes, store the partial test for later */
                strcpy( partial, argv[argIndex] );
                strchr( partial, '*' )[0] = 0;
            }
            else
            {
                /* Nope, loop through looking for an exact name match */
                for (i=0; i<num_fns; i++)
                {
                    if (strcmp(argv[argIndex], fnNames[i]) == 0)
                    {
                        fn_to_test = i;
                        break;
                    }
                }
                if (i == num_fns)
                {
                    log_error("invalid test name: %s \n", argv[argIndex]);
                    ret = 1;
                    continue;    /* Keep processing other arguments */
                }
                else if( ( fn_to_test == (int)num_fns - 1 ) && ( strcmp( fnNames[i], "all" ) == 0 ) )
                {
                    fn_to_test = -1;
                }
            }

            /* Execute this particular test loop  (remember to remove 1 from the function count for the lack of "all" at the end!) */
            ret += callTestFunctions( fnList, num_fns - 1, fnNames,
                                     device, forceNoContextCreation, num_elements,
                                     fn_to_test, partial, queueProps );
        }
    }

    if (gTestsFailed == 0) {
        if (gTestsPassed > 1)
            log_info("PASSED %d of %d tests.\n", gTestsPassed, gTestsPassed);
        else if (gTestsPassed > 0)
            log_info("PASSED test.\n");
    } else if (gTestsFailed > 0) {
        if (gTestsFailed+gTestsPassed > 1)
            log_error("FAILED %d of %d tests.\n", gTestsFailed, gTestsFailed+gTestsPassed);
        else
            log_error("FAILED test.\n");
    }

    test_finish();

    return ret;
}

// The actual function that loops through tests and executes them
int callTestFunctions( basefn functionList[], int numFunctions,
                      const char *functionNames[],
                      cl_device_id deviceToUse, int forceNoContextCreation,
                      int numElementsToUse,
                      int functionIndexToCall, const char *partialName, cl_command_queue_properties queueProps )
{
    int numErrors = 0, found = 0, i;

    if( functionIndexToCall >= numFunctions )
    {
        log_error( "ERROR: Invalid function index to test!\n" );
        return 1;
    }

    if (functionIndexToCall == -1)
    {
        for (i=0; i<numFunctions; i++)
        {
            /* If we're matching partial names, skip any that don't match */
            if( partialName != NULL && strncmp( functionNames[i], partialName, strlen( partialName ) ) != 0 )
                continue;

            /* Skip any unimplemented tests */
            if (functionList[i] == 0)
            {
                log_info("%s test currently not implemented\n", functionNames[i]);
                continue;
            }

            found = 1;
            numErrors += callSingleTestFunction( functionList[i], functionNames[i], deviceToUse, forceNoContextCreation, numElementsToUse, queueProps );
        }
        if( found == 0 && partialName != NULL )
        {
            log_error( "ERROR: Wildcard test name does not match any tests: %s\n", partialName );
            return numErrors + 1;
        }
    }
    else
    {
        /* Run a single test */
        if (functionList[functionIndexToCall])
        {
            numErrors += callSingleTestFunction( functionList[functionIndexToCall], functionNames[functionIndexToCall],
                                                deviceToUse, forceNoContextCreation, numElementsToUse, queueProps );
        }
        else
            log_info("%s test currently not implemented\n", functionNames[functionIndexToCall]);
    }
    return numErrors;
}

void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    log_info( "%s\n", errinfo );
}

// Actual function execution
int callSingleTestFunction( basefn functionToCall, const char *functionName,
                           cl_device_id deviceToUse, int forceNoContextCreation,
                           int numElementsToUse, cl_command_queue_properties queueProps )
{
    int numErrors = 0, ret;
    cl_int error;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    /* Create a context to work with, unless we're told not to */
    if( !forceNoContextCreation )
    {
        context = clCreateContext(NULL, 1, &deviceToUse, notify_callback, NULL, &error );
        if (!context)
        {
            print_error( error, "Unable to create testing context" );
            return 1;
        }

        queue = clCreateCommandQueue( context, deviceToUse, queueProps, &error );
        if( queue == NULL )
        {
            print_error( error, "Unable to create testing command queue" );
            return 1;
        }
    }

    /* Run the test and print the result */
    log_info( "%s...\n", functionName );
    fflush( stdout );

    ret = functionToCall( deviceToUse, context, queue, numElementsToUse);        //test_threaded_function( ptr_basefn_list[i], group, context, num_elements);
    if( ret == TEST_NOT_IMPLEMENTED )
    {
        /* Tests can also let us know they're not implemented yet */
        log_info("%s test currently not implemented\n\n", functionName);
    }
    else
    {
        /* Print result */
        if( ret == 0 ) {
            log_info( "%s passed\n", functionName );
            gTestsPassed++;
        }
        else
        {
            numErrors++;
            log_error( "%s FAILED\n", functionName );
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
    error = clGetDeviceIDs( plat, CL_DEVICE_TYPE_ALL, actualCount, otherDevices, NULL );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to get list of devices" );
        free(otherDevices);
        return NULL;
    }

    if( actualCount == 1 )
    {
        free(otherDevices);
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
                free(otherDevices);
                return NULL;
            }
            cl_device_id result = otherDevices[ i ];
            free(otherDevices);
            return result;
        }
    }

    // Should never get here
    free(otherDevices);
    return NULL;
}


void PrintArch( void )
{
    vlog( "\nHost info:\n" );
    vlog( "\tsizeof( void*) = %ld\n", sizeof( void *) );
#if defined( __ppc__ )
    vlog( "\tARCH:\tppc\n" );
#elif defined( __ppc64__ )
    vlog( "\tARCH:\tppc64\n" );
#elif defined( __PPC__ )
    vlog( "ARCH:\tppc\n" );
#elif defined( __i386__ )
    vlog( "\tARCH:\ti386\n" );
#elif defined( __x86_64__ )
    vlog( "\tARCH:\tx86_64\n" );
#elif defined( __arm__ )
    vlog( "\tARCH:\tarm\n" );
#elif defined( __aarch64__ )
    vlog( "\tARCH:\taarch64\n" );
#else
    vlog( "\tARCH:\tunknown\n" );
#endif

#if defined( __APPLE__ )
    int type = 0;
    size_t typeSize = sizeof( type );
    sysctlbyname( "hw.cputype", &type, &typeSize, NULL, 0 );
    vlog( "\tcpu type:\t%d\n", type );
    typeSize = sizeof( type );
    sysctlbyname( "hw.cpusubtype", &type, &typeSize, NULL, 0 );
    vlog( "\tcpu subtype:\t%d\n", type );

#elif defined( __linux__ ) // && !defined(__aarch64__)
   struct utsname buffer;

   if (uname(&buffer) != 0) {
      vlog("uname");
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


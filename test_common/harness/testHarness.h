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
#ifndef _testHarness_h
#define _testHarness_h

#include "threadTesting.h"
#include "clImageHelper.h"

#ifdef __cplusplus
extern "C" {
#endif

extern cl_uint gReSeed;
extern cl_uint gRandomSeed;

// Supply a list of functions to test here. This will allocate a CL device, create a context, all that
// setup work, and then call each function in turn as dictatated by the passed arguments.
extern int runTestHarness( int argc, const char *argv[], unsigned int num_fns,
                            basefn fnList[], const char *fnNames[],
                            int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps );

// Device checking function. See runTestHarnessWithCheck. If this function returns anything other than CL_SUCCESS (0), the harness exits.
typedef int (*DeviceCheckFn)( cl_device_id device );

// Same as runTestHarness, but also supplies a function that checks the created device for required functionality.
extern int runTestHarnessWithCheck( int argc, const char *argv[], unsigned int num_fns,
                              basefn fnList[], const char *fnNames[],
                              int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps, DeviceCheckFn deviceCheckFn );

// The command line parser used by runTestHarness to break up parameters into calls to callTestFunctions
extern int parseAndCallCommandLineTests( int argc, const char *argv[], cl_device_id device, unsigned int num_fns,
                                        basefn *fnList, const char *fnNames[],
                                        int forceNoContextCreation, cl_command_queue_properties queueProps, int num_elements );

// Call this function if you need to do all the setup work yourself, and just need the function list called/
// managed.
//    functionIndexToCall can be a valid index into the function list, or -1 to run all of them.
//    partialName can be a string to partially match function names against and only execute functions who
//        match, or NULL to not restrict execution (ignored if functionIndexToCall is not -1)
//    functionList is the actual array of functions
//    numFunctions is the number of functions in the list (which should NOT have NULL at the end for "all")
//    functionNames is an array of strings representing the name of each function, to be used in partial matching
//    contextProps are used to create a testing context for each test
//    deviceToUse, deviceGroupToUse and numElementsToUse are all just passed to each test function

extern int callTestFunctions( basefn functionList[], int numFunctions,
                              const char *functionNames[],
                             cl_device_id deviceToUse, int forceNoContextCreation,
                             int numElementsToUse,
                             int functionIndexToCall, const char *partialName, cl_command_queue_properties queueProps );

// This function is called by callTestFunctions, once per function, to do setup, call, logging and cleanup
extern int callSingleTestFunction( basefn functionToCall, const char *functionName,
                                  cl_device_id deviceToUse, int forceNoContextCreation,
                                  int numElementsToUse, cl_command_queue_properties queueProps );

///// Miscellaneous steps

// Given a pre-existing device type choice, check the environment for an override, then print what
// choice was made and how (and return the overridden choice, if there is one)
extern void checkDeviceTypeOverride( cl_device_type *inOutType );

// standard callback function for context pfn_notify
extern void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data);

extern cl_device_type GetDeviceType( cl_device_id );

// Given a device (most likely passed in by the harness, but not required), will attempt to find
// a DIFFERENT device and return it. Useful for finding another device to run multi-device tests against.
// Note that returning NULL means an error was hit, but if no error was hit and the device passed in
// is the only device available, the SAME device is returned, so check!
extern cl_device_id GetOpposingDevice( cl_device_id device );


extern int      gFlushDenormsToZero;    // This is set to 1 if the device does not support denorms (CL_FP_DENORM)
extern int      gInfNanSupport;         // This is set to 1 if the device supports infinities and NaNs
extern int        gIsEmbedded;            // This is set to 1 if the device is an embedded device
extern int        gHasLong;               // This is set to 1 if the device suppots long and ulong types in OpenCL C.
extern int      gIsOpenCL_C_1_0_Device; // This is set to 1 if the device supports only OpenCL C 1.0.

#if ! defined( __APPLE__ )
    void     memset_pattern4(void *, const void *, size_t);
#endif

#ifdef __cplusplus
}
#endif

extern void PrintArch( void );


#endif // _testHarness_h



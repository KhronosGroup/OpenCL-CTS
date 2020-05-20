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
#ifndef _errorHelpers_h
#define _errorHelpers_h

#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#define LOWER_IS_BETTER     0
#define HIGHER_IS_BETTER    1

#include <stdio.h>
#define test_start()
#define log_info printf
#define log_error printf
#define log_missing_feature printf
#define log_perf(_number, _higherBetter, _numType, _format, ...) printf("Performance Number " _format " (in %s, %s): %g\n",##__VA_ARGS__, _numType,        \
                    _higherBetter?"higher is better":"lower is better", _number )
#define vlog_perf(_number, _higherBetter, _numType, _format, ...) printf("Performance Number " _format " (in %s, %s): %g\n",##__VA_ARGS__, _numType,    \
                    _higherBetter?"higher is better":"lower is better" , _number)
#ifdef _WIN32
    #ifdef __MINGW32__
        // Use __mingw_printf since it supports "%a" format specifier
        #define vlog __mingw_printf
        #define vlog_error __mingw_printf
    #else
        // Use home-baked function that treats "%a" as "%f"
    static int vlog_win32(const char *format, ...);
    #define vlog vlog_win32
    #define vlog_error vlog_win32
    #endif
#else
    #define vlog_error printf
    #define vlog printf
#endif

#define ct_assert(b)          ct_assert_i(b, __LINE__)
#define ct_assert_i(b, line)  ct_assert_ii(b, line)
#define ct_assert_ii(b, line) int _compile_time_assertion_on_line_##line[b ? 1 : -1];

#define test_error(errCode,msg)    test_error_ret(errCode,msg,errCode)
#define test_error_ret(errCode,msg,retValue)    { if( errCode != CL_SUCCESS ) { print_error( errCode, msg ); return retValue ; } }
#define print_error(errCode,msg)    log_error( "ERROR: %s! (%s from %s:%d)\n", msg, IGetErrorString( errCode ), __FILE__, __LINE__ );

#define test_missing_feature(errCode, msg) test_missing_feature_ret(errCode, msg, errCode)
// this macro should always return CL_SUCCESS, but print the missing feature message
#define test_missing_feature_ret(errCode,msg,retValue)    { if( errCode != CL_SUCCESS ) { print_missing_feature( errCode, msg ); return CL_SUCCESS ; } }
#define print_missing_feature(errCode, msg) log_missing_feature("ERROR: Subtest %s tests a feature not supported by the device version! (from %s:%d)\n", msg, __FILE__, __LINE__ );

#define test_missing_support_offline_cmpiler(errCode, msg) test_missing_support_offline_cmpiler_ret(errCode, msg, errCode)
// this macro should always return CL_SUCCESS, but print the skip message on test not supported with offline compiler
#define test_missing_support_offline_cmpiler_ret(errCode,msg,retValue)    { if( errCode != CL_SUCCESS ) { log_info( "INFO: Subtest %s tests is not supported in offline compiler execution path! (from %s:%d)\n", msg, __FILE__, __LINE__ ); return TEST_SKIP ; } }

// expected error code vs. what we got
#define test_failure_error(errCode, expectedErrCode, msg) test_failure_error_ret(errCode, expectedErrCode, msg, errCode != expectedErrCode)
#define test_failure_error_ret(errCode, expectedErrCode, msg, retValue) { if( errCode != expectedErrCode ) { print_failure_error( errCode, expectedErrCode, msg ); return retValue ; } }
#define print_failure_error(errCode, expectedErrCode, msg) log_error( "ERROR: %s! (Got %s, expected %s from %s:%d)\n", msg, IGetErrorString( errCode ), IGetErrorString( expectedErrCode ), __FILE__, __LINE__ );
#define test_failure_warning(errCode, expectedErrCode, msg) test_failure_warning_ret(errCode, expectedErrCode, msg, errCode != expectedErrCode)
#define test_failure_warning_ret(errCode, expectedErrCode, msg, retValue) { if( errCode != expectedErrCode ) { print_failure_warning( errCode, expectedErrCode, msg ); warnings++ ; } }
#define print_failure_warning(errCode, expectedErrCode, msg) log_error( "WARNING: %s! (Got %s, expected %s from %s:%d)\n", msg, IGetErrorString( errCode ), IGetErrorString( expectedErrCode ), __FILE__, __LINE__ );

#define ASSERT_SUCCESS(expr, msg)                                                                  \
    do                                                                                             \
    {                                                                                              \
        cl_int _temp_retval = (expr);                                                              \
        if (_temp_retval != CL_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream ss;                                                                  \
            ss << "ERROR: " << msg << "=" << IGetErrorString(_temp_retval)                         \
               << " at " << __FILE__ << ":" << __LINE__ << "\n";                                   \
            throw std::runtime_error(ss.str());                                                    \
        }                                                                                          \
    } while (0)

extern const char    *IGetErrorString( int clErrorCode );

extern float Ulp_Error_Half( cl_ushort test, float reference );
extern float Ulp_Error( float test, double reference );
extern float Ulp_Error_Double( double test, long double reference );

extern const char *GetChannelTypeName( cl_channel_type type );
extern int IsChannelTypeSupported( cl_channel_type type );
extern const char *GetChannelOrderName( cl_channel_order order );
extern int IsChannelOrderSupported( cl_channel_order order );
extern const char *GetAddressModeName( cl_addressing_mode mode );

extern const char *GetDeviceTypeName( cl_device_type type );
int check_functions_for_offline_compiler(const char *subtestname, cl_device_id device);

// NON-REENTRANT UNLESS YOU PROVIDE A BUFFER PTR (pass null to use static storage, but it's not reentrant then!)
extern const char *GetDataVectorString( void *dataBuffer, size_t typeSize, size_t vecSize, char *buffer );

#if defined (_WIN32) && !defined(__MINGW32__)
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
static int vlog_win32(const char *format, ...)
{
    const char *new_format = format;

    if (strstr(format, "%a")) {
        char *temp;
        if ((temp = strdup(format)) == NULL) {
            printf("vlog_win32: Failed to allocate memory for strdup\n");
            return -1;
        }
        new_format = temp;
        while (*temp) {
            // replace %a with %f
            if ((*temp == '%') && (*(temp+1) == 'a')) {
                *(temp+1) = 'f';
            }
            temp++;
        }
    }

    va_list args;
    va_start(args, format);
    vprintf(new_format, args);
    va_end(args);

    if (new_format != format) {
        free((void*)new_format);
    }

    return 0;
}
#endif


#endif // _errorHelpers_h



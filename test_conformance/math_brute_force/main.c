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
#include "Utility.h"

#include <stdio.h>
#include <string.h>

#include <stdlib.h>
#include <time.h>
#include "FunctionList.h"
#include "Sleep.h"

#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/parseParameters.h"

#if defined( __APPLE__ )
    #include <sys/sysctl.h>
    #include <sys/mman.h>
    #include <libgen.h>
    #include <sys/time.h>
#elif defined( __linux__ )
    #include <unistd.h>
    #include <sys/syscall.h>
    #include <linux/sysctl.h>
    #include <sys/param.h>
#endif

#if defined (__linux__) || (defined WIN32 && defined __MINGW32__)
#include <sys/param.h>
#endif

#include "../../test_common/harness/testHarness.h"

#define kPageSize           4096
#define DOUBLE_REQUIRED_FEATURES    ( CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM  )

const char      **gTestNames = NULL;
unsigned int    gTestNameCount = 0;
char            appName[ MAXPATHLEN ] = "";
cl_device_id    gDevice = NULL;
cl_context      gContext = NULL;
cl_command_queue gQueue = NULL;
int             gTestCount = 0;
int             gFailCount = 0;
static int32_t  gStartTestNumber;
static int32_t  gEndTestNumber;
int             gSkipCorrectnessTesting = 0;
int             gStopOnError = 0;
static bool     gSkipRestOfTests;
#if defined( __APPLE__ )
int             gMeasureTimes = 1;
#else
int             gMeasureTimes = 0;
#endif
int             gReportAverageTimes = 0;
int             gForceFTZ = 0;
int             gWimpyMode = 0;
int             gHasDouble = 0;
int             gTestFloat = 1;
//This flag should be 'ON' by default and it can be changed through the command line arguments.
volatile int             gTestFastRelaxed = 1;
/*This flag corresponds to defining if the implementation has Derived Fast Relaxed functions.
  The spec does not specify ULP for derived function.  The derived functions are composed of base functions which are tested for ULP, thus when this flag is enabled,
  Derived functions will not be tested for ULP, as per table 7.1 of OpenCL 2.0 spec.
  Since there is no way of quering the device whether it is a derived or non-derived implementation according to OpenCL 2.0 spec then it has to be changed through a command line argument.
*/
int             gFastRelaxedDerived = 1;
int             gToggleCorrectlyRoundedDivideSqrt = 0;
int             gDeviceILogb0 = 1;
int             gDeviceILogbNaN = 1;
int             gCheckTininessBeforeRounding = 1;
int             gIsInRTZMode = 0;
uint32_t        gMaxVectorSizeIndex = VECTOR_SIZE_COUNT;
uint32_t        gMinVectorSizeIndex = 0;
const char      *method[] = { "Best", "Average" };
void            *gIn = NULL;
void            *gIn2 = NULL;
void            *gIn3 = NULL;
void            *gOut_Ref = NULL;
void            *gOut[VECTOR_SIZE_COUNT] = {NULL, NULL, NULL, NULL, NULL, NULL };
void            *gOut_Ref2 = NULL;
void            *gOut2[VECTOR_SIZE_COUNT] = {NULL, NULL, NULL, NULL, NULL, NULL };
cl_mem          gInBuffer = NULL;
cl_mem          gInBuffer2 = NULL;
cl_mem          gInBuffer3 = NULL;
cl_mem          gOutBuffer[VECTOR_SIZE_COUNT]= {NULL, NULL, NULL, NULL, NULL, NULL };
cl_mem          gOutBuffer2[VECTOR_SIZE_COUNT]= {NULL, NULL, NULL, NULL, NULL, NULL };
uint32_t        gComputeDevices = 0;
uint32_t        gSimdSize = 1;
uint32_t        gDeviceFrequency = 0;
static MTdata   gMTdata;
cl_device_fp_config gFloatCapabilities = 0;
cl_device_fp_config gDoubleCapabilities = 0;
int             gWimpyReductionFactor = 32;
int             gWimpyBufferSize = BUFFER_SIZE;
int             gVerboseBruteForce = 0;
#if defined( __APPLE__ )
int             gHasBasicDouble = 0;
char*           gBasicDoubleFuncs[] = {
                    "add",
                    "assignment",
                    "divide",
                    "isequal",
                    "isgreater",
                    "isgreaterequal",
                    "isless",
                    "islessequal",
                    "isnotequal",
                    "multiply",
                    "sqrt",
                    "subtract" };
size_t          gNumBasicDoubleFuncs = sizeof(gBasicDoubleFuncs)/sizeof(char*);
#endif


static int ParseArgs( int argc, const char **argv );
static void PrintArch( void );
static void PrintUsage( void );
static void PrintFunctions( void );
test_status InitCL( cl_device_id device );
static void ReleaseCL( void );
static int InitILogbConstants( void );
static int IsTininessDetectedBeforeRounding( void );
static int IsInRTZMode( void );         //expensive. Please check gIsInRTZMode global instead.
static void TestFinishAtExit(void);


int doTest( const char* name )
{
    if( gSkipRestOfTests )
    {
        vlog( "Skipping function because of an earlier error.\n" );
        return 1;
    }

    int error = 0;
    const Func* func_data = NULL;

    for( size_t i = 0; i < functionListCount; i++ )
    {
        const Func* const temp_func = functionList + i;
        if( strcmp( temp_func->name, name ) == 0 )
        {
            if( i < gStartTestNumber || i > gEndTestNumber )
            {
                vlog( "Skipping function #%d\n", i );
                return 0;
            }

            func_data = temp_func;
            break;
        }
    }

    if( func_data == NULL )
    {
        vlog( "Function '%s' doesn't exist!\n", name );
        exit( EXIT_FAILURE );
    }

    if( func_data->func.p == NULL )
    {
        vlog( "'%s' is missing implementation, skipping function.\n", func_data->name );
        return 0;
    }

    // if correctly rounded divide & sqrt are supported by the implementation
    // then test it; otherwise skip the test
    if( strcmp( func_data->name, "sqrt_cr" ) == 0 || strcmp( func_data->name, "divide_cr" ) == 0 )
    {
        if( ( gFloatCapabilities & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ) == 0 )
        {
            vlog( "Correctly rounded divide and sqrt are not supported, skipping function.\n" );
            return 0;
        }
    }

    {
        extern int my_ilogb(double);
        if( 0 == strcmp( "ilogb", func_data->name ) )
        {
            InitILogbConstants();
        }

        if ( gTestFastRelaxed )
        {
            if( func_data->relaxed )
            {
                gTestCount++;
                vlog( "%3d: ", gTestCount );
                if( func_data->vtbl_ptr->TestFunc( func_data, gMTdata )  )
                {
                    gFailCount++;
                    error++;
                    if( gStopOnError )
                    {
                        gSkipRestOfTests = true;
                        return error;
                    }
                }
            }
        }

        if( gTestFloat )
        {
            int testFastRelaxedTmp = gTestFastRelaxed;
            gTestFastRelaxed = 0;

            gTestCount++;
            vlog( "%3d: ", gTestCount );
            if( func_data->vtbl_ptr->TestFunc( func_data, gMTdata )  )
            {
                gFailCount++;
                error++;
                if( gStopOnError )
                {
                    gTestFastRelaxed = testFastRelaxedTmp;
                    gSkipRestOfTests = true;
                    return error;
                }
            }
            gTestFastRelaxed = testFastRelaxedTmp;
        }

        if( gHasDouble && NULL != func_data->vtbl_ptr->DoubleTestFunc && NULL != func_data->dfunc.p )
        {
            //Disable fast-relaxed-math for double precision floating-point
            int testFastRelaxedTmp = gTestFastRelaxed;
            gTestFastRelaxed = 0;

            gTestCount++;
            vlog( "%3d: ", gTestCount );
            if( func_data->vtbl_ptr->DoubleTestFunc( func_data, gMTdata )  )
            {
                gFailCount++;
                error++;
                if( gStopOnError )
                {
                    gTestFastRelaxed = testFastRelaxedTmp;
                    gSkipRestOfTests = true;
                    return error;
                }
            }

            //Re-enable testing fast-relaxed-math mode
            gTestFastRelaxed = testFastRelaxedTmp;
        }

#if defined( __APPLE__ )
        {
            if( gHasBasicDouble && NULL != func_data->vtbl_ptr->DoubleTestFunc && NULL != func_data->dfunc.p)
            {
                //Disable fast-relaxed-math for double precision floating-point
                int testFastRelaxedTmp = gTestFastRelaxed;
                gTestFastRelaxed = 0;

                int isBasicTest = 0;
                for( size_t j = 0; j < gNumBasicDoubleFuncs; j++ ) {
                    if( 0 == strcmp(gBasicDoubleFuncs[j], func_data->name ) ) {
                        isBasicTest = 1;
                        break;
                    }
                }
                if (isBasicTest) {
                    gTestCount++;
                    if( gTestFloat )
                        vlog( "    " );
                    if( func_data->vtbl_ptr->DoubleTestFunc( func_data, gMTdata )  )
                    {
                        gFailCount++;
                        error++;
                        if( gStopOnError )
                        {
                            gTestFastRelaxed = testFastRelaxedTmp;
                            gSkipRestOfTests = true;
                            return error;
                        }
                    }
                }

                //Re-enable testing fast-relaxed-math mode
                gTestFastRelaxed = testFastRelaxedTmp;
            }
        }
#endif
    }

    return error;
}

int test_acos( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "acos" );
}
int test_acosh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "acosh" );
}
int test_acospi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "acospi" );
}
int test_asin( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "asin" );
}
int test_asinh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "asinh" );
}
int test_asinpi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "asinpi" );
}
int test_atan( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "atan" );
}
int test_atanh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "atanh" );
}
int test_atanpi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "atanpi" );
}
int test_atan2( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "atan2" );
}
int test_atan2pi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "atan2pi" );
}
int test_cbrt( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "cbrt" );
}
int test_ceil( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "ceil" );
}
int test_copysign( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "copysign" );
}
int test_cos( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "cos" );
}
int test_cosh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "cosh" );
}
int test_cospi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "cospi" );
}
int test_exp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "exp" );
}
int test_exp2( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "exp2" );
}
int test_exp10( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "exp10" );
}
int test_expm1( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "expm1" );
}
int test_fabs( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fabs" );
}
int test_fdim( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fdim" );
}
int test_floor( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "floor" );
}
int test_fma( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fma" );
}
int test_fmax( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fmax" );
}
int test_fmin( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fmin" );
}
int test_fmod( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fmod" );
}
int test_fract( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "fract" );
}
int test_frexp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "frexp" );
}
int test_hypot( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "hypot" );
}
int test_ilogb( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "ilogb" );
}
int test_isequal( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isequal" );
}
int test_isfinite( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isfinite" );
}
int test_isgreater( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isgreater" );
}
int test_isgreaterequal( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isgreaterequal" );
}
int test_isinf( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isinf" );
}
int test_isless( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isless" );
}
int test_islessequal( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "islessequal" );
}
int test_islessgreater( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "islessgreater" );
}
int test_isnan( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isnan" );
}
int test_isnormal( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isnormal" );
}
int test_isnotequal( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isnotequal" );
}
int test_isordered( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isordered" );
}
int test_isunordered( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "isunordered" );
}
int test_ldexp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "ldexp" );
}
int test_lgamma( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "lgamma" );
}
int test_lgamma_r( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "lgamma_r" );
}
int test_log( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "log" );
}
int test_log2( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "log2" );
}
int test_log10( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "log10" );
}
int test_log1p( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "log1p" );
}
int test_logb( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "logb" );
}
int test_mad( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "mad" );
}
int test_maxmag( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "maxmag" );
}
int test_minmag( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "minmag" );
}
int test_modf( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "modf" );
}
int test_nan( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "nan" );
}
int test_nextafter( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "nextafter" );
}
int test_pow( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "pow" );
}
int test_pown( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "pown" );
}
int test_powr( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "powr" );
}
int test_remainder( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "remainder" );
}
int test_remquo( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "remquo" );
}
int test_rint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "rint" );
}
int test_rootn( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "rootn" );
}
int test_round( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "round" );
}
int test_rsqrt( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "rsqrt" );
}
int test_signbit( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "signbit" );
}
int test_sin( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sin" );
}
int test_sincos( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sincos" );
}
int test_sinh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sinh" );
}
int test_sinpi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sinpi" );
}
int test_sqrt( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sqrt" );
}
int test_sqrt_cr( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "sqrt_cr" );
}
int test_tan( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "tan" );
}
int test_tanh( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "tanh" );
}
int test_tanpi( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "tanpi" );
}
int test_trunc( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "trunc" );
}
int test_half_cos( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_cos" );
}
int test_half_divide( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_divide" );
}
int test_half_exp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_exp" );
}
int test_half_exp2( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_exp2" );
}
int test_half_exp10( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_exp10" );
}
int test_half_log( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_log" );
}
int test_half_log2( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_log2" );
}
int test_half_log10( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_log10" );
}
int test_half_powr( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_powr" );
}
int test_half_recip( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_recip" );
}
int test_half_rsqrt( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_rsqrt" );
}
int test_half_sin( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_sin" );
}
int test_half_sqrt( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_sqrt" );
}
int test_half_tan( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "half_tan" );
}
int test_add( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "add" );
}
int test_subtract( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "subtract" );
}
int test_divide( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "divide" );
}
int test_divide_cr( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "divide_cr" );
}
int test_multiply( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "multiply" );
}
int test_assignment( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "assignment" );
}
int test_not( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    return doTest( "not" );
}

test_definition test_list[] = {
    ADD_TEST( acos ),
    ADD_TEST( acosh ),
    ADD_TEST( acospi ),
    ADD_TEST( asin ),
    ADD_TEST( asinh ),
    ADD_TEST( asinpi ),
    ADD_TEST( atan ),
    ADD_TEST( atanh ),
    ADD_TEST( atanpi ),
    ADD_TEST( atan2 ),
    ADD_TEST( atan2pi ),
    ADD_TEST( cbrt ),
    ADD_TEST( ceil ),
    ADD_TEST( copysign ),
    ADD_TEST( cos ),
    ADD_TEST( cosh ),
    ADD_TEST( cospi ),
    ADD_TEST( exp ),
    ADD_TEST( exp2 ),
    ADD_TEST( exp10 ),
    ADD_TEST( expm1 ),
    ADD_TEST( fabs ),
    ADD_TEST( fdim ),
    ADD_TEST( floor ),
    ADD_TEST( fma ),
    ADD_TEST( fmax ),
    ADD_TEST( fmin ),
    ADD_TEST( fmod ),
    ADD_TEST( fract ),
    ADD_TEST( frexp ),
    ADD_TEST( hypot ),
    ADD_TEST( ilogb ),
    ADD_TEST( isequal ),
    ADD_TEST( isfinite ),
    ADD_TEST( isgreater ),
    ADD_TEST( isgreaterequal ),
    ADD_TEST( isinf ),
    ADD_TEST( isless ),
    ADD_TEST( islessequal ),
    ADD_TEST( islessgreater ),
    ADD_TEST( isnan ),
    ADD_TEST( isnormal ),
    ADD_TEST( isnotequal ),
    ADD_TEST( isordered ),
    ADD_TEST( isunordered ),
    ADD_TEST( ldexp ),
    ADD_TEST( lgamma ),
    ADD_TEST( lgamma_r ),
    ADD_TEST( log ),
    ADD_TEST( log2 ),
    ADD_TEST( log10 ),
    ADD_TEST( log1p ),
    ADD_TEST( logb ),
    ADD_TEST( mad ),
    ADD_TEST( maxmag ),
    ADD_TEST( minmag ),
    ADD_TEST( modf ),
    ADD_TEST( nan ),
    ADD_TEST( nextafter ),
    ADD_TEST( pow ),
    ADD_TEST( pown ),
    ADD_TEST( powr ),
    ADD_TEST( remainder ),
    ADD_TEST( remquo ),
    ADD_TEST( rint ),
    ADD_TEST( rootn ),
    ADD_TEST( round ),
    ADD_TEST( rsqrt ),
    ADD_TEST( signbit ),
    ADD_TEST( sin ),
    ADD_TEST( sincos ),
    ADD_TEST( sinh ),
    ADD_TEST( sinpi ),
    ADD_TEST( sqrt ),
    ADD_TEST( sqrt_cr ),
    ADD_TEST( tan ),
    ADD_TEST( tanh ),
    ADD_TEST( tanpi ),
    ADD_TEST( trunc ),
    ADD_TEST( half_cos ),
    ADD_TEST( half_divide ),
    ADD_TEST( half_exp ),
    ADD_TEST( half_exp2 ),
    ADD_TEST( half_exp10 ),
    ADD_TEST( half_log ),
    ADD_TEST( half_log2 ),
    ADD_TEST( half_log10 ),
    ADD_TEST( half_powr ),
    ADD_TEST( half_recip ),
    ADD_TEST( half_rsqrt ),
    ADD_TEST( half_sin ),
    ADD_TEST( half_sqrt ),
    ADD_TEST( half_tan ),
    ADD_TEST( add ),
    ADD_TEST( subtract ),
    ADD_TEST( divide ),
    ADD_TEST( divide_cr ),
    ADD_TEST( multiply ),
    ADD_TEST( assignment ),
    ADD_TEST( not ),
};

const int test_num = ARRAY_SIZE( test_list );

#pragma mark -

int main (int argc, const char * argv[])
{
    int error;

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return -1;
    }
    atexit(TestFinishAtExit);

#if defined( __APPLE__ )
    struct timeval startTime;
    gettimeofday( &startTime, NULL );
#endif

    error = ParseArgs( argc, argv );
    if( error )
        return error;

    // This takes a while, so prevent the machine from going to sleep.
    PreventSleep();
    atexit( ResumeSleep );

    if( gSkipCorrectnessTesting )
        vlog( "*** Skipping correctness testing! ***\n\n" );
    else if( gStopOnError )
        vlog( "Stopping at first error.\n" );

    if( gMeasureTimes )
    {
        vlog( "%s times are reported at right (cycles per element):\n", method[gReportAverageTimes] );
        vlog( "\n" );
        if( gSkipCorrectnessTesting )
            vlog( "   \t               ");
        else
            vlog( "   \t                                        ");
        if( gWimpyMode )
            vlog( "   " );
        for( int i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
            vlog( "\t  float%s", sizeNames[i] );
    }
    else
    {
        vlog( "   \t                                        ");
        if( gWimpyMode )
            vlog( "   " );
    }
    if( ! gSkipCorrectnessTesting )
        vlog( "\t  max_ulps" );

    vlog( "\n-----------------------------------------------------------------------------------------------------------\n" );

    gMTdata = init_genrand( gRandomSeed );
    if( gEndTestNumber == 0 )
    {
        gEndTestNumber = functionListCount;
    }

    FPU_mode_type oldMode;
    DisableFTZ( &oldMode );

    int ret = runTestHarnessWithCheck( gTestNameCount, gTestNames, test_num, test_list, false, true, 0, InitCL );

    RestoreFPState( &oldMode );

    free_mtdata(gMTdata);
    free(gTestNames);

    int error_code = clFinish(gQueue);
    if (error_code)
        vlog_error("clFinish failed:%d\n", error_code);

    if (gFailCount == 0)
    {
        if (gTestCount > 1)
            vlog("PASSED %d of %d sub-tests.\n", gTestCount, gTestCount);
        else
            vlog("PASSED sub-test.\n");
    }
    else if (gFailCount > 0)
    {
        if (gTestCount > 1)
            vlog_error("FAILED %d of %d sub-tests.\n", gFailCount, gTestCount);
        else
            vlog_error("FAILED sub-test.\n");
    }

    ReleaseCL();

#if defined( __APPLE__ )
    struct timeval endTime;
    gettimeofday( &endTime, NULL );
    double time = (double) endTime.tv_sec - (double) startTime.tv_sec;
    time += 1e-6 * ((double) endTime.tv_usec - (double) startTime.tv_usec);
    vlog( "time: %f s\n", time );
#endif

    return ret;
}

static int ParseArgs( int argc, const char **argv )
{
    int i;
    gTestNames = (const char**) calloc( argc - 1, sizeof( char*) );
    if( NULL == gTestNames )
    {
        vlog( "Failed to allocate memory for gTestNames array.\n" );
        return 1;
    }
    gTestNames[0] = argv[0];
    gTestNameCount = 1;
    int singleThreaded = 0;

    { // Extract the app name
        strncpy( appName, argv[0], MAXPATHLEN );

#if defined( __APPLE__ )
        char baseName[MAXPATHLEN];
        char *base = NULL;
        strncpy( baseName, argv[0], MAXPATHLEN );
        base = basename( baseName );
        if( NULL != base )
        {
            strncpy( appName, base, sizeof( appName )  );
            appName[ sizeof( appName ) -1 ] = '\0';
        }
#endif
    }

    vlog( "\n%s\t", appName );
    for( i = 1; i < argc; i++ )
    {
        const char *arg = argv[i];
        if( NULL == arg )
            break;

        vlog( "\t%s", arg );
        int optionFound = 0;
        if( arg[0] == '-' )
        {
            while( arg[1] != '\0' )
            {
                arg++;
                optionFound = 1;
                switch( *arg )
                {
                    case 'a':
                        gReportAverageTimes ^= 1;
                        break;

                    case 'c':
                        gToggleCorrectlyRoundedDivideSqrt ^= 1;
                        break;

                    case 'd':
                        gHasDouble ^= 1;
                        break;

                    case 'e':
                        gFastRelaxedDerived ^= 1;
                        break;

                    case 'f':
                        gTestFloat ^= 1;
                        break;

                    case 'h':
                        PrintUsage();
                        return -1;

                    case 'p':
                      PrintFunctions();
                      return -1;

                    case 'l':
                        gSkipCorrectnessTesting ^= 1;
                        break;

                    case 'm':
                        singleThreaded ^= 1;
                        break;

                    case 'r':
                        gTestFastRelaxed ^= 1;
                        break;

                    case 's':
                        gStopOnError ^= 1;
                        break;

                    case 't':
                        gMeasureTimes ^= 1;
                        break;

                    case 'v':
                        gVerboseBruteForce ^= 1;
                        break;

                    case 'w':   // wimpy mode
                        gWimpyMode ^= 1;
                        break;

                    case '[':
                        parseWimpyReductionFactor(arg, gWimpyReductionFactor);
                        break;

                    case 'z':
                        gForceFTZ ^= 1;
                        break;

                    case '1':
                        if( arg[1] == '6' )
                        {
                            gMinVectorSizeIndex = 5;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            arg++;
                        }
                        else
                        {
                            gMinVectorSizeIndex = 0;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        }
                        break;
                    case '2':
                            gMinVectorSizeIndex = 1;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            break;
                    case '3':
                            gMinVectorSizeIndex = 2;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            break;
                    case '4':
                            gMinVectorSizeIndex = 3;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            break;
                    case '8':
                            gMinVectorSizeIndex = 4;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            break;
                        break;

                    default:
                        vlog( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        PrintUsage();
                        return -1;
                }
            }
        }

        if( ! optionFound )
        {
            char *t = NULL;
            long number = strtol( arg, &t, 0 );
            if( t != arg )
            {
                if( 0 == gStartTestNumber )
                    gStartTestNumber = (int32_t) number;
                else
                    gEndTestNumber = gStartTestNumber + (int32_t) number;
            }
            else
            {
                // Make sure this is a valid name
                unsigned int k;
                for (k=0; k<functionListCount; k++)
                {
                    const Func *f = functionList+k;
                    if (strcmp(arg, f->name) == 0)
                    {
                        gTestNames[ gTestNameCount ] = arg;
                        gTestNameCount++;
                        break;
                    }
                }
                // If we didn't find it in the list of test names
                if (k >= functionListCount)
                {
                    gTestNames[gTestNameCount] = arg;
                    gTestNameCount++;
                }
            }
        }
    }

    // Check for the wimpy mode environment variable
    if (getenv("CL_WIMPY_MODE")) {
      vlog( "\n" );
      vlog( "*** Detected CL_WIMPY_MODE env                          ***\n" );
      gWimpyMode = 1;
    }

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
                    vlog_error( "Error: Unknown CL_MAX_SSE setting: %s\n", env );
                    return -2;
                }

                vlog( "*** Environment: CL_MAX_SSE = %s ***\n", env );
                _cpu_capabilities &= ~mask;
            }
        }
    #endif
#endif   /* __APPLE__ */

    vlog( "\nTest binary built %s %s\n", __DATE__, __TIME__ );

    PrintArch();

    if( gWimpyMode )
    {
        vlog( "\n" );
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
        vlog( "*** Wimpy Reduction Factor: %-27u ***\n\n", gWimpyReductionFactor );
    }

    if( singleThreaded )
        SetThreadCount(1);

    return 0;
}

static void PrintArch( void )
{
    vlog( "\nHost info:\n" );
    vlog( "\tsizeof( void*) = %zd\n", sizeof( void *) );
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
        vlog( "\tARCH:\taarch64\n");
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

#elif defined( __linux__ ) && !defined(__aarch64__)
        int _sysctl(struct __sysctl_args *args );
        #define OSNAMESZ 100

        struct __sysctl_args args;
        char osname[OSNAMESZ];
        size_t osnamelth;
        int name[] = { CTL_KERN, KERN_OSTYPE };
        memset(&args, 0, sizeof(struct __sysctl_args));
        args.name = name;
        args.nlen = sizeof(name)/sizeof(name[0]);
        args.oldval = osname;
        args.oldlenp = &osnamelth;

        osnamelth = sizeof(osname);

        if (syscall(SYS__sysctl, &args) == -1) {
           vlog( "_sysctl error\n" );
        }
        else {
           vlog("this machine is running %*s\n", osnamelth, osname);
        }


#endif
}

static void PrintFunctions ( void )
{
  vlog( "\nMath function names:\n" );
  for( int i = 0; i < functionListCount; i++ )
  {
    vlog( "\t%s\n", functionList[ i ].name );
  }
}

static void PrintUsage( void )
{
    vlog( "%s [-acglstz]: <optional: math function names>\n", appName );
    vlog( "\toptions:\n" );
    vlog( "\t\t-a\tReport average times instead of best times\n" );
    vlog( "\t\t-c\tToggle test fp correctly rounded divide and sqrt (Default: off)\n");
    vlog( "\t\t-d\tToggle double precision testing. (Default: on iff khr_fp_64 on)\n" );
    vlog( "\t\t-f\tToggle float precision testing. (Default: on)\n" );
    vlog( "\t\t-r\tToggle fast relaxed math precision testing. (Default: on)\n" );
    vlog( "\t\t-e\tToggle test as derived implementations for fast relaxed math precision. (Default: on)\n" );
    vlog( "\t\t-h\tPrint this message and quit\n" );
    vlog( "\t\t-p\tPrint all math function names and quit\n" );
    vlog( "\t\t-l\tlink check only (make sure functions are present, skip accuracy checks.)\n" );
    vlog( "\t\t-m\tToggle run multi-threaded. (Default: on) )\n" );
    vlog( "\t\t-s\tStop on error\n" );
    vlog( "\t\t-t\tToggle timing  (on by default)\n" );
    vlog( "\t\t-w\tToggle Wimpy Mode, * Not a valid test * \n");
    vlog( "\t\t-[2^n]\tSet wimpy reduction factor, recommended range of n is 1-10, default factor(%u)\n",gWimpyReductionFactor );
    vlog( "\t\t-z\tToggle FTZ mode (Section 6.5.3) for all functions. (Set by device capabilities by default.)\n" );
    vlog( "\t\t-v\tToggle Verbosity (Default: off)\n ");
    vlog( "\t\t-#\tTest only vector sizes #, e.g. \"-1\" tests scalar only, \"-16\" tests 16-wide vectors only.\n" );
    vlog( "\n\tYou may also pass a number instead of a function name.\n" );
    vlog( "\tThis causes the first N tests to be skipped. The tests are numbered.\n" );
    vlog( "\tIf you pass a second number, that is the number tests to run after the first one.\n" );
    vlog( "\tA name list may be used in conjunction with a number range. In that case,\n" );
    vlog( "\tonly the named cases in the number range will run.\n" );
    vlog( "\tYou may also choose to pass no arguments, in which case all tests will be run.\n" );
    vlog( "\tYou may pass CL_DEVICE_TYPE_CPU/GPU/ACCELERATOR to select the device.\n" );
    vlog( "\n" );
}

static void CL_CALLBACK bruteforce_notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    vlog( "%s  (%p, %zd, %p)\n", errinfo, private_info, cb, user_data );
}

test_status InitCL( cl_device_id device )
{
    int error;
    uint32_t i;
    size_t configSize = sizeof( gComputeDevices );
    cl_device_type device_type;

    error = clGetDeviceInfo( device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL );
    if( error )
    {
        print_error( error, "Unable to get device type" );
        return TEST_FAIL;
    }

    gDevice = device;


    if( (error = clGetDeviceInfo( gDevice, CL_DEVICE_MAX_COMPUTE_UNITS, configSize, &gComputeDevices, NULL )) )
        gComputeDevices = 1;

    // Check extensions
    size_t extSize = 0;
    if((error = clGetDeviceInfo( gDevice, CL_DEVICE_EXTENSIONS, 0, NULL, &extSize)))
    {   vlog_error( "Unable to get device extension string to see if double present. (%d) \n", error ); }
    else
    {
        char *ext = (char*) malloc( extSize );
        if( NULL == ext )
        { vlog_error( "malloc failed at %s:%d\nUnable to determine if double present.\n", __FILE__, __LINE__ ); }
        else
        {
            if((error = clGetDeviceInfo( gDevice, CL_DEVICE_EXTENSIONS, extSize, ext, NULL)))
            {    vlog_error( "Unable to get device extension string to see if double present. (%d) \n", error ); }
            else
            {
                if( strstr( ext, "cl_khr_fp64" ))
                {
                    gHasDouble ^= 1;

#if defined( CL_DEVICE_DOUBLE_FP_CONFIG )
                    if( (error = clGetDeviceInfo(gDevice, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(gDoubleCapabilities), &gDoubleCapabilities, NULL)))
                    {
                        vlog_error( "ERROR: Unable to get device CL_DEVICE_DOUBLE_FP_CONFIG. (%d)\n", error );
                        return TEST_FAIL;
                    }

                    if( DOUBLE_REQUIRED_FEATURES != (gDoubleCapabilities & DOUBLE_REQUIRED_FEATURES) )
                    {
                        char list[300] = "";
                        if( 0 == (gDoubleCapabilities & CL_FP_FMA) )
                            strncat( list, "CL_FP_FMA, ", sizeof( list )-1 );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_NEAREST) )
                            strncat( list, "CL_FP_ROUND_TO_NEAREST, ", sizeof( list )-1 );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_ZERO) )
                            strncat( list, "CL_FP_ROUND_TO_ZERO, ", sizeof( list )-1 );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_INF) )
                            strncat( list, "CL_FP_ROUND_TO_INF, ", sizeof( list )-1 );
                        if( 0 == (gDoubleCapabilities & CL_FP_INF_NAN) )
                            strncat( list, "CL_FP_INF_NAN, ", sizeof( list )-1 );
                        if( 0 == (gDoubleCapabilities & CL_FP_DENORM) )
                            strncat( list, "CL_FP_DENORM, ", sizeof( list )-1 );
                        vlog_error( "ERROR: required double features are missing: %s\n", list );

                        free(ext);
                        return TEST_FAIL;
                    }
#else
                    vlog_error( "FAIL: device says it supports cl_khr_fp64 but CL_DEVICE_DOUBLE_FP_CONFIG is not in the headers!\n" );
                    return TEST_FAIL;
#endif
                }
#if defined( __APPLE__ )
                else if( strstr( ext, "cl_APPLE_fp64_basic_ops" ))
                {
                    gHasBasicDouble ^= 1;

#if defined( CL_DEVICE_DOUBLE_FP_CONFIG )
                    if( (error = clGetDeviceInfo(gDevice, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(gDoubleCapabilities), &gDoubleCapabilities, NULL)))
                    {
                        vlog_error( "ERROR: Unable to get device CL_DEVICE_DOUBLE_FP_CONFIG. (%d)\n", error );
                        return TEST_FAIL;
                    }

                    if( DOUBLE_REQUIRED_FEATURES != (gDoubleCapabilities & DOUBLE_REQUIRED_FEATURES) )
                    {
                        char list[300] = "";
                        if( 0 == (gDoubleCapabilities & CL_FP_FMA) )
                            strncat( list, "CL_FP_FMA, ", sizeof( list ) );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_NEAREST) )
                            strncat( list, "CL_FP_ROUND_TO_NEAREST, ", sizeof( list ) );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_ZERO) )
                            strncat( list, "CL_FP_ROUND_TO_ZERO, ", sizeof( list ) );
                        if( 0 == (gDoubleCapabilities & CL_FP_ROUND_TO_INF) )
                            strncat( list, "CL_FP_ROUND_TO_INF, ", sizeof( list ) );
                        if( 0 == (gDoubleCapabilities & CL_FP_INF_NAN) )
                            strncat( list, "CL_FP_INF_NAN, ", sizeof( list ) );
                        if( 0 == (gDoubleCapabilities & CL_FP_DENORM) )
                            strncat( list, "CL_FP_DENORM, ", sizeof( list ) );
                        vlog_error( "ERROR: required double features are missing: %s\n", list );

                        free(ext);
                        return TEST_FAIL;
                    }
#else
                    vlog_error( "FAIL: device says it supports cl_khr_fp64 but CL_DEVICE_DOUBLE_FP_CONFIG is not in the headers!\n" );
                    return TEST_FAIL;
#endif
                }
#endif /* __APPLE__ */
            }
            free(ext);
        }
    }

    configSize = sizeof( gDeviceFrequency );
    if( (error = clGetDeviceInfo( gDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, configSize, &gDeviceFrequency, NULL )) )
        gDeviceFrequency = 0;

    if( (error = clGetDeviceInfo(gDevice, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(gFloatCapabilities), &gFloatCapabilities, NULL)))
    {
        vlog_error( "ERROR: Unable to get device CL_DEVICE_SINGLE_FP_CONFIG. (%d)\n", error );
        return TEST_FAIL;
    }

    gContext = clCreateContext( NULL, 1, &gDevice, bruteforce_notify_callback, NULL, &error );
    if( NULL == gContext || error )
    {
        vlog_error( "clCreateContext failed. (%d) \n", error );
        return TEST_FAIL;
    }

    gQueue = clCreateCommandQueueWithProperties(gContext, gDevice, 0, &error);
    if( NULL == gQueue || error )
    {
        vlog_error( "clCreateCommandQueue failed. (%d)\n", error );
        return TEST_FAIL;
    }

#if defined( __APPLE__ )
    // FIXME: use clProtectedArray
#endif
    //Allocate buffers
    cl_uint min_alignment = 0;
    error = clGetDeviceInfo (gDevice, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), (void*)&min_alignment, NULL);
    if (CL_SUCCESS != error)
    {
        vlog_error( "clGetDeviceInfo failed. (%d)\n", error );
        return TEST_FAIL;
    }
    min_alignment >>= 3;    // convert bits to bytes

    gIn   = align_malloc( BUFFER_SIZE, min_alignment );
    if( NULL == gIn )
        return TEST_FAIL;
    gIn2   = align_malloc( BUFFER_SIZE, min_alignment );
    if( NULL == gIn2 )
        return TEST_FAIL;
    gIn3   = align_malloc( BUFFER_SIZE, min_alignment );
    if( NULL == gIn3 )
        return TEST_FAIL;
    gOut_Ref   = align_malloc( BUFFER_SIZE, min_alignment );
    if( NULL == gOut_Ref )
        return TEST_FAIL;
    gOut_Ref2   = align_malloc( BUFFER_SIZE, min_alignment );
    if( NULL == gOut_Ref2 )
        return TEST_FAIL;

    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        gOut[i] = align_malloc( BUFFER_SIZE, min_alignment );
        if( NULL == gOut[i] )
            return TEST_FAIL;
        gOut2[i] = align_malloc( BUFFER_SIZE, min_alignment );
        if( NULL == gOut2[i] )
            return TEST_FAIL;
    }

    cl_mem_flags device_flags = CL_MEM_READ_ONLY;
    // save a copy on the host device to make this go faster
    if( CL_DEVICE_TYPE_CPU == device_type )
        device_flags |= CL_MEM_USE_HOST_PTR;
      else
          device_flags |= CL_MEM_COPY_HOST_PTR;

    // setup input buffers
    gInBuffer = clCreateBuffer(gContext, device_flags, BUFFER_SIZE, gIn, &error);
    if( gInBuffer == NULL || error )
    {
        vlog_error( "clCreateBuffer1 failed for input (%d)\n", error );
        return TEST_FAIL;
    }

    gInBuffer2 = clCreateBuffer( gContext, device_flags, BUFFER_SIZE, gIn2, &error );
    if( gInBuffer2 == NULL || error )
    {
        vlog_error( "clCreateArray2 failed for input (%d)\n" , error );
        return TEST_FAIL;
    }

    gInBuffer3 = clCreateBuffer( gContext, device_flags, BUFFER_SIZE, gIn3, &error );
    if( gInBuffer3 == NULL  || error)
    {
        vlog_error( "clCreateArray3 failed for input (%d)\n", error );
        return TEST_FAIL;
    }


    // setup output buffers
    device_flags = CL_MEM_READ_WRITE;
    // save a copy on the host device to make this go faster
    if( CL_DEVICE_TYPE_CPU == device_type )
        device_flags |= CL_MEM_USE_HOST_PTR;
      else
          device_flags |= CL_MEM_COPY_HOST_PTR;
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        gOutBuffer[i] = clCreateBuffer( gContext, device_flags, BUFFER_SIZE, gOut[i], &error );
        if( gOutBuffer[i] == NULL || error )
        {
            vlog_error( "clCreateArray failed for output (%d)\n", error  );
            return TEST_FAIL;
        }
        gOutBuffer2[i] = clCreateBuffer( gContext, device_flags, BUFFER_SIZE, gOut2[i], &error );
        if( gOutBuffer2[i] == NULL || error)
        {
            vlog_error( "clCreateArray2 failed for output (%d)\n", error );
            return TEST_FAIL;
        }
    }

    // we are embedded, check current rounding mode
    if( gIsEmbedded )
    {
        gIsInRTZMode = IsInRTZMode();
    }

    //Check tininess detection
    IsTininessDetectedBeforeRounding();

    cl_platform_id platform;
    int err = clGetPlatformIDs(1, &platform, NULL);
    if( err )
    {
        print_error(err, "clGetPlatformIDs failed");
        return TEST_FAIL;
    }

    char c[1024];
    static const char *no_yes[] = { "NO", "YES" };
    vlog( "\nCompute Device info:\n" );
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(c), &c, NULL);
    vlog( "\tPlatform Version: %s\n", c );
    clGetDeviceInfo(gDevice, CL_DEVICE_NAME, sizeof(c), &c, NULL);
    vlog( "\tDevice Name: %s\n", c );
    clGetDeviceInfo(gDevice, CL_DEVICE_VENDOR, sizeof(c), &c, NULL);
    vlog( "\tVendor: %s\n", c );
    clGetDeviceInfo(gDevice, CL_DEVICE_VERSION, sizeof(c), &c, NULL);
    vlog( "\tDevice Version: %s\n", c );
    clGetDeviceInfo(gDevice, CL_DEVICE_OPENCL_C_VERSION, sizeof(c), &c, NULL);
    vlog( "\tCL C Version: %s\n", c );
    clGetDeviceInfo(gDevice, CL_DRIVER_VERSION, sizeof(c), &c, NULL);
    vlog( "\tDriver Version: %s\n", c );
    vlog( "\tDevice Frequency: %d MHz\n", gDeviceFrequency );
    vlog( "\tSubnormal values supported for floats? %s\n", no_yes[0 != (CL_FP_DENORM & gFloatCapabilities)] );
    vlog( "\tCorrectly rounded divide and sqrt supported for floats? %s\n", no_yes[0 != (CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT & gFloatCapabilities)] );
    if( gToggleCorrectlyRoundedDivideSqrt )
    {
        gFloatCapabilities ^= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    vlog( "\tTesting with correctly rounded float divide and sqrt? %s\n", no_yes[0 != (CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT & gFloatCapabilities)] );
    vlog( "\tTesting with FTZ mode ON for floats? %s\n", no_yes[0 != gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities)] );
    vlog( "\tTesting single precision? %s\n", no_yes[0 != gTestFloat] );
    vlog( "\tTesting fast relaxed math? %s\n", no_yes[0 != gTestFastRelaxed] );
    if(gTestFastRelaxed)
    {
      vlog( "\tFast relaxed math has derived implementations? %s\n", no_yes[0 != gFastRelaxedDerived] );
    }
    vlog( "\tTesting double precision? %s\n", no_yes[0 != gHasDouble] );
    if( sizeof( long double) == sizeof( double ) && gHasDouble )
    {
        vlog( "\n\t\tWARNING: Host system long double does not have better precision than double!\n" );
        vlog( "\t\t         All double results that do not match the reference result have their reported\n" );
        vlog( "\t\t         error inflated by 0.5 ulps to account for the fact that this system\n" );
        vlog( "\t\t         can not accurately represent the right result to an accuracy closer\n" );
        vlog( "\t\t         than half an ulp. See comments in Bruteforce_Ulp_Error_Double() for more details.\n\n" );
    }
#if defined( __APPLE__ )
    vlog( "\tTesting basic double precision? %s\n", no_yes[0 != gHasBasicDouble] );
#endif

    vlog( "\tIs Embedded? %s\n", no_yes[0 != gIsEmbedded] );
    if( gIsEmbedded )
        vlog( "\tRunning in RTZ mode? %s\n", no_yes[0 != gIsInRTZMode] );
    vlog( "\tTininess is detected before rounding? %s\n", no_yes[0 != gCheckTininessBeforeRounding] );
    vlog( "\tWorker threads: %d\n", GetThreadCount() );
    vlog( "\tTesting vector sizes:" );
    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
        vlog( "\t%d", sizeValues[i] );

    vlog("\n");
    vlog("\tVerbose? %s\n", no_yes[0 != gVerboseBruteForce]);
    vlog( "\n\n" );

    // Check to see if we are using single threaded mode on other than a 1.0 device
    if (getenv( "CL_TEST_SINGLE_THREADED" )) {

      char device_version[1024] = { 0 };
      clGetDeviceInfo( gDevice, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL );

      if (strcmp("OpenCL 1.0 ",device_version)) {
        vlog("ERROR: CL_TEST_SINGLE_THREADED is set in the environment. Running single threaded.\n");
      }
    }

    return TEST_PASS;
}

static void ReleaseCL( void )
{
    uint32_t i;
    clReleaseMemObject(gInBuffer);
    clReleaseMemObject(gInBuffer2);
    clReleaseMemObject(gInBuffer3);
    for ( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++) {
        clReleaseMemObject(gOutBuffer[i]);
        clReleaseMemObject(gOutBuffer2[i]);
    }
    clReleaseCommandQueue(gQueue);
    clReleaseContext(gContext);

    align_free(gIn);
    align_free(gIn2);
    align_free(gIn3);
    align_free(gOut_Ref);
    align_free(gOut_Ref2);

    for( i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++ )
    {
        align_free(gOut[i]);
        align_free(gOut2[i]);
    }
}

void _LogBuildError( cl_program p, int line, const char *file )
{
    char the_log[2048] = "";

    vlog_error( "%s:%d: Build Log:\n", file, line );
    if( 0 == clGetProgramBuildInfo(p, gDevice, CL_PROGRAM_BUILD_LOG, sizeof(the_log), the_log, NULL) )
        vlog_error( "%s", the_log );
    else
        vlog_error( "*** Error getting build log for program %p\n", p );
}

int InitILogbConstants( void )
{
    int error;
    const char *kernel =
    "__kernel void GetILogBConstants( __global int *out )\n"
    "{\n"
    "   out[0] = FP_ILOGB0;\n"
    "   out[1] = FP_ILOGBNAN;\n"
    "}\n";

    cl_program query;
    error = create_single_kernel_helper(gContext, &query, NULL, 1, &kernel, NULL);
    if (NULL == query || error)
    {
        vlog_error( "Error: Unable to create program to get FP_ILOGB0 and FP_ILOGBNAN for the device. (%d)", error );
        return error;
    }

    cl_kernel k = clCreateKernel( query, "GetILogBConstants", &error );
    if( NULL == k || error)
    {
      vlog_error( "Error: Unable to create kernel to get FP_ILOGB0 and FP_ILOGBNAN for the device. Err = %d", error );
        return error;
    }

    if((error = clSetKernelArg(k, 0, sizeof( gOutBuffer[gMinVectorSizeIndex]), &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error( "Error: Unable to set kernel arg to get FP_ILOGB0 and FP_ILOGBNAN for the device. Err = %d", error );
        return error;
    }

    size_t dim = 1;
    if((error = clEnqueueNDRangeKernel(gQueue, k, 1, NULL, &dim, NULL, 0, NULL, NULL) ))
    {
        vlog_error( "Error: Unable to execute kernel to get FP_ILOGB0 and FP_ILOGBNAN for the device. Err = %d", error );
        return error;
    }

    struct{ cl_int ilogb0, ilogbnan; }data;
    if(( error = clEnqueueReadBuffer( gQueue, gOutBuffer[gMinVectorSizeIndex], CL_TRUE, 0, sizeof( data ), &data, 0, NULL, NULL)))
    {
        vlog_error( "Error: unable to read FP_ILOGB0 and FP_ILOGBNAN from the device. Err = %d", error );
        return error;
    }

    gDeviceILogb0 = data.ilogb0;
    gDeviceILogbNaN = data.ilogbnan;

    clReleaseKernel(k);
    clReleaseProgram(query);

    return 0;
}

int IsTininessDetectedBeforeRounding( void )
{
    int error;
    const char *kernel =
    "__kernel void IsTininessDetectedBeforeRounding( __global float *out )\n"
    "{\n"
    "   volatile float a = 0x1.000002p-126f;\n"
    "   volatile float b = 0x1.fffffcp-1f;\n"       // product is 0x1.fffffffffff8p-127
    "   out[0] = a * b;\n"
    "}\n";

    cl_program query;
    error = create_single_kernel_helper(gContext, &query, NULL, 1, &kernel, NULL);
    if (error != CL_SUCCESS) {
        vlog_error( "Error: Unable to create program to detect how tininess is detected for the device. (%d)", error );
        return error;
    }

    cl_kernel k = clCreateKernel( query, "IsTininessDetectedBeforeRounding", &error );
    if( NULL == k || error)
    {
      vlog_error( "Error: Unable to create kernel to detect how tininess is detected  for the device. Err = %d", error );
        return error;
    }

    if((error = clSetKernelArg(k, 0, sizeof( gOutBuffer[gMinVectorSizeIndex]), &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error( "Error: Unable to set kernel arg to detect how tininess is detected  for the device. Err = %d", error );
        return error;
    }

    size_t dim = 1;
    if((error = clEnqueueNDRangeKernel(gQueue, k, 1, NULL, &dim, NULL, 0, NULL, NULL) ))
    {
        vlog_error( "Error: Unable to execute kernel to detect how tininess is detected  for the device. Err = %d", error );
        return error;
    }

    struct{ cl_uint f; }data;
    if(( error = clEnqueueReadBuffer( gQueue, gOutBuffer[gMinVectorSizeIndex], CL_TRUE, 0, sizeof( data ), &data, 0, NULL, NULL)))
    {
        vlog_error( "Error: unable to read result from tininess test from the device. Err = %d", error );
        return error;
    }

    gCheckTininessBeforeRounding = 0 == (data.f & 0x7fffffff);

    clReleaseKernel(k);
    clReleaseProgram(query);

    return 0;
}


int MakeKernel( const char **c, cl_uint count, const char *name, cl_kernel *k, cl_program *p )
{
    int error = 0;
    char options[200];

    strcpy(options, "-cl-std=CL2.0");

    if( gForceFTZ )
    {
      strcat(options," -cl-denorms-are-zero");
    }

    if( gTestFastRelaxed )
    {
      strcat(options, " -cl-fast-relaxed-math");
    }

    error = create_single_kernel_helper(gContext, p, NULL, count, c, NULL, options);
    if (error != CL_SUCCESS)
    {
        vlog_error("\t\tFAILED -- Failed to create program. (%d)\n", error);
        return error;
    }

    *k = clCreateKernel( *p, name, &error );
    if( NULL == *k || error )
    {
        char    buffer[2048] = "";

        vlog_error("\t\tFAILED -- clCreateKernel() failed: (%d)\n", error);
        clGetProgramBuildInfo(*p, gDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        vlog_error("Log: %s\n", buffer);
        clReleaseProgram( *p );
        return error;
    }

    return error;
}

int MakeKernels( const char **c, cl_uint count, const char *name, cl_uint kernel_count, cl_kernel *k, cl_program *p )
{
    int error = 0;
    cl_uint i;
    char options[200];

    strcpy(options, "-cl-std=CL2.0");

    if (gForceFTZ)
    {
      strcat(options," -cl-denorms-are-zero ");
    }

    if( gFloatCapabilities & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT )
    {
      strcat(options," -cl-fp32-correctly-rounded-divide-sqrt ");
    }

    if( gTestFastRelaxed )
    {
      strcat(options, " -cl-fast-relaxed-math");
    }

    error = create_single_kernel_helper(gContext, p, NULL, count, c, NULL, options);
    if ( error != CL_SUCCESS )
    {
        vlog_error( "\t\tFAILED -- Failed to create program. (%d)\n", error );
        return error;
    }


    memset( k, 0, kernel_count * sizeof( *k) );
    for( i = 0; i< kernel_count; i++ )
    {
        k[i] = clCreateKernel( *p, name, &error );
        if( NULL == k[i]|| error )
        {
            char    buffer[2048] = "";

            vlog_error("\t\tFAILED -- clCreateKernel() failed: (%d)\n", error);
            clGetProgramBuildInfo(*p, gDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
            vlog_error("Log: %s\n", buffer);
            clReleaseProgram( *p );
            return error;
        }
    }

    return error;
}


static int IsInRTZMode( void )
{
    int error;
    const char *kernel =
    "__kernel void GetRoundingMode( __global int *out )\n"
    "{\n"
    "   volatile float a = 0x1.0p23f;\n"
    "   volatile float b = -0x1.0p23f;\n"
    "   out[0] = (a + 0x1.fffffep-1f == a) && (b - 0x1.fffffep-1f == b);\n"
    "}\n";

    cl_program query;
    error = create_single_kernel_helper(gContext, &query, NULL, 1, &kernel, NULL);
    if (error != CL_SUCCESS) {
        vlog_error( "Error: Unable to create program to detect RTZ mode for the device. (%d)", error );
        return error;
    }

    cl_kernel k = clCreateKernel( query, "GetRoundingMode", &error );
    if( NULL == k || error)
    {
        vlog_error( "Error: Unable to create kernel to gdetect RTZ mode for the device. Err = %d", error );
        return error;
    }

    if((error = clSetKernelArg(k, 0, sizeof( gOutBuffer[gMinVectorSizeIndex]), &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error( "Error: Unable to set kernel arg to detect RTZ mode for the device. Err = %d", error );
        return error;
    }

    size_t dim = 1;
    if((error = clEnqueueNDRangeKernel(gQueue, k, 1, NULL, &dim, NULL, 0, NULL, NULL) ))
    {
        vlog_error( "Error: Unable to execute kernel to detect RTZ mode for the device. Err = %d", error );
        return error;
    }

    struct{ cl_int isRTZ; }data;
    if(( error = clEnqueueReadBuffer( gQueue, gOutBuffer[gMinVectorSizeIndex], CL_TRUE, 0, sizeof( data ), &data, 0, NULL, NULL)))
    {
        vlog_error( "Error: unable to read RTZ mode data from the device. Err = %d", error );
        return error;
    }

    clReleaseKernel(k);
    clReleaseProgram(query);

    return data.isRTZ;
}

#pragma mark -

const char *sizeNames[ VECTOR_SIZE_COUNT] = { "", "2", "3", "4", "8", "16" };
const int  sizeValues[ VECTOR_SIZE_COUNT] = { 1, 2, 3, 4, 8, 16 };

// TODO: There is another version of Ulp_Error_Double defined in test_common/harness/errorHelpers.c
float Bruteforce_Ulp_Error_Double( double test, long double reference )
{
//Check for Non-power-of-two and NaN

  // Note: This function presumes that someone has already tested whether the result is correctly,
  // rounded before calling this function.  That test:
  //
  //    if( (float) reference == test )
  //        return 0.0f;
  //
  // would ensure that cases like fabs(reference) > FLT_MAX are weeded out before we get here.
  // Otherwise, we'll return inf ulp error here, for what are otherwise correctly rounded
  // results.

  // Deal with long double = double
  // On most systems long double is a higher precision type than double. They provide either
  // a 80-bit or greater floating point type, or they provide a head-tail double double format.
  // That is sufficient to represent the accuracy of a floating point result to many more bits
  // than double and we can calculate sub-ulp errors. This is the standard system for which this
  // test suite is designed.
  //
  // On some systems double and long double are the same thing. Then we run into a problem,
  // because our representation of the infinitely precise result (passed in as reference above)
  // can be off by as much as a half double precision ulp itself.  In this case, we inflate the
  // reported error by half an ulp to take this into account.  A more correct and permanent fix
  // would be to undertake refactoring the reference code to return results in this format:
  //
  //    typedef struct DoubleReference
  //    { // true value = correctlyRoundedResult + ulps * ulp(correctlyRoundedResult)        (infinitely precise)
  //        double  correctlyRoundedResult;     // as best we can
  //        double  ulps;                       // plus a fractional amount to account for the difference
  //    }DoubleReference;                       //     between infinitely precise result and correctlyRoundedResult, in units of ulps.
  //
  // This would provide a useful higher-than-double precision format for everyone that we can use,
  // and would solve a few problems with representing absolute errors below DBL_MIN and over DBL_MAX for systems
  // that use a head to tail double double for long double.

    int x;
    long double testVal = test;

    // First, handle special reference values
    if (isinf(reference))
    {
    if (reference == testVal)
        return 0.0f;

    return INFINITY;
    }

    if (isnan(reference))
    {
    if (isnan(testVal))
        return 0.0f;

    return INFINITY;
    }

    if ( 0.0L != reference && 0.5L != frexpl(reference, &x) )
    { // Non-zero and Non-power of two

       // allow correctly rounded results to pass through unmolested. (We might add error to it below.)
       // There is something of a performance optimization here.
        if( testVal == reference )
            return 0.0f;

        // The unbiased exponent of the ulp unit place
        int ulp_exp = DBL_MANT_DIG - 1 - MAX( ilogbl( reference), DBL_MIN_EXP-1 );

        // Scale the exponent of the error
        float result = (float) scalbnl( testVal - reference, ulp_exp );

        // account for rounding error in reference result on systems that do not have a higher precision floating point type (see above)
        if( sizeof(long double) == sizeof( double ) )
            result += copysignf( 0.5f, result);

        return result;
    }

    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp =  DBL_MANT_DIG - 1 - MAX( ilogbl( reference) - 1, DBL_MIN_EXP-1 );

   // allow correctly rounded results to pass through unmolested. (We might add error to it below.)
   // There is something of a performance optimization here too.
    if( testVal == reference )
        return 0.0f;

    // Scale the exponent of the error
    float result = (float) scalbnl( testVal - reference, ulp_exp );

    // account for rounding error in reference result on systems that do not have a higher precision floating point type (see above)
    if( sizeof(long double) == sizeof( double ) )
        result += copysignf( 0.5f, result);

    return result;
}

float Abs_Error( float test, double reference )
{
  if( isnan(test) && isnan(reference) )
    return 0.0f;
  return fabs((float)(reference-(double)test));
}

/*
#define HALF_MIN_EXP    -13
#define HALF_MANT_DIG    11
float Ulp_Error_Half( float test, double reference )
{
    union{ double d; uint64_t u; }u;     u.d = reference;

  // Note: This function presumes that someone has already tested whether the result is correctly,
  // rounded before calling this function.  That test:
  //
  //    if( (float) reference == test )
  //        return 0.0f;
  //
  // would ensure that cases like fabs(reference) > FLT_MAX are weeded out before we get here.
  // Otherwise, we'll return inf ulp error here, for what are otherwise correctly rounded
  // results.

    double testVal = test;
    if( u.u & 0x000fffffffffffffULL )
    { // Non-power of two and NaN
        if( isnan( reference ) && isnan( test ) )
            return 0.0f;    // if we are expecting a NaN, any NaN is fine

        // The unbiased exponent of the ulp unit place
        int ulp_exp = HALF_MANT_DIG - 1 - MAX( ilogb( reference), HALF_MIN_EXP-1 );

        // Scale the exponent of the error
        return (float) scalbn( testVal - reference, ulp_exp );
    }

    if( isinf( reference ) )
    {
        if( (double) test == reference )
            return 0.0f;

        return (float) (testVal - reference );
    }

    // reference is a normal power of two or a zero
    int ulp_exp =  HALF_MANT_DIG - 1 - MAX( ilogb( reference) - 1, HALF_MIN_EXP-1 );

    // Scale the exponent of the error
    return (float) scalbn( testVal - reference, ulp_exp );
}
*/


#if defined( __APPLE__ )
    #include <mach/mach_time.h>
#endif

uint64_t GetTime( void )
{
#if defined( __APPLE__ )
    return mach_absolute_time();
#elif defined(_WIN32) && defined(_MSC_VER)
    return  ReadTime();
#else
    //mach_absolute_time is a high precision timer with precision < 1 microsecond.
    #warning need accurate clock here.  Times are invalid.
    return 0;
#endif
}


#if defined(_WIN32) && defined (_MSC_VER)
/* function is defined in "compat.h" */
#else
double SubtractTime( uint64_t endTime, uint64_t startTime )
{
    uint64_t diff = endTime - startTime;
    static double conversion = 0.0;

    if( 0.0 == conversion )
    {
#if defined( __APPLE__ )
        mach_timebase_info_data_t info = {0,0};
        kern_return_t   err = mach_timebase_info( &info );
        if( 0 == err )
            conversion = 1e-9 * (double) info.numer / (double) info.denom;
#else
    // This function consumes output from GetTime() above, and converts the time to secionds.
    #warning need accurate ticks to seconds conversion factor here. Times are invalid.
#endif
    }

    // strictly speaking we should also be subtracting out timer latency here
    return conversion * (double) diff;
}
#endif

cl_uint RoundUpToNextPowerOfTwo( cl_uint x )
{
    if( 0 == (x & (x-1)))
        return x;

    while( x & (x-1) )
        x &= x-1;

    return x+x;
}

void TestFinishAtExit(void) {
  test_finish();
}


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
#include "procs.h"

#define TEST_VALUE_POSITIVE( string_name, name, value ) \
{ \
if (name < value) { \
log_error("FAILED: " string_name ": " #name " < " #value "\n"); \
errors++;\
} else { \
log_info("\t" string_name ": " #name " >= " #value "\n"); \
} \
}

#define TEST_VALUE_NEGATIVE( string_name, name, value ) \
{ \
if (name > value) { \
log_error("FAILED: " string_name ": " #name " > " #value "\n"); \
errors++;\
} else { \
log_info("\t" string_name ": " #name " <= " #value "\n"); \
} \
}

#define TEST_VALUE_EQUAL_LITERAL( string_name, name, value ) \
{ \
if (name != value) { \
log_error("FAILED: " string_name ": " #name " != " #value "\n"); \
errors++;\
} else { \
log_info("\t" string_name ": " #name " = " #value "\n"); \
} \
}

#define TEST_VALUE_EQUAL( string_name, name, value ) \
{ \
if (name != value) { \
log_error("FAILED: " string_name ": " #name " != %a   (%17.21g)\n", value, value); \
errors++;\
} else { \
log_info("\t" string_name ": " #name " = %a  (%17.21g)\n", value, value); \
} \
}

int test_host_numeric_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int errors = 0;
    TEST_VALUE_EQUAL_LITERAL( "CL_CHAR_BIT",     CL_CHAR_BIT,    8)
    TEST_VALUE_EQUAL_LITERAL( "CL_SCHAR_MAX",    CL_SCHAR_MAX,   127)
    TEST_VALUE_EQUAL_LITERAL( "CL_SCHAR_MIN",    CL_SCHAR_MIN,   (-127-1))
    TEST_VALUE_EQUAL_LITERAL( "CL_CHAR_MAX",     CL_CHAR_MAX,    CL_SCHAR_MAX)
    TEST_VALUE_EQUAL_LITERAL( "CL_CHAR_MIN",     CL_CHAR_MIN,    CL_SCHAR_MIN)
    TEST_VALUE_EQUAL_LITERAL( "CL_UCHAR_MAX",    CL_UCHAR_MAX,   255)
    TEST_VALUE_EQUAL_LITERAL( "CL_SHRT_MAX",     CL_SHRT_MAX,    32767)
    TEST_VALUE_EQUAL_LITERAL( "CL_SHRT_MIN",     CL_SHRT_MIN,    (-32767-1))
    TEST_VALUE_EQUAL_LITERAL( "CL_USHRT_MAX",    CL_USHRT_MAX,   65535)
    TEST_VALUE_EQUAL_LITERAL( "CL_INT_MAX",      CL_INT_MAX,     2147483647)
    TEST_VALUE_EQUAL_LITERAL( "CL_INT_MIN",      CL_INT_MIN,     (-2147483647-1))
    TEST_VALUE_EQUAL_LITERAL( "CL_UINT_MAX",     CL_UINT_MAX,    0xffffffffU)
    TEST_VALUE_EQUAL_LITERAL( "CL_LONG_MAX",     CL_LONG_MAX,    ((cl_long) 0x7FFFFFFFFFFFFFFFLL))
    TEST_VALUE_EQUAL_LITERAL( "CL_LONG_MIN",     CL_LONG_MIN,    ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL))
    TEST_VALUE_EQUAL_LITERAL( "CL_ULONG_MAX",    CL_ULONG_MAX,   ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL))

    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_DIG",         CL_FLT_DIG,         6)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MANT_DIG",    CL_FLT_MANT_DIG,    24)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MAX_10_EXP",  CL_FLT_MAX_10_EXP,  +38)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MAX_EXP",     CL_FLT_MAX_EXP,     +128)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MIN_10_EXP",  CL_FLT_MIN_10_EXP,  -37)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MIN_EXP",     CL_FLT_MIN_EXP,     -125)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_RADIX",       CL_FLT_RADIX,       2)
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MAX",         CL_FLT_MAX,         MAKE_HEX_FLOAT( 0x1.fffffep127f, 0x1fffffeL, 103))
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_MIN",         CL_FLT_MIN,         MAKE_HEX_FLOAT(0x1.0p-126f, 0x1L, -126))
    TEST_VALUE_EQUAL_LITERAL( "CL_FLT_EPSILON",     CL_FLT_EPSILON,     MAKE_HEX_FLOAT(0x1.0p-23f, 0x1L, -23))

    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_DIG",         CL_DBL_DIG,         15)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_MANT_DIG",    CL_DBL_MANT_DIG,    53)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_MAX_10_EXP",  CL_DBL_MAX_10_EXP,  +308)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_MAX_EXP",     CL_DBL_MAX_EXP,     +1024)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_MIN_10_EXP",  CL_DBL_MIN_10_EXP,  -307)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_MIN_EXP",     CL_DBL_MIN_EXP,     -1021)
    TEST_VALUE_EQUAL_LITERAL( "CL_DBL_RADIX",       CL_DBL_RADIX,       2)
    TEST_VALUE_EQUAL( "CL_DBL_MAX",         CL_DBL_MAX,         MAKE_HEX_DOUBLE(0x1.fffffffffffffp1023, 0x1fffffffffffffLL, 971))
    TEST_VALUE_EQUAL( "CL_DBL_MIN",         CL_DBL_MIN,         MAKE_HEX_DOUBLE(0x1.0p-1022, 0x1LL, -1022))
    TEST_VALUE_EQUAL( "CL_DBL_EPSILON",     CL_DBL_EPSILON,     MAKE_HEX_DOUBLE(0x1.0p-52, 0x1LL, -52))

    TEST_VALUE_EQUAL( "CL_M_E",          CL_M_E,         MAKE_HEX_DOUBLE(0x1.5bf0a8b145769p+1, 0x15bf0a8b145769LL, -51) );
    TEST_VALUE_EQUAL( "CL_M_LOG2E",      CL_M_LOG2E,     MAKE_HEX_DOUBLE(0x1.71547652b82fep+0, 0x171547652b82feLL, -52) );
    TEST_VALUE_EQUAL( "CL_M_LOG10E",     CL_M_LOG10E,    MAKE_HEX_DOUBLE(0x1.bcb7b1526e50ep-2, 0x1bcb7b1526e50eLL, -54) );
    TEST_VALUE_EQUAL( "CL_M_LN2",        CL_M_LN2,       MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1, 0x162e42fefa39efLL, -53) );
    TEST_VALUE_EQUAL( "CL_M_LN10",       CL_M_LN10,      MAKE_HEX_DOUBLE(0x1.26bb1bbb55516p+1, 0x126bb1bbb55516LL, -51) );
    TEST_VALUE_EQUAL( "CL_M_PI",         CL_M_PI,        MAKE_HEX_DOUBLE(0x1.921fb54442d18p+1, 0x1921fb54442d18LL, -51) );
    TEST_VALUE_EQUAL( "CL_M_PI_2",       CL_M_PI_2,      MAKE_HEX_DOUBLE(0x1.921fb54442d18p+0, 0x1921fb54442d18LL, -52) );
    TEST_VALUE_EQUAL( "CL_M_PI_4",       CL_M_PI_4,      MAKE_HEX_DOUBLE(0x1.921fb54442d18p-1, 0x1921fb54442d18LL, -53) );
    TEST_VALUE_EQUAL( "CL_M_1_PI",       CL_M_1_PI,      MAKE_HEX_DOUBLE(0x1.45f306dc9c883p-2, 0x145f306dc9c883LL, -54) );
    TEST_VALUE_EQUAL( "CL_M_2_PI",       CL_M_2_PI,      MAKE_HEX_DOUBLE(0x1.45f306dc9c883p-1, 0x145f306dc9c883LL, -53) );
    TEST_VALUE_EQUAL( "CL_M_2_SQRTPI",   CL_M_2_SQRTPI,  MAKE_HEX_DOUBLE(0x1.20dd750429b6dp+0, 0x120dd750429b6dLL, -52) );
    TEST_VALUE_EQUAL( "CL_M_SQRT2",      CL_M_SQRT2,     MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp+0, 0x16a09e667f3bcdLL, -52) );
    TEST_VALUE_EQUAL( "CL_M_SQRT1_2",    CL_M_SQRT1_2,   MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp-1, 0x16a09e667f3bcdLL, -53) );

    TEST_VALUE_EQUAL( "CL_M_E_F",        CL_M_E_F,       MAKE_HEX_FLOAT(0x1.5bf0a8p+1f, 0x15bf0a8L, -23));
    TEST_VALUE_EQUAL( "CL_M_LOG2E_F",    CL_M_LOG2E_F,   MAKE_HEX_FLOAT(0x1.715476p+0f, 0x1715476L, -24));
    TEST_VALUE_EQUAL( "CL_M_LOG10E_F",   CL_M_LOG10E_F,  MAKE_HEX_FLOAT(0x1.bcb7b2p-2f, 0x1bcb7b2L, -26));
    TEST_VALUE_EQUAL( "CL_M_LN2_F",      CL_M_LN2_F,     MAKE_HEX_FLOAT(0x1.62e43p-1f, 0x162e43L, -21) );
    TEST_VALUE_EQUAL( "CL_M_LN10_F",     CL_M_LN10_F,    MAKE_HEX_FLOAT(0x1.26bb1cp+1f, 0x126bb1cL, -23));
    TEST_VALUE_EQUAL( "CL_M_PI_F",       CL_M_PI_F,      MAKE_HEX_FLOAT(0x1.921fb6p+1f, 0x1921fb6L, -23));
    TEST_VALUE_EQUAL( "CL_M_PI_2_F",     CL_M_PI_2_F,    MAKE_HEX_FLOAT(0x1.921fb6p+0f, 0x1921fb6L, -24));
    TEST_VALUE_EQUAL( "CL_M_PI_4_F",     CL_M_PI_4_F,    MAKE_HEX_FLOAT(0x1.921fb6p-1f, 0x1921fb6L, -25));
    TEST_VALUE_EQUAL( "CL_M_1_PI_F",     CL_M_1_PI_F,    MAKE_HEX_FLOAT(0x1.45f306p-2f, 0x145f306L, -26));
    TEST_VALUE_EQUAL( "CL_M_2_PI_F",     CL_M_2_PI_F,    MAKE_HEX_FLOAT(0x1.45f306p-1f, 0x145f306L, -25));
    TEST_VALUE_EQUAL( "CL_M_2_SQRTPI_F", CL_M_2_SQRTPI_F,MAKE_HEX_FLOAT(0x1.20dd76p+0f, 0x120dd76L, -24));
    TEST_VALUE_EQUAL( "CL_M_SQRT2_F",    CL_M_SQRT2_F,   MAKE_HEX_FLOAT(0x1.6a09e6p+0f, 0x16a09e6L, -24));
    TEST_VALUE_EQUAL( "CL_M_SQRT1_2_F",  CL_M_SQRT1_2_F, MAKE_HEX_FLOAT(0x1.6a09e6p-1f, 0x16a09e6L, -25));

    return errors;
}


const char *kernel_int_float[] = {
  "__kernel void test( __global float *float_out, __global int *int_out, __global uint *uint_out) \n"
  "{\n"
  "  int_out[0] = CHAR_BIT;\n"
  "  int_out[1] = SCHAR_MAX;\n"
  "  int_out[2] = SCHAR_MIN;\n"
  "  int_out[3] = CHAR_MAX;\n"
  "  int_out[4] = CHAR_MIN;\n"
  "  int_out[5] = UCHAR_MAX;\n"
  "  int_out[6] = SHRT_MAX;\n"
  "  int_out[7] = SHRT_MIN;\n"
  "  int_out[8] = USHRT_MAX;\n"
  "  int_out[9] = INT_MAX;\n"
  "  int_out[10] = INT_MIN;\n"
  "  uint_out[0] = UINT_MAX;\n"

  "  int_out[11] = FLT_DIG;\n"
  "  int_out[12] = FLT_MANT_DIG;\n"
  "  int_out[13] = FLT_MAX_10_EXP;\n"
  "  int_out[14] = FLT_MAX_EXP;\n"
  "  int_out[15] = FLT_MIN_10_EXP;\n"
  "  int_out[16] = FLT_MIN_EXP;\n"
  "  int_out[17] = FLT_RADIX;\n"
  "#ifdef __IMAGE_SUPPORT__\n"
  "  int_out[18] = __IMAGE_SUPPORT__;\n"
  "#else\n"
  "  int_out[18] = 0xf00baa;\n"
  "#endif\n"
  "  float_out[0] = FLT_MAX;\n"
  "  float_out[1] = FLT_MIN;\n"
  "  float_out[2] = FLT_EPSILON;\n"
  "  float_out[3] = M_E_F;\n"
  "  float_out[4] = M_LOG2E_F;\n"
  "  float_out[5] = M_LOG10E_F;\n"
  "  float_out[6] = M_LN2_F;\n"
  "  float_out[7] = M_LN10_F;\n"
  "  float_out[8] = M_PI_F;\n"
  "  float_out[9] = M_PI_2_F;\n"
  "  float_out[10] = M_PI_4_F;\n"
  "  float_out[11] = M_1_PI_F;\n"
  "  float_out[12] = M_2_PI_F;\n"
  "  float_out[13] = M_2_SQRTPI_F;\n"
  "  float_out[14] = M_SQRT2_F;\n"
  "  float_out[15] = M_SQRT1_2_F;\n"
  "}\n"
};

const char *kernel_long[] = {
  "__kernel void test(__global long *long_out, __global ulong *ulong_out) \n"
  "{\n"
  "  long_out[0] = LONG_MAX;\n"
  "  long_out[1] = LONG_MIN;\n"
  "  ulong_out[0] = ULONG_MAX;\n"
  "}\n"
};

const char *kernel_double[] = {
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  "__kernel void test( __global double *double_out, __global long *long_out ) \n    "
  "{\n"
  "  long_out[0] = DBL_DIG;\n"
  "  long_out[1] = DBL_MANT_DIG;\n"
  "  long_out[2] = DBL_MAX_10_EXP;\n"
  "  long_out[3] = DBL_MAX_EXP;\n"
  "  long_out[4] = DBL_MIN_10_EXP;\n"
  "  long_out[5] = DBL_MIN_EXP;\n"
  "  long_out[6] = DBL_RADIX;\n"
  "  double_out[0] = DBL_MAX;\n"
  "  double_out[1] = DBL_MIN;\n"
  "  double_out[2] = DBL_EPSILON;\n"
  "  double_out[3] = M_E;\n"
  "  double_out[4] = M_LOG2E;\n"
  "  double_out[5] = M_LOG10E;\n"
  "  double_out[6] = M_LN2;\n"
  "  double_out[7] = M_LN10;\n"
  "  double_out[8] = M_PI;\n"
  "  double_out[9] = M_PI_2;\n"
  "  double_out[10] = M_PI_4;\n"
  "  double_out[11] = M_1_PI;\n"
  "  double_out[12] = M_2_PI;\n"
  "  double_out[13] = M_2_SQRTPI;\n"
  "  double_out[14] = M_SQRT2;\n"
  "  double_out[15] = M_SQRT1_2;\n"
  "}\n"
};


int test_kernel_numeric_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error, errors = 0;
    //    clProgramWrapper program;
    //    clKernelWrapper kernel;
    //    clMemWrapper    streams[3];
    cl_program program;
    cl_kernel kernel;
    cl_mem    streams[3];

    size_t    threads[] = {1,1,1};
    cl_float float_out[16];
    cl_int int_out[19];
    cl_uint uint_out[1];
    cl_long long_out[7];
    cl_ulong ulong_out[1];
    cl_double double_out[16];

    /** INTs and FLOATs **/

    // Create the kernel
    if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_int_float, "test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(float_out), NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(int_out), NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(uint_out), NULL, &error);
    test_error( error, "Creating test array failed" );

    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 2, sizeof( streams[2] ), &streams[2]);
    test_error( error, "Unable to set indexed kernel arguments" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(float_out), (void*)float_out, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );
    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(int_out), (void*)int_out, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );
    error = clEnqueueReadBuffer( queue, streams[2], CL_TRUE, 0, sizeof(uint_out), (void*)uint_out, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );

    TEST_VALUE_EQUAL_LITERAL( "CHAR_BIT", int_out[0],         8)
    TEST_VALUE_EQUAL_LITERAL( "SCHAR_MAX", int_out[1],        127)
    TEST_VALUE_EQUAL_LITERAL( "SCHAR_MIN", int_out[2],        (-127-1))
    TEST_VALUE_EQUAL_LITERAL( "CHAR_MAX", int_out[3],         CL_SCHAR_MAX)
    TEST_VALUE_EQUAL_LITERAL( "CHAR_MIN", int_out[4],         CL_SCHAR_MIN)
    TEST_VALUE_EQUAL_LITERAL( "UCHAR_MAX", int_out[5],        255)
    TEST_VALUE_EQUAL_LITERAL( "SHRT_MAX", int_out[6],         32767)
    TEST_VALUE_EQUAL_LITERAL( "SHRT_MIN",int_out[7],          (-32767-1))
    TEST_VALUE_EQUAL_LITERAL( "USHRT_MAX", int_out[8],        65535)
    TEST_VALUE_EQUAL_LITERAL( "INT_MAX", int_out[9],          2147483647)
    TEST_VALUE_EQUAL_LITERAL( "INT_MIN", int_out[10],         (-2147483647-1))
    TEST_VALUE_EQUAL_LITERAL( "UINT_MAX", uint_out[0],        0xffffffffU)

    TEST_VALUE_EQUAL_LITERAL( "FLT_DIG", int_out[11],         6)
    TEST_VALUE_EQUAL_LITERAL( "FLT_MANT_DIG", int_out[12],    24)
    TEST_VALUE_EQUAL_LITERAL( "FLT_MAX_10_EXP", int_out[13],  +38)
    TEST_VALUE_EQUAL_LITERAL( "FLT_MAX_EXP", int_out[14],     +128)
    TEST_VALUE_EQUAL_LITERAL( "FLT_MIN_10_EXP", int_out[15],  -37)
    TEST_VALUE_EQUAL_LITERAL( "FLT_MIN_EXP", int_out[16],     -125)
    TEST_VALUE_EQUAL_LITERAL( "FLT_RADIX", int_out[17],       2)
    TEST_VALUE_EQUAL( "FLT_MAX", float_out[0],           MAKE_HEX_FLOAT(0x1.fffffep127f, 0x1fffffeL, 103))
    TEST_VALUE_EQUAL( "FLT_MIN", float_out[1],           MAKE_HEX_FLOAT(0x1.0p-126f, 0x1L, -126))
    TEST_VALUE_EQUAL( "FLT_EPSILON", float_out[2],       MAKE_HEX_FLOAT(0x1.0p-23f, 0x1L, -23))
    TEST_VALUE_EQUAL( "M_E_F", float_out[3],             CL_M_E_F )
    TEST_VALUE_EQUAL( "M_LOG2E_F", float_out[4],         CL_M_LOG2E_F )
    TEST_VALUE_EQUAL( "M_LOG10E_F", float_out[5],        CL_M_LOG10E_F )
    TEST_VALUE_EQUAL( "M_LN2_F", float_out[6],           CL_M_LN2_F )
    TEST_VALUE_EQUAL( "M_LN10_F", float_out[7],          CL_M_LN10_F )
    TEST_VALUE_EQUAL( "M_PI_F", float_out[8],            CL_M_PI_F )
    TEST_VALUE_EQUAL( "M_PI_2_F", float_out[9],          CL_M_PI_2_F )
    TEST_VALUE_EQUAL( "M_PI_4_F", float_out[10],         CL_M_PI_4_F )
    TEST_VALUE_EQUAL( "M_1_PI_F", float_out[11],         CL_M_1_PI_F )
    TEST_VALUE_EQUAL( "M_2_PI_F", float_out[12],         CL_M_2_PI_F )
    TEST_VALUE_EQUAL( "M_2_SQRTPI_F", float_out[13],     CL_M_2_SQRTPI_F )
    TEST_VALUE_EQUAL( "M_SQRT2_F", float_out[14],        CL_M_SQRT2_F )
    TEST_VALUE_EQUAL( "M_SQRT1_2_F", float_out[15],      CL_M_SQRT1_2_F )

    // We need to check these values against what we know is supported on the device
    if( checkForImageSupport( deviceID ) == 0 )
    { // has images
        // If images are supported, the constant should have been defined to the value 1
        if( int_out[18] == 0xf00baa )
        {
            log_error( "FAILURE: __IMAGE_SUPPORT__ undefined even though images are supported\n" );
            return -1;
        }
        else if( int_out[18] != 1 )
        {
            log_error( "FAILURE: __IMAGE_SUPPORT__ defined, but to the wrong value (defined as %d, spec states it should be 1)\n", int_out[18] );
            return -1;
        }
    }
    else
    { // no images
        // If images aren't supported, the constant should be undefined
        if( int_out[18] != 0xf00baa )
        {
            log_error( "FAILURE: __IMAGE_SUPPORT__ defined to value %d even though images aren't supported", int_out[18] );
            return -1;
        }
    }
    log_info( "\t__IMAGE_SUPPORT__: %d\n", int_out[18]);

    clReleaseMemObject(streams[0]); streams[0] = NULL;
    clReleaseMemObject(streams[1]); streams[1] = NULL;
    clReleaseMemObject(streams[2]); streams[2] = NULL;
    clReleaseKernel(kernel); kernel = NULL;
    clReleaseProgram(program); program = NULL;

    /** LONGs **/

    if(!gHasLong) {
        log_info("Longs not supported; skipping long tests.\n");
    }
    else
    {
        // Create the kernel
        if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_long, "test" ) != 0 )
        {
            return -1;
        }

        streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(long_out), NULL, &error);
        test_error( error, "Creating test array failed" );
        streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(ulong_out), NULL, &error);
        test_error( error, "Creating test array failed" );

        error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
        test_error( error, "Unable to set indexed kernel arguments" );
        error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
        test_error( error, "Unable to set indexed kernel arguments" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(long_out), &long_out, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );
        error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(ulong_out), &ulong_out, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );

        TEST_VALUE_EQUAL_LITERAL( "LONG_MAX", long_out[0],        ((cl_long) 0x7FFFFFFFFFFFFFFFLL))
        TEST_VALUE_EQUAL_LITERAL( "LONG_MIN", long_out[1],        ((cl_long) -0x7FFFFFFFFFFFFFFFLL - 1LL))
        TEST_VALUE_EQUAL_LITERAL( "ULONG_MAX", ulong_out[0],       ((cl_ulong) 0xFFFFFFFFFFFFFFFFULL))

        clReleaseMemObject(streams[0]); streams[0] = NULL;
        clReleaseMemObject(streams[1]); streams[1] = NULL;
        clReleaseKernel(kernel); kernel = NULL;
        clReleaseProgram(program); program = NULL;
    }

    /** DOUBLEs **/

    if(!is_extension_available(deviceID, "cl_khr_fp64")) {
        log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
    }
    else
    {
        // Create the kernel
        if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_double, "test" ) != 0 )
        {
            return -1;
        }

        streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(double_out), NULL, &error);
        test_error( error, "Creating test array failed" );
        streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(long_out), NULL, &error);
        test_error( error, "Creating test array failed" );

        error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
        test_error( error, "Unable to set indexed kernel arguments" );
        error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
        test_error( error, "Unable to set indexed kernel arguments" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(double_out), &double_out, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );
        error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(long_out), &long_out, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );

        TEST_VALUE_EQUAL_LITERAL( "DBL_DIG", long_out[0],          15)
        TEST_VALUE_EQUAL_LITERAL( "DBL_MANT_DIG", long_out[1],     53)
        TEST_VALUE_EQUAL_LITERAL( "DBL_MAX_10_EXP", long_out[2],   +308)
        TEST_VALUE_EQUAL_LITERAL( "DBL_MAX_EXP", long_out[3],      +1024)
        TEST_VALUE_EQUAL_LITERAL( "DBL_MIN_10_EXP", long_out[4],   -307)
        TEST_VALUE_EQUAL_LITERAL( "DBL_MIN_EXP", long_out[5],      -1021)
        TEST_VALUE_EQUAL_LITERAL( "DBL_RADIX", long_out[6],        2)
        TEST_VALUE_EQUAL( "DBL_MAX", double_out[0],         MAKE_HEX_DOUBLE(0x1.fffffffffffffp1023, 0x1fffffffffffffLL, 971))
        TEST_VALUE_EQUAL( "DBL_MIN", double_out[1],         MAKE_HEX_DOUBLE(0x1.0p-1022, 0x1LL, -1022))
        TEST_VALUE_EQUAL( "DBL_EPSILON", double_out[2],     MAKE_HEX_DOUBLE(0x1.0p-52, 0x1LL, -52))
        //TEST_VALUE_EQUAL( "M_E", double_out[3], CL_M_E )
        TEST_VALUE_EQUAL( "M_LOG2E", double_out[4],         CL_M_LOG2E )
        TEST_VALUE_EQUAL( "M_LOG10E", double_out[5],        CL_M_LOG10E )
        TEST_VALUE_EQUAL( "M_LN2", double_out[6],           CL_M_LN2 )
        TEST_VALUE_EQUAL( "M_LN10", double_out[7],          CL_M_LN10 )
        TEST_VALUE_EQUAL( "M_PI", double_out[8],            CL_M_PI )
        TEST_VALUE_EQUAL( "M_PI_2", double_out[9],          CL_M_PI_2 )
        TEST_VALUE_EQUAL( "M_PI_4", double_out[10],         CL_M_PI_4 )
        TEST_VALUE_EQUAL( "M_1_PI", double_out[11],         CL_M_1_PI )
        TEST_VALUE_EQUAL( "M_2_PI", double_out[12],         CL_M_2_PI )
        TEST_VALUE_EQUAL( "M_2_SQRTPI", double_out[13],     CL_M_2_SQRTPI )
        TEST_VALUE_EQUAL( "M_SQRT2", double_out[14],        CL_M_SQRT2 )
        TEST_VALUE_EQUAL( "M_SQRT1_2", double_out[15],      CL_M_SQRT1_2 )

        clReleaseMemObject(streams[0]); streams[0] = NULL;
        clReleaseMemObject(streams[1]); streams[1] = NULL;
        clReleaseKernel(kernel); kernel = NULL;
        clReleaseProgram(program); program = NULL;
    }

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    return errors;
}


const char *kernel_constant_limits[] = {
    "__kernel void test( __global int *intOut, __global float *floatOut ) \n"
    "{\n"
    "  intOut[0] = isinf( MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[1] = isnormal( MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[2] = isnan( MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[3] = sizeof( MAXFLOAT );\n"
    "  intOut[4] = ( MAXFLOAT == FLT_MAX ) ? 1 : 0;\n"
    //    "  intOut[5] = ( MAXFLOAT == CL_FLT_MAX ) ? 1 : 0;\n"
    "  intOut[6] = ( MAXFLOAT == MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[7] = ( MAXFLOAT == 0x1.fffffep127f ) ? 1 : 0;\n"
    "  floatOut[0] = MAXFLOAT;\n"
    "}\n"
};

const char *kernel_constant_extended_limits[] = {
    "__kernel void test( __global int *intOut, __global float *floatOut ) \n"
    "{\n"
    "  intOut[0] = ( INFINITY == HUGE_VALF ) ? 1 : 0;\n"
    "  intOut[1] = sizeof( INFINITY );\n"
    "  intOut[2] = isinf( INFINITY ) ? 1 : 0;\n"
    "  intOut[3] = isnormal( INFINITY ) ? 1 : 0;\n"
    "  intOut[4] = isnan( INFINITY ) ? 1 : 0;\n"
    "  intOut[5] = ( INFINITY > MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[6] = ( -INFINITY < -MAXFLOAT ) ? 1 : 0;\n"
    "  intOut[7] = ( ( MAXFLOAT + MAXFLOAT ) == INFINITY ) ? 1 : 0;\n"
    "  intOut[8] = ( nextafter( MAXFLOAT, INFINITY ) == INFINITY ) ? 1 : 0;\n"
    "  intOut[9] = ( nextafter( -MAXFLOAT, -INFINITY ) == -INFINITY ) ? 1 : 0;\n"
    "  intOut[10] = ( INFINITY == INFINITY ) ? 1 : 0;\n"
    "  intOut[11] = ( as_uint( INFINITY ) == 0x7f800000 ) ? 1 : 0;\n"
    "  floatOut[0] = INFINITY;\n"
    "\n"
    "  intOut[12] = sizeof( HUGE_VALF );\n"
    "  intOut[13] = ( HUGE_VALF == INFINITY ) ? 1 : 0;\n"
    "  floatOut[1] = HUGE_VALF;\n"
    "\n"
    "  intOut[14] = ( NAN == NAN ) ? 1 : 0;\n"
    "  intOut[15] = ( NAN != NAN ) ? 1 : 0;\n"
    "  intOut[16] = isnan( NAN ) ? 1 : 0;\n"
    "  intOut[17] = isinf( NAN ) ? 1 : 0;\n"
    "  intOut[18] = isnormal( NAN ) ? 1 : 0;\n"
    "  intOut[19] = ( ( as_uint( NAN ) & 0x7fffffff ) > 0x7f800000 ) ? 1 : 0;\n"
    "  intOut[20] = sizeof( NAN );\n"
    "  floatOut[2] = NAN;\n"
    "\n"
    "  intOut[21] = isnan( INFINITY / INFINITY ) ? 1 : 0;\n"
    "  intOut[22] = isnan( INFINITY - INFINITY ) ? 1 : 0;\n"
    "  intOut[23] = isnan( 0.f / 0.f ) ? 1 : 0;\n"
    "  intOut[24] = isnan( INFINITY * 0.f ) ? 1 : 0;\n"
    "  intOut[25] = ( INFINITY == NAN ); \n"
    "  intOut[26] = ( -INFINITY == NAN ); \n"
    "  intOut[27] = ( INFINITY > NAN ); \n"
    "  intOut[28] = ( -INFINITY < NAN ); \n"
    "  intOut[29] = ( INFINITY != NAN ); \n"
    "  intOut[30] = ( NAN > INFINITY ); \n"
    "  intOut[31] = ( NAN < -INFINITY ); \n"

    "}\n"
};

const char *kernel_constant_double_limits[] = {
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void test( __global int *intOut, __global double *doubleOut ) \n"
    "{\n"
    "  intOut[0] = sizeof( HUGE_VAL );\n"
    "  intOut[1] = ( HUGE_VAL == INFINITY ) ? 1 : 0;\n"
    "  intOut[2] = isinf( HUGE_VAL ) ? 1 : 0;\n"
    "  intOut[3] = isnormal( HUGE_VAL ) ? 1 : 0;\n"
    "  intOut[4] = isnan( HUGE_VAL ) ? 1 : 0;\n"
    "  intOut[5] = ( HUGE_VAL == HUGE_VALF ) ? 1 : 0;\n"
    "  intOut[6] = ( as_ulong( HUGE_VAL ) == 0x7ff0000000000000UL ) ? 1 : 0;\n"
    "  doubleOut[0] = HUGE_VAL;\n"
    "}\n"
};

#define TEST_FLOAT_ASSERTION( a, msg, f ) if( !( a ) ) { log_error( "ERROR: Float constant failed requirement: %s (bitwise value is 0x%8.8x)\n", msg, *( (uint32_t *)&f ) ); return -1; }
#define TEST_DOUBLE_ASSERTION( a, msg, f ) if( !( a ) ) { log_error( "ERROR: Double constant failed requirement: %s (bitwise value is 0x%16.16llx)\n", msg, *( (uint64_t *)&f ) ); return -1; }

int test_kernel_limit_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t              threads[] = {1,1,1};
    clMemWrapper        intStream, floatStream, doubleStream;
    cl_int              intOut[ 32 ];
    cl_float            floatOut[ 3 ];
    cl_double           doubleOut[ 1 ];


    /* Create some I/O streams */
    intStream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(intOut), NULL, &error );
    test_error( error, "Creating test array failed" );
    floatStream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(floatOut), NULL, &error );
    test_error( error, "Creating test array failed" );

    // Stage 1: basic limits on MAXFLOAT
    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_constant_limits, "test" ) != 0 )
        {
            return -1;
        }

        error = clSetKernelArg( kernel, 0, sizeof( intStream ), &intStream );
        test_error( error, "Unable to set indexed kernel arguments" );
        error = clSetKernelArg( kernel, 1, sizeof( floatStream ), &floatStream );
        test_error( error, "Unable to set indexed kernel arguments" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        error = clEnqueueReadBuffer( queue, intStream, CL_TRUE, 0, sizeof(intOut), intOut, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );
        error = clEnqueueReadBuffer( queue, floatStream, CL_TRUE, 0, sizeof(floatOut), floatOut, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );

        // Test MAXFLOAT properties
        TEST_FLOAT_ASSERTION( intOut[0] == 0, "isinf( MAXFLOAT ) = false", floatOut[0] )
        TEST_FLOAT_ASSERTION( intOut[1] == 1, "isnormal( MAXFLOAT ) = true", floatOut[0] )
        TEST_FLOAT_ASSERTION( intOut[2] == 0, "isnan( MAXFLOAT ) = false", floatOut[0] )
        TEST_FLOAT_ASSERTION( intOut[3] == 4, "sizeof( MAXFLOAT ) = 4", floatOut[0] )
        TEST_FLOAT_ASSERTION( intOut[4] == 1, "MAXFLOAT = FLT_MAX", floatOut[0] )
        TEST_FLOAT_ASSERTION( floatOut[0] == CL_FLT_MAX, "MAXFLOAT = CL_FLT_MAX", floatOut[0] )
        TEST_FLOAT_ASSERTION( intOut[6] == 1, "MAXFLOAT = MAXFLOAT", floatOut[0] )
        TEST_FLOAT_ASSERTION( floatOut[0] == MAKE_HEX_FLOAT( 0x1.fffffep127f, 0x1fffffeL, 103), "MAXFLOAT = 0x1.fffffep127f", floatOut[0] )
    }

    // Stage 2: INFINITY and NAN
    char profileStr[128] = "";
    error = clGetDeviceInfo( deviceID, CL_DEVICE_PROFILE, sizeof( profileStr ), &profileStr, NULL );
    test_error( error, "Unable to run INFINITY/NAN tests (unable to get CL_DEVICE_PROFILE" );

    bool testInfNan = true;
    if( strcmp( profileStr, "EMBEDDED_PROFILE" ) == 0 )
    {
        // We test if we're not an embedded profile, OR if the inf/nan flag in the config is set
        cl_device_fp_config single = 0;
        error = clGetDeviceInfo( deviceID, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( single ), &single, NULL );
        test_error( error, "Unable to run INFINITY/NAN tests (unable to get FP_CONFIG bits)" );

        if( ( single & CL_FP_INF_NAN ) == 0 )
        {
            log_info( "Skipping INFINITY and NAN tests on embedded device (INF/NAN not supported on this device)" );
            testInfNan = false;
        }
    }

    if( testInfNan )
    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_constant_extended_limits, "test" ) != 0 )
        {
            return -1;
        }

        error = clSetKernelArg( kernel, 0, sizeof( intStream ), &intStream );
        test_error( error, "Unable to set indexed kernel arguments" );
        error = clSetKernelArg( kernel, 1, sizeof( floatStream ), &floatStream );
        test_error( error, "Unable to set indexed kernel arguments" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Kernel execution failed" );

        error = clEnqueueReadBuffer( queue, intStream, CL_TRUE, 0, sizeof(intOut), intOut, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );
        error = clEnqueueReadBuffer( queue, floatStream, CL_TRUE, 0, sizeof(floatOut), floatOut, 0, NULL, NULL );
        test_error( error, "Unable to get result data" );

        TEST_FLOAT_ASSERTION( intOut[0] == 1, "INFINITY == HUGE_VALF", intOut[0] )
        TEST_FLOAT_ASSERTION( intOut[1] == 4, "sizeof( INFINITY ) == 4", intOut[1] )
        TEST_FLOAT_ASSERTION( intOut[2] == 1, "isinf( INFINITY ) == true", intOut[2] )
        TEST_FLOAT_ASSERTION( intOut[3] == 0, "isnormal( INFINITY ) == false", intOut[3] )
        TEST_FLOAT_ASSERTION( intOut[4] == 0, "isnan( INFINITY ) == false", intOut[4] )
        TEST_FLOAT_ASSERTION( intOut[5] == 1, "INFINITY > MAXFLOAT", intOut[5] )
        TEST_FLOAT_ASSERTION( intOut[6] == 1, "-INFINITY < -MAXFLOAT", intOut[6] )
        TEST_FLOAT_ASSERTION( intOut[7] == 1, "( MAXFLOAT + MAXFLOAT ) == INFINITY", intOut[7] )
        TEST_FLOAT_ASSERTION( intOut[8] == 1, "nextafter( MAXFLOAT, INFINITY ) == INFINITY", intOut[8] )
        TEST_FLOAT_ASSERTION( intOut[9] == 1, "nextafter( -MAXFLOAT, -INFINITY ) == -INFINITY", intOut[9] )
        TEST_FLOAT_ASSERTION( intOut[10] == 1, "INFINITY = INFINITY", intOut[10] )
        TEST_FLOAT_ASSERTION( intOut[11] == 1, "asuint( INFINITY ) == 0x7f800000", intOut[11] )
        TEST_FLOAT_ASSERTION( *( (uint32_t *)&floatOut[0] ) == 0x7f800000, "asuint( INFINITY ) == 0x7f800000", floatOut[0] )
        TEST_FLOAT_ASSERTION( floatOut[1] == INFINITY, "INFINITY == INFINITY", floatOut[1] )

        TEST_FLOAT_ASSERTION( intOut[12] == 4, "sizeof( HUGE_VALF ) == 4", intOut[12] )
        TEST_FLOAT_ASSERTION( intOut[13] == 1, "HUGE_VALF == INFINITY", intOut[13] )
        TEST_FLOAT_ASSERTION( floatOut[1] == HUGE_VALF, "HUGE_VALF == HUGE_VALF", floatOut[1] )

        TEST_FLOAT_ASSERTION( intOut[14] == 0, "(NAN == NAN) = false", intOut[14] )
        TEST_FLOAT_ASSERTION( intOut[15] == 1, "(NAN != NAN) = true", intOut[15] )
        TEST_FLOAT_ASSERTION( intOut[16] == 1, "isnan( NAN ) = true", intOut[16] )
        TEST_FLOAT_ASSERTION( intOut[17] == 0, "isinf( NAN ) = false", intOut[17] )
        TEST_FLOAT_ASSERTION( intOut[18] == 0, "isnormal( NAN ) = false", intOut[18] )
        TEST_FLOAT_ASSERTION( intOut[19] == 1, "( as_uint( NAN ) & 0x7fffffff ) > 0x7f800000", intOut[19] )
        TEST_FLOAT_ASSERTION( intOut[20] == 4, "sizeof( NAN ) = 4", intOut[20] )
        TEST_FLOAT_ASSERTION( ( *( (uint32_t *)&floatOut[2] ) & 0x7fffffff ) > 0x7f800000, "( as_uint( NAN ) & 0x7fffffff ) > 0x7f800000", floatOut[2] )

        TEST_FLOAT_ASSERTION( intOut[ 21 ] == 1, "isnan( INFINITY / INFINITY ) = true", intOut[ 21 ] )
        TEST_FLOAT_ASSERTION( intOut[ 22 ] == 1, "isnan( INFINITY - INFINITY ) = true", intOut[ 22 ] )
        TEST_FLOAT_ASSERTION( intOut[ 23 ] == 1, "isnan( 0.f / 0.f ) = true", intOut[ 23 ] )
        TEST_FLOAT_ASSERTION( intOut[ 24 ] == 1, "isnan( INFINITY * 0.f ) = true", intOut[ 24 ] )
        TEST_FLOAT_ASSERTION( intOut[ 25 ] == 0, "( INFINITY == NAN ) = false", intOut[ 25 ] )
        TEST_FLOAT_ASSERTION( intOut[ 26 ] == 0, "(-INFINITY == NAN ) = false", intOut[ 26 ] )
        TEST_FLOAT_ASSERTION( intOut[ 27 ] == 0, "( INFINITY > NAN ) = false", intOut[ 27 ] )
        TEST_FLOAT_ASSERTION( intOut[ 28 ] == 0, "(-INFINITY < NAN ) = false", intOut[ 28 ] )
        TEST_FLOAT_ASSERTION( intOut[ 29 ] == 1, "( INFINITY != NAN ) = true", intOut[ 29 ] )
        TEST_FLOAT_ASSERTION( intOut[ 30 ] == 0, "( NAN < INFINITY ) = false", intOut[ 30 ] )
        TEST_FLOAT_ASSERTION( intOut[ 31 ] == 0, "( NAN > -INFINITY ) = false", intOut[ 31 ] )
    }

    // Stage 3: limits on HUGE_VAL (double)
    if( !is_extension_available( deviceID, "cl_khr_fp64" ) )
        log_info( "Note: Skipping double HUGE_VAL tests (doubles unsupported on device)\n" );
    else
    {
        cl_device_fp_config config = 0;
        error = clGetDeviceInfo( deviceID, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof( config ), &config, NULL );
        test_error( error, "Unable to run INFINITY/NAN tests (unable to get double FP_CONFIG bits)" );

        if( ( config & CL_FP_INF_NAN ) == 0 )
            log_info( "Skipping HUGE_VAL tests (INF/NAN not supported on this device)" );
        else
        {
            clProgramWrapper program;
            clKernelWrapper kernel;

            if( create_single_kernel_helper( context, &program, &kernel, 1, kernel_constant_double_limits, "test" ) != 0 )
            {
                return -1;
            }

            doubleStream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(doubleOut), NULL, &error );
            test_error( error, "Creating test array failed" );

            error = clSetKernelArg( kernel, 0, sizeof( intStream ), &intStream );
            test_error( error, "Unable to set indexed kernel arguments" );
            error = clSetKernelArg( kernel, 1, sizeof( doubleStream ), &doubleStream );
            test_error( error, "Unable to set indexed kernel arguments" );

            error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
            test_error( error, "Kernel execution failed" );

            error = clEnqueueReadBuffer( queue, intStream, CL_TRUE, 0, sizeof(intOut), intOut, 0, NULL, NULL );
            test_error( error, "Unable to get result data" );
            error = clEnqueueReadBuffer( queue, doubleStream, CL_TRUE, 0, sizeof(doubleOut), doubleOut, 0, NULL, NULL );
            test_error( error, "Unable to get result data" );

            TEST_DOUBLE_ASSERTION( intOut[0] == 8, "sizeof( HUGE_VAL ) = 8", intOut[0] )
            TEST_DOUBLE_ASSERTION( intOut[1] == 1, "HUGE_VAL = INFINITY", intOut[1] )
            TEST_DOUBLE_ASSERTION( intOut[2] == 1, "isinf( HUGE_VAL ) = true", intOut[2] )
            TEST_DOUBLE_ASSERTION( intOut[3] == 0, "isnormal( HUGE_VAL ) = false", intOut[3] )
            TEST_DOUBLE_ASSERTION( intOut[4] == 0, "isnan( HUGE_VAL ) = false", intOut[4] )
            TEST_DOUBLE_ASSERTION( intOut[5] == 1, "HUGE_VAL = HUGE_VAL", intOut[5] )
            TEST_DOUBLE_ASSERTION( intOut[6] == 1, "as_ulong( HUGE_VAL ) = 0x7ff0000000000000UL", intOut[6] )
            TEST_DOUBLE_ASSERTION( *( (uint64_t *)&doubleOut[0] ) == 0x7ff0000000000000ULL, "as_ulong( HUGE_VAL ) = 0x7ff0000000000000UL", doubleOut[0] )
        }
    }

    return 0;
}



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
#include "harness/rounding_mode.h"
#include "harness/kernelHelpers.h"

#include "test_printf.h"
#include <assert.h>
#include <CL/cl_half.h>


// Helpers for generating runtime reference results
static void intRefBuilder(printDataGenParameters&, char*, const size_t);
static void halfRefBuilder(printDataGenParameters&, char* rResult,
                           const size_t);
static void floatRefBuilder(printDataGenParameters&, char* rResult, const size_t);
static void doubleRefBuilder(printDataGenParameters&, char* rResult,
                             const size_t);
static void octalRefBuilder(printDataGenParameters&, char*, const size_t);
static void unsignedRefBuilder(printDataGenParameters&, char*, const size_t);
static void hexRefBuilder(printDataGenParameters&, char*, const size_t);

//==================================

// int

//==================================

//------------------------------------------------------

// [string] format  | [string] int-data representation |

//------------------------------------------------------

std::vector<printDataGenParameters> printIntGenParameters = {

    //(Minimum)Five-wide,default(right)-justified

    { { "%5d" }, "10" },

    //(Minimum)Five-wide,left-justified

    { { "%-5d" }, "10" },

    //(Minimum)Five-wide,default(right)-justified,zero-filled

    { { "%05d" }, "10" },

    //(Minimum)Five-wide,default(right)-justified,with sign

    { { "%+5d" }, "10" },

    //(Minimum)Five-wide ,left-justified,with sign

    { { "%-+5d" }, "10" },

    //(Minimum)Five-digit(zero-filled in absent digits),default(right)-justified

    { { "%.5i" }, "100" },

    //(Minimum)Six-wide,Five-digit(zero-filled in absent
    // digits),default(right)-justified

    { { "%6.5i" }, "100" },

    // 0 and - flag both apper ==>0 is ignored,left-justified,capital I

    { { "%-06i" }, "100" },

    //(Minimum)Six-wide,Five-digit(zero-filled in absent
    // digits),default(right)-justified

    { { "%06.5i" }, "100" },

    //(Minimum)Ten-wide, left-justified, with a blank space inserted before the
    // value

    { { "% 10d" }, "42" },
};

//-----------------------------------------------

//test case for int                             |

//-----------------------------------------------

testCase testCaseInt = {

    TYPE_INT,

    correctBufferInt,

    printIntGenParameters,

    intRefBuilder,

    kint

};


//==================================

// long

//==================================

//------------------------------------------------------

// [string] format  | [string] int-data representation |

//------------------------------------------------------

std::vector<printDataGenParameters> printLongGenParameters = {

    //(Minimum) fifteen-wide,default(right)-justified

    { { "%5ld" }, "10000000000L" },

    //(Minimum) fifteen-wide,left-justified

    { { "%-15ld" }, "-10000000000L" },

    //(Minimum) fifteen-wide,default(right)-justified,zero-filled

    { { "%015ld" }, "10000000000L" },

    //(Minimum) fifteen-wide,default(right)-justified,with sign

    { { "%+15ld" }, "-10000000000L" },

    //(Minimum) fifteen-wide ,left-justified,with sign

    { { "%-+15ld" }, "10000000000L" },

    //(Minimum) fifteen-digit(zero-filled in absent
    // digits),default(right)-justified

    { { "%.15li" }, "10000000000L" },

    //(Minimum)Sixteen-wide, fifteen-digit(zero-filled in absent
    // digits),default(right)-justified

    { { "%-+16.15li" }, "-10000000000L" },

};

//--------------------------------------------------------

//  Lookup table - [string]long-correct buffer             |

//--------------------------------------------------------

// The table below is used to accommodate differences in `printf` output when
// using the `%ld` format specifier in 32-bit versus 64-bit compiled binaries

std::vector<std::string> correctBufferLong = {

    "10000000000",

    "-10000000000   ",

    "000010000000000",

    "   -10000000000",

    "+10000000000   ",

    "000010000000000",

    "-000010000000000"

};

//-----------------------------------------------

// test case for long                             |

//-----------------------------------------------

testCase testCaseLong = {

    TYPE_LONG,

    correctBufferLong,

    printLongGenParameters,

    NULL,

    klong

};


//==============================================

// half

//==============================================

//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------

std::vector<printDataGenParameters> printHalfGenParameters = {

    // Default(right)-justified

    { { "%f" }, "1.234h" },

    // One position after the decimal,default(right)-justified

    { { "%4.2f" }, "1.2345h" },

    // Zero positions after the
    // decimal([floor]rounding),default(right)-justified

    { { "%.0f" }, "0.1h" },

    // Zero positions after the decimal([ceil]rounding),default(right)-justified

    { { "%.0f" }, "0.6h" },

    // Zero-filled,default positions number after the
    // decimal,default(right)-justified

    { { "%0f" }, "0.6h" },

    // Double argument representing floating-point,used by f
    // style,default(right)-justified

    { { "%4g" }, "5.678h" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4.2g" }, "5.678h" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4G" }, "0.000062h" },

    // Double argument representing floating-point,with
    // exponent,left-justified,default(right)-justified

    { { "%-#20.15e" }, "65504.0h" },

    // Double argument representing floating-point,with
    // exponent,left-justified,with sign,capital E,default(right)-justified

    { { "%+#21.15E" }, "-65504.0h" },

    //(Minimum)Ten-wide,two positions after the decimal,with
    // a blank space inserted before the value, default(right)-justified

    { { "% 10.2f" }, "1.25h" },

    //(Minimum)Eight-wide,two positions after the decimal, with
    // zeros inserted before the value, default(right)-justified

    { { "%08.2f" }, "3.14h" },
};

//---------------------------------------------------------

// Test case for float                                     |

//---------------------------------------------------------

testCase testCaseHalf = {

    TYPE_HALF,

    correctBufferHalf,

    printHalfGenParameters,

    halfRefBuilder,

    kfloat

};


//==============================================

// half limits

//==============================================


//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------


std::vector<printDataGenParameters> printHalfLimitsGenParameters = {

    // Infinity (1.0/0.0)
    { { "%f", "%e", "%g", "%a" }, "1.0h/0.0h" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "nan((ushort)0)" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "acospi(2.0h)" },

    // Infinity (1.0/0.0)
    { { "%F", "%E", "%G", "%A" }, "1.0h/0.0h" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "nan((ushort)0)" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "acospi(2.0h)" }
};
//--------------------------------------------------------

//  Lookup table - [string]float-correct buffer             |

//--------------------------------------------------------

std::vector<std::string> correctBufferHalfLimits = {

    "inf",

    "nan",

    "nan",

    "INF",

    "NAN",

    "NAN"

};

//---------------------------------------------------------

// Test case for float                                     |

//---------------------------------------------------------

testCase testCaseHalfLimits = {

    TYPE_HALF_LIMITS,

    correctBufferHalfLimits,

    printHalfLimitsGenParameters,

    NULL

};


//==============================================

// float

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------

std::vector<printDataGenParameters> printFloatGenParameters = {

    // Default(right)-justified

    { { "%f" }, "10.3456f" },

    // One position after the decimal,default(right)-justified

    { { "%.1f" }, "10.3456f" },

    // Two positions after the decimal,default(right)-justified

    { { "%.2f" }, "10.3456f" },

    //(Minimum)Eight-wide,three positions after the
    // decimal,default(right)-justified

    { { "%8.3f" }, "10.3456f" },

    //(Minimum)Eight-wide,two positions after the
    // decimal,zero-filled,default(right)-justified

    { { "%08.2f" }, "10.3456f" },

    //(Minimum)Eight-wide,two positions after the decimal,left-justified

    { { "%-8.2f" }, "10.3456f" },

    //(Minimum)Eight-wide,two positions after the decimal,with
    // sign,default(right)-justified

    { { "%+8.2f" }, "-10.3456f" },

    // Zero positions after the
    // decimal([floor]rounding),default(right)-justified

    { { "%.0f" }, "0.1f" },

    // Zero positions after the decimal([ceil]rounding),default(right)-justified

    { { "%.0f" }, "0.6f" },

    // Zero-filled,default positions number after the
    // decimal,default(right)-justified

    { { "%0f" }, "0.6f" },

    // Double argument representing floating-point,used by f
    // style,default(right)-justified

    { { "%4g" }, "12345.6789f" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4.2g" }, "12345.6789f" },

    // Double argument representing floating-point,used by f
    // style,default(right)-justified

    { { "%4G" }, "0.0000023f" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4G" }, "0.023f" },

    // Double argument representing floating-point,with
    // exponent,left-justified,default(right)-justified
    // Use a value that is exactly representable as 32-bit float.

    { { "%-#20.15e" }, "789456128.f" },

    // Double argument representing floating-point,with
    // exponent,left-justified,with sign,capital E,default(right)-justified
    // Use a value that is exactly representable as 32-bit float.

    { { "%+#21.15E" }, "789456128.f" },

    // Double argument representing floating-point,in [-]xh.hhhhpAd style

    { { "%.6a" }, "0.1f" },

    //(Minimum)Ten-wide,Double argument representing floating-point,in
    // xh.hhhhpAd style,default(right)-justified

    { { "%10.2a" }, "9990.235f" },

    //(Minimum)Ten-wide,two positions after the decimal,with
    // a blank space inserted before the value, default(right)-justified

    { { "% 10.2f" }, "1.25f" },

    //(Minimum)Eight-wide,two positions after the decimal,with
    // zeros inserted before the value, default(right)-justified

    { { "%08.2f" }, "3.14f" },
};

//---------------------------------------------------------

//Test case for float                                     |

//---------------------------------------------------------

testCase testCaseFloat = {

    TYPE_FLOAT,

    correctBufferFloat,

    printFloatGenParameters,

    floatRefBuilder,

    kfloat

};

//==============================================

// float limits

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------


std::vector<printDataGenParameters> printFloatLimitsGenParameters = {

    // Infinity (1.0/0.0)
    { { "%f", "%e", "%g", "%a" }, "1.0f/0.0f" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "nan(0U)" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "acospi(2.0f)" },

    // Infinity (1.0/0.0)
    { { "%F", "%E", "%G", "%A" }, "1.0f/0.0f" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "nan(0U)" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "acospi(2.0f)" }
};
//--------------------------------------------------------

//  Lookup table - [string]float-correct buffer             |

//--------------------------------------------------------

std::vector<std::string> correctBufferFloatLimits = {

    "inf",

    "nan",

    "nan",

    "INF",

    "NAN",

    "NAN"

};

//---------------------------------------------------------

//Test case for float                                     |

//---------------------------------------------------------

testCase testCaseFloatLimits = {

    TYPE_FLOAT_LIMITS,

    correctBufferFloatLimits,

    printFloatLimitsGenParameters,

    NULL

};

//==============================================

// double

//==============================================


//--------------------------------------------------------

// [string] format |  [string] double-data representation |

//--------------------------------------------------------

std::vector<printDataGenParameters> printDoubleGenParameters = {

    // Default(right)-justified

    { { "%f" }, "10.3456" },

    // One position after the decimal,default(right)-justified

    { { "%.1f" }, "10.3456" },

    // Two positions after the decimal,default(right)-justified

    { { "%.2f" }, "10.3456" },

    //(Minimum)Eight-wide,three positions after the
    // decimal,default(right)-justified

    { { "%8.3f" }, "10.3456" },

    //(Minimum)Eight-wide,two positions after the
    // decimal,zero-filled,default(right)-justified

    { { "%08.2f" }, "10.3456" },

    //(Minimum)Eight-wide,two positions after the decimal,left-justified

    { { "%-8.2f" }, "10.3456" },

    //(Minimum)Eight-wide,two positions after the decimal,with
    // sign,default(right)-justified

    { { "%+8.2f" }, "-10.3456" },

    // Zero positions after the
    // decimal([floor]rounding),default(right)-justified

    { { "%.0f" }, "0.1" },

    // Zero positions after the decimal([ceil]rounding),default(right)-justified

    { { "%.0f" }, "0.6" },

    // Zero-filled,default positions number after the
    // decimal,default(right)-justified

    { { "%0f" }, "0.6" },

    // Double argument representing floating-point,used by f
    // style,default(right)-justified

    { { "%4g" }, "12345.6789" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4.2g" }, "12345.6789" },

    // Double argument representing floating-point,used by f
    // style,default(right)-justified

    { { "%4G" }, "0.0000023" },

    // Double argument representing floating-point,used by e
    // style,default(right)-justified

    { { "%4G" }, "0.023" },

    // Double argument representing floating-point,with
    // exponent,left-justified,default(right)-justified

    { { "%-#20.15e" }, "789456123.0" },

    // Double argument representing floating-point,with
    // exponent,left-justified,with sign,capital E,default(right)-justified

    { { "%+#21.15E" }, "789456123.0" },

    // Double argument representing floating-point,in [-]xh.hhhhpAd style

    { { "%.6a" }, "0.1" },

    //(Minimum)Ten-wide,Double argument representing floating-point,in
    // xh.hhhhpAd style,default(right)-justified

    { { "%10.2a" }, "9990.235" },
};

//---------------------------------------------------------

// Test case for double                                    |

//---------------------------------------------------------

testCase testCaseDouble = {

    TYPE_DOUBLE,

    correctBufferDouble,

    printDoubleGenParameters,

    doubleRefBuilder,

    kdouble

};

//==============================================

// double limits

//==============================================


//--------------------------------------------------------

// [string] double |  [string] double-data representation |

//--------------------------------------------------------


std::vector<printDataGenParameters> printDoubleLimitsGenParameters = {

    // Infinity (1.0/0.0)
    { { "%f", "%e", "%g", "%a" }, "1.0/0.0" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "nan(0UL)" },

    // NaN
    { { "%f", "%e", "%g", "%a" }, "acospi(2.0)" },

    // Infinity (1.0/0.0)
    { { "%F", "%E", "%G", "%A" }, "1.0/0.0" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "nan(0UL)" },

    // NaN
    { { "%F", "%E", "%G", "%A" }, "acospi(2.0)" }
};
//--------------------------------------------------------

//  Lookup table - [string]double-correct buffer             |

//--------------------------------------------------------

std::vector<std::string> correctBufferDoubleLimits = {

    "inf",

    "nan",

    "nan",

    "INF",

    "NAN",

    "NAN"

};

//---------------------------------------------------------

// Test case for float                                     |

//---------------------------------------------------------

testCase testCaseDoubleLimits = {

    TYPE_DOUBLE_LIMITS,

    correctBufferDoubleLimits,

    printDoubleLimitsGenParameters,

    NULL

};

//=========================================================

// octal

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] octal-data representation  |

//---------------------------------------------------------

std::vector<printDataGenParameters> printOctalGenParameters = {

    // Default(right)-justified

    { { "%o" }, "10" },

    // Five-digit,default(right)-justified

    { { "%.5o" }, "10" },

    // Default(right)-justified,increase precision

    { { "%#o" }, "100000000" },

    //(Minimum)Four-wide,Five-digit,0-flag ignored(because of
    // precision),default(right)-justified

    { { "%04.5o" }, "10" },

    //(Minimum)Ten-wide, zeros inserted before the value,
    // default(right)-justified

    { { "%010o" }, "10" }
};

//-------------------------------------------------------

//Test case for octal                                   |

//-------------------------------------------------------

testCase testCaseOctal = {

    TYPE_OCTAL,

    correctBufferOctal,

    printOctalGenParameters,

    octalRefBuilder,

    kulong

};



//=========================================================

// unsigned

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] unsined-data representation  |

//---------------------------------------------------------

std::vector<printDataGenParameters> printUnsignedGenParameters = {

    // Default(right)-justified

    { { "%u" }, "10" },

    // Zero precision for zero,default(right)-justified

    { { "%.0u" }, "0" },

};

//-------------------------------------------------------

// Test case for unsigned                                 |

//-------------------------------------------------------

testCase testCaseUnsigned = {

    TYPE_UNSIGNED,

    correctBufferUnsigned,

    printUnsignedGenParameters,

    unsignedRefBuilder,

    kulong

};



//=======================================================

// hexadecimal

//=======================================================



//--------------------------------------------------------------

// [string] format  | [string] hexadecimal-data representation |

//--------------------------------------------------------------

std::vector<printDataGenParameters> printHexadecimalGenParameters = {

    // Add 0x,low x,default(right)-justified

    { { "%#x" }, "0xABCDEF" },

    // Add 0x,capital X,default(right)-justified

    { { "%#X" }, "0xABCDEF" },

    // Not add 0x,if zero,default(right)-justified

    { { "%#X" }, "0" },

    //(Minimum)Eight-wide,default(right)-justified

    { { "%8x" }, "399" },

    //(Minimum)Four-wide,zero-filled,default(right)-justified

    { { "%04x" }, "399" }

};

//--------------------------------------------------------------

//Test case for hexadecimal                                    |

//--------------------------------------------------------------

testCase testCaseHexadecimal = {

    TYPE_HEXADEC,

    correctBufferHexadecimal,

    printHexadecimalGenParameters,

    hexRefBuilder,

    kulong

};



//=============================================================

// char

//=============================================================



//-----------------------------------------------------------

// [string] format  | [string] string-data representation   |

//-----------------------------------------------------------

std::vector<printDataGenParameters> printCharGenParameters = {

    // Four-wide,zero-filled,default(right)-justified

    { { "%4c" }, "\'1\'" },

    // Four-wide,left-justified

    { { "%-4c" }, "\'1\'" },

    //(unsigned) int argument,default(right)-justified

    { { "%c" }, "66" }

};

//---------------------------------------------------------

// Lookup table -[string] char-correct buffer             |

//---------------------------------------------------------

std::vector<std::string> correctBufferChar = {

    "   1",

    "1   ",

    "B",

};




//----------------------------------------------------------

//Test case for char                                       |

//----------------------------------------------------------

testCase testCaseChar = {

    TYPE_CHAR,

    correctBufferChar,

    printCharGenParameters,

    NULL,

    kchar

};



//==========================================================

// string

//==========================================================



//--------------------------------------------------------

// [string]format | [string] string-data representation  |

//--------------------------------------------------------
// clang-format off

std::vector<printDataGenParameters> printStringGenParameters = {

    // empty format
    { {""}, "\"foo\"" },

    // empty argument
    { {"%s"}, "\"\"" },

    //(Minimum)Four-wide,zero-filled,default(right)-justified

    { { "%4s" }, "\"foo\"" },

    // One-digit(precision ignored),left-justified

    { { "%.1s" }, "\"foo\"" },

    //%% specification

    { {"%s"}, "\"%%\"" },

    { {"%s"}, "\"foo%%bar%%bar%%foo\"" },

    { {"%%%s%%"}, "\"foo\"" },

    { {"%%s%s"}, "\"foo\"" },

    // special symbols
    // nested

    { {"%s"}, "\"\\\"%%\\\"\"" },

    { {"%s"}, "\"\\\'%%\\\'\"" },

    // tabs

    { {"%s"}, "\"foo\\tfoo\"" },

    // newlines

    { {"%s"}, "\"foo\\nfoo\"" },

    // terminator
    { {"%s"}, "\"foo\\0foo\"" },

    // all ascii characters
    { {"%s"},
      "\" "
      "!\\\"#$%&\'()*+,-./"
      "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`"
      "abcdefghijklmnopqrstuvwxyz{|}~\"" }
};

//---------------------------------------------------------

// Lookup table -[string] string-correct buffer           |

//---------------------------------------------------------

std::vector<std::string> correctBufferString = {

    "",

    "",

    " foo",

    "f",

    "%%",

    "foo%%bar%%bar%%foo",

    "%foo%",

    "%sfoo",

    "\"%%\"",

    "\'%%\'",

    "foo\tfoo",

R"(foo
foo)",

    "foo",

    " !\"#$%&\'()*+,-./"
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
};

//---------------------------------------------------------

//Test case for string                                    |

//---------------------------------------------------------

testCase testCaseString = {

    TYPE_STRING,

    correctBufferString,

    printStringGenParameters,

    NULL,

    kchar

};

//--------------------------------------------------------

// [string]format |

//--------------------------------------------------------

std::vector<printDataGenParameters> printFormatStringGenParameters = {

    //%% specification

    { {"%%"} },

    // special symbols
    // nested

    { {"\\\"%%\\\""} },

    { {"\'%%\'"} },

    { {"\'foo%%bar%%bar%%foo\'"} },

    // tabs

    { {"foo\\t\\t\\tfoo"} },

    // newlines

    { {"foo\\nfoo"} },

    // all ascii characters
    { {
          " !\\\"#$%%&\'()*+,-./"
          "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]^_`"
          "abcdefghijklmnopqrstuvwxyz{|}~"
      } }
};

//---------------------------------------------------------

// Lookup table -[string] string-correct buffer           |

//---------------------------------------------------------

std::vector<std::string> correctBufferFormatString = {

    "%",

    "\"%\"",

    "\'%\'",

    "\'foo%bar%bar%foo\'",

    "foo\t\t\tfoo",

R"(foo
foo)",

    " !\"#$%&\'()*+,-./"
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
};

//---------------------------------------------------------

//Test case for string                                    |

//---------------------------------------------------------

testCase testCaseFormatString = {

    TYPE_FORMAT_STRING,

    correctBufferFormatString,

    printFormatStringGenParameters,

    NULL,

    kchar

};

// clang-format on

//=========================================================

// vector

//=========================================================



//-------------------------------------------------------------------------------------------------------------------

//[string] flag | [string] specifier | [string] type | [string] vector-data representation | [string] vector size   |

//-------------------------------------------------------------------------------------------------------------------

std::vector<printDataGenParameters> printVectorGenParameters = {

    //(Minimum)Two-wide,two positions after decimal

    { { "" }, "(1.0f,2.0f,3.0f,4.0f)", "%2.2", "hlf", "float", "4" },

    // Alternative form,uchar argument

    { { "" }, "(0xFA,0xFB)", "%#", "hhx", "uchar", "2" },

    // Alternative form,ushort argument

    { { "" }, "(0x1234,0x8765)", "%#", "hx", "ushort", "2" },

    // Alternative form,uint argument

    { { "" }, "(0x12345678,0x87654321)", "%#", "hlx", "uint", "2" },

    // Alternative form,long argument

    { { "" }, "(12345678,98765432)", "%", "ld", "long", "2" },

    //(Minimum)Two-wide,two positions after decimal

    { { "" }, "(1.0h,2.0h,3.0h,4.0h)", "%2.2", "hf", "half", "4" },

    // Three component vector in scientific notation

    { { "" }, "(1234.25f,987654.5f,0.0005f)", "%.2", "hle", "float", "3" },

    // Four component vector in hexadecimal floating point, lowercase format

    { { "" }, "(0.25f,0.5f,1.f,1.5f)", "%.1", "hla", "float", "4" },

    // Eight component vector in the shortest float representation

    { { "" },
      "(1.f,2.f,3.f,4.f,1.5f,3.14f,2.5f,3.5f)",
      "%",
      "hlg",
      "float",
      "8" },

    // Sixteen component vector in unsigned octal format

    { { "" },
      "(1,2,3,4,5,6,7,8,9,0,32,64,128,256,512,1024)",
      "%",
      "hlo",
      "uint",
      "16" },

    // Eight component vector in signed decimal integer format

    { { "" }, "(1,-2,3,-4,5,-6,7,-8)", "%+", "hli", "int", "8" },

    // Four component vector in unsigned decimal integer format

    { { "" }, "(512,1024,262144,1048576)", "%05", "hlu", "uint", "4" },

};

//------------------------------------------------------------

// Lookup table -[string] vector-correct buffer              |

//------------------------------------------------------------

std::vector<std::string> correctBufferVector = {

    "1.00,2.00,3.00,4.00",

    "0xfa,0xfb",

    "0x1234,0x8765",

    "0x12345678,0x87654321",

    "12345678,98765432",

    "1.00,2.00,3.00,4.00",

    "1.23e+03,9.88e+05,5.00e-04",

    "0x1.0p-2,0x1.0p-1,0x1.0p+0,0x1.8p+0",

    "1,2,3,4,1.5,3.14,2.5,3.5",

    "1,2,3,4,5,6,7,10,11,0,40,100,200,400,1000,2000",

    "+1,-2,+3,-4,+5,-6,+7,-8",

    "00512,01024,262144,1048576"
};

//-----------------------------------------------------------

//Test case for vector                                      |

//-----------------------------------------------------------

testCase testCaseVector = {

    TYPE_VECTOR,

    correctBufferVector,

    printVectorGenParameters,

    NULL

};



//==================================================================

// address space

//==================================================================



//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// [string] argument type qualifier |[string] variable type qualifier + initialization | [string] format | [string] parameter |[string]%p indicator/additional code |

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------


std::vector<printDataGenParameters> printAddrSpaceGenParameters = {

    // Global memory region

    { { "\"%d\\n\"" },
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      "__global int* x",
      "",
      "*x",
      "" },

    // Global,constant, memory region

    { { "\"%d\\n\"" },
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      "constant int* x",
      "",
      "*x",
      "" },

    // Local memory region

    { { "\"%+d\\n\"" },
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      "",
      "local int x;\n x= (int)3;\n",
      "x",
      "" },

    // Private memory region

    { { "\"%i\\n\"" },
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      "",
      "private int x;\n x = (int)-1;\n",
      "x",
      "" },

    // Address of void * from global memory region

    { { "\"%p\\n\"" },
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      "__global void* x,__global intptr_t*  xAddr",
      "",
      "x",
      "*xAddr = (intptr_t)x;\n" }

};

//-------------------------------------------------------------------------------

//  Lookup table -[string] address space -correct buffer                        |

//-------------------------------------------------------------------------------

std::vector<std::string> correctAddrSpace = {

    "2","2","+3","-1",""

};

//-------------------------------------------------------------------------------

//Test case for address space                                                   |

//-------------------------------------------------------------------------------

testCase testCaseAddrSpace = {

    TYPE_ADDRESS_SPACE,

    correctAddrSpace,

    printAddrSpaceGenParameters,

    NULL

};

//=========================================================
// mixed format
//=========================================================

//----------------------------------------------------------
// Container related to mixed format tests.
// Empty records for which the format string and reference string are generated
// at run time. The size of this vector specifies the number of random tests
// that will be run.
std::vector<printDataGenParameters> printMixedFormatGenParameters(64,
                                                                  { { "" } });

std::vector<std::string> correctBufferMixedFormat;

//----------------------------------------------------------
// Test case for mixed-args
//----------------------------------------------------------
testCase testCaseMixedFormat = { TYPE_MIXED_FORMAT_RANDOM,
                                 correctBufferMixedFormat,
                                 printMixedFormatGenParameters, NULL };


//=============================================================

// length sub-specifier format

//=============================================================

std::vector<printDataGenParameters> printLenSpecGenParameters = {

    { { "%hd" }, "32767" },

    { { "%hhd" }, "127" },

    { { "%ld" }, "9223372036854775807L" },

    { { "%hd" }, "-32767" },

    { { "%hhd" }, "-128" },

    { { "%ld" }, "-9223372036854775807L" },

    { { "%hx" }, "32767" },

    { { "%hhx" }, "127" },

    { { "%lx" }, "9223372036854775807L" },

    { { "%hx" }, "-32767" },

    { { "%hhx" }, "-128" },

    { { "%lx" }, "-9223372036854775807L" },

    { { "%ho" }, "32767" },

    { { "%hho" }, "127" },

    { { "%lo" }, "9223372036854775807L" },

    { { "%ho" }, "-32767" },

    { { "%hho" }, "-128" },

    { { "%lo" }, "-9223372036854775807L" },
};

//---------------------------------------------------------

// Lookup table -[string] length specified correct buffer

//---------------------------------------------------------

std::vector<std::string> correctBufferLenSpec = {

    "32767",

    "127",

    "9223372036854775807",

    "-32767",

    "-128",

    "-9223372036854775807",

    "7fff",

    "7f",

    "7fffffffffffffff",

    "8001",

    "80",

    "8000000000000001",

    "77777",

    "177",

    "777777777777777777777",

    "100001",

    "200",

    "1000000000000000000001",
};


//----------------------------------------------------------

// Test case for length specified values

//----------------------------------------------------------

testCase testCaseLenSpec = {

    TYPE_LENGTH_SPECIFIER,

    correctBufferLenSpec,

    printLenSpecGenParameters,

    NULL

};

//-------------------------------------------------------------------------------

//All Test cases                                                                |

//-------------------------------------------------------------------------------

std::vector<testCase*> allTestCase = {
    &testCaseInt,        &testCaseLong,         &testCaseHalf,
    &testCaseHalfLimits, &testCaseFloat,        &testCaseFloatLimits,
    &testCaseDouble,     &testCaseDoubleLimits, &testCaseOctal,
    &testCaseUnsigned,   &testCaseHexadecimal,  &testCaseChar,
    &testCaseString,     &testCaseFormatString, &testCaseVector,
    &testCaseAddrSpace,  &testCaseMixedFormat,  &testCaseLenSpec
};

//-----------------------------------------

cl_half_rounding_mode half_rounding_mode = CL_HALF_RTE;

//-----------------------------------------

// Check functions

//-----------------------------------------

size_t verifyOutputBuffer(char *analysisBuffer,testCase* pTestCase,size_t testId,cl_ulong pAddr)
{
    int terminatePos = strlen(analysisBuffer);
        if(terminatePos > 0)
    {
        analysisBuffer[terminatePos - 1] = '\0';
    }

    //Convert analysis buffer to long for address space
    if(pTestCase->_type == TYPE_ADDRESS_SPACE && strcmp(pTestCase->_genParameters[testId].addrSpacePAdd,""))

    {
        char analysisBufferTmp[ANALYSIS_BUFFER_SIZE + 1];

        if(strstr(analysisBuffer,"0x") == NULL)
        // Need to prepend 0x to ASCII number before calling strtol.
        strcpy(analysisBufferTmp,"0x");

        else analysisBufferTmp[0]='\0';
        strncat(analysisBufferTmp, analysisBuffer, ANALYSIS_BUFFER_SIZE);
        if (sizeof(long) == 8) {
            if(strtoul(analysisBufferTmp,NULL,0) == pAddr) return 0;
        }
        else {
            if(strtoull(analysisBufferTmp,NULL,0) == pAddr) return 0;
        }
        return 1;

    }

    char* exp = nullptr;
    std::string copy_str;
    std::vector<char> staging(strlen(analysisBuffer) + 1);
    std::vector<char> staging_correct(pTestCase->_correctBuffer[testId].size()
                                      + 1);
    std::snprintf(staging.data(), staging.size(), "%s", analysisBuffer);
    std::snprintf(staging_correct.data(), staging_correct.size(), "%s",
                  pTestCase->_correctBuffer[testId].c_str());
    // Exponenent representation
    while ((exp = strstr(staging.data(), "E+")) != NULL
           || (exp = strstr(staging.data(), "e+")) != NULL
           || (exp = strstr(staging.data(), "E-")) != NULL
           || (exp = strstr(staging.data(), "e-")) != NULL)
    {
        char correctExp[3]={0};
        strncpy(correctExp,exp,2);

        // check if leading data is equal
        int ret = strncmp(staging_correct.data(), staging.data(),
                          exp - staging.data());
        if (ret) return ret;

        char* eCorrectBuffer = strstr(staging_correct.data(), correctExp);
        if(eCorrectBuffer == NULL)
            return 1;

        eCorrectBuffer+=2;
        exp += 2;

        //Exponent always contains at least two digits
        if(strlen(exp) < 2)
            return 1;
        //Skip leading zeros in the exponent
        while(*exp == '0')
            ++exp;
        while(*eCorrectBuffer == '0')
            ++eCorrectBuffer;

        copy_str = std::string(eCorrectBuffer);
        std::snprintf(staging_correct.data(), staging_correct.size(), "%s",
                      copy_str.c_str());

        copy_str = std::string(exp);
        std::snprintf(staging.data(), staging.size(), "%s", copy_str.c_str());

        if (strstr(staging.data(), "E+") != NULL
            || strstr(staging.data(), "e+") != NULL
            || strstr(staging.data(), "E-") != NULL
            || strstr(staging.data(), "e-") != NULL)
            continue;

        return strcmp(staging_correct.data(), copy_str.c_str());
    }

    if (pTestCase->_correctBuffer[testId] == "inf")
        return strcmp(analysisBuffer, "inf")
            && strcmp(analysisBuffer, "infinity");
    else if (pTestCase->_correctBuffer[testId] == "INF")
        return strcmp(analysisBuffer, "INF")
            && strcmp(analysisBuffer, "INFINITY");
    else if (pTestCase->_correctBuffer[testId] == "nan")
        return strcmp(analysisBuffer, "nan") && strcmp(analysisBuffer, "-nan");
    else if (pTestCase->_correctBuffer[testId] == "NAN")
        return strcmp(analysisBuffer, "NAN") && strcmp(analysisBuffer, "-NAN");

    return strcmp(analysisBuffer, pTestCase->_correctBuffer[testId].c_str());
}

static void intRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    snprintf(refResult, refSize, params.genericFormats.front().c_str(),
             atoi(params.dataRepresentation));
}

static void halfRefBuilder(printDataGenParameters& params, char* refResult,
                           const size_t refSize)
{
    cl_half val = cl_half_from_float(strtof(params.dataRepresentation, NULL),
                                     half_rounding_mode);
    snprintf(refResult, refSize, params.genericFormats.front().c_str(),
             cl_half_to_float(val));
}

static void floatRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    snprintf(refResult, refSize, params.genericFormats.front().c_str(),
             strtof(params.dataRepresentation, NULL));
}

static void doubleRefBuilder(printDataGenParameters& params, char* refResult,
                             const size_t refSize)
{
    snprintf(refResult, refSize, params.genericFormats.front().c_str(),
             strtod(params.dataRepresentation, NULL));
}

static void octalRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormats.front().c_str(), data);
}

static void unsignedRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormats.front().c_str(), data);
}

static void hexRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 0);
    snprintf(refResult, refSize, params.genericFormats.front().c_str(), data);
}

/*
    Generate reference results.

    Results are only generated for test cases
    that can easily be generated by using CPU
    printf.

    If that is not the case, results are constants
    that have been hard-coded.
*/
void generateRef(const cl_device_id device)
{
    bool fp16_supported = is_extension_available(device, "cl_khr_fp16");
    bool fp64_supported = is_extension_available(device, "cl_khr_fp64");

    const cl_device_fp_config fpConfigSingle =
        get_default_rounding_mode(device);
    const cl_device_fp_config fpConfigHalf = (half_rounding_mode == CL_HALF_RTE)
        ? CL_FP_ROUND_TO_NEAREST
        : CL_FP_ROUND_TO_ZERO;
    const cl_device_fp_config fpConfigDouble = fp64_supported
        ? get_default_rounding_mode(device, CL_DEVICE_DOUBLE_FP_CONFIG)
        : 0;
    const RoundingMode hostRound = get_round();

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

    // Loop through all test cases
    for (auto &caseToTest: allTestCase)
    {
        /*
            Cases that have a NULL function pointer
            already have their reference results
            as they're constant and hard-coded
        */
        if (caseToTest->printFN == NULL)
            continue;

        // Make sure the reference result is empty
        assert(caseToTest->_correctBuffer.size() == 0);

        const cl_device_fp_config* fpConfig = nullptr;
        switch (caseToTest->_type)
        {
            case TYPE_HALF:
            case TYPE_HALF_LIMITS:
                if (!fp16_supported) continue;
                fpConfig = &fpConfigHalf;
                break;
            case TYPE_DOUBLE:
            case TYPE_DOUBLE_LIMITS:
                if (!fp64_supported) continue;
                fpConfig = &fpConfigDouble;
                break;
            default: fpConfig = &fpConfigSingle; break;
        }
        RoundingMode deviceRound = get_rounding(*fpConfig);

        // Loop through each input
        for (auto &params: caseToTest->_genParameters)
        {
            char refResult[ANALYSIS_BUFFER_SIZE];
            // Set CPU rounding mode to match that of the device
            set_round(deviceRound, caseToTest->dataType);
            // Generate the result
            caseToTest->printFN(params, refResult, ARRAY_SIZE(refResult));
            // Restore the original CPU rounding mode
            set_round(hostRound, kfloat);
            // Save the reference result
            caseToTest->_correctBuffer.push_back(refResult);
        }
    }
}

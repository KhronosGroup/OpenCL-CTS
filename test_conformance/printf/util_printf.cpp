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
#include "harness/compat.h"
#include "harness/rounding_mode.h"
#include "harness/kernelHelpers.h"

#include "test_printf.h"
#include <assert.h>

// Helpers for generating runtime reference results
static void intRefBuilder(printDataGenParameters&, char*, const size_t);
static void floatRefBuilder(printDataGenParameters&, char* rResult, const size_t);
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

    {"%5d","10"},

        //(Minimum)Five-wide,left-justified

    {"%-5d","10"},

        //(Minimum)Five-wide,default(right)-justified,zero-filled

    {"%05d","10"},

        //(Minimum)Five-wide,default(right)-justified,with sign

    {"%+5d","10"},

         //(Minimum)Five-wide ,left-justified,with sign

    {"%-+5d","10"},

        //(Minimum)Five-digit(zero-filled in absent digits),default(right)-justified

    {"%.5i","100"},

        //(Minimum)Six-wide,Five-digit(zero-filled in absent digits),default(right)-justified

    {"%6.5i","100"},

        //0 and - flag both apper ==>0 is ignored,left-justified,capital I

    {"%-06i","100"},

        //(Minimum)Six-wide,Five-digit(zero-filled in absent digits),default(right)-justified

    {"%06.5i","100"}

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





//==============================================

// float

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------

std::vector<printDataGenParameters> printFloatGenParameters = {

    //Default(right)-justified

    {"%f","10.3456"},

    //One position after the decimal,default(right)-justified

    {"%.1f","10.3456"},

    //Two positions after the decimal,default(right)-justified

    {"%.2f","10.3456"},

    //(Minimum)Eight-wide,three positions after the decimal,default(right)-justified

    {"%8.3f","10.3456"},

    //(Minimum)Eight-wide,two positions after the decimal,zero-filled,default(right)-justified

    {"%08.2f","10.3456"},

    //(Minimum)Eight-wide,two positions after the decimal,left-justified

    {"%-8.2f","10.3456"},

    //(Minimum)Eight-wide,two positions after the decimal,with sign,default(right)-justified

    {"%+8.2f","-10.3456"},

    //Zero positions after the decimal([floor]rounding),default(right)-justified

    {"%.0f","0.1"},

    //Zero positions after the decimal([ceil]rounding),default(right)-justified

    {"%.0f","0.6"},

    //Zero-filled,default positions number after the decimal,default(right)-justified

    {"%0f","0.6"},

    //Double argument representing floating-point,used by f style,default(right)-justified

    {"%4g","12345.6789"},

    //Double argument representing floating-point,used by e style,default(right)-justified

    {"%4.2g","12345.6789"},

    //Double argument representing floating-point,used by f style,default(right)-justified

    {"%4G","0.0000023"},

    //Double argument representing floating-point,used by e style,default(right)-justified

    {"%4G","0.023"},

    //Double argument representing floating-point,with exponent,left-justified,default(right)-justified

    {"%-#20.15e","789456123.0"},

    //Double argument representing floating-point,with exponent,left-justified,with sign,capital E,default(right)-justified

    {"%+#21.15E","789456123.0"},

    //Double argument representing floating-point,in [-]xh.hhhhpAd style

    {"%.6a","0.1"},

    //(Minimum)Ten-wide,Double argument representing floating-point,in xh.hhhhpAd style,default(right)-justified

    {"%10.2a","9990.235"},
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

    //Infinity (1.0/0.0)

    {"%f","1.0f/0.0f"},

    //NaN

    {"%f","sqrt(-1.0f)"},

    //NaN
    {"%f","acospi(2.0f)"}
    };
//--------------------------------------------------------

//  Lookup table - [string]float-correct buffer             |

//--------------------------------------------------------

std::vector<std::string> correctBufferFloatLimits = {

    "inf",

    "-nan",

    "nan"
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

//=========================================================

// octal

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] octal-data representation  |

//---------------------------------------------------------

std::vector<printDataGenParameters> printOctalGenParameters = {

    //Default(right)-justified

    {"%o","10"},

    //Five-digit,default(right)-justified

    {"%.5o","10"},

    //Default(right)-justified,increase precision

    {"%#o","100000000"},

    //(Minimum)Four-wide,Five-digit,0-flag ignored(because of precision),default(right)-justified

    {"%04.5o","10"}

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

    //Default(right)-justified

    {"%u","10"},

    //Zero precision for zero,default(right)-justified

    {"%.0u","0"},

};

//-------------------------------------------------------

//Test case for octal                                   |

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

    //Add 0x,low x,default(right)-justified

    {"%#x","0xABCDEF"},

    //Add 0x,capital X,default(right)-justified

    {"%#X","0xABCDEF"},

    //Not add 0x,if zero,default(right)-justified

    {"%#X","0"},

    //(Minimum)Eight-wide,default(right)-justified

    {"%8x","399"},

    //(Minimum)Four-wide,zero-filled,default(right)-justified

    {"%04x","399"}

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

    //Four-wide,zero-filled,default(right)-justified

    {"%4c","\'1\'"},

        //Four-wide,left-justified

    {"%-4c","\'1\'"},

        //(unsigned) int argument,default(right)-justified

    {"%c","66"}

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

std::vector<printDataGenParameters> printStringGenParameters = {

    //(Minimum)Four-wide,zero-filled,default(right)-justified

    {"%4s","\"foo\""},

    //One-digit(precision ignored),left-justified

    {"%.1s","\"foo\""},

    //%% specification

    {"%s","\"%%\""},
};

//---------------------------------------------------------

// Lookup table -[string] string-correct buffer           |

//---------------------------------------------------------

std::vector<std::string> correctBufferString = {

    " foo",

    "f",

    "%%",
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



//=========================================================

// vector

//=========================================================



//-------------------------------------------------------------------------------------------------------------------

//[string] flag | [string] specifier | [string] type | [string] vector-data representation | [string] vector size   |

//-------------------------------------------------------------------------------------------------------------------

std::vector<printDataGenParameters> printVectorGenParameters = {

    //(Minimum)Two-wide,two positions after decimal

    {NULL,"(1.0f,2.0f,3.0f,4.0f)","%2.2","hlf","float","4"},

    //Alternative form,uchar argument

    {NULL,"(0xFA,0xFB)","%#","hhx","uchar","2"},

    //Alternative form,ushort argument

    {NULL,"(0x1234,0x8765)","%#","hx","ushort","2"},

  //Alternative form,uint argument

    {NULL,"(0x12345678,0x87654321)","%#","hlx","uint","2"},

    //Alternative form,long argument

    {NULL,"(12345678,98765432)","%","ld","long","2"}

};

//------------------------------------------------------------

// Lookup table -[string] vector-correct buffer              |

//------------------------------------------------------------

std::vector<std::string> correctBufferVector = {

    "1.00,2.00,3.00,4.00",

    "0xfa,0xfb",

    "0x1234,0x8765",

  "0x12345678,0x87654321",

    "12345678,98765432"

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

    //Global memory region

    {"\"%d\\n\"",NULL,NULL,NULL,NULL,NULL,"__global int* x","","*x",""},

    //Global,constant, memory region

    {"\"%d\\n\"",NULL,NULL,NULL,NULL,NULL,"constant int* x","","*x",""},

    //Local memory region

    {"\"%+d\\n\"",NULL,NULL,NULL,NULL,NULL,"","local int x;\n x= (int)3;\n","x",""},

    //Private memory region

    {"\"%i\\n\"",NULL,NULL,NULL,NULL,NULL,"","private int x;\n x = (int)-1;\n","x",""},

    //Address of void * from global memory region

    {"\"%p\\n\"",NULL,NULL,NULL,NULL,NULL,"__global void* x,__global intptr_t*  xAddr","","x","*xAddr = (intptr_t)x;\n"}

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



//-------------------------------------------------------------------------------

//All Test cases                                                                |

//-------------------------------------------------------------------------------

std::vector<testCase*> allTestCase = {&testCaseInt,&testCaseFloat,&testCaseFloatLimits,&testCaseOctal,&testCaseUnsigned,&testCaseHexadecimal,&testCaseChar,&testCaseString,&testCaseVector,&testCaseAddrSpace};


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
        char analysisBufferTmp[ANALYSIS_BUFFER_SIZE];

        if(strstr(analysisBuffer,"0x") == NULL)
        // Need to prepend 0x to ASCII number before calling strtol.
        strcpy(analysisBufferTmp,"0x");

        else analysisBufferTmp[0]='\0';
        strcat(analysisBufferTmp,analysisBuffer);
        if (sizeof(long) == 8) {
            if(strtoul(analysisBufferTmp,NULL,0) == pAddr) return 0;
        }
        else {
            if(strtoull(analysisBufferTmp,NULL,0) == pAddr) return 0;
        }
        return 1;

    }

    char* exp;
    //Exponenent representation
    if((exp = strstr(analysisBuffer,"E+")) != NULL || (exp = strstr(analysisBuffer,"e+")) != NULL || (exp = strstr(analysisBuffer,"E-")) != NULL || (exp = strstr(analysisBuffer,"e-")) != NULL)
    {
        char correctExp[3]={0};
        strncpy(correctExp,exp,2);

        char* eCorrectBuffer = strstr((char*)pTestCase->_correctBuffer[testId].c_str(),correctExp);
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
        return strcmp(eCorrectBuffer,exp);
    }
    if(!strcmp(pTestCase->_correctBuffer[testId].c_str(),"inf"))
    return strcmp(analysisBuffer,"inf")&&strcmp(analysisBuffer,"infinity")&&strcmp(analysisBuffer,"1.#INF00")&&strcmp(analysisBuffer,"Inf");
    if(!strcmp(pTestCase->_correctBuffer[testId].c_str(),"nan") || !strcmp(pTestCase->_correctBuffer[testId].c_str(),"-nan")) {
       return strcmp(analysisBuffer,"nan")&&strcmp(analysisBuffer,"-nan")&&strcmp(analysisBuffer,"1.#IND00")&&strcmp(analysisBuffer,"-1.#IND00")&&strcmp(analysisBuffer,"NaN")&&strcmp(analysisBuffer,"nan(ind)")&&strcmp(analysisBuffer,"nan(snan)")&&strcmp(analysisBuffer,"-nan(ind)");
    }
    return strcmp(analysisBuffer,pTestCase->_correctBuffer[testId].c_str());
}

static void intRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    snprintf(refResult, refSize, params.genericFormat, atoi(params.dataRepresentation));
}

static void floatRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    snprintf(refResult, refSize, params.genericFormat, strtof(params.dataRepresentation, NULL));
}

static void octalRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormat, data);
}

static void unsignedRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 10);
    snprintf(refResult, refSize, params.genericFormat, data);
}

static void hexRefBuilder(printDataGenParameters& params, char* refResult, const size_t refSize)
{
    const unsigned long int data = strtoul(params.dataRepresentation, NULL, 0);
    snprintf(refResult, refSize, params.genericFormat, data);
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
    int fd = -1;
    char _refBuffer[ANALYSIS_BUFFER_SIZE];
    const cl_device_fp_config fpConfig = get_default_rounding_mode(device);
    const RoundingMode hostRound = get_round();
    RoundingMode deviceRound;

    // Map device rounding to CTS rounding type
    // get_default_rounding_mode supports RNE and RTZ
    if (fpConfig == CL_FP_ROUND_TO_NEAREST)
    {
        deviceRound = kRoundToNearestEven;
    }
    else if (fpConfig == CL_FP_ROUND_TO_ZERO)
    {
        deviceRound = kRoundTowardZero;
    }
    else
    {
        assert(false && "Unreachable");
    }

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

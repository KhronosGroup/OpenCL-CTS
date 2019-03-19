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
#if !defined(_WIN32)

#include <stdbool.h>

#include <stdint.h>

#endif



#include <math.h>

#include "test_printf.h"


#if defined (_WIN32)
#define strtoull _strtoi64
#endif

const char* strType[] = {"int","float","octal","unsigned","hexadecimal","char","string","vector","address space"};



//==================================

// int

//==================================

//------------------------------------------------------

// [string] format  | [string] int-data representation |

//------------------------------------------------------

struct printDataGenParameters printIntGenParameters[] = {

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

//------------------------------------------------

// Lookup table - [string]int-correct buffer     |

//------------------------------------------------

const char *correctBufferInt[] = {

    "   10",

    "10   ",

    "00010",

    "  +10",

    "+10  ",

    "00100",

    " 00100",

    "100   ",

    " 00100"

};





//-----------------------------------------------

//test case for int                             |

//-----------------------------------------------

testCase testCaseInt = {

    sizeof(correctBufferInt)/sizeof(char*),

    INT,

    correctBufferInt,

    printIntGenParameters

};





//==============================================

// float

//==============================================



//--------------------------------------------------------

// [string] format |  [string] float-data representation |

//--------------------------------------------------------

struct printDataGenParameters printFloatGenParameters[] = {

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

    //Infinity (1.0/0.0)

    {"%f","1.0f/0.0f"},

    //NaN

    {"%f","sqrt(-1.0f)"}
    };
//--------------------------------------------------------

//  Lookup table - [string]float-correct buffer             |

//--------------------------------------------------------

const char* correctBufferFloat[] = {

    "10.345600",

    "10.3",

    "10.35",

    "  10.346",

    "00010.35",

    "10.35   ",

    "  -10.35",

    "0",

    "1",

    "0.600000",

    "12345.7",

    "1.2e+4",

    "2.3E-6",

    "0.023",

    "7.894561230000000e+8",

    "+7.894561230000000E+8",

    "0x1.99999ap-4",

    "0x1.38p+13",

    "inf",

    "nan"
};

//---------------------------------------------------------

//Test case for float                                     |

//---------------------------------------------------------

testCase testCaseFloat = {

    sizeof(correctBufferFloat)/sizeof(char*),

    FLOAT,

    correctBufferFloat,

    printFloatGenParameters

};



//=========================================================

// octal

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] octal-data representation  |

//---------------------------------------------------------

struct printDataGenParameters printOctalGenParameters[] = {

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

// Lookup table - [string] octal-correct buffer            |

//-------------------------------------------------------



const char* correctBufferOctal[] = {

    "12",

    "00012",

    "0575360400",

    "00012"

};

//-------------------------------------------------------

//Test case for octal                                   |

//-------------------------------------------------------

testCase testCaseOctal = {

    sizeof(correctBufferOctal)/sizeof(char*),

    OCTAL,

    correctBufferOctal,

    printOctalGenParameters

};



//=========================================================

// unsigned

//=========================================================



//---------------------------------------------------------

// [string] format  | [string] unsined-data representation  |

//---------------------------------------------------------

struct printDataGenParameters printUnsignedGenParameters[] = {

    //Default(right)-justified

    {"%u","10"},

    //Zero precision for zero,default(right)-justified

    {"%.0u","0"},

};

//-------------------------------------------------------

// Lookup table - [string] octal-correct buffer            |

//-------------------------------------------------------



const char* correctBufferUnsigned[] = {

    "10",

    ""

};

//-------------------------------------------------------

//Test case for octal                                   |

//-------------------------------------------------------

testCase testCaseUnsigned = {

    sizeof(correctBufferUnsigned)/sizeof(char*),

    UNSIGNED,

    correctBufferUnsigned,

    printUnsignedGenParameters

};



//=======================================================

// hexadecimal

//=======================================================



//--------------------------------------------------------------

// [string] format  | [string] hexadecimal-data representation |

//--------------------------------------------------------------

struct printDataGenParameters printHexadecimalGenParameters[] = {

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

// Lookup table - [string]hexadecimal-correct buffer           |

//--------------------------------------------------------------



const char* correctBufferHexadecimal[] = {

    "0xabcdef",

    "0XABCDEF",

    "0",

    "     18f",

    "018f"

};

//--------------------------------------------------------------

//Test case for hexadecimal                                    |

//--------------------------------------------------------------

testCase testCaseHexadecimal = {

    sizeof(correctBufferHexadecimal)/sizeof(char*),

    HEXADEC,

    correctBufferHexadecimal,

    printHexadecimalGenParameters

};



//=============================================================

// char

//=============================================================



//-----------------------------------------------------------

// [string] format  | [string] string-data representation   |

//-----------------------------------------------------------

struct printDataGenParameters printCharGenParameters[] = {

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

const char * correctBufferChar[] = {

    "   1",

    "1   ",

    "B",

};



//----------------------------------------------------------

//Test case for char                                       |

//----------------------------------------------------------

testCase testCaseChar = {

    sizeof(correctBufferChar)/sizeof(char*),

    CHAR,

    correctBufferChar,

    printCharGenParameters

};



//==========================================================

// string

//==========================================================



//--------------------------------------------------------

// [string]format | [string] string-data representation  |

//--------------------------------------------------------

struct printDataGenParameters printStringGenParameters[] = {

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

const char * correctBufferString[] = {

    " foo",

    "f",

    "%%",
};

//---------------------------------------------------------

//Test case for string                                    |

//---------------------------------------------------------

testCase testCaseString = {

    sizeof(correctBufferString)/sizeof(char*),

    STRING,

    correctBufferString,

    printStringGenParameters

};



//=========================================================

// vector

//=========================================================



//-------------------------------------------------------------------------------------------------------------------

//[string] flag | [string] specifier | [string] type | [string] vector-data representation | [string] vector size   |

//-------------------------------------------------------------------------------------------------------------------

struct printDataGenParameters printVectorGenParameters[]={

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

const char * correctBufferVector[] = {

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

    sizeof(correctBufferVector)/(sizeof(char *)),

    VECTOR,

    correctBufferVector,

    printVectorGenParameters

};



//==================================================================

// address space

//==================================================================



//-------------------------------------------------------------------------------------------------------------------------------------------------------------------

// [string] argument type qualifier |[string] variable type qualifier + initialization | [string] format | [string] parameter |[string]%p indicator/additional code |

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------



struct printDataGenParameters printAddrSpaceGenParameters[]={

    //Global memory region

    {"\"%d\\n\"",NULL,NULL,NULL,NULL,NULL,"__global int* x","","*x",""},

    //Global,constant, memory region

    {"\"%d\\n\"",NULL,NULL,NULL,NULL,NULL,"constant int* x","","*x",""},

    //Local memory region

    {"\"%+d\\n\"",NULL,NULL,NULL,NULL,NULL,"","local int x;\n x= (int)3;\n","x",""},

    //Private memory region

    {"\"%i\\n\"",NULL,NULL,NULL,NULL,NULL,"","private int x;\n x = (int)-1;\n","x",""},

    //Address of void * from global memory region

    {"\"%p\\n\"",NULL,NULL,NULL,NULL,NULL,"__global void* x,__global long* xAddr","","x","*xAddr = x;\n"}

};

//-------------------------------------------------------------------------------

//  Lookup table -[string] address space -correct buffer                        |

//-------------------------------------------------------------------------------

const char * correctAddrSpace[] = {

    "2","2","+3","-1",""

};

//-------------------------------------------------------------------------------

//Test case for address space                                                   |

//-------------------------------------------------------------------------------

testCase testCaseAddrSpace = {

    sizeof(correctAddrSpace)/(sizeof(char *)),

    ADDRESS_SPACE,

    correctAddrSpace,

    printAddrSpaceGenParameters

};



//-------------------------------------------------------------------------------

//All Test cases                                                                |

//-------------------------------------------------------------------------------

testCase* allTestCase[] = {&testCaseInt,&testCaseFloat,&testCaseOctal,&testCaseUnsigned,&testCaseHexadecimal,&testCaseChar,&testCaseString,&testCaseVector,&testCaseAddrSpace};


//-----------------------------------------

// Check functions

//-----------------------------------------

size_t verifyOutputBuffer(char *analysisBuffer,testCase* pTestCase,size_t testId,cl_ulong pAddr)
{

    analysisBuffer[strlen(analysisBuffer)-1] = '\0';
    //Convert analysis buffer to long for address space
    if(pTestCase->_type == ADDRESS_SPACE && strcmp(pTestCase->_genParameters[testId].addrSpacePAdd,""))

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


        char* eCorrectBuffer = strstr((char*)pTestCase->_correctBuffer[testId],correctExp);
        if(eCorrectBuffer == NULL)
            return 1;
        eCorrectBuffer+=2;
        exp += 2;
        //Exponent always contains at least two digits
        if(strlen(exp) < 2)
            return 1;
        //Scip leading zeros in the exponent
        while(*exp == '0') 
            ++exp; 
    return strcmp(eCorrectBuffer,exp);
    }
    if(!strcmp(pTestCase->_correctBuffer[testId],"inf"))
        return strcmp(analysisBuffer,"inf")&&strcmp(analysisBuffer,"infinity")&&strcmp(analysisBuffer,"1.#INF00")&&strcmp(analysisBuffer,"Inf");
    if(!strcmp(pTestCase->_correctBuffer[testId],"nan") || !strcmp(pTestCase->_correctBuffer[testId],"-nan")) {
        return strcmp(analysisBuffer,"nan")&&strcmp(analysisBuffer,"-nan")&&strcmp(analysisBuffer,"1.#IND00")&&strcmp(analysisBuffer,"-1.#IND00")&&strcmp(analysisBuffer,"NaN")&&strcmp(analysisBuffer,"nan(ind)")&&strcmp(analysisBuffer,"nan(snan)")&&strcmp(analysisBuffer,"-nan(ind)");
    }
    return strcmp(analysisBuffer,pTestCase->_correctBuffer[testId]);
}


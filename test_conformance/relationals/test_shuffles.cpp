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

#include <iomanip>
#include <vector>
#include "testBase.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"

// #define USE_NEW_SYNTAX    1
// The number of shuffles to test per test
#define NUM_TESTS 32
// The number of times to run each combination of shuffles
#define NUM_ITERATIONS_PER_TEST 2
#define MAX_PROGRAM_SIZE NUM_TESTS*1024
#define PRINT_SHUFFLE_KERNEL_SOURCE 0
#define SPEW_ORDER_DETAILS 0

enum ShuffleMode
{
    kNormalMode = 0,
    kFunctionCallMode,
    kArrayAccessMode,
    kBuiltInFnMode,
    kBuiltInDualInputFnMode
};

static const char *shuffleKernelPattern[3] =  {
    "__kernel void sample_test( __global %s%s *source, __global %s%s *dest )\n"
    "{\n"
    "    if (get_global_id(0) != 0) return;\n"
    "     //%s%s src1 %s, src2%s;\n",// Here's a comma...
                                    // Above code is commented out for now, but keeping around for testing local storage options
    "}\n" };

static const char *shuffleTempPattern = "  %s%s tmp;\n";

static const char *clearTempPattern = "        tmp = (%s%s)((%s)0);\n";

static const char *shuffleSinglePattern =
"        tmp%s%s = source[%d]%s%s;\n"
"        dest[%d] = tmp;\n"
;

static const char * shuffleSinglePatternV3src =
"           tmp%s%s = vload3(%d, source)%s%s;\n"
"        dest[%d] = tmp;\n";

static const char * shuffleSinglePatternV3dst =
"        tmp%s%s = source[%d]%s%s;\n"
"           vstore3(tmp, %d, dest);\n";


static const char * shuffleSinglePatternV3srcV3dst =
"tmp%s%s = vload3(%d, source)%s%s;\n"
"vstore3(tmp, %d, dest);\n";

static const char *shuffleFnLinePattern = "%s%s shuffle_fn( %s%s source );\n%s%s shuffle_fn( %s%s source ) { return source; }\n\n";

static const char *shuffleFnPattern =
"        tmp%s%s = shuffle_fn( source[%d] )%s%s;\n"
"        dest[%d] = tmp;\n"
;


static const char *shuffleFnPatternV3src =
"        tmp%s%s = shuffle_fn( vload3(%d, source) )%s%s;\n"
"        dest[%d] = tmp;\n"
;


static const char *shuffleFnPatternV3dst =
"        tmp%s%s = shuffle_fn( source[%d] )%s%s;\n"
"               vstore3(tmp, %d, dest);\n"
;


static const char *shuffleFnPatternV3srcV3dst =
"        tmp%s%s = shuffle_fn(vload3(%d, source) )%s%s;\n"
"               vstore3(tmp, %d, dest);\n"
;

// shuffle() built-in function patterns
static const char *shuffleBuiltInPattern =
"        {\n"
"            %s%s src1 = %s;\n"
"            %s%s%s mask = (%s%s%s)( %s );\n"
"            tmp = shuffle( src1, mask );\n"
"            %s;\n"
"        }\n"
;

// shuffle() built-in dual-input function patterns
static const char *shuffleBuiltInDualPattern =
"        {\n"
"            %s%s src1 = %s;\n"
"            %s%s src2 = %s;\n"
"            %s%s%s mask = (%s%s%s)( %s );\n"
"            tmp = shuffle2( src1, src2, mask );\n"
"            %s;\n"
"        }\n"
;


typedef unsigned char ShuffleOrder[ 16 ];

void incrementShuffleOrder( ShuffleOrder &order, size_t orderSize, size_t orderRange )
{
    for( size_t i = 0; i < orderSize; i++ )
    {
        order[ i ]++;
        if( order[ i ] < orderRange )
            return;
        order[ i ] = 0;
    }
}

bool shuffleOrderContainsDuplicates( ShuffleOrder &order, size_t orderSize )
{
    bool flags[ 16 ] = { false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
    for( size_t i = 0; i < orderSize; i++ )
    {
        if( flags[ order[ i ] ] )
            return true;
        flags[ order[ i ] ] = true;
    }
    return false;
}

static void shuffleVector( unsigned char *inVector, unsigned char *outVector, ShuffleOrder order, size_t vecSize, size_t typeSize, cl_uint lengthToUse )
{
    for(size_t i = 0; i < lengthToUse; i++ )
    {
        unsigned char *inPtr = inVector + typeSize *order[ i ];
        memcpy( outVector, inPtr, typeSize );
        outVector += typeSize;
    }
}

static void shuffleVector2( unsigned char *inVector, unsigned char *outVector, ShuffleOrder order, size_t vecSize, size_t typeSize, cl_uint lengthToUse )
{
    for(size_t i = 0; i < lengthToUse; i++ )
    {
        unsigned char *outPtr = outVector + typeSize *order[ i ];
        memcpy( outPtr, inVector, typeSize );
        inVector += typeSize;
    }
}

static void shuffleVectorDual( unsigned char *inVector, unsigned char *inSecondVector, unsigned char *outVector, ShuffleOrder order, size_t vecSize, size_t typeSize, cl_uint lengthToUse )
{
    // This is tricky: the indices of each shuffle are in a range (0-srcVecSize * 2-1),
    // where (srcVecSize-srcVecSize*2-1) refers to the second input.
    size_t uphalfMask = (size_t)vecSize;
    size_t lowerBits = (size_t)( vecSize - 1 );

    for(size_t i = 0; i < lengthToUse; i++ )
    {
        unsigned char *inPtr;
#if SPEW_ORDER_DETAILS
        log_info("order[%d] is %d, or %d of %s\n", (int)i,
                 (int)(order[i]),
                 (int)(order[i] & lowerBits),
                 ((order[i]&uphalfMask) == 0)?"lower num":"upper num");
#endif
        if( order[ i ] & uphalfMask )
            inPtr = inSecondVector + typeSize * ( order[ i ] & lowerBits );
        else
            inPtr = inVector + typeSize * ( order[ i ] & lowerBits );
        memcpy( outVector, inPtr, typeSize );
        outVector += typeSize;
    }
}


static ShuffleOrder sNaturalOrder = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

static int useNumbersFlip = 0;
const char *get_order_string( ShuffleOrder &order, size_t vecSize, cl_uint lengthToUse, bool byNumber, MTdata d )
{
    // NOTE: names are only valid for hex characters (up to F) but for debugging, we use
    // this to print out orders for dual inputs, which actually can be valid up to position 31 (two 16-element vectors)
    // so we go ahead and fake the rest of the alphabet for those other 16 positions, so we have
    // some (indirectly) meaningful output
    char names[] = "0123456789abcdefghijklmnopqrstuv";
    char namesUpperCase[] = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
    char names2[] = "xyzw!!!!!!!!!!!!";

    static char orderString[ 18 ];

    size_t j, idx;

    // Assume we don't have to use numbered indices (.s0123...).
    byNumber = false;
    // Check if any index is beyond xyzw, which requires to use numbers.
    for( j = 0; j < lengthToUse; j++ )
    {
        if (order[j] > 3) {
            byNumber = true;
            break;
        }
    }
    // If we can use numbers, do so half the time.
    if (!byNumber) {
        byNumber = (useNumbersFlip++)%2;
    }

    if (byNumber)
    {
        idx = 0;
        // Randomly chose upper and lower case S.
        orderString[ idx++ ] = random_in_range(0, 1, d) ? 's' : 'S';
        for( j = 0; j < vecSize && j < lengthToUse; j++ ) {
            // Randomly choose upper and lower case.
            orderString[ idx++ ] = random_in_range(0, 1, d) ? names[ (int)order[ j ] ] : namesUpperCase[ (int)order[ j ] ];
        }
        orderString[ idx++ ] = 0;
    }
    else
    {
        // Use xyzw.
        for( j = 0; j < vecSize && j < lengthToUse; j++ ) {
            orderString[ j ] = names2[ (int)order[ j ] ];
        }
        orderString[ j ] = 0;
    }

    return orderString;
}

char * get_order_name( ExplicitType vecType, size_t inVecSize, size_t outVecSize, ShuffleOrder &inOrder, ShuffleOrder &outOrder, cl_uint lengthToUse, MTdata d, bool inUseNumerics, bool outUseNumerics )
{
    static char orderName[ 512 ] = "";
    char inOrderStr[64], outOrderStr[64];

    if( inVecSize == 1 )
        inOrderStr[ 0 ] = 0;
    else
        sprintf(inOrderStr, "%d.%s", (int)inVecSize,
                get_order_string(inOrder, inVecSize, lengthToUse, inUseNumerics,
                                 d));
    if( outVecSize == 1 )
        outOrderStr[ 0 ] = 0;
    else
        sprintf( outOrderStr, "%d.%s", (int)outVecSize, get_order_string( outOrder, outVecSize, lengthToUse, outUseNumerics, d ) );

    sprintf( orderName, "order %s%s -> %s%s",
            get_explicit_type_name( vecType ), inOrderStr, get_explicit_type_name( vecType ), outOrderStr );
    return orderName;
}

void print_hex_mem_dump(const unsigned char *inDataPtr,
                        const unsigned char *inDataPtr2,
                        const unsigned char *expected,
                        const unsigned char *outDataPtr, size_t inVecSize,
                        size_t outVecSize, size_t typeSize)
{
    auto byte_to_hex_str = [](unsigned char v) {
        // Use a new stream to avoid manipulating state of outer stream.
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(2) << std::right << std::hex << +v;
        return ss.str();
    };

    std::ostringstream error;
    error << "      Source: ";
    for (size_t j = 0; j < inVecSize * typeSize; j++)
    {
        error << (j % typeSize ? "" : " ") << byte_to_hex_str(inDataPtr[j])
              << " ";
    }
    if (inDataPtr2 != NULL)
    {
        error << "\n    Source 2: ";
        for (size_t j = 0; j < inVecSize * typeSize; j++)
        {
            error << (j % typeSize ? "" : " ") << byte_to_hex_str(inDataPtr2[j])
                  << " ";
        }
    }
    error << "\n    Expected: ";
    for (size_t j = 0; j < outVecSize * typeSize; j++)
    {
        error << (j % typeSize ? "" : " ") << byte_to_hex_str(expected[j])
              << " ";
    }
    error << "\n      Actual: ";
    for (size_t j = 0; j < outVecSize * typeSize; j++)
    {
        error << (j % typeSize ? "" : " ") << byte_to_hex_str(outDataPtr[j])
              << " ";
    }
    log_info("%s\n", error.str().c_str());
}

void generate_shuffle_mask( char *outMaskString, size_t maskSize, const ShuffleOrder *order )
{
    outMaskString[ 0 ] = 0;
    if( order != NULL )
    {
        for( size_t jj = 0; jj < maskSize; jj++ )
        {
            char thisMask[ 16 ];
            sprintf( thisMask, "%s%d", ( jj == 0 ) ? "" : ", ", (*order)[ jj ] );
            strcat( outMaskString, thisMask );
        }
    }
    else
    {
        for( size_t jj = 0; jj < maskSize; jj++ )
        {
            char thisMask[ 16 ];
            sprintf( thisMask, "%s%ld", ( jj == 0 ) ? "" : ", ", jj );
            strcat( outMaskString, thisMask );
        }
    }
}

static int create_shuffle_kernel( cl_context context, cl_program *outProgram, cl_kernel *outKernel,
                                 size_t *outRealVecSize,
                                 ExplicitType vecType, size_t inVecSize, size_t outVecSize, cl_uint *lengthToUse, bool inUseNumerics, bool outUseNumerics,
                                 size_t numOrders, ShuffleOrder *inOrders, ShuffleOrder *outOrders,
                                 MTdata d, ShuffleMode shuffleMode = kNormalMode )
{
    char inOrder[18], shuffledOrder[18];
    char kernelSource[MAX_PROGRAM_SIZE], progLine[ 10240 ];
    char *programPtr;
    char inSizeName[4], outSizeName[4], outRealSizeName[4], inSizeArgName[4];
    char outSizeNameTmpVar[4];


    /* Create the source; note vec size is the vector length we are testing */
    if( inVecSize == 1 ) //|| (inVecSize == 3)) // just have arrays if we go with size 3
        inSizeName[ 0 ] = 0;
    else
        sprintf( inSizeName, "%ld", inVecSize );
    if( inVecSize == 3 )
        inSizeArgName[ 0 ] = 0;
    else
        strcpy( inSizeArgName, inSizeName );

    *outRealVecSize = outVecSize;

    if( outVecSize == 1 ||  (outVecSize == 3))
        outSizeName[ 0 ] = 0;
    else
        sprintf( outSizeName, "%d", (int)outVecSize );

    if(outVecSize == 1) {
        outSizeNameTmpVar[0] = 0;
    } else {
        sprintf(outSizeNameTmpVar, "%d", (int)outVecSize);
    }

    if( *outRealVecSize == 1 || ( *outRealVecSize == 3))
        outRealSizeName[ 0 ] = 0;
    else
        sprintf( outRealSizeName, "%d", (int)*outRealVecSize );


    // Loop through and create the source for all order strings
    kernelSource[ 0 ] = 0;
    if (vecType == kDouble) {
        strcat(kernelSource, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    }

    if( shuffleMode == kFunctionCallMode )
    {
        sprintf( progLine, shuffleFnLinePattern, get_explicit_type_name( vecType ), inSizeName, get_explicit_type_name( vecType ), inSizeName,
                get_explicit_type_name( vecType ), inSizeName, get_explicit_type_name( vecType ), inSizeName );
        strcat(kernelSource, progLine);
    }

    // We're going to play a REALLY NASTY trick here. We're going to use the inSize insert point
    // to put in an entire third parameter if we need it
    char inParamSizeString[ 1024 ];
    if( shuffleMode == kBuiltInDualInputFnMode )
        sprintf( inParamSizeString, "%s *secondSource, __global %s%s", inSizeArgName, get_explicit_type_name( vecType ), inSizeArgName );
    else
        strcpy( inParamSizeString, inSizeArgName );

    // These two take care of unused variable warnings
    const char * src2EnableA = ( shuffleMode == kBuiltInDualInputFnMode ) ? "" : "/*";
    const char * src2EnableB = ( shuffleMode == kBuiltInDualInputFnMode ) ? "" : "*/";

    sprintf( progLine, shuffleKernelPattern[ 0 ], get_explicit_type_name( vecType ), inParamSizeString,
            get_explicit_type_name( vecType ), outRealSizeName, get_explicit_type_name( vecType ), inSizeName,
            src2EnableA, src2EnableB );
    strcat(kernelSource, progLine);
    if( inOrders == NULL )
        strcpy( inOrder, get_order_string( sNaturalOrder, outVecSize, (cl_uint)outVecSize, inUseNumerics, d ) );

    sprintf( progLine, shuffleTempPattern, get_explicit_type_name( vecType ), outSizeNameTmpVar);
    strcat(kernelSource, progLine);

    for( unsigned int i = 0; i < numOrders; i++ )
    {
        if( inOrders != NULL )
            strcpy(inOrder,
                   get_order_string(inOrders[i], inVecSize, lengthToUse[i],
                                    inUseNumerics, d));
        strcpy( shuffledOrder, get_order_string( outOrders[ i ], outVecSize, lengthToUse[i], outUseNumerics, d ) );


        sprintf( progLine, clearTempPattern, get_explicit_type_name( vecType ), outSizeName,get_explicit_type_name( vecType ));
        strcat(kernelSource, progLine);


        if( shuffleMode == kNormalMode )
        {
            if(outVecSize == 3 && inVecSize == 3) {
                // shuffleSinglePatternV3srcV3dst
                sprintf( progLine, shuffleSinglePatternV3srcV3dst,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "", (int)i );
            } else if(inVecSize == 3) {
                // shuffleSinglePatternV3src
                sprintf( progLine, shuffleSinglePatternV3src,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "", (int)i );
            } else if(outVecSize == 3) {
                // shuffleSinglePatternV3dst
                sprintf( progLine, shuffleSinglePatternV3dst,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "",
                        (int)i );
            } else {
                sprintf( progLine, shuffleSinglePattern,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "", (int)i );
            }
        }
        else if( shuffleMode == kFunctionCallMode )
        {
            // log_info("About to make a shuffle line\n");
            // fflush(stdout);
            if(inVecSize == 3 && outVecSize == 3) { // swap last two
                sprintf( progLine, shuffleFnPatternV3srcV3dst,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "",
                        (int)i );
            } else if(outVecSize == 3)  { // swap last two
                                          // log_info("Here\n\n");
                                          // fflush(stdout);
                sprintf( progLine, shuffleFnPatternV3dst,
                        outVecSize > 1 ? "." : "",
                        outVecSize > 1 ? shuffledOrder : "",
                        (int)i,
                        inVecSize > 1 ? "." : "",
                        inVecSize > 1 ? inOrder : "",
                        (int)i );
                // log_info("\n%s\n", progLine);
                // fflush(stdout);
            } else if(inVecSize == 3) {
                sprintf( progLine, shuffleFnPatternV3src,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "", (int)i );
            } else  {
                sprintf( progLine, shuffleFnPattern,
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "", (int)i,
                        inVecSize > 1 ? "." : "", inVecSize > 1 ? inOrder : "", (int)i );
            }
        }
        else if( shuffleMode == kArrayAccessMode )
        { // now we want to replace inSizeName with inSizeNameShuffleFn
            int vectorSizeToCastTo = 16;
            cl_uint item;
            for (item =0; item<lengthToUse[i]; item++) {
                int absoluteIndex = i*(int)inVecSize+(int)inOrders[i][item];
                int castVectorIndex = absoluteIndex/vectorSizeToCastTo;
                size_t castElementIndex = absoluteIndex % vectorSizeToCastTo;
                ShuffleOrder myOutOrders, myInOrders;
                myOutOrders[0]  = outOrders[i][item];
                myInOrders[0] = castElementIndex;

                strcpy( inOrder, get_order_string( myInOrders, 1, 1, 0, d ) );
                strcpy( shuffledOrder, get_order_string( myOutOrders, 1, 1, 0, d ) );

                sprintf(progLine, "     tmp%s%s = ((__global %s%d *)source)[%d]%s%s;\n",
                        outVecSize > 1 ? "." : "", outVecSize > 1 ? shuffledOrder : "",
                        get_explicit_type_name( vecType ), vectorSizeToCastTo,
                        castVectorIndex,
                        vectorSizeToCastTo > 1 ? "." : "", vectorSizeToCastTo > 1 ? inOrder : "");
                strcat(kernelSource, progLine);
            }
            if(outVecSize == 3) {
                sprintf(progLine,"     vstore3(tmp, %d, (__global %s *)dest);\n",
                        i, get_explicit_type_name( vecType ));
                // probably don't need that last
                // cast to (__global %s *) where %s is get_explicit_type_name( vecType)
            } else {
                sprintf(progLine,"     dest[%d] = tmp;\n", i );
            }
        }
        else // shuffleMode == kBuiltInFnMode or kBuiltInDualInputFnMode
        {
            if(inVecSize == 3 || outVecSize == 3 ||
               inVecSize == 1 || outVecSize == 1) {
                // log_info("Skipping test for size 3\n");
                continue;
            }
            ExplicitType maskType = vecType;
            if( maskType == kFloat )
                maskType = kUInt;
            if( maskType == kDouble) {
                maskType = kULong;
            }

            char maskString[ 1024 ] = "";
            size_t maskSize = outVecSize;// ( shuffleMode == kBuiltInDualInputFnMode ) ? ( outVecSize << 1 ) : outVecSize;
            generate_shuffle_mask( maskString, maskSize, ( outOrders != NULL ) ? &outOrders[ i ] : NULL );

            // Set up a quick prefix, so mask gets unsigned type regardless of the input/output type
            char maskPrefix[ 2 ] = "u";
            if( get_explicit_type_name( maskType )[ 0 ] == 'u' )
                maskPrefix[ 0 ] = 0;

            char progLine2[ 10240 ];
            if( shuffleMode == kBuiltInDualInputFnMode )
            {
                sprintf( progLine2, shuffleBuiltInDualPattern, get_explicit_type_name( vecType ), inSizeName,
                        ( inVecSize == 3 ) ? "vload3( %ld, (__global %s *)source )" : "source[ %ld ]",
                        get_explicit_type_name( vecType ), inSizeName,
                        ( inVecSize == 3 ) ? "vload3( %ld, (__global %s *)secondSource )" : "secondSource[ %ld ]",
                        maskPrefix, get_explicit_type_name( maskType ), outSizeName, maskPrefix, get_explicit_type_name( maskType ), outSizeName,
                        maskString,
                        ( outVecSize == 3 ) ? "vstore3( tmp, %ld, (__global %s *)dest )" : "dest[ %ld ] = tmp" );

                if( outVecSize == 3 )
                {
                    if( inVecSize == 3 )
                        sprintf( progLine, progLine2, i, get_explicit_type_name( vecType ), i, get_explicit_type_name( vecType ), i, get_explicit_type_name( vecType ) );
                    else
                        sprintf( progLine, progLine2, i, i, i, get_explicit_type_name( vecType ) );
                }
                else
                {
                    if( inVecSize == 3 )
                        sprintf( progLine, progLine2, i, get_explicit_type_name( vecType ), i, get_explicit_type_name( vecType ), i );
                    else
                        sprintf( progLine, progLine2, i, i, i );
                }
            }
            else
            {
                sprintf( progLine2, shuffleBuiltInPattern, get_explicit_type_name( vecType ), inSizeName,
                        ( inVecSize == 3 ) ? "vload3( %ld, (__global %s *)source )" : "source[ %ld ]",
                        maskPrefix, get_explicit_type_name( maskType ), outSizeName, maskPrefix, get_explicit_type_name( maskType ), outSizeName,
                        maskString,
                        ( outVecSize == 3 ) ? "vstore3( tmp, %ld, (__global %s *)dest )" : "dest[ %ld ] = tmp" );

                if( outVecSize == 3 )
                {
                    if( inVecSize == 3 )
                        sprintf( progLine, progLine2, i, get_explicit_type_name( vecType ), i, get_explicit_type_name( vecType ) );
                    else
                        sprintf( progLine, progLine2, i, i, get_explicit_type_name( vecType ) );
                }
                else
                {
                    if( inVecSize == 3 )
                        sprintf( progLine, progLine2, i, get_explicit_type_name( vecType ), i );
                    else
                        sprintf( progLine, progLine2, i, i );
                }
            }
        }

        strcat( kernelSource, progLine );
        if (strlen(kernelSource) > 0.9*MAX_PROGRAM_SIZE)
            log_info("WARNING: Program has grown to 90%% (%d) of the defined max program size of %d\n", (int)strlen(kernelSource), (int)MAX_PROGRAM_SIZE);
    }
    strcat( kernelSource, shuffleKernelPattern[ 1 ] );

    // Print the kernel source
    if (PRINT_SHUFFLE_KERNEL_SOURCE)
        log_info( "Kernel:%s\n", kernelSource );

    /* Create kernel */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, outProgram, outKernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }
    return 0;
}

int test_shuffle_dual_kernel(cl_context context, cl_command_queue queue,
                             ExplicitType vecType, size_t inVecSize, size_t outVecSize, cl_uint *lengthToUse, size_t numOrders,
                             ShuffleOrder *inOrderIdx, ShuffleOrder *outOrderIdx, bool inUseNumerics, bool outUseNumerics, MTdata d,
                             ShuffleMode shuffleMode = kNormalMode )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    int error;
    size_t threads[1], localThreads[1];
    size_t typeSize, outRealVecSize;
    clMemWrapper streams[ 3 ];

    /* Create the source */
    error = create_shuffle_kernel( context, &program, &kernel, &outRealVecSize, vecType,
                                  inVecSize, outVecSize, lengthToUse, inUseNumerics, outUseNumerics, numOrders, inOrderIdx, outOrderIdx,
                                  d, shuffleMode );
    if( error != 0 )
        return error;

    typeSize = get_explicit_type_size(vecType);
    std::vector<cl_long> inData(inVecSize * numOrders);
    std::vector<cl_long> inSecondData(inVecSize * numOrders);
    std::vector<cl_long> outData(outRealVecSize * numOrders);

    generate_random_data(vecType, (unsigned int)(numOrders * inVecSize), d,
                         inData.data());
    if( shuffleMode == kBuiltInDualInputFnMode )
        generate_random_data(vecType, (unsigned int)(numOrders * inVecSize), d,
                             inSecondData.data());

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       typeSize * inVecSize * numOrders, inData.data(), &error);
    test_error( error, "Unable to create input stream" );

    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                typeSize * outRealVecSize * numOrders,
                                outData.data(), &error);
    test_error( error, "Unable to create output stream" );

    int argIndex = 0;
    if( shuffleMode == kBuiltInDualInputFnMode )
    {
        streams[2] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                    typeSize * inVecSize * numOrders,
                                    inSecondData.data(), &error);
        test_error( error, "Unable to create second input stream" );

        error = clSetKernelArg( kernel, argIndex++, sizeof( streams[ 2 ] ), &streams[ 2 ] );
        test_error( error, "Unable to set kernel argument" );
    }

    // Set kernel arguments
    error = clSetKernelArg( kernel, argIndex++, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, argIndex++, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel argument" );


    /* Run the kernel */
    threads[0] = numOrders;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );


    // Read the results back
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                typeSize * numOrders * outRealVecSize,
                                outData.data(), 0, NULL, NULL);
    test_error( error, "Unable to read results" );

    unsigned char *inDataPtr = (unsigned char *)inData.data();
    unsigned char *inSecondDataPtr = (unsigned char *)inSecondData.data();
    unsigned char *outDataPtr = (unsigned char *)outData.data();
    int ret = 0;
    int errors_printed = 0;
    for( size_t i = 0; i < numOrders; i++ )
    {
        unsigned char expected[ 1024 ];
        unsigned char temp[ 1024 ];
        memset(expected, 0, sizeof(expected));
        memset(temp, 0, sizeof(temp));
        if( shuffleMode == kBuiltInFnMode )
            shuffleVector( inDataPtr, expected, outOrderIdx[ i ], outVecSize, typeSize, lengthToUse[i] );
        else if( shuffleMode == kBuiltInDualInputFnMode )
            shuffleVectorDual( inDataPtr, inSecondDataPtr, expected, outOrderIdx[ i ], inVecSize, typeSize, lengthToUse[i] );
        else
        {
            shuffleVector( inDataPtr, temp, inOrderIdx[ i ], inVecSize, typeSize, lengthToUse[i] );
            shuffleVector2( temp, expected, outOrderIdx[ i ], outVecSize, typeSize, lengthToUse[i] );
        }

        if( memcmp( expected, outDataPtr, outVecSize * typeSize ) != 0 )
        {
            log_error( " ERROR: Shuffle test %d FAILED for %s (memory hex dump follows)\n", (int)i,
                      get_order_name( vecType, inVecSize, outVecSize, inOrderIdx[ i ], outOrderIdx[ i ], lengthToUse[i], d, inUseNumerics, outUseNumerics ) );

            print_hex_mem_dump( inDataPtr, ( shuffleMode == kBuiltInDualInputFnMode ) ? inSecondDataPtr : NULL, expected, outDataPtr, inVecSize, outVecSize, typeSize );

            if( ( shuffleMode == kBuiltInFnMode ) || ( shuffleMode == kBuiltInDualInputFnMode ) )
            {
                // Mask would've been different for every shuffle done, so we have to regen it to print it
                char maskString[ 1024 ];
                generate_shuffle_mask( maskString, outVecSize, ( outOrderIdx != NULL ) ? &outOrderIdx[ i ] : NULL );
                log_error( "        Mask:  %s\n", maskString );
            }

            ret++;
            errors_printed++;
            if (errors_printed > MAX_ERRORS_TO_PRINT)
            {
                log_info("Further errors suppressed.\n");
                return ret;
            }
        }
        inDataPtr += inVecSize * typeSize;
        inSecondDataPtr += inVecSize * typeSize;
        outDataPtr += outRealVecSize * typeSize;
    }

    return ret;
}

void    build_random_shuffle_order( ShuffleOrder &outIndices, unsigned int length, unsigned int selectLength, bool allowRepeats, MTdata d )
{
    char flags[ 16 ];

    memset( flags, 0, sizeof( flags ) );

    for( unsigned int i = 0; i < length; i++ )
    {
        char selector = (char)random_in_range( 0, selectLength - 1, d );
        if( !allowRepeats )
        {
            while( flags[ (int)selector ] )
                selector = (char)random_in_range( 0, selectLength - 1, d );
            flags[ (int)selector ] = true;
        }
        outIndices[ i ] = selector;
    }
}

class shuffleBuffer
{
public:

    shuffleBuffer( cl_context ctx, cl_command_queue queue, ExplicitType type, size_t inSize, size_t outSize, ShuffleMode mode )
    {
        mContext = ctx;
        mQueue = queue;
        mVecType = type;
        mInVecSize = inSize;
        mOutVecSize = outSize;
        mShuffleMode = mode;

        mCount = 0;

        // Here's the deal with mLengthToUse[i].
        // if you have, for instance
        // uchar4 dst;
        // uchar8 src;
        // you can do
        // src.s0213 = dst.s1045;
        // but you can also do
        // src.s02 = dst.s10;
        // which has a different effect
        // The intent with these "sub lengths" is to test all such
        // possibilities
        // Calculate a range of sub-lengths within the vector to copy.
        int i;
        size_t maxSize = (mInVecSize < mOutVecSize) ? mInVecSize : mOutVecSize;
        for(i=0; i<NUM_TESTS; i++)
        {
            // Built-in fns can't select sub-lengths (the mask must be the length of the dest vector).
            // Well, at least for these tests...
            if( ( mode == kBuiltInFnMode ) || ( mode == kBuiltInDualInputFnMode ) )
                mLengthToUse[i]    = (cl_int)mOutVecSize;
            else
            {
                mLengthToUse[i] = (cl_uint)(((double)i/NUM_TESTS)*maxSize) + 1;
                // Force the length to be a valid vector length.
                if( ( mLengthToUse[i] == 1 ) && ( mode != kBuiltInFnMode ) )
                    mLengthToUse[i] = 1;
                else if (mLengthToUse[i] < 4)
                    mLengthToUse[i] = 2;
                else if (mLengthToUse[i] < 8)
                    mLengthToUse[i] = 4;
                else if (mLengthToUse[i] < 16)
                    mLengthToUse[i] = 8;
                else
                    mLengthToUse[i] = 16;
            }
        }
    }

    int    AddRun( ShuffleOrder &inOrder, ShuffleOrder &outOrder, MTdata d )
    {
        memcpy( &mInOrders[ mCount ], &inOrder, sizeof( inOrder ) );
        memcpy( &mOutOrders[ mCount ], &outOrder, sizeof( outOrder ) );
        mCount++;

        if( mCount == NUM_TESTS )
            return Flush(d);

        return CL_SUCCESS;
    }

    int Flush( MTdata d )
    {
        int err = CL_SUCCESS;
        if( mCount > 0 )
        {
            err = test_shuffle_dual_kernel( mContext, mQueue, mVecType, mInVecSize, mOutVecSize, mLengthToUse,
                                           mCount, mInOrders, mOutOrders, true, true, d, mShuffleMode );
            mCount = 0;
        }
        return err;
    }

protected:
    cl_context            mContext;
    cl_command_queue    mQueue;
    ExplicitType        mVecType;
    size_t                mInVecSize, mOutVecSize, mCount;
    ShuffleMode            mShuffleMode;
    cl_uint             mLengthToUse[ NUM_TESTS ];

    ShuffleOrder        mInOrders[ NUM_TESTS ], mOutOrders[ NUM_TESTS ];
};


int test_shuffle_random(cl_device_id device, cl_context context, cl_command_queue queue, ShuffleMode shuffleMode, MTdata d )
{
    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int srcIdx, dstIdx, typeIndex;
    int error = 0, totalError = 0, prevTotalError = 0;
    RandomSeed seed(gRandomSeed);

    for( typeIndex = 0; typeIndex < 10; typeIndex++ )
    {
        //log_info( "\n\t%s... ", get_explicit_type_name( vecType[ typeIndex ] ) );
        //fflush( stdout );
        if (vecType[typeIndex] == kDouble) {
            if (!is_extension_available(device, "cl_khr_fp64")) {
                log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
                continue;
            }
            log_info("Testing doubles.\n");
        }

        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong) && !gHasLong )
        {
            log_info("Long types are unsupported, skipping.");
            continue;
        }

        error = 0;
        for( srcIdx = 0; vecSizes[ srcIdx ] != 0 /*&& error == 0*/; srcIdx++ )
        {
            for( dstIdx = 0; vecSizes[ dstIdx ] != 0 /*&& error == 0*/; dstIdx++ )
            {
                if( ( ( shuffleMode == kBuiltInDualInputFnMode ) || ( shuffleMode == kBuiltInFnMode ) ) &&
                   ( ( vecSizes[ dstIdx ] & 1 ) || ( vecSizes[ srcIdx ] & 1 ) ) )
                {
                    // Built-in shuffle functions don't work on size 1 (scalars) or size 3 (vec3s)
                    continue;
                }

                log_info("Testing [%s%d to %s%d]... ", get_explicit_type_name( vecType[ typeIndex ] ) , vecSizes[srcIdx], get_explicit_type_name( vecType[ typeIndex ] ) , vecSizes[dstIdx]);
                shuffleBuffer buffer( context, queue, vecType[ typeIndex ], vecSizes[ srcIdx ], vecSizes[ dstIdx ], shuffleMode );

                int numTests = NUM_TESTS*NUM_ITERATIONS_PER_TEST;
                for( int i = 0; i < numTests /*&& error == 0*/; i++ )
                {
                    ShuffleOrder src{ 0 };
                    ShuffleOrder dst;
                    if( shuffleMode == kBuiltInFnMode )
                    {
                        build_random_shuffle_order( dst, vecSizes[ dstIdx ], vecSizes[ srcIdx ], true, d );
                    }
                    else if(shuffleMode == kBuiltInDualInputFnMode)
                    {
                        build_random_shuffle_order(dst, vecSizes[dstIdx], 2*vecSizes[srcIdx], true, d);
                    }
                    else
                    {
                        build_random_shuffle_order( src, vecSizes[ dstIdx ], vecSizes[ srcIdx ], true, d );
                        build_random_shuffle_order( dst, vecSizes[ dstIdx ], vecSizes[ dstIdx ], false, d );
                    }

                    error = buffer.AddRun( src, dst, seed );
                    if (error)
                        totalError++;
                }
                int test_error = buffer.Flush(seed);
                if (test_error)
                    totalError++;

                if (totalError == prevTotalError)
                    log_info("\tPassed.\n");
                else
                {
                    log_error("\tFAILED.\n");
                    prevTotalError = totalError;
                }
            }
        }
    }
    return totalError;
}

int test_shuffle_copy(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    RandomSeed seed(gRandomSeed);
    return test_shuffle_random( device, context, queue, kNormalMode, seed );
}

int test_shuffle_function_call(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    RandomSeed seed(gRandomSeed);
    return test_shuffle_random( device, context, queue, kFunctionCallMode, seed );
}

int test_shuffle_array_cast(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    RandomSeed seed(gRandomSeed);
    return test_shuffle_random( device, context, queue, kArrayAccessMode, seed );
}

int test_shuffle_built_in(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    RandomSeed seed(gRandomSeed);
    return test_shuffle_random( device, context, queue, kBuiltInFnMode, seed );
}

int test_shuffle_built_in_dual_input(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    RandomSeed seed(gRandomSeed);
    return test_shuffle_random( device, context, queue, kBuiltInDualInputFnMode, seed );
}


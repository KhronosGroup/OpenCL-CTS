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
#ifndef BASIC_TEST_CONVERSIONS_H
#define BASIC_TEST_CONVERSIONS_H

#include "harness/compat.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "harness/errorHelpers.h"
#include "harness/rounding_mode.h"

#include <stdio.h>
#if defined( __APPLE__ )
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "harness/mt19937.h"
#include "harness/testHarness.h"

#include <tuple>
#include <vector>
#include <memory>

typedef void (*Convert)( void *dest, void *src, size_t );

#define kVectorSizeCount 6
#define kMaxVectorSize 16
#define kPageSize 4096

#define BUFFER_SIZE (1024 * 1024)
#define EMBEDDED_REDUCTION_FACTOR 16
#define PERF_LOOP_COUNT 100

typedef enum
{
    kUnsaturated = 0,
    kSaturated,

    kSaturationModeCount
}SaturationMode;

// extern Convert gConversions[kTypeCount][kTypeCount];                // [dest
// format][source format] extern Convert
// gSaturatedConversions[kTypeCount][kTypeCount];       // [dest format][source
// format]
extern const char *gTypeNames[ kTypeCount ];
extern const char *gRoundingModeNames[ kRoundingModeCount ];        // { "", "_rte", "_rtp", "_rtn", "_rtz" }
extern const char *gSaturationNames[ kSaturationModeCount ];        // { "", "_sat" }
extern const char *gVectorSizeNames[kVectorSizeCount];              // { "", "2", "4", "8", "16" }
extern size_t gTypeSizes[ kTypeCount ];
extern int gIsEmbedded;

//Functions for clamping floating point numbers into the representable range for the type
typedef float (*clampf)( float );
typedef double (*clampd)( double );

extern clampf gClampFloat[ kTypeCount ][kRoundingModeCount];
extern clampd gClampDouble[ kTypeCount ][kRoundingModeCount];

typedef void (*InitDataFunc)( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
extern InitDataFunc gInitFunctions[ kTypeCount ];

typedef int (*CheckResults)( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
extern CheckResults gCheckResults[ kTypeCount ];

#define kCallStyleCount (kVectorSizeCount + 1 /* for implicit scalar */)

extern MTdata gMTdata;
extern cl_command_queue gQueue;
extern cl_context gContext;
extern cl_mem gInBuffer;
extern cl_mem gOutBuffers[];
extern int gHasDouble;
extern int gTestDouble;
extern int gWimpyMode;
extern int gWimpyReductionFactor;
extern int gSkipTesting;
extern int gMinVectorSize;
extern int gMaxVectorSize;
extern int gForceFTZ;
extern int gTimeResults;
extern int gReportAverageTimes;
extern int gStartTestNumber;
extern int gEndTestNumber;
extern int gIsRTZ;
extern void *gIn;
extern void *gRef;
extern void *gAllowZ;
extern void *gOut[];

extern const char **argList;
extern int argCount;

extern const char *sizeNames[];
extern int vectorSizes[];

extern size_t gComputeDevices;
extern uint32_t gDeviceFrequency;

////////////////////////////////////////////////////////////////////////////////

struct CalcReferenceValuesInfo
{
    // pointer back to the parent WriteInputBufferInfo struct
    struct WriteInputBufferInfo *parent;
    cl_kernel kernel; // the kernel for this vector size
    cl_program program; // the program for this vector size
    cl_uint vectorSize; // the vector size for this callback chain
    void *p; // the pointer to mapped result data for this vector size
    cl_int result;
};

struct CalcRefValsBase : CalcReferenceValuesInfo
{
    virtual int check_result(void *, uint32_t, int) { return 0; }
};

template <typename InType, typename OutType>
struct CalcRefValsPat : CalcRefValsBase
{
    int check_result(void *, uint32_t, int) override;
};

struct WriteInputBufferInfo
{
    WriteInputBufferInfo()
        : calcReferenceValues(nullptr), doneBarrier(nullptr), count(0),
          outType(kuchar), inType(kuchar), barrierCount(0)
    {}

    volatile cl_event
        calcReferenceValues; // user event which signals when main thread is
                             // done calculating reference values
    volatile cl_event
        doneBarrier; // user event which signals when worker threads are done
    cl_uint count; // the number of elements in the array
    Type outType; // the data type of the conversion result
    Type inType; // the data type of the conversion input
    volatile int barrierCount;

    std::vector<std::unique_ptr<CalcRefValsBase>> calcInfo;
};

//--------------------------------------------------------------------------

#pragma pack(push)
#pragma pack(1)
struct DataInitInfo
{
    cl_ulong start;
    cl_uint size;
    Type outType;
    Type inType;
    SaturationMode sat;
    RoundingMode round;
    MTdata *d;
};

//--------------------------------------------------------------------------

struct DataInitBase : public DataInitInfo
{
    DataInitBase(const DataInitInfo &agg): DataInitInfo(agg) {}
    virtual void conv_array(void *out, void *in, size_t n) {}
    virtual void conv_array_sat(void *out, void *in, size_t n) {}
    virtual void init(const cl_uint &, const cl_uint &) {}
};

//--------------------------------------------------------------------------

template <typename InType, typename OutType>
struct DataInfoSpec : public DataInitBase
{

    DataInfoSpec(const DataInitInfo &agg);

    // helpers
    float round_to_int(float f);
    long long round_to_int_and_clamp(double d);

    OutType absolute(const OutType &x);

    // actual conversion of reference values
    void conv(OutType *out, InType *in);
    void conv_sat(OutType *out, InType *in);

    // min/max ranges for output type of data
    std::pair<OutType, OutType> ranges;

    // matrix of clamping ranges for each rounding type
    std::vector<std::pair<InType, InType>> clamp_ranges;

    ////////////////////////////////////////////////////////////////////////////
    void conv_array(void *out, void *in, size_t n) override
    {
        for (size_t i = 0; i < n; i++)
            conv(&((OutType *)out)[i], &((InType *)in)[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    void conv_array_sat(void *out, void *in, size_t n) override
    {
        for (size_t i = 0; i < n; i++)
            conv_sat(&((OutType *)out)[i], &((InType *)in)[i]);
    }

    ////////////////////////////////////////////////////////////////////////////
    void init(const cl_uint &, const cl_uint &) override;
    InType clamp(const InType &);
};
#pragma pack(pop)

//--------------------------------------------------------------------------

//    kuchar = 0,
//    kchar = 1,
//    kushort = 2,
//    kshort = 3,
//    kuint = 4,
//    kint = 5,
//    kfloat = 6,
//    kdouble = 7,
//    kulong = 8,
//    klong = 9,

using TypeIter = std::tuple<cl_uchar, cl_char, cl_ushort, cl_short, cl_uint,
                            cl_int, cl_float, cl_double, cl_ulong, cl_long>;

//--------------------------------------------------------------------------

// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct ConversionsTest
{
    ConversionsTest(cl_device_id device, cl_context context,
                    cl_command_queue queue);

    virtual cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    virtual cl_int Run();

    template <typename InType, typename OutType>
    int DoTest(Type outType, Type inType, SaturationMode sat,
               RoundingMode round, MTdata d);

    template <typename InType, typename OutType>
    cl_int TestTypesConversion(const Type &inType, const Type &outType,
                               int &tn);

protected:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    size_t num_elements;

    TypeIter typeIterator;
};

//--------------------------------------------------------------------------

struct CustomConversionsTest : ConversionsTest
{
    CustomConversionsTest(cl_device_id device, cl_context context,
                          cl_command_queue queue)
        : ConversionsTest(device, context, queue)
    {}

    cl_int Run() override;
};

//--------------------------------------------------------------------------

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    auto test_fixture = T(device, context, queue);

    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    error = test_fixture.Run();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}


#endif /* BASIC_TEST_CONVERSIONS_H */


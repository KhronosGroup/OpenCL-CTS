//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "harness/typeWrappers.h"

#include <memory>
#include <tuple>
#include <vector>

#include "conversions_data_info.h"

#define kVectorSizeCount 6
#define kMaxVectorSize 16
#define kPageSize 4096

#define BUFFER_SIZE (1024 * 1024)
#define EMBEDDED_REDUCTION_FACTOR 16
#define PERF_LOOP_COUNT 100

extern const char *gTypeNames[ kTypeCount ];
extern const char *gRoundingModeNames[ kRoundingModeCount ];        // { "", "_rte", "_rtp", "_rtn", "_rtz" }
extern const char *gSaturationNames[ kSaturationModeCount ];        // { "", "_sat" }
extern const char *gVectorSizeNames[kVectorSizeCount];              // { "", "2", "4", "8", "16" }
extern size_t gTypeSizes[ kTypeCount ];

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

namespace conv_test {

cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                       RoundingMode round, int vectorSize,
                       cl_kernel *outKernel);

int RunKernel(cl_kernel kernel, void *inBuf, void *outBuf, size_t blockCount);

int GetTestCase(const char *name, Type *outType, Type *inType,
                SaturationMode *sat, RoundingMode *round);

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p);
cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p);
uint64_t GetTime(void);

void WriteInputBufferComplete(void *);
void *FlushToZero(void);
void UnFlushToZero(void *);
}

struct CalcRefValsBase
{
    virtual ~CalcRefValsBase() = default;
    virtual int check_result(void *, uint32_t, int) { return 0; }

    // pointer back to the parent WriteInputBufferInfo struct
    struct WriteInputBufferInfo *parent;
    clKernelWrapper kernel; // the kernel for this vector size
    clProgramWrapper program; // the program for this vector size
    cl_uint vectorSize; // the vector size for this callback chain
    void *p; // the pointer to mapped result data for this vector size
    cl_int result;
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

// Must be aligned with Type enums!
using TypeIter = std::tuple<cl_uchar, cl_char, cl_ushort, cl_short, cl_uint,
                            cl_int, cl_float, cl_double, cl_ulong, cl_long>;

// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct ConversionsTest
{
    virtual ~ConversionsTest() = default;

    ConversionsTest(cl_device_id device, cl_context context,
                    cl_command_queue queue);

    cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    cl_int Run();

    template <typename InType, typename OutType>
    int DoTest(Type outType, Type inType, SaturationMode sat,
               RoundingMode round);

    template <typename InType, typename OutType>
    void TestTypesConversion(const Type &inType, const Type &outType, int &tn,
                             const int smvs);

protected:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    size_t num_elements;

    TypeIter typeIterator;
};

struct CustomConversionsTest : ConversionsTest
{
    CustomConversionsTest(cl_device_id device, cl_context context,
                          cl_command_queue queue)
        : ConversionsTest(device, context, queue)
    {}

    cl_int Run();
};

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    T test_fixture(device, context, queue);

    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    return test_fixture.Run();
}

struct TestType
{
    template <typename T> bool testType(Type in)
    {
        switch (in)
        {
            default: return false;
            case kuchar: return std::is_same<cl_uchar, T>::value;
            case kchar: return std::is_same<cl_char, T>::value;
            case kushort: return std::is_same<cl_ushort, T>::value;
            case kshort: return std::is_same<cl_short, T>::value;
            case kuint: return std::is_same<cl_uint, T>::value;
            case kint: return std::is_same<cl_int, T>::value;
            case kfloat: return std::is_same<cl_float, T>::value;
            case kdouble: return std::is_same<cl_double, T>::value;
            case kulong: return std::is_same<cl_ulong, T>::value;
            case klong: return std::is_same<cl_long, T>::value;
        }
    }
};

// Helper structures to iterate over all tuple attributes of different types
struct IterOverTypes : public TestType
{
    IterOverTypes(const TypeIter &typeIter, ConversionsTest &test)
        : inType((Type)0), outType((Type)0), typeIter(typeIter), test(test),
          testNumber(-1), startMinVectorSize(gMinVectorSize)
    {}

    void Run() { for_each_out_elem(typeIter); }

protected:
    template <std::size_t Out = 0, typename OutType>
    void iterate_out_type(const OutType &t)
    {
        for_each_in_elem<0, Out, OutType>(typeIter);
        outType = (Type)(outType + 1);
        inType = (Type)0;
    }

    template <std::size_t In, std::size_t Out, typename OutType,
              typename InType>
    void iterate_in_type(const InType &t)
    {
        if (!testType<InType>(inType)) vlog_error("Unexpected data type!\n");

        if (!testType<OutType>(outType)) vlog_error("Unexpected data type!\n");

        // run the conversions
        test.TestTypesConversion<InType, OutType>(inType, outType, testNumber,
                                                  startMinVectorSize);
        inType = (Type)(inType + 1);
    }

    template <std::size_t Out = 0, typename... Tp>
    inline typename std::enable_if<Out == sizeof...(Tp), void>::type
    for_each_out_elem(
        const std::tuple<Tp...> &) // Unused arguments are given no names.
    {}

    template <std::size_t Out = 0, typename... Tp>
        inline typename std::enable_if < Out<sizeof...(Tp), void>::type
        for_each_out_elem(const std::tuple<Tp...> &t)
    {
        iterate_out_type<Out>(std::get<Out>(t));
        for_each_out_elem<Out + 1, Tp...>(t);
    }

    template <std::size_t In = 0, std::size_t Out, typename OutType,
              typename... Tp>
    inline typename std::enable_if<In == sizeof...(Tp), void>::type
    for_each_in_elem(
        const std::tuple<Tp...> &) // Unused arguments are given no names.
    {}

    template <std::size_t In = 0, std::size_t Out, typename OutType,
              typename... Tp>
        inline typename std::enable_if < In<sizeof...(Tp), void>::type
        for_each_in_elem(const std::tuple<Tp...> &t)
    {
        iterate_in_type<In, Out, OutType>(std::get<In>(t));
        for_each_in_elem<In + 1, Out, OutType, Tp...>(t);
    }

protected:
    Type inType;
    Type outType;
    const TypeIter &typeIter;
    ConversionsTest &test;
    int testNumber;
    int startMinVectorSize;
};


// Helper structures to select type 2 type conversion test case
struct IterOverSelectedTypes : public TestType
{
    IterOverSelectedTypes(const TypeIter &typeIter, ConversionsTest &test,
                          const Type in, const Type out,
                          const RoundingMode round, const SaturationMode sat)
        : inType(in), outType(out), rounding(round), saturation(sat),
          typeIter(typeIter), test(test), testNumber(-1),
          startMinVectorSize(gMinVectorSize)
    {}

    void Run() { for_each_out_elem(typeIter); }

protected:
    template <std::size_t Out = 0, typename OutType>
    void iterate_out_type(const OutType &t)
    {
        for_each_in_elem<0, Out, OutType>(typeIter);
    }

    template <std::size_t In, std::size_t Out, typename OutType,
              typename InType>
    void iterate_in_type(const InType &t)
    {
        if (testType<InType>(inType) && testType<OutType>(outType))
        {
            // run selected conversion
            // testing of the result will happen afterwards
            test.DoTest<InType, OutType>(outType, inType, saturation, rounding);
        }
    }

    template <std::size_t Out = 0, typename... Tp>
    inline typename std::enable_if<Out == sizeof...(Tp), void>::type
    for_each_out_elem(const std::tuple<Tp...> &)
    {}

    template <std::size_t Out = 0, typename... Tp>
        inline typename std::enable_if < Out<sizeof...(Tp), void>::type
        for_each_out_elem(const std::tuple<Tp...> &t)
    {
        iterate_out_type<Out>(std::get<Out>(t));
        for_each_out_elem<Out + 1, Tp...>(t);
    }

    template <std::size_t In = 0, std::size_t Out, typename OutType,
              typename... Tp>
    inline typename std::enable_if<In == sizeof...(Tp), void>::type
    for_each_in_elem(const std::tuple<Tp...> &)
    {}

    template <std::size_t In = 0, std::size_t Out, typename OutType,
              typename... Tp>
        inline typename std::enable_if < In<sizeof...(Tp), void>::type
        for_each_in_elem(const std::tuple<Tp...> &t)
    {
        iterate_in_type<In, Out, OutType>(std::get<In>(t));
        for_each_in_elem<In + 1, Out, OutType, Tp...>(t);
    }

protected:
    Type inType;
    Type outType;
    RoundingMode rounding;
    SaturationMode saturation;

    const TypeIter &typeIter;
    ConversionsTest &test;
    int testNumber;
    int startMinVectorSize;
};


#endif /* BASIC_TEST_CONVERSIONS_H */


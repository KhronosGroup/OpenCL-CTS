//
// Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef _TEST_GEOMETRICS_FP_H
#define _TEST_GEOMETRICS_FP_H

#include <vector>
#include <map>
#include <memory>
#include <CL/cl_half.h>

#include "testBase.h"

#define HALF_P_NAN 0x7e00
#define HALF_N_NAN 0xfe00
#define HALF_P_INF 0x7c00
#define HALF_N_INF 0xfc00

struct GeometricsFPTest;

//--------------------------------------------------------------------------/

using half = cl_half;

//--------------------------------------------------------------------------

struct GeomTestBase
{
    GeomTestBase(const ExplicitTypes &dt, const std::string &name,
                 const float &ulp)
        : dataType(dt), fnName(name), ulpLimit(ulp)
    {}

    ExplicitTypes dataType;
    std::string fnName;
    float ulpLimit;

    static cl_half_rounding_mode halfRoundingMode;
};

//--------------------------------------------------------------------------

template <typename T> struct GeomTestParams : public GeomTestBase
{
    GeomTestParams(const ExplicitTypes &dt, const std::string &name,
                   const float &ulp);

    std::vector<T> trickyValues;
};

//--------------------------------------------------------------------------

template <typename T>
using TwoArgsVerifyFunc = double (*)(const T *, const T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct TwoArgsTestParams : public GeomTestParams<T>
{
    TwoArgsTestParams(const TwoArgsVerifyFunc<T> &fn, const ExplicitTypes &dt,
                      const std::string &name, const float &ulp)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn)
    {}

    TwoArgsVerifyFunc<T> verifyFunc;
};

//--------------------------------------------------------------------------

template <typename T> using OneArgVerifyFunc = double (*)(const T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct OneArgTestParams : public GeomTestParams<T>
{
    OneArgTestParams(const OneArgVerifyFunc<T> &fn, const ExplicitTypes &dt,
                     const std::string &name, const float &ulp, const float &um)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn), ulpMult(um)
    {}

    OneArgVerifyFunc<T> verifyFunc;
    float ulpMult;
};

//--------------------------------------------------------------------------

template <typename T>
using OneToOneArgVerifyFunc = void (*)(const T *, T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct OneToOneTestParams : public GeomTestParams<T>
{
    OneToOneTestParams(const OneToOneArgVerifyFunc<T> &fn,
                       const ExplicitTypes &dt, const std::string &name,
                       const float &ulp)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn)
    {}

    OneToOneArgVerifyFunc<T> verifyFunc;
};

//--------------------------------------------------------------------------

// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct GeometricsFPTest
{
    GeometricsFPTest(cl_device_id device, cl_context context,
                     cl_command_queue queue);

    virtual cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    virtual cl_int Run();

    virtual cl_int RunSingleTest(const GeomTestBase *p) = 0;

    template <typename T>
    void FillWithTrickyNums(T *const, T *const, const size_t, const size_t,
                            const MTdata &, const GeomTestParams<T> &);

    template <typename T> float UlpError(const T &, const double &);

    template <typename T> double ToDouble(const T &);

    cl_int VerifyTestSize(size_t &test_size, const size_t &max_alloc,
                          const size_t &total_buf_size);

protected:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    cl_device_fp_config floatRoundingMode;
    cl_device_fp_config halfRoundingMode;

    bool floatHasInfNan;
    bool halfHasInfNan;

    int halfFlushDenormsToZero;

    size_t num_elements;

    std::vector<std::unique_ptr<GeomTestBase>> params;

    static cl_ulong maxAllocSize;
    static cl_ulong maxGlobalMemSize;
};

//--------------------------------------------------------------------------

struct CrossFPTest : public GeometricsFPTest
{
    CrossFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}
    cl_int SetUp(int elements) override;
    cl_int RunSingleTest(const GeomTestBase *p) override;

    template <typename T> int CrossKernel(const GeomTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct TwoArgsFPTest : public GeometricsFPTest
{
    TwoArgsFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}
    cl_int RunSingleTest(const GeomTestBase *p) override;

    template <typename T> int TwoArgs(const TwoArgsTestParams<T> &p);

    template <typename T>
    T GetMaxValue(const T *const, const T *const, const size_t &,
                  const TwoArgsTestParams<T> &);

    template <typename T>
    int TwoArgsKernel(const size_t &, const MTdata &,
                      const TwoArgsTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct DotProdFPTest : public TwoArgsFPTest
{
    DotProdFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoArgsFPTest(d, c, q)
    {}
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

struct FastDistanceFPTest : public TwoArgsFPTest
{
    FastDistanceFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoArgsFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

struct DistanceFPTest : public TwoArgsFPTest
{
    DistanceFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoArgsFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int DistTest(TwoArgsTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct OneArgFPTest : public GeometricsFPTest
{
    OneArgFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}

    template <typename T>
    int OneArgKernel(const size_t &, const MTdata &,
                     const OneArgTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct LengthFPTest : public OneArgFPTest
{
    LengthFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : OneArgFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int LenghtTest(OneArgTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct FastLengthFPTest : public OneArgFPTest
{
    FastLengthFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : OneArgFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

struct OneToOneArgFPTest : public GeometricsFPTest
{
    OneToOneArgFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}

    template <typename T>
    int OneToOneArgKernel(const size_t &, const MTdata &,
                          const OneToOneTestParams<T> &p);

    template <typename T>
    int VerifySubnormals(int &fail, const size_t &vecsize, const T *const inA,
                         T *const out, T *const expected,
                         const OneToOneTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct NormalizeFPTest : public OneToOneArgFPTest
{
    NormalizeFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : OneToOneArgFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int NormalizeTest(OneToOneTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct FastNormalizeFPTest : public OneToOneArgFPTest
{
    FastNormalizeFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : OneToOneArgFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
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

#endif // _TEST_GEOMETRICS_FP_H

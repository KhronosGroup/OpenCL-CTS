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

#ifndef TEST_COMPARISONS_FP_H
#define TEST_COMPARISONS_FP_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <CL/cl_half.h>

#include "testBase.h"

#define HALF_NAN 0x7e00
template <typename T> using VerifyFunc = bool (*)(const T &, const T &);

struct RelTestBase
{
    explicit RelTestBase(const ExplicitTypes &dt): dataType(dt) {}
    virtual ~RelTestBase() = default;
    ExplicitTypes dataType;
};

template <typename T> struct RelTestParams : public RelTestBase
{
    RelTestParams(const VerifyFunc<T> &vfn, const ExplicitTypes &dt,
                  const T &nan_)
        : RelTestBase(dt), verifyFn(vfn), nan(nan_)
    {}

    VerifyFunc<T> verifyFn;
    T nan;
};

struct RelationalsFPTest
{
    RelationalsFPTest(cl_context context, cl_device_id device,
                      cl_command_queue queue, const char *fn, const char *op);

    virtual cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    virtual cl_int Run();

    template <typename T>
    void generate_equiv_test_data(T *, unsigned int, bool,
                                  const RelTestParams<T> &, const MTdata &);

    template <typename T, typename U>
    void verify_equiv_values(unsigned int, const T *const, const T *const,
                             U *const, const VerifyFunc<T> &);

    template <typename T>
    int test_equiv_kernel(unsigned int vecSize, const RelTestParams<T> &param,
                          const MTdata &d);

    template <typename T>
    int test_relational(int numElements, const RelTestParams<T> &param);

protected:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    std::string fnName;
    std::string opName;

    std::vector<std::unique_ptr<RelTestBase>> params;
    std::map<ExplicitTypes, std::string> eqTypeNames;
    size_t num_elements;

    int halfFlushDenormsToZero;
};

struct IsEqualFPTest : public RelationalsFPTest
{
    IsEqualFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "isequal", "==")
    {}
    cl_int SetUp(int elements) override;

    // for correct handling nan/inf we need fp value
    struct half_equals_to
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) == cl_half_to_float(rhs);
        }
    };
};

struct IsNotEqualFPTest : public RelationalsFPTest
{
    IsNotEqualFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "isnotequal", "!=")
    {}
    cl_int SetUp(int elements) override;

    // for correct handling nan/inf we need fp value
    struct half_not_equals_to
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) != cl_half_to_float(rhs);
        }
    };
};

struct IsGreaterFPTest : public RelationalsFPTest
{
    IsGreaterFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "isgreater", ">")
    {}
    cl_int SetUp(int elements) override;

    struct half_greater
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) > cl_half_to_float(rhs);
        }
    };
};

struct IsGreaterEqualFPTest : public RelationalsFPTest
{
    IsGreaterEqualFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "isgreaterequal", ">=")
    {}
    cl_int SetUp(int elements) override;

    struct half_greater_equal
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) >= cl_half_to_float(rhs);
        }
    };
};

struct IsLessFPTest : public RelationalsFPTest
{
    IsLessFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "isless", "<")
    {}
    cl_int SetUp(int elements) override;

    struct half_less
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) < cl_half_to_float(rhs);
        }
    };
};

struct IsLessEqualFPTest : public RelationalsFPTest
{
    IsLessEqualFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "islessequal", "<=")
    {}
    cl_int SetUp(int elements) override;

    struct half_less_equal
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            return cl_half_to_float(lhs) <= cl_half_to_float(rhs);
        }
    };
};

struct IsLessGreaterFPTest : public RelationalsFPTest
{
    IsLessGreaterFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : RelationalsFPTest(c, d, q, "islessgreater", "<>")
    {}
    cl_int SetUp(int elements) override;

    struct half_less_greater
    {
        bool operator()(const cl_half &lhs, const cl_half &rhs) const
        {
            float flhs = cl_half_to_float(lhs), frhs = cl_half_to_float(rhs);
            return (flhs < frhs) || (flhs > frhs);
        }
    };

    template <typename T> struct less_greater
    {
        bool operator()(const T &lhs, const T &rhs) const
        {
            return (lhs < rhs) || (lhs > rhs);
        }
    };
};

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

#endif // TEST_COMPARISONS_FP_H

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

#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "harness/stringHelpers.h"

#include <CL/cl_half.h>

#include "test_comparisons_fp.h"

#define TEST_SIZE 512

static char ftype[32] = { 0 };
static char ftype_vec[32] = { 0 };
static char itype[32] = { 0 };
static char itype_vec[32] = { 0 };
static char extension[128] = { 0 };

// clang-format off
// for readability sake keep this section unformatted
const char* equivTestKernPat[] = {
extension,
"__kernel void sample_test(__global ", ftype_vec, " *sourceA, __global ", ftype_vec,
" *sourceB, __global ", itype_vec, " *destValues, __global ", itype_vec, " *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"    destValuesB[tid] = sourceA[tid] %s sourceB[tid];\n"
"}\n"};

const char* equivTestKernPatLessGreater[] = {
extension,
"__kernel void sample_test(__global ", ftype_vec, " *sourceA, __global ", ftype_vec,
" *sourceB, __global ", itype_vec, " *destValues, __global ", itype_vec, " *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"    destValuesB[tid] = (sourceA[tid] < sourceB[tid]) | (sourceA[tid] > sourceB[tid]);\n"
"}\n"};

const char* equivTestKerPat_3[] = {
extension,
"__kernel void sample_test(__global ", ftype_vec, " *sourceA, __global ", ftype_vec,
" *sourceB, __global ", itype_vec, " *destValues, __global ", itype_vec, " *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    ",ftype_vec," sampA = vload3(tid, (__global ",ftype," *)sourceA);\n"
"    ",ftype_vec," sampB = vload3(tid, (__global ",ftype," *)sourceB);\n"
"    vstore3(%s( sampA, sampB ), tid, (__global ",itype," *)destValues);\n"
"    vstore3(( sampA %s sampB ), tid, (__global ",itype," *)destValuesB);\n"
"}\n"};

const char* equivTestKerPatLessGreater_3[] = {
extension,
"__kernel void sample_test(__global ", ftype_vec, " *sourceA, __global ", ftype_vec,
" *sourceB, __global ", itype_vec, " *destValues, __global ", itype_vec, " *destValuesB)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    ", ftype_vec, " sampA = vload3(tid, (__global ", ftype, " *)sourceA);\n"
"    ", ftype_vec, " sampB = vload3(tid, (__global ", ftype, " *)sourceB);\n"
"    vstore3(%s( sampA, sampB ), tid, (__global ", itype, " *)destValues);\n"
"    vstore3(( sampA < sampB ) | (sampA > sampB), tid, (__global ", itype, " *)destValuesB);\n"
"}\n"
};
// clang-format on


template <typename T, typename F> bool verify(const T& A, const T& B)
{
    return F()(A, B);
}

RelationalsFPTest::RelationalsFPTest(cl_context context, cl_device_id device,
                                     cl_command_queue queue, const char* fn,
                                     const char* op)
    : context(context), device(device), queue(queue), fnName(fn), opName(op),
      halfFlushDenormsToZero(0)
{
    // hardcoded for now, to be changed into typeid().name solution in future
    // for now C++ spec doesn't guarantee human readable type name

    eqTypeNames = { { kHalf, "short" },
                    { kFloat, "int" },
                    { kDouble, "long" } };
}

template <typename T>
void RelationalsFPTest::generate_equiv_test_data(T* outData,
                                                 unsigned int vecSize,
                                                 bool alpha,
                                                 const RelTestParams<T>& param,
                                                 const MTdata& d)
{
    unsigned int i;

    generate_random_data(param.dataType, vecSize * TEST_SIZE, d, outData);

    // Fill the first few vectors with NAN in each vector element (or the second
    // set if we're alpha, so we can test either case)
    if (alpha) outData += vecSize * vecSize;
    for (i = 0; i < vecSize; i++)
    {
        outData[0] = param.nan;
        outData += vecSize + 1;
    }
    // Make sure the third set is filled regardless, to test the case where both
    // have NANs
    if (!alpha) outData += vecSize * vecSize;
    for (i = 0; i < vecSize; i++)
    {
        outData[0] = param.nan;
        outData += vecSize + 1;
    }
}

template <typename T, typename U>
void RelationalsFPTest::verify_equiv_values(unsigned int vecSize,
                                            const T* const inDataA,
                                            const T* const inDataB,
                                            U* const outData,
                                            const VerifyFunc<T>& verifyFn)
{
    unsigned int i;
    int trueResult;
    bool result;

    trueResult = (vecSize == 1) ? 1 : -1;
    for (i = 0; i < vecSize; i++)
    {
        result = verifyFn(inDataA[i], inDataB[i]);
        outData[i] = result ? trueResult : 0;
    }
}

template <typename T>
int RelationalsFPTest::test_equiv_kernel(unsigned int vecSize,
                                         const RelTestParams<T>& param,
                                         const MTdata& d)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[4];
    T inDataA[TEST_SIZE * 16], inDataB[TEST_SIZE * 16];

    // support half, float, double equivalents - otherwise assert
    typedef typename std::conditional<
        (sizeof(T) == sizeof(std::int16_t)), std::int16_t,
        typename std::conditional<(sizeof(T) == sizeof(std::int32_t)),
                                  std::int32_t, std::int64_t>::type>::type U;

    U outData[TEST_SIZE * 16], expected[16];
    int error, i, j;
    size_t threads[1], localThreads[1];
    std::string kernelSource;
    char sizeName[4];

    /* Create the source */
    if (vecSize == 1)
        sizeName[0] = 0;
    else
        sprintf(sizeName, "%d", vecSize);

    if (eqTypeNames.find(param.dataType) == eqTypeNames.end())
        log_error(
            "RelationalsFPTest::test_equiv_kernel: unsupported fp data type");

    sprintf(ftype, "%s", get_explicit_type_name(param.dataType));
    sprintf(ftype_vec, "%s%s", get_explicit_type_name(param.dataType),
            sizeName);

    sprintf(itype, "%s", eqTypeNames[param.dataType].c_str());
    sprintf(itype_vec, "%s%s", eqTypeNames[param.dataType].c_str(), sizeName);

    if (std::is_same<T, double>::value)
        strcpy(extension, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    else if (std::is_same<T, cl_half>::value)
        strcpy(extension, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
    else
        extension[0] = '\0';

    if (DENSE_PACK_VECS && vecSize == 3)
    {
        if (strcmp(fnName.c_str(), "islessgreater"))
        {
            auto str =
                concat_kernel(equivTestKerPat_3,
                              sizeof(equivTestKerPat_3) / sizeof(const char*));
            kernelSource = str_sprintf(str, fnName.c_str(), opName.c_str());
        }
        else
        {
            auto str = concat_kernel(equivTestKerPatLessGreater_3,
                                     sizeof(equivTestKerPatLessGreater_3)
                                         / sizeof(const char*));
            kernelSource = str_sprintf(str, fnName.c_str());
        }
    }
    else
    {
        if (strcmp(fnName.c_str(), "islessgreater"))
        {
            auto str =
                concat_kernel(equivTestKernPat,
                              sizeof(equivTestKernPat) / sizeof(const char*));
            kernelSource = str_sprintf(str, fnName.c_str(), opName.c_str());
        }
        else
        {
            auto str = concat_kernel(equivTestKernPatLessGreater,
                                     sizeof(equivTestKernPatLessGreater)
                                         / sizeof(const char*));
            kernelSource = str_sprintf(str, fnName.c_str());
        }
    }

    /* Create kernels */
    const char* programPtr = kernelSource.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char**)&programPtr, "sample_test"))
    {
        return -1;
    }

    /* Generate some streams */
    generate_equiv_test_data<T>(inDataA, vecSize, true, param, d);
    generate_equiv_test_data<T>(inDataB, vecSize, false, param, d);

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(T) * vecSize * TEST_SIZE, &inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(T) * vecSize * TEST_SIZE, &inDataB, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "Creating input array A failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(U) * vecSize * TEST_SIZE, NULL, &error);
    if (streams[2] == NULL)
    {
        print_error(error, "Creating output array failed!\n");
        return -1;
    }
    streams[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(U) * vecSize * TEST_SIZE, NULL, &error);
    if (streams[3] == NULL)
    {
        print_error(error, "Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 2, sizeof(streams[2]), &streams[2]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 3, sizeof(streams[3]), &streams[3]);
    test_error(error, "Unable to set indexed kernel arguments");

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[2], true, 0,
                                sizeof(U) * TEST_SIZE * vecSize, outData, 0,
                                NULL, NULL);
    test_error(error, "Unable to read output array!");

    auto verror_msg = [](const int& i, const int& j, const unsigned& vs,
                         const U& e, const U& o, const T& iA, const T& iB) {
        std::stringstream sstr;
        sstr << "ERROR: Data sample " << i << ":" << j << " at size " << vs
             << " does not validate! Expected " << e << ", got " << o
             << ", source " << iA << ":" << iB << std::endl;
        log_error(sstr.str().c_str());
    };

    /* And verify! */
    for (i = 0; i < TEST_SIZE; i++)
    {
        verify_equiv_values<T, U>(vecSize, &inDataA[i * vecSize],
                                  &inDataB[i * vecSize], expected,
                                  param.verifyFn);

        for (j = 0; j < (int)vecSize; j++)
        {
            if (expected[j] != outData[i * vecSize + j])
            {
                bool acceptFail = true;
                if (std::is_same<T, cl_half>::value)
                {
                    bool in_denorm = IsHalfSubnormal(inDataA[i * vecSize + j])
                        || IsHalfSubnormal(inDataB[i * vecSize + j]);

                    if (halfFlushDenormsToZero && in_denorm)
                    {
                        acceptFail = false;
                    }
                }

                if (acceptFail)
                {
                    verror_msg(
                        i, j, vecSize, expected[j], outData[i * vecSize + j],
                        inDataA[i * vecSize + j], inDataB[i * vecSize + j]);
                    return -1;
                }
            }
        }
    }

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[3], true, 0,
                                sizeof(U) * TEST_SIZE * vecSize, outData, 0,
                                NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    int fail = 0;
    for (i = 0; i < TEST_SIZE; i++)
    {
        verify_equiv_values<T, U>(vecSize, &inDataA[i * vecSize],
                                  &inDataB[i * vecSize], expected,
                                  param.verifyFn);

        for (j = 0; j < (int)vecSize; j++)
        {
            if (expected[j] != outData[i * vecSize + j])
            {
                if (std::is_same<T, float>::value)
                {
                    if (gInfNanSupport == 0)
                    {
                        if (isnan(inDataA[i * vecSize + j])
                            || isnan(inDataB[i * vecSize + j]))
                            fail = 0;
                        else
                            fail = 1;
                    }
                    if (fail)
                    {
                        verror_msg(i, j, vecSize, expected[j],
                                   outData[i * vecSize + j],
                                   inDataA[i * vecSize + j],
                                   inDataB[i * vecSize + j]);
                        return -1;
                    }
                }
                else if (std::is_same<T, cl_half>::value)
                {
                    bool in_denorm = IsHalfSubnormal(inDataA[i * vecSize + j])
                        || IsHalfSubnormal(inDataB[i * vecSize + j]);

                    if (!(halfFlushDenormsToZero && in_denorm))
                    {
                        verror_msg(i, j, vecSize, expected[j],
                                   outData[i * vecSize + j],
                                   inDataA[i * vecSize + j],
                                   inDataB[i * vecSize + j]);
                        return -1;
                    }
                }
                else
                {
                    verror_msg(
                        i, j, vecSize, expected[j], outData[i * vecSize + j],
                        inDataA[i * vecSize + j], inDataB[i * vecSize + j]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

template <typename T>
int RelationalsFPTest::test_relational(int numElements,
                                       const RelTestParams<T>& param)
{
    RandomSeed seed(gRandomSeed);
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index;
    int retVal = 0;

    for (index = 0; vecSizes[index] != 0; index++)
    {
        // Test!
        if (test_equiv_kernel<T>(vecSizes[index], param, seed) != 0)
        {
            log_error("   Vector %s%d FAILED\n", ftype, vecSizes[index]);
            retVal = -1;
        }
    }
    return retVal;
}

cl_int RelationalsFPTest::SetUp(int elements)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        cl_device_fp_config config = 0;
        cl_int error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                       sizeof(config), &config, NULL);
        test_error(error, "Unable to get device CL_DEVICE_HALF_FP_CONFIG");

        halfFlushDenormsToZero = (0 == (config & CL_FP_DENORM));
        log_info("Supports half precision denormals: %s\n",
                 halfFlushDenormsToZero ? "NO" : "YES");
    }

    return CL_SUCCESS;
}

cl_int RelationalsFPTest::Run()
{
    cl_int error = CL_SUCCESS;
    for (auto&& param : params)
    {
        switch (param->dataType)
        {
            case kHalf:
                error = test_relational<cl_half>(
                    num_elements, *((RelTestParams<cl_half>*)param.get()));
                break;
            case kFloat:
                error = test_relational<float>(
                    num_elements, *((RelTestParams<float>*)param.get()));
                break;
            case kDouble:
                error = test_relational<double>(
                    num_elements, *((RelTestParams<double>*)param.get()));
                break;
            default:
                test_error(-1, "RelationalsFPTest::Run: incorrect fp type");
                break;
        }
        test_error(error, "RelationalsFPTest::Run: test_relational failed");
    }
    return CL_SUCCESS;
}

cl_int IsEqualFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_equals_to>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::equal_to<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::equal_to<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsNotEqualFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_not_equals_to>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::not_equal_to<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::not_equal_to<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsGreaterFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_greater>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::greater<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::greater<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsGreaterEqualFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_greater_equal>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::greater_equal<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::greater_equal<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsLessFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_less>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::less<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::less<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsLessEqualFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_less_equal>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, std::less_equal<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, std::less_equal<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

cl_int IsLessGreaterFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new RelTestParams<cl_half>(
            &verify<cl_half, half_less_greater>, kHalf, HALF_NAN));

    params.emplace_back(new RelTestParams<float>(
        &verify<float, less_greater<float>>, kFloat, NAN));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new RelTestParams<double>(
            &verify<double, less_greater<double>>, kDouble, NAN));

    return RelationalsFPTest::SetUp(elements);
}

int test_relational_isequal(cl_device_id device, cl_context context,
                            cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsEqualFPTest>(device, context, queue, numElements);
}

int test_relational_isnotequal(cl_device_id device, cl_context context,
                               cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsNotEqualFPTest>(device, context, queue,
                                            numElements);
}

int test_relational_isgreater(cl_device_id device, cl_context context,
                              cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsGreaterFPTest>(device, context, queue, numElements);
}

int test_relational_isgreaterequal(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsGreaterEqualFPTest>(device, context, queue,
                                                numElements);
}

int test_relational_isless(cl_device_id device, cl_context context,
                           cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsLessFPTest>(device, context, queue, numElements);
}

int test_relational_islessequal(cl_device_id device, cl_context context,
                                cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsLessEqualFPTest>(device, context, queue,
                                             numElements);
}

int test_relational_islessgreater(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int numElements)
{
    return MakeAndRunTest<IsLessGreaterFPTest>(device, context, queue,
                                               numElements);
}

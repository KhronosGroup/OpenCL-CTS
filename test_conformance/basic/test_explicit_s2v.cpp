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
#include <cmath>
using std::isnan;
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include <CL/cl_half.h>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

extern cl_half_rounding_mode halfRoundingMode;

namespace {

// clang-format off

#define DECLARE_S2V_IDENT_KERNEL(srctype,dsttype,size) \
"__kernel void test_conversion(__global " srctype " *sourceValues, __global " dsttype #size " *destValues )\n"        \
"{\n"                                                                                                                 \
"    int  tid = get_global_id(0);\n"                                                                                  \
"    " srctype "  src = sourceValues[tid];\n"                                                                         \
"\n"                                                                                                                  \
"    destValues[tid] = (" dsttype #size ")src;\n"                                                                     \
"\n"                                                                                                                  \
"}\n"

#define DECLARE_S2V_IDENT_KERNELS(srctype, dsttype)                            \
    {                                                                          \
        DECLARE_S2V_IDENT_KERNEL(srctype, #dsttype, 2),                        \
        DECLARE_S2V_IDENT_KERNEL(srctype, #dsttype, 4),                        \
        DECLARE_S2V_IDENT_KERNEL(srctype, #dsttype, 8),                        \
        DECLARE_S2V_IDENT_KERNEL(srctype, #dsttype, 16)                        \
    }

#define DECLARE_EMPTY                                                          \
    {                                                                          \
        NULL, NULL, NULL, NULL, NULL                                           \
    }

/* Note: the next four arrays all must match in order and size to the
 * ExplicitTypes enum in conversions.h!!! */

#define DECLARE_S2V_IDENT_KERNELS_SET(srctype)                                 \
    {                                                                          \
        DECLARE_S2V_IDENT_KERNELS(#srctype, char),                             \
        DECLARE_S2V_IDENT_KERNELS(#srctype, uchar),                            \
        DECLARE_S2V_IDENT_KERNELS(#srctype, short),                            \
        DECLARE_S2V_IDENT_KERNELS(#srctype, ushort),                           \
        DECLARE_S2V_IDENT_KERNELS(#srctype, int),                              \
        DECLARE_S2V_IDENT_KERNELS(#srctype, uint),                             \
        DECLARE_S2V_IDENT_KERNELS(#srctype, long),                             \
        DECLARE_S2V_IDENT_KERNELS(#srctype, ulong),                            \
        DECLARE_S2V_IDENT_KERNELS(#srctype, float),                            \
        DECLARE_S2V_IDENT_KERNELS(#srctype, half),                             \
        DECLARE_S2V_IDENT_KERNELS(#srctype, double)                            \
    }

#define DECLARE_EMPTY_SET                                                      \
    {                                                                          \
        DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY,            \
        DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY,            \
        DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY,            \
        DECLARE_EMPTY, DECLARE_EMPTY, DECLARE_EMPTY                            \
    }

#define NUM_VEC_TYPES 11

/* The overall array */
const char * kernel_explicit_s2v_set[NUM_VEC_TYPES][NUM_VEC_TYPES][5] = {
    DECLARE_S2V_IDENT_KERNELS_SET(char),
    DECLARE_S2V_IDENT_KERNELS_SET(uchar),
    DECLARE_S2V_IDENT_KERNELS_SET(short),
    DECLARE_S2V_IDENT_KERNELS_SET(ushort),
    DECLARE_S2V_IDENT_KERNELS_SET(int),
    DECLARE_S2V_IDENT_KERNELS_SET(uint),
    DECLARE_S2V_IDENT_KERNELS_SET(long),
    DECLARE_S2V_IDENT_KERNELS_SET(ulong),
    DECLARE_S2V_IDENT_KERNELS_SET(float),
    DECLARE_S2V_IDENT_KERNELS_SET(half),
    DECLARE_S2V_IDENT_KERNELS_SET(double)
};

// clang-format on

bool IsHalfNaN(cl_half v)
{
    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = ((cl_half)v) & 0x3FF;

    // NaN test
    return (h_exp == 0x1F && h_mant != 0);
}

int test_explicit_s2v_function(cl_context context, cl_command_queue queue,
                               cl_kernel kernel, ExplicitType srcType,
                               unsigned int count, ExplicitType destType,
                               unsigned int vecSize, void *inputData)
{
    int error;
    clMemWrapper streams[2];
    size_t threadSize[3], groupSize[3];
    unsigned char convertedData[8]; /* Max type size is 8 bytes */
    unsigned int i, s;
    unsigned char *inPtr, *outPtr;
    size_t paramSize, destTypeSize;

    paramSize = get_explicit_type_size(srcType);
    destTypeSize = get_explicit_type_size(destType);

    size_t destStride = destTypeSize * vecSize;
    std::vector<char> outData(destStride * count);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                paramSize * count, inputData, &error);
    test_error(error, "clCreateBuffer failed");
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, destStride * count,
                                NULL, &error);
    test_error(error, "clCreateBuffer failed");

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    /* Run the kernel */
    threadSize[0] = count;

    error = get_max_common_work_group_size(context, kernel, threadSize[0],
                                           &groupSize[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threadSize,
                                   groupSize, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now verify the results. Each value should have been duplicated four
     times, and we should be able to just
     do a memcpy instead of relying on the actual type of data */
    error =
        clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, destStride * count,
                            outData.data(), 0, NULL, NULL);
    test_error(error, "Unable to read output values!");

    inPtr = (unsigned char *)inputData;
    outPtr = (unsigned char *)outData.data();

    for (i = 0; i < count; i++)
    {
        /* Convert the input data element to our output data type to compare
         * against */
        convert_explicit_value((void *)inPtr, (void *)convertedData, srcType,
                               false, kDefaultRoundingType, halfRoundingMode,
                               destType);

        /* Now compare every element of the vector */
        for (s = 0; s < vecSize; s++)
        {
            if (memcmp(convertedData, outPtr + destTypeSize * s, destTypeSize)
                != 0)
            {
                bool isSrcNaN =
                    (((srcType == kHalf)
                      && IsHalfNaN(*reinterpret_cast<cl_half *>(inPtr)))
                     || ((srcType == kFloat)
                         && isnan(*reinterpret_cast<cl_float *>(inPtr)))
                     || ((srcType == kDouble)
                         && isnan(*reinterpret_cast<cl_double *>(inPtr))));
                bool isDestNaN = (((destType == kHalf)
                                   && IsHalfNaN(*reinterpret_cast<cl_half *>(
                                       outPtr + destTypeSize * s)))
                                  || ((destType == kFloat)
                                      && isnan(*reinterpret_cast<cl_float *>(
                                          outPtr + destTypeSize * s)))
                                  || ((destType == kDouble)
                                      && isnan(*reinterpret_cast<cl_double *>(
                                          outPtr + destTypeSize * s))));

                if (isSrcNaN && isDestNaN)
                {
                    continue;
                }

                unsigned int *p = (unsigned int *)outPtr;
                log_error("ERROR: Output value %d:%d does not validate for "
                          "size %d:%d!\n",
                          i, s, vecSize, (int)destTypeSize);
                log_error("       Input:   0x%0*x\n", (int)(paramSize * 2),
                          *(unsigned int *)inPtr
                              & (0xffffffff >> (32 - paramSize * 8)));
                log_error("       Actual:  0x%08x 0x%08x 0x%08x 0x%08x\n", p[0],
                          p[1], p[2], p[3]);
                return -1;
            }
        }
        inPtr += paramSize;
        outPtr += destStride;
    }
    return 0;
}

struct TypesIterator
{
    using TypeIter =
        std::tuple<cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint,
                   cl_long, cl_ulong, cl_float, cl_half, cl_double>;

    TypesIterator(cl_device_id deviceID, cl_context context,
                  cl_command_queue queue)
        : dstType(0), srcType(0), context(context), queue(queue)
    {
        vecTypes = { kChar, kUChar, kShort, kUShort, kInt,   kUInt,
                     kLong, kULong, kFloat, kHalf,   kDouble };
        fp16Support = is_extension_available(deviceID, "cl_khr_fp16");
        fp64Support = is_extension_available(deviceID, "cl_khr_fp64");

        for_each_src_elem(it);
    }

    bool skip_type(ExplicitType type)
    {
        if ((type == kLong || type == kULong) && !gHasLong)
            return true;
        else if (type == kDouble && !fp64Support)
            return true;
        else if (type == kHalf && !fp16Support)
            return true;
        else if (strchr(get_explicit_type_name(type), ' ') != 0)
            return true;
        return false;
    }

    template <std::size_t Src = 0, typename SrcType>
    void iterate_src_type(const SrcType &t)
    {
        bool doTest = !skip_type(vecTypes[srcType]);
        if (doTest)
        {
            SrcType inputData[sample_count];
            RandomSeed seed(gRandomSeed);
            generate_random_data(vecTypes[srcType], 128, seed, inputData);

            for_each_dst_elem<0, Src, SrcType>(it, inputData);
        }

        srcType++;
        dstType = 0;
    }

    // crucial to keep it in-sync with ExplicitType
    bool isExplicitTypeFloating(ExplicitType type) { return (type >= kFloat); }

    template <std::size_t Dst, std::size_t Src, typename SrcType,
              typename DstType>
    void iterate_dst_type(const DstType &t, SrcType *inputData)
    {
        bool doTest = !skip_type(vecTypes[dstType]);

        doTest = doTest
            && ((isExplicitTypeFloating(vecTypes[srcType])
                 && isExplicitTypeFloating(vecTypes[dstType]))
                || (!isExplicitTypeFloating(vecTypes[srcType])
                    && !isExplicitTypeFloating(vecTypes[dstType])));

        if (doTest)
            test_explicit_s2v_function_set<SrcType, DstType>(
                vecTypes[srcType], vecTypes[dstType], inputData);
        dstType++;
    }

    template <std::size_t Out = 0, typename... Tp>
    inline typename std::enable_if<Out == sizeof...(Tp), void>::type
    for_each_src_elem(
        const std::tuple<Tp...> &) // Unused arguments are given no names.
    {}

    template <std::size_t Out = 0, typename... Tp>
        inline typename std::enable_if < Out<sizeof...(Tp), void>::type
        for_each_src_elem(const std::tuple<Tp...> &t)
    {
        iterate_src_type<Out>(std::get<Out>(t));
        for_each_src_elem<Out + 1, Tp...>(t);
    }

    template <std::size_t In = 0, std::size_t Out, typename SrcType,
              typename... Tp>
    inline typename std::enable_if<In == sizeof...(Tp), void>::type
    for_each_dst_elem(const std::tuple<Tp...> &, SrcType *)
    {}

    template <std::size_t In = 0, std::size_t Out, typename SrcType,
              typename... Tp>
        inline typename std::enable_if < In<sizeof...(Tp), void>::type
        for_each_dst_elem(const std::tuple<Tp...> &t, SrcType *inputData)
    {
        iterate_dst_type<In, Out, SrcType>(std::get<In>(t), inputData);
        for_each_dst_elem<In + 1, Out, SrcType, Tp...>(t, inputData);
    }

    template <typename SrcType, typename DstType>
    void test_explicit_s2v_function_set(ExplicitType srcT, ExplicitType dstT,
                                        SrcType *inputData)
    {
        unsigned int sizes[] = { 2, 4, 8, 16, 0 };

        for (int i = 0; sizes[i] != 0; i++)
        {
            clProgramWrapper program;
            clKernelWrapper kernel;

            char pragma[256] = { 0 };
            const char *finalProgramSrc[2] = {
                pragma, // optional pragma
                kernel_explicit_s2v_set[srcType][dstType][i]
            };

            std::stringstream sstr;
            if (srcT == kDouble || dstT == kDouble)
                sstr << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

            if (srcT == kHalf || dstT == kHalf)
                sstr << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

            snprintf(pragma, sizeof(pragma), "%s", sstr.str().c_str());

            if (create_single_kernel_helper(context, &program, &kernel, 2,
                                            finalProgramSrc, "test_conversion"))
            {
                log_info("****** %s%s *******\n", finalProgramSrc[0],
                         finalProgramSrc[1]);
                throw std::runtime_error(
                    "create_single_kernel_helper failed\n");
            }

            if (test_explicit_s2v_function(context, queue, kernel, srcT,
                                           sample_count, dstT, sizes[i],
                                           inputData)
                != 0)
            {
                log_error("ERROR: Explicit cast of scalar %s to vector %s%d "
                          "FAILED; skipping other %s vector tests\n",
                          get_explicit_type_name(srcT),
                          get_explicit_type_name(dstT), sizes[i],
                          get_explicit_type_name(dstT));
                throw std::runtime_error("test_explicit_s2v_function failed\n");
            }
        }
    }

protected:
    bool fp16Support;
    bool fp64Support;

    TypeIter it;
    unsigned int dstType, srcType;
    cl_context context;
    cl_command_queue queue;

    std::vector<ExplicitType> vecTypes;

    constexpr static unsigned int sample_count =
        128; // hardcoded in original test
};

} // anonymous namespace

int test_explicit_s2v(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    try
    {
        TypesIterator(deviceID, context, queue);
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

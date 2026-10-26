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
#include "testBase.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cinttypes>


// The tests we are running
const TestDef test_defs[NUM_TESTS] = {
    { kOpAdd, "+", "+", kKernelBinary, false, kGroupMath },
    { kOpSub, "-", "-", kKernelBinary, false, kGroupMath },
    { kOpMul, "*", "*", kKernelBinary, false, kGroupMath },
    { kOpDiv, "/", "/", kKernelBinary, false, kGroupMath },
    { kOpRem, "%", "%", kKernelBinary, false, kGroupMath },
    { kOpBitAnd, "&", "&", kKernelBinary, false, kGroupLogic },
    { kOpBitOr, "|", "|", kKernelBinary, false, kGroupLogic },
    { kOpBitXor, "^", "^", kKernelBinary, false, kGroupLogic },
    { kOpShrVec, ">>", ">> by vector", kKernelBinary, false, kGroupShift },
    { kOpShlVec, "<<", "<< by vector", kKernelBinary, false, kGroupShift },
    { kOpShrScalar, ">>", ">> by scalar", kKernelScalarShift, false,
      kGroupShift },
    { kOpShlScalar, "<<", "<< by scalar", kKernelScalarShift, false,
      kGroupShift },
    { kOpBitNot, "~", "~", kKernelUnary, false, kGroupLogic },
    { kOpTernary, "?:", "?:", kKernelQuestionColon, false, kGroupCompare },
    { kOpLogAnd, "&&", "&&", kKernelBinary, true, kGroupLogic },
    { kOpLogOr, "||", "||", kKernelBinary, true, kGroupLogic },
    { kOpLt, "<", "<", kKernelBinary, true, kGroupCompare },
    { kOpGt, ">", ">", kKernelBinary, true, kGroupCompare },
    { kOpLe, "<=", "<=", kKernelBinary, true, kGroupCompare },
    { kOpGe, ">=", ">=", kKernelBinary, true, kGroupCompare },
    { kOpEq, "==", "==", kKernelBinary, true, kGroupCompare },
    { kOpNe, "!=", "!=", kKernelBinary, true, kGroupCompare },
    { kOpLogNot, "!", "!", kKernelUnary, true, kGroupLogic },
};

template <typename T> struct VerifyInput
{
    const T *ptr;
    bool is_scalar;
    size_t vector_size;

    VerifyInput(const T *p, bool is_s, size_t v)
        : ptr(p), is_scalar(is_s), vector_size(v)
    {}

    T operator[](size_t idx) const
    {
        return is_scalar ? ptr[idx / vector_size] : ptr[idx];
    }
};

#define WRAP_INPUTS(type)                                                      \
    VerifyInput<type> inptrA(inptrA_raw, a_scalar, vector_size);               \
    VerifyInput<type> inptrB(inptrB_raw, b_scalar, vector_size)

template <typename T> bool is_valid_div(T a, T b)
{
    if (b == 0) return false;
    if (std::is_signed<T>::value)
    {
        if (b == (T)-1 && a == std::numeric_limits<T>::min())
        {
            return false;
        }
    }
    return true;
}

template <typename T, typename Func>
static void compute_loop_i(T *ref, size_t n, Func fn)
{
    for (size_t i = 0; i < n; i++)
    {
        ref[i] = fn(i);
    }
}

template <typename T, typename Func>
static void compute_loop_ij(T *ref, size_t n, size_t vector_size, Func fn)
{
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            ref[i] = fn(i, j);
        }
    }
}

template <typename T>
static void compute_references(int test, size_t vector_size,
                               const VerifyInput<T> &inptrA,
                               const VerifyInput<T> &inptrB, const T *outptr,
                               T *ref, size_t n, cl_uint shift_mask)
{
    const TestDef &test_def = test_defs[test];
    const T true_val = (vector_size == 1) ? (T)1 : (T)-1;
    switch (test_def.id)
    {
        case kOpAdd:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] + inptrB[i]; });
            break;
        case kOpSub:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] - inptrB[i]; });
            break;
        case kOpMul:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] * inptrB[i]; });
            break;
        case kOpDiv:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return is_valid_div<T>(inptrA[i], inptrB[i])
                    ? (T)(inptrA[i] / inptrB[i])
                    : outptr[i];
            });
            break;
        case kOpRem:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return is_valid_div<T>(inptrA[i], inptrB[i])
                    ? (T)(inptrA[i] % inptrB[i])
                    : outptr[i];
            });
            break;
        case kOpBitAnd:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] & inptrB[i]; });
            break;
        case kOpBitOr:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] | inptrB[i]; });
            break;
        case kOpBitXor:
            compute_loop_i(
                ref, n, [&](size_t i) -> T { return inptrA[i] ^ inptrB[i]; });
            break;
        case kOpShrVec:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return inptrA[i] >> (inptrB[i] & shift_mask);
            });
            break;
        case kOpShlVec:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return inptrA[i] << (inptrB[i] & shift_mask);
            });
            break;
        case kOpShrScalar:
            compute_loop_ij(ref, n, vector_size, [&](size_t i, size_t j) -> T {
                return inptrA[i] >> (inptrB[j] & shift_mask);
            });
            break;
        case kOpShlScalar:
            compute_loop_ij(ref, n, vector_size, [&](size_t i, size_t j) -> T {
                return inptrA[i] << (inptrB[j] & shift_mask);
            });
            break;
        case kOpBitNot:
            compute_loop_i(ref, n, [&](size_t i) -> T { return ~inptrA[i]; });
            break;
        case kOpTernary:
            compute_loop_ij(ref, n, vector_size, [&](size_t i, size_t j) -> T {
                return (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
            });
            break;
        case kOpLogAnd:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] && inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpLogOr:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] || inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpLt:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] < inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpGt:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] > inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpLe:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] <= inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpGe:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] >= inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpEq:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] == inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpNe:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (inptrA[i] != inptrB[i]) ? true_val : (T)0;
            });
            break;
        case kOpLogNot:
            compute_loop_i(ref, n, [&](size_t i) -> T {
                return (!inptrA[i]) ? true_val : (T)0;
            });
            break;
        case NUM_TESTS: break;
    }
}

static void wait_for_event(cl_event event)
{
    if (event != nullptr)
    {
        clWaitForEvents(1, &event);
    }
}

// =======================================
// defined arithmetic inputs
// =======================================

// floor(sqrt(TYPE_MAX)): the largest operand whose square is representable.
constexpr cl_long kIntMulBound = 46340;
constexpr cl_long kLongMulBound = 3037000499LL;
// A scalar ushort is promoted to int before the multiply, so the product has to
// fit in an int rather than in a ushort.
constexpr cl_ulong kUShortMulBound = 46340;

// Writes `count` random values in [-bound, bound] to each operand, then seeds
// the leading elements with the cross product of the corner values, mirroring
// what the unbounded init_*_data() generators do.
template <typename T>
static void fill_bounded_signed(void *inputA, void *inputB, size_t count,
                                cl_long bound, MTdata d)
{
    T *a = (T *)inputA;
    T *b = (T *)inputB;

    for (size_t i = 0; i < count; i++)
    {
        a[i] = (T)get_random_long(-bound, bound, d);
        b[i] = (T)get_random_long(-bound, bound, d);
    }

    const cl_long corners[] = { 0, -1, 1, -bound, bound };
    const size_t num_corners = sizeof(corners) / sizeof(corners[0]);
    size_t index = 0;
    for (size_t x = 0; x < num_corners; x++)
        for (size_t y = 0; y < num_corners && index < count; y++)
        {
            a[index] = (T)corners[x];
            b[index] = (T)corners[y];
            index++;
        }
}

template <typename T>
static void fill_bounded_unsigned(void *inputA, void *inputB, size_t count,
                                  cl_ulong bound, MTdata d)
{
    T *a = (T *)inputA;
    T *b = (T *)inputB;

    for (size_t i = 0; i < count; i++)
    {
        a[i] = (T)get_random_ulong(0, bound, d);
        b[i] = (T)get_random_ulong(0, bound, d);
    }

    const cl_ulong corners[] = { 0, 1, bound };
    const size_t num_corners = sizeof(corners) / sizeof(corners[0]);
    size_t index = 0;
    for (size_t x = 0; x < num_corners; x++)
        for (size_t y = 0; y < num_corners && index < count; y++)
        {
            a[index] = (T)corners[x];
            b[index] = (T)corners[y];
            index++;
        }
}

std::optional<DefinedArithmeticOp> get_defined_arithmetic_op(const char *opName)
{
    if (strcmp(opName, "+") == 0) return kDefinedAdd;
    if (strcmp(opName, "-") == 0) return kDefinedSub;
    if (strcmp(opName, "*") == 0) return kDefinedMul;

    return std::nullopt;
}

void init_defined_arithmetic_data(ExplicitType type, DefinedArithmeticOp op,
                                  size_t num_elements, void *inputA,
                                  void *inputB, MTdata d)
{
    switch (type)
    {
        case kInt:
            fill_bounded_signed<cl_int>(
                inputA, inputB, num_elements,
                op == kDefinedMul ? kIntMulBound : CL_INT_MAX / 2, d);
            break;
        case kLong:
            fill_bounded_signed<cl_long>(
                inputA, inputB, num_elements,
                op == kDefinedMul ? kLongMulBound : CL_LONG_MAX / 2, d);
            break;
        case kUShort:
            if (op == kDefinedMul)
                fill_bounded_unsigned<cl_ushort>(inputA, inputB, num_elements,
                                                 kUShortMulBound, d);
            break;
        default:
            // Cannot overflow: keep the unbounded inputs.
            break;
    }
}

template <typename T> const char *get_cl_type_name();
template <> const char *get_cl_type_name<cl_char>() { return "cl_char"; }
template <> const char *get_cl_type_name<cl_uchar>() { return "cl_uchar"; }
template <> const char *get_cl_type_name<cl_short>() { return "cl_short"; }
template <> const char *get_cl_type_name<cl_ushort>() { return "cl_ushort"; }
template <> const char *get_cl_type_name<cl_int>() { return "cl_int"; }
template <> const char *get_cl_type_name<cl_uint>() { return "cl_uint"; }
template <> const char *get_cl_type_name<cl_long>() { return "cl_long"; }
template <> const char *get_cl_type_name<cl_ulong>() { return "cl_ulong"; }

template <typename T>
int verify(int test, size_t vector_size, T *inptrA_raw, T *inptrB_raw,
           T *outptr, T *ref, size_t n, bool a_scalar, bool b_scalar,
           cl_event event, bool compute_ref)
{
    cl_uint shift_mask = (vector_size == 1 && sizeof(T) < sizeof(cl_int))
        ? (cl_uint)(sizeof(cl_int) * 8) - 1
        : (cl_uint)(sizeof(T) * 8) - 1;
    wait_for_event(event);
    WRAP_INPUTS(T);
    if (compute_ref)
    {
        compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                           shift_mask);
    }
    if (memcmp(ref, outptr, n * sizeof(T)) == 0)
    {
        return 0;
    }

    const char *type_name = get_cl_type_name<T>();
    int count = 0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            T r = ref[i];
            if (r != outptr[i])
            {
                // Shift is tricky
                if (test == kOpShrVec || test == kOpShlVec)
                {
                    log_error("%s Verification failed at element %zu of "
                              "%zu: 0x%" PRIx64 " %s 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              type_name, i, n, (cl_ulong)inptrA[i],
                              test_defs[test].op, (cl_ulong)inptrB[i],
                              (cl_ulong)r, (cl_ulong)outptr[i]);
                    log_error(
                        "\t1) Vector shift failure at element %zu: original is "
                        "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                        i, (cl_ulong)inptrA[i], test_defs[test].op,
                        (int)inptrB[i], (cl_ulong)inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRIu64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(T) * 8),
                              (cl_ulong)(inptrB[i] & shift_mask),
                              (cl_ulong)(inptrB[i] & shift_mask));
                }
                else if (test == kOpShrScalar || test == kOpShlScalar)
                {
                    log_error("%s Verification failed at element %zu of "
                              "%zu (%zu): 0x%" PRIx64 " %s 0x%" PRIx64
                              " = 0x%" PRIx64 ", got 0x%" PRIx64 "\n",
                              type_name, i, n, j, (cl_ulong)inptrA[i],
                              test_defs[test].op, (cl_ulong)inptrB[j],
                              (cl_ulong)r, (cl_ulong)outptr[i]);
                    log_error(
                        "\t1) Scalar shift failure at element %zu: original is "
                        "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                        i, (cl_ulong)inptrA[i], test_defs[test].op,
                        (int)inptrB[j], (cl_ulong)inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRIu64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(T) * 8),
                              (cl_ulong)(inptrB[j] & shift_mask),
                              (cl_ulong)(inptrB[j] & shift_mask));
                }
                else if (test == kOpTernary)
                {
                    log_error("%s Verification failed at element %zu of "
                              "%zu (%zu): (0x%" PRIx64 " < 0x%" PRIx64
                              ") ? 0x%" PRIx64 " : 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              type_name, i, n, j, (cl_ulong)inptrA[j],
                              (cl_ulong)inptrB[j], (cl_ulong)inptrA[i],
                              (cl_ulong)inptrB[i], (cl_ulong)r,
                              (cl_ulong)outptr[i]);
                }
                else
                {
                    log_error("%s Verification failed at element %zu of "
                              "%zu: 0x%" PRIx64 " %s 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              type_name, i, n, (cl_ulong)inptrA[i],
                              test_defs[test].op, (cl_ulong)inptrB[i],
                              (cl_ulong)r, (cl_ulong)outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT)
                {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    return count ? -1 : 0;
}

template int verify<cl_char>(int, size_t, cl_char *, cl_char *, cl_char *,
                             cl_char *, size_t, bool, bool, cl_event, bool);
template int verify<cl_uchar>(int, size_t, cl_uchar *, cl_uchar *, cl_uchar *,
                              cl_uchar *, size_t, bool, bool, cl_event, bool);
template int verify<cl_short>(int, size_t, cl_short *, cl_short *, cl_short *,
                              cl_short *, size_t, bool, bool, cl_event, bool);
template int verify<cl_ushort>(int, size_t, cl_ushort *, cl_ushort *,
                               cl_ushort *, cl_ushort *, size_t, bool, bool,
                               cl_event, bool);
template int verify<cl_int>(int, size_t, cl_int *, cl_int *, cl_int *, cl_int *,
                            size_t, bool, bool, cl_event, bool);
template int verify<cl_uint>(int, size_t, cl_uint *, cl_uint *, cl_uint *,
                             cl_uint *, size_t, bool, bool, cl_event, bool);
template int verify<cl_long>(int, size_t, cl_long *, cl_long *, cl_long *,
                             cl_long *, size_t, bool, bool, cl_event, bool);
template int verify<cl_ulong>(int, size_t, cl_ulong *, cl_ulong *, cl_ulong *,
                              cl_ulong *, size_t, bool, bool, cl_event, bool);

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
const char *tests[] = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    ">>",
    "<<",
    ">>",
    "<<",
    "~",
    "?:",
    "&&",
    "||",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "!",
};

// The names of the tests
const char *test_names[] = {
    "+", // 0
    "-", // 1
    "*", // 2
    "/", // 3
    "%", // 4
    "&", // 5
    "|", // 6
    "^", // 7
    ">> by vector", // 8
    "<< by vector", // 9
    ">> by scalar", // 10
    "<< by scalar", // 11
    "~",  // 12
    "?:", // 13
    "&&", // 14
    "||", // 15
    "<",  // 16
    ">",  // 17
    "<=", // 18
    ">=", // 19
    "==", // 20
    "!=", // 21
    "!",  // 22
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

template <typename T>
static void compute_references(int test, size_t vector_size,
                               const VerifyInput<T> &inptrA,
                               const VerifyInput<T> &inptrB, const T *outptr,
                               T *ref, size_t n, cl_uint shift_mask)
{
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            T r = 0;
            switch (test)
            {
                case 0: r = inptrA[i] + inptrB[i]; break;
                case 1: r = inptrA[i] - inptrB[i]; break;
                case 2: r = inptrA[i] * inptrB[i]; break;
                case 3:
                    if (!is_valid_div<T>(inptrA[i], inptrB[i]))
                        r = outptr[i];
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (!is_valid_div<T>(inptrA[i], inptrB[i]))
                        r = outptr[i];
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5: r = inptrA[i] & inptrB[i]; break;
                case 6: r = inptrA[i] | inptrB[i]; break;
                case 7: r = inptrA[i] ^ inptrB[i]; break;
                case 8: r = inptrA[i] >> (inptrB[i] & shift_mask); break;
                case 9: r = inptrA[i] << (inptrB[i] & shift_mask); break;
                case 10: r = inptrA[i] >> (inptrB[j] & shift_mask); break;
                case 11: r = inptrA[i] << (inptrB[j] & shift_mask); break;
                case 12: r = ~inptrA[i]; break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    r = inptrA[i] && inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 15:
                    r = inptrA[i] || inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 16:
                    r = inptrA[i] < inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 17:
                    r = inptrA[i] > inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 18:
                    r = inptrA[i] <= inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 19:
                    r = inptrA[i] >= inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 20:
                    r = inptrA[i] == inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 21:
                    r = inptrA[i] != inptrB[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
                case 22:
                    r = !inptrA[i];
                    if (vector_size != 1 && r) r = -1;
                    break;
            }
            ref[i] = r;
        }
    }
}

// =======================================
// long
// =======================================
int verify_long(int test, size_t vector_size, cl_long *inptrA_raw,
                cl_long *inptrB_raw, cl_long *outptr, cl_long *ref, size_t n,
                bool a_scalar, bool b_scalar)
{
    cl_long shift_mask = (sizeof(cl_long) * 8) - 1;
    WRAP_INPUTS(cl_long);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_long)) == 0)
    {
        return 0;
    }

    int count=0;
    for (size_t j=0; j<n; j += vector_size )
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_long r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_long Verification failed at element %zu of "
                              "%zu : 0x%" PRIx64 " %s 0x%" PRIx64
                              " = 0x%" PRIx64 ", got 0x%" PRIx64 "\n",
                              i, n, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error(
                        "\t1) Vector shift failure at element %zu: original is "
                        "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                        i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRId64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(cl_long) * 8),
                              inptrB[i] & shift_mask, inptrB[i] & shift_mask);
                }
                else if (test == 10 || test == 11) {

                    log_error("cl_long Verification failed at element %zu of "
                              "%zu (%zu): 0x%" PRIx64 " %s 0x%" PRIx64
                              " = 0x%" PRIx64 ", got 0x%" PRIx64 "\n",
                              i, n, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error(
                        "\t1) Scalar shift failure at element %zu: original is "
                        "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                        i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRId64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(cl_long) * 8),
                              inptrB[j] & shift_mask, inptrB[j] & shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %zu "
                              "(%zu): (0x%" PRIx64 " < 0x%" PRIx64
                              ") ? 0x%" PRIx64 " : 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i],
                              r, outptr[i]);
                } else {
                    log_error("cl_long Verification failed at element %zu of "
                              "%zu: 0x%" PRIx64 " %s 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              i, n, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    if (count) return -1; else return 0;
}

// =======================================
// ulong
// =======================================
int verify_ulong(int test, size_t vector_size, cl_ulong *inptrA_raw,
                 cl_ulong *inptrB_raw, cl_ulong *outptr, cl_ulong *ref,
                 size_t n, bool a_scalar, bool b_scalar)
{
    cl_ulong shift_mask = (sizeof(cl_ulong)*8)-1;
    WRAP_INPUTS(cl_ulong);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_ulong)) == 0)
    {
        return 0;
    }

    int count=0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_ulong r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_ulong Verification failed at element %zu of "
                              "%zu: 0x%" PRIx64 " %s 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              i, n, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRIu64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(cl_ulong) * 8),
                              inptrB[i] & shift_mask, inptrB[i] & shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_ulong Verification failed at element %zu of "
                              "%zu (%zu): 0x%" PRIx64 " %s 0x%" PRIx64
                              " = 0x%" PRIx64 ", got 0x%" PRIx64 "\n",
                              i, n, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error(
                        "\t1) Scalar shift failure at element %zu: original is "
                        "0x%" PRIx64 " %s %d (0x%" PRIx64 ")\n",
                        i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the "
                              "final shift amount %" PRId64 " (0x%" PRIx64
                              ").\n",
                              (int)log2(sizeof(cl_long) * 8),
                              inptrB[j] & shift_mask, inptrB[j] & shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %zu of "
                              "%zu (%zu): (0x%" PRIx64 " < 0x%" PRIx64
                              ") ? 0x%" PRIx64 " : 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              i, n, j, inptrA[j], inptrB[j], inptrA[i],
                              inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_ulong Verification failed at element %zu of "
                              "%zu: 0x%" PRIx64 " %s 0x%" PRIx64 " = 0x%" PRIx64
                              ", got 0x%" PRIx64 "\n",
                              i, n, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }
    if (count) return -1; else return 0;
}

// =======================================
// int
// =======================================
int verify_int(int test, size_t vector_size, cl_int *inptrA_raw,
               cl_int *inptrB_raw, cl_int *outptr, cl_int *ref, size_t n,
               bool a_scalar, bool b_scalar)
{
    cl_int shift_mask = (sizeof(cl_int)*8)-1;
    WRAP_INPUTS(cl_int);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_int)) == 0)
    {
        return 0;
    }

    int count=0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_int r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_int Verification failed at element %zu: 0x%x "
                              "%s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_int)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_int Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_int)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_int Verification failed at element %zu: 0x%x "
                              "%s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    if (count) return -1; else return 0;
}

// =======================================
// uint
// =======================================
int verify_uint(int test, size_t vector_size, cl_uint *inptrA_raw,
                cl_uint *inptrB_raw, cl_uint *outptr, cl_uint *ref, size_t n,
                bool a_scalar, bool b_scalar)
{
    cl_uint shift_mask = (sizeof(cl_uint)*8)-1;
    WRAP_INPUTS(cl_uint);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_uint)) == 0)
    {
        return 0;
    }
    int count=0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_uint r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_uint Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uint)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_uint Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uint)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_uint Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }
    if (count) return -1; else return 0;
}

// =======================================
// short
// =======================================
int verify_short(int test, size_t vector_size, cl_short *inptrA_raw,
                 cl_short *inptrB_raw, cl_short *outptr, cl_short *ref,
                 size_t n, bool a_scalar, bool b_scalar)
{
    cl_int   shift_mask = vector_size == 1 ? (cl_int)(sizeof(cl_int)*8)-1
    : (cl_int)(sizeof(cl_short)*8)-1;
    WRAP_INPUTS(cl_short);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_short)) == 0)
    {
        return 0;
    }
    int      count=0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_short r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_short Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_short)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_short Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_short)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_short Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    if (count) return -1; else return 0;
}

// =======================================
// ushort
// =======================================
int verify_ushort(int test, size_t vector_size, cl_ushort *inptrA_raw,
                  cl_ushort *inptrB_raw, cl_ushort *outptr, cl_ushort *ref,
                  size_t n, bool a_scalar, bool b_scalar)
{
    cl_uint   shift_mask = vector_size == 1 ? (cl_uint)(sizeof(cl_uint)*8)-1
    : (cl_uint)(sizeof(cl_ushort)*8)-1;
    WRAP_INPUTS(cl_ushort);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_ushort)) == 0)
    {
        return 0;
    }
    int             count=0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_ushort r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_ushort Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_ushort)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_ushort Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_ushort)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_ushort Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    if (count) return -1; else return 0;
}

// =======================================
// char
// =======================================
int verify_char(int test, size_t vector_size, cl_char *inptrA_raw,
                cl_char *inptrB_raw, cl_char *outptr, cl_char *ref, size_t n,
                bool a_scalar, bool b_scalar)
{
    cl_int    shift_mask = vector_size == 1 ? (cl_int)(sizeof(cl_int)*8)-1
    : (cl_int)(sizeof(cl_char)*8)-1;
    WRAP_INPUTS(cl_char);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_char)) == 0)
    {
        return 0;
    }
    int count = 0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_char r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_char Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_char)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_char Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_long)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_char Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }
    if (count) return -1; else return 0;
}

// =======================================
// uchar
// =======================================
int verify_uchar(int test, size_t vector_size, cl_uchar *inptrA_raw,
                 cl_uchar *inptrB_raw, cl_uchar *outptr, cl_uchar *ref,
                 size_t n, bool a_scalar, bool b_scalar)
{
    cl_uint shift_mask = vector_size == 1 ? (cl_uint)(sizeof(cl_uint) * 8) - 1
                                          : (cl_uint)(sizeof(cl_uchar) * 8) - 1;
    WRAP_INPUTS(cl_uchar);
    compute_references(test, vector_size, inptrA, inptrB, outptr, ref, n,
                       shift_mask);
    if (memcmp(ref, outptr, n * sizeof(cl_uchar)) == 0)
    {
        return 0;
    }
    int count = 0;
    for (size_t j = 0; j < n; j += vector_size)
    {
        for (size_t i = j; i < j + vector_size; i++)
        {
            cl_uchar r = ref[i];
            if (r != outptr[i]) {
                // Shift is tricky
                if (test == 8 || test == 9) {
                    log_error("cl_uchar Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                    log_error("\t1) Shift failure at element %zu: original is "
                              "0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[i],
                              inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uchar)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_uchar Verification failed at element %zu "
                              "(%zu): 0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, j, inptrA[i], tests[test], inptrB[j], r,
                              outptr[i]);
                    log_error("\t1) Scalar shift failure at element %zu: "
                              "original is 0x%x %s %d (0x%x)\n",
                              i, inptrA[i], tests[test], (int)inptrB[j],
                              inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uchar)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error(
                        "cl_int Verification failed at element %zu (%zu): "
                        "(0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n",
                        i, j, inptrA[j], inptrB[j], inptrA[i], inptrB[i], r,
                        outptr[i]);
                } else {
                    log_error("cl_uchar Verification failed at element %zu: "
                              "0x%x %s 0x%x = 0x%x, got 0x%x\n",
                              i, inptrA[i], tests[test], inptrB[i], r,
                              outptr[i]);
                }
                count++;
                if (count >= MAX_ERRORS_TO_PRINT) {
                    log_error("Further errors ignored.\n");
                    return -1;
                }
            }
        }
    }

    if (count) return -1; else return 0;
}

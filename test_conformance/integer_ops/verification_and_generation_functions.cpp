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

// =======================================
// long
// =======================================
int
verify_long(int test, size_t vector_size, cl_long *inptrA, cl_long *inptrB, cl_long *outptr, size_t n)
{
    cl_long            r, shift_mask = (sizeof(cl_long)*8)-1;
    size_t         i, j;
    int count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_LONG_MIN))
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_LONG_MIN))
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_ulong(int test, size_t vector_size, cl_ulong *inptrA, cl_ulong *inptrB, cl_ulong *outptr, size_t n)
{
    cl_ulong        r, shift_mask = (sizeof(cl_ulong)*8)-1;
    size_t          i, j;
    int count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_int(int test, size_t vector_size, cl_int *inptrA, cl_int *inptrB, cl_int *outptr, size_t n)
{
    cl_int            r, shift_mask = (sizeof(cl_int)*8)-1;
    size_t          i, j;
    int count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_INT_MIN))
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_INT_MIN))
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_uint(int test, size_t vector_size, cl_uint *inptrA, cl_uint *inptrB, cl_uint *outptr, size_t n)
{
    cl_uint            r, shift_mask = (sizeof(cl_uint)*8)-1;
    size_t          i, j;
    int count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_short(int test, size_t vector_size, cl_short *inptrA, cl_short *inptrB, cl_short *outptr, size_t n)
{
    cl_short r;
    cl_int   shift_mask = vector_size == 1 ? (cl_int)(sizeof(cl_int)*8)-1
    : (cl_int)(sizeof(cl_short)*8)-1;
    size_t   i, j;
    int      count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_SHRT_MIN))
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_SHRT_MIN))
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_ushort(int test, size_t vector_size, cl_ushort *inptrA, cl_ushort *inptrB, cl_ushort *outptr, size_t n)
{
    cl_ushort       r;
    cl_uint   shift_mask = vector_size == 1 ? (cl_uint)(sizeof(cl_uint)*8)-1
    : (cl_uint)(sizeof(cl_ushort)*8)-1;
    size_t          i, j;
    int             count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_char(int test, size_t vector_size, cl_char *inptrA, cl_char *inptrB, cl_char *outptr, size_t n)
{
    cl_char   r;
    cl_int    shift_mask = vector_size == 1 ? (cl_int)(sizeof(cl_int)*8)-1
    : (cl_int)(sizeof(cl_char)*8)-1;
    size_t    i, j;
    int       count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {

            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_CHAR_MIN))
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0 || (inptrB[i] == -1 && inptrA[i] == CL_CHAR_MIN))
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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
int
verify_uchar(int test, size_t vector_size, cl_uchar *inptrA, cl_uchar *inptrB, cl_uchar *outptr, size_t n)
{
    cl_uchar r;
    cl_uint shift_mask = vector_size == 1 ? (cl_uint)(sizeof(cl_uint) * 8) - 1
                                          : (cl_uint)(sizeof(cl_uchar) * 8) - 1;
    size_t   i, j;
    int      count=0;

    for (j=0; j<n; j += vector_size )
    {
        for( i = j; i < j + vector_size; i++ )
        {
            switch (test) {
                case 0:
                    r = inptrA[i] + inptrB[i];
                    break;
                case 1:
                    r = inptrA[i] - inptrB[i];
                    break;
                case 2:
                    r = inptrA[i] * inptrB[i];
                    break;
                case 3:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] / inptrB[i];
                    break;
                case 4:
                    if (inptrB[i] == 0)
                        continue;
                    else
                        r = inptrA[i] % inptrB[i];
                    break;
                case 5:
                    r = inptrA[i] & inptrB[i];
                    break;
                case 6:
                    r = inptrA[i] | inptrB[i];
                    break;
                case 7:
                    r = inptrA[i] ^ inptrB[i];
                    break;
                case 8:
                    r = inptrA[i] >> (inptrB[i] & shift_mask);
                    break;
                case 9:
                    r = inptrA[i] << (inptrB[i] & shift_mask);
                    break;
                case 10:
                    r = inptrA[i] >> (inptrB[j] & shift_mask);
                    break;
                case 11:
                    r = inptrA[i] << (inptrB[j] & shift_mask);
                    break;
                case 12:
                    r = ~inptrA[i];
                    break;
                case 13:
                    r = (inptrA[j] < inptrB[j]) ? inptrA[i] : inptrB[i];
                    break;
                case 14:
                    // Scalars are set to 1/0
                    r = inptrA[i] && inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 15:
                    // Scalars are set to 1/0
                    r = inptrA[i] || inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 16:
                    // Scalars are set to 1/0
                    r = inptrA[i] < inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 17:
                    // Scalars are set to 1/0
                    r = inptrA[i] > inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 18:
                    // Scalars are set to 1/0
                    r = inptrA[i] <= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 19:
                    // Scalars are set to 1/0
                    r = inptrA[i] >= inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 20:
                    // Scalars are set to 1/0
                    r = inptrA[i] == inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 21:
                    // Scalars are set to 1/0
                    r = inptrA[i] != inptrB[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                case 22:
                    // Scalars are set to 1/0
                    r = !inptrA[i];
                    // Vectors are set to -1/0
                    if (vector_size != 1 && r) {
                        r = -1;
                    }
                    break;
                default:
                    log_error("Invalid test: %d\n", test);
                    return -1;
                    break;
            }
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

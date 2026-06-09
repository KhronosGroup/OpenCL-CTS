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
#include "mt19937.h"
#include "testBase.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cinttypes>
#include <mutex>

#include "harness/conversions.h"
#include "harness/parseParameters.h"

static std::recursive_mutex gLock;

extern     MTdata          d;
#define MASK(n) ((1 << (n)) - 1)

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

void init_long_data(uint64_t indx, uint32_t num_elements, cl_long *input_ptr[],
                    MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint64_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_long *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint64_t, cl_long>(specialValues[index + offset]);
        }
        for (; index < num_elements; index++)
        {
            ptr[index] = bitcast<cl_ulong, cl_long>(genrand_int64(d));
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_ulong_data(uint64_t indx, uint32_t num_elements,
                     cl_ulong *input_ptr[], MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint64_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_ulong *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint64_t, cl_ulong>(specialValues[index + offset]);
        }
        for (; index < num_elements; index++)
        {
            ptr[index] = bitcast<cl_ulong, cl_ulong>(genrand_int64(d));
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_int_data(uint64_t indx, uint32_t num_elements, cl_int *input_ptr[],
                   MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint32_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_int *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint32_t, cl_int>(specialValues[index + offset]);
        }
        for (; index < num_elements; index++)
        {
            ptr[index] = bitcast<cl_uint, cl_int>(genrand_int32(d));
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_uint_data(uint64_t indx, uint32_t num_elements, cl_uint *input_ptr[],
                    MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint32_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_uint *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint32_t, cl_uint>(specialValues[index + offset]);
        }
        for (; index < num_elements; index++)
        {
            ptr[index] = bitcast<cl_uint, cl_uint>(genrand_int32(d));
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_short_data(uint64_t indx, uint32_t num_elements,
                     cl_short *input_ptr[], MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint16_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_short *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint16_t, cl_short>(specialValues[index + offset]);
        }
        for (; (index + 1) < num_elements; index += 2)
        {
            cl_uint random = genrand_int32(d);
            ptr[index] = bitcast<cl_ushort, cl_short>(random & 0xffff);
            ptr[index + 1] = bitcast<cl_ushort, cl_short>(random >> 16);
        }
        if (index < num_elements)
        {
            ptr[index] =
                bitcast<cl_ushort, cl_short>(genrand_int32(d) & 0xffff);
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_ushort_data(uint64_t indx, uint32_t num_elements,
                      cl_ushort *input_ptr[], MTdata d, int num_runs_shift)
{
    auto specialValues = GetIntSpecialValues<uint16_t>(gLock, 1, gWimpyMode);
    assert((1ULL << (num_runs_shift / 2)) >= specialValues.size());
    auto init = [&specialValues, &num_elements, &d](cl_ushort *ptr,
                                                    uint64_t offset) {
        uint32_t index;
        for (index = 0;
             ((index + offset) < specialValues.size()) && index < num_elements;
             index++)
        {
            ptr[index] =
                bitcast<uint16_t, cl_ushort>(specialValues[index + offset]);
        }
        for (; (index + 1) < num_elements; index += 2)
        {
            cl_uint random = genrand_int32(d);
            ptr[index] = bitcast<cl_ushort, cl_ushort>(random & 0xffff);
            ptr[index + 1] = bitcast<cl_ushort, cl_ushort>(random >> 16);
        }
        if (index < num_elements)
        {
            ptr[index] =
                bitcast<cl_ushort, cl_ushort>(genrand_int32(d) & 0xffff);
        }
    };
    init(input_ptr[0], indx & MASK(num_runs_shift / 2));
    init(input_ptr[1], indx >> (num_runs_shift / 2));
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

void init_char_data(uint64_t indx, uint32_t num_elements, cl_char *input_ptr[],
                    MTdata d)
{
    for (uint32_t j = 0; j < num_elements; j++)
    {
        cl_ushort bits = (indx + j) & 0xffff;
        ((cl_char *)input_ptr[0])[j] = bitcast<cl_uchar, cl_char>(bits & 0xff);
        ((cl_char *)input_ptr[1])[j] = bitcast<cl_uchar, cl_char>(bits >> 8);
    }
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

void init_uchar_data(uint64_t indx, uint32_t num_elements,
                     cl_uchar *input_ptr[], MTdata d)
{
    for (uint32_t j = 0; j < num_elements; j++)
    {
        cl_ushort bits = (indx + j) & 0xffff;
        ((cl_uchar *)input_ptr[0])[j] =
            bitcast<cl_uchar, cl_uchar>(bits & 0xff);
        ((cl_uchar *)input_ptr[1])[j] = bitcast<cl_uchar, cl_uchar>(bits >> 8);
    }
}


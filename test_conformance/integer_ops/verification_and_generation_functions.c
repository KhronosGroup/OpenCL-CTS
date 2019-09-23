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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/conversions.h"

extern     MTdata          d;

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

const size_t vector_aligns[] = {0, 1, 2, 4, 4,
    8, 8, 8, 8,
    16, 16, 16, 16,
    16, 16, 16, 16};

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
                    log_error("cl_long Verification failed at element %ld of %ld : 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Vector shift failure at element %ld: original is 0x%llx %s %d (0x%llx)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %lld (0x%llx).\n", (int)log2(sizeof(cl_long)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {

                    log_error("cl_long Verification failed at element %ld of %ld (%ld): 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%llx %s %d (0x%llx)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %lld (0x%llx).\n", (int)log2(sizeof(cl_long)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%llx < 0x%llx) ? 0x%llx : 0x%llx = 0x%llx, got 0x%llx\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_long Verification failed at element %ld of %ld: 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_long_data(uint64_t indx, int num_elements, cl_long *input_ptr[], MTdata d)
{
    cl_ulong        *p = (cl_ulong *)input_ptr[0];
    int         j;

    if (indx == 0) {
        // Do the tricky values the first time around
        fill_test_values( input_ptr[ 0 ], input_ptr[ 1 ], (size_t)num_elements, d );
    } else {
        // Then just test lots of random ones.
        for (j=0; j<num_elements; j++) {
            cl_uint a = (cl_uint)genrand_int32(d);
            cl_uint b = (cl_uint)genrand_int32(d);
            p[j] = ((cl_ulong)a <<32 | b);
        }
        p = (cl_ulong *)input_ptr[1];
        for (j=0; j<num_elements; j++) {
            cl_uint a = (cl_uint)genrand_int32(d);
            cl_uint b = (cl_uint)genrand_int32(d);
            p[j] = ((cl_ulong)a <<32 | b);
        }
    }
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
                    log_error("cl_ulong Verification failed at element %ld of %ld: 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%llx %s %d (0x%llx)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %llu (0x%llx).\n", (int)log2(sizeof(cl_ulong)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_ulong Verification failed at element %ld of %ld (%ld): 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%llx %s %d (0x%llx)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %lld (0x%llx).\n", (int)log2(sizeof(cl_long)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld of %ld (%ld): (0x%llx < 0x%llx) ? 0x%llx : 0x%llx = 0x%llx, got 0x%llx\n", i, n, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_ulong Verification failed at element %ld of %ld: 0x%llx %s 0x%llx = 0x%llx, got 0x%llx\n", i, n, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_ulong_data(uint64_t indx, int num_elements, cl_ulong *input_ptr[], MTdata d)
{
    cl_ulong        *p = (cl_ulong *)input_ptr[0];
    int            j;

    if (indx == 0)
    {
        // Do the tricky values the first time around
        fill_test_values( (cl_long*)input_ptr[ 0 ], (cl_long*)input_ptr[ 1 ], (size_t)num_elements, d );
    }
    else
    {
        // Then just test lots of random ones.
        for (j=0; j<num_elements; j++)
        {
            cl_ulong a = genrand_int32(d);
            cl_ulong b = genrand_int32(d);
            // Fill in the top, bottom, and middle, remembering that random only sets 31 bits.
            p[j] = (a <<32) | b;
        }
        p = (cl_ulong *)input_ptr[1];
        for (j=0; j<num_elements; j++)
        {
            cl_ulong a = genrand_int32(d);
            cl_ulong b = genrand_int32(d);
            // Fill in the top, bottom, and middle, remembering that random only sets 31 bits.
            p[j] = (a <<32) | b;
        }
    }
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
                    log_error("cl_int Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_int)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_int Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_int)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_int Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_int_data(uint64_t indx, int num_elements, cl_int *input_ptr[], MTdata d)
{
    static const cl_int specialCaseList[] = { 0, -1, 1, CL_INT_MIN, CL_INT_MIN + 1, CL_INT_MAX };
    int            j;

    // Set the inputs to a random number
    for (j=0; j<num_elements; j++)
    {
        ((cl_int *)input_ptr[0])[j] = (cl_int)genrand_int32(d);
        ((cl_int *)input_ptr[1])[j] = (cl_int)genrand_int32(d);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_int *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_int *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
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
                    log_error("cl_uint Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uint)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_uint Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uint)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_uint Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_uint_data(uint64_t indx, int num_elements, cl_uint *input_ptr[], MTdata d)
{
    static cl_uint specialCaseList[] = { 0, (cl_uint) CL_INT_MAX, (cl_uint) CL_INT_MAX + 1, CL_UINT_MAX-1, CL_UINT_MAX };
    int            j;

    // Set the first input to an incrementing number
    // Set the second input to a random number
    for (j=0; j<num_elements; j++)
    {
        ((cl_uint *)input_ptr[0])[j] = genrand_int32(d);
        ((cl_uint *)input_ptr[1])[j] = genrand_int32(d);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_uint *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_uint *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
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
                    log_error("cl_short Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_short)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_short Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_short)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_short Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_short_data(uint64_t indx, int num_elements, cl_short *input_ptr[], MTdata d)
{
    static const cl_short specialCaseList[] = { 0, -1, 1, CL_SHRT_MIN, CL_SHRT_MIN + 1, CL_SHRT_MAX };
    int            j;

    // Set the inputs to a random number
    for (j=0; j<num_elements; j++)
    {
        cl_uint bits = genrand_int32(d);
        ((cl_short *)input_ptr[0])[j] = (cl_short) bits;
        ((cl_short *)input_ptr[1])[j] = (cl_short) (bits >> 16);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_short *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_short *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
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
                    log_error("cl_ushort Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_ushort)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_ushort Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_ushort)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_ushort Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_ushort_data(uint64_t indx, int num_elements, cl_ushort *input_ptr[], MTdata d)
{
    static const cl_ushort specialCaseList[] = { 0, -1, 1, CL_SHRT_MAX, CL_SHRT_MAX + 1, CL_USHRT_MAX };
    int            j;

    // Set the inputs to a random number
    for (j=0; j<num_elements; j++)
    {
        cl_uint bits = genrand_int32(d);
        ((cl_ushort *)input_ptr[0])[j] = (cl_ushort) bits;
        ((cl_ushort *)input_ptr[1])[j] = (cl_ushort) (bits >> 16);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_ushort *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_ushort *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
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
                    log_error("cl_char Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_char)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_char Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_long)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_char Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_char_data(uint64_t indx, int num_elements, cl_char *input_ptr[], MTdata d)
{
    static const cl_char specialCaseList[] = { 0, -1, 1, CL_CHAR_MIN, CL_CHAR_MIN + 1, CL_CHAR_MAX };
    int            j;

    // FIXME comment below might not be appropriate for
    // vector data.  Yes, checking every scalar char against every
    // scalar char is only 2^16 ~ 64000 tests, but once we get to vec3,
    // vec4, vec8...

    // in the meantime, this means I can use [] to access vec3 instead of
    // vload3 / vstore3 :D

    // FIXME: we really should just check every char against every char here
    // Set the inputs to a random number
    for (j=0; j<num_elements; j++)
    {
        cl_uint bits = genrand_int32(d);
        ((cl_char *)input_ptr[0])[j] = (cl_char) bits;
        ((cl_char *)input_ptr[1])[j] = (cl_char) (bits >> 16);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_char *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_char *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
}


// =======================================
// uchar
// =======================================
int
verify_uchar(int test, size_t vector_size, cl_uchar *inptrA, cl_uchar *inptrB, cl_uchar *outptr, size_t n)
{
    cl_uchar r;
    cl_uint  shift_mask = vector_size == 1 ? (cl_uint)(sizeof(cl_uint)*8)-1
    : (cl_uint)(sizeof(cl_uchar)*8)-1;;
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
                    log_error("cl_uchar Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
                    log_error("\t1) Shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[i], inptrB[i]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uchar)*8),  inptrB[i]&shift_mask, inptrB[i]&shift_mask);
                }
                else if (test == 10 || test == 11) {
                    log_error("cl_uchar Verification failed at element %ld (%ld): 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[i], tests[test], inptrB[j], r, outptr[i]);
                    log_error("\t1) Scalar shift failure at element %ld: original is 0x%x %s %d (0x%x)\n", i, inptrA[i], tests[test], (int)inptrB[j], inptrB[j]);
                    log_error("\t2) Take the %d LSBs of the shift to get the final shift amount %d (0x%x).\n", (int)log2(sizeof(cl_uchar)*8),  inptrB[j]&shift_mask, inptrB[j]&shift_mask);
                } else if (test == 13) {
                    log_error("cl_int Verification failed at element %ld (%ld): (0x%x < 0x%x) ? 0x%x : 0x%x = 0x%x, got 0x%x\n", i, j, inptrA[j], inptrB[j],
                              inptrA[i], inptrB[i], r, outptr[i]);
                } else {
                    log_error("cl_uchar Verification failed at element %ld: 0x%x %s 0x%x = 0x%x, got 0x%x\n", i, inptrA[i], tests[test], inptrB[i], r, outptr[i]);
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

void
init_uchar_data(uint64_t indx, int num_elements, cl_uchar *input_ptr[], MTdata d)
{
    static const cl_uchar specialCaseList[] = { 0, -1, 1, CL_CHAR_MAX, CL_CHAR_MAX + 1, CL_UCHAR_MAX };
    int            j;

    // FIXME: we really should just check every char against every char here

    // Set the inputs to a random number
    for (j=0; j<num_elements; j++)
    {
        cl_uint bits = genrand_int32(d);
        ((cl_uchar *)input_ptr[0])[j] = (cl_uchar) bits;
        ((cl_uchar *)input_ptr[1])[j] = (cl_uchar) (bits >> 16);
    }

    // Init the first few values to test special cases
    {
        size_t x, y, index = 0;
        for( x = 0; x < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); x++ )
            for( y = 0; y < sizeof( specialCaseList ) / sizeof( specialCaseList[0] ); y++ )
            {
                ((cl_uchar *)input_ptr[0])[index] = specialCaseList[x];
                ((cl_uchar *)input_ptr[1])[index++] = specialCaseList[y];
            }
    }
}


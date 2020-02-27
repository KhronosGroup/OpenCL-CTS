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
#include "harness/errorHelpers.h"

#include <stdio.h>
#include "test_select.h"


//-----------------------------------------
// Definitions and initializations
//-----------------------------------------


const char *type_name[kTypeCount] = {
    "uchar", "char",
    "ushort", "short",
    "uint",   "int",
    "float",  "ulong", "long", "double" };

const size_t type_size[kTypeCount] = {
    sizeof(cl_uchar), sizeof(cl_char),
    sizeof(cl_ushort), sizeof(cl_short),
    sizeof(cl_uint), sizeof(cl_int),
    sizeof(cl_float), sizeof(cl_ulong), sizeof(cl_long), sizeof( cl_double ) };

const Type ctype[kTypeCount][2] = {
    { kuchar,  kchar },     // uchar
    { kuchar,  kchar },     // char
    { kushort, kshort},     // ushort
    { kushort, kshort},     // short
    { kuint,   kint  },     // uint
    { kuint,   kint  },     // int
    { kuint,   kint  },     // float
    { kulong,  klong },     // ulong
    { kulong,  klong },     // long
    { kulong,  klong }     // double
};


//-----------------------------------------
// Reference functions
//-----------------------------------------

void refselect_1i8(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_char *d, *x, *y, *m;
    d = (cl_char*) dest;
    x = (cl_char*) src1;
    y = (cl_char*) src2;
    m = (cl_char*) cmp;
    for (i=0; i < count; ++i) {
        d[i] = m[i] ? y[i] : x[i];
    }
}

void refselect_1u8(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uchar *d, *x, *y;
    cl_char *m;
    d = (cl_uchar*) dest;
    x = (cl_uchar*) src1;
    y = (cl_uchar*) src2;
    m = (cl_char*) cmp;
    for (i=0; i < count; ++i) {
        d[i] = m[i] ? y[i] : x[i];
    }
}

void refselect_1i16(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_short *d, *x, *y, *m;
    d = (cl_short*) dest;
    x = (cl_short*) src1;
    y = (cl_short*) src2;
    m = (cl_short*) cmp;

    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u16(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ushort *d, *x, *y;
    cl_short *m;
    d = (cl_ushort*) dest;
    x = (cl_ushort*) src1;
    y = (cl_ushort*) src2;
    m = (cl_short*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i32(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_int *d, *x, *y, *m;
    d = (cl_int*)dest;
    x = (cl_int*)src1;
    y = (cl_int*)src2;
    m = (cl_int*)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u32(void *dest, void *src1, void *src2, void *cmp, size_t count){
    size_t i;
    cl_uint *d, *x, *y;
    cl_int *m;
    d = (cl_uint*)dest;
    x = (cl_uint*)src1;
    y = (cl_uint*)src2;
    m = (cl_int*)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i64(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y, *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u64(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y;
    cl_long *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i8u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_char *d, *x, *y;
    cl_uchar *m;
    d = (cl_char*) dest;
    x = (cl_char*) src1;
    y = (cl_char*) src2;
    m = (cl_uchar*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u8u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uchar *d, *x, *y, *m;
    d = (cl_uchar*) dest;
    x = (cl_uchar*) src1;
    y = (cl_uchar*) src2;
    m = (cl_uchar*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i16u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_short *d, *x, *y;
    cl_ushort *m;
    d = (cl_short*) dest;
    x = (cl_short*) src1;
    y = (cl_short*) src2;
    m = (cl_ushort*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u16u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ushort *d, *x, *y, *m;
    d = (cl_ushort*) dest;
    x = (cl_ushort*) src1;
    y = (cl_ushort*) src2;
    m = (cl_ushort*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i32u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_int *d, *x, *y;
    cl_uint *m;
    d = (cl_int*) dest;
    x = (cl_int*) src1;
    y = (cl_int*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u32u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uint *d, *x, *y, *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i64u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y;
    cl_ulong *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u64u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y, *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ffi(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_int *d, *x, *y;
    cl_int *m;
    d = (cl_int*) dest;
    x = (cl_int*) src1;
    y = (cl_int*) src2;
    m = (cl_int*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ffu(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uint *d, *x, *y;
    cl_uint *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ddi(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y;
    cl_long *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ddu(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y;
    cl_ulong *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void vrefselect_1i8(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_char *d, *x, *y, *m;
    d = (cl_char*) dest;
    x = (cl_char*) src1;
    y = (cl_char*) src2;
    m = (cl_char*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80) ? y[i] : x[i];
}

void vrefselect_1u8(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uchar *d, *x, *y;
    cl_char *m;
    d = (cl_uchar*) dest;
    x = (cl_uchar*) src1;
    y = (cl_uchar*) src2;
    m = (cl_char*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80) ? y[i] : x[i];
}

void vrefselect_1i16(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_short *d, *x, *y, *m;
    d = (cl_short*) dest;
    x = (cl_short*) src1;
    y = (cl_short*) src2;
    m = (cl_short*) cmp;

    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000) ? y[i] : x[i];
}

void vrefselect_1u16(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ushort *d, *x, *y;
    cl_short *m;
    d = (cl_ushort*) dest;
    x = (cl_ushort*)src1;
    y = (cl_ushort*)src2;
    m = (cl_short*)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000) ? y[i] : x[i];
}

void vrefselect_1i32(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_int *d, *x, *y, *m;
    d = (cl_int*) dest;
    x = (cl_int*) src1;
    y = (cl_int*) src2;
    m = (cl_int*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_1u32(void *dest, void *src1, void *src2, void *cmp, size_t count){
    size_t i;
    cl_uint *d, *x, *y;
    cl_int *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_int*) cmp;

    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_1i64(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y, *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_1u64(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y;
    cl_long *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_1i8u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_char *d, *x, *y;
    cl_uchar *m;
    d = (cl_char*) dest;
    x = (cl_char*) src1;
    y = (cl_char*) src2;
    m = (cl_uchar*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80U) ? y[i] : x[i];
}

void vrefselect_1u8u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uchar *d, *x, *y, *m;
    d = (cl_uchar*) dest;
    x = (cl_uchar*) src1;
    y = (cl_uchar*) src2;
    m = (cl_uchar*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80U) ? y[i] : x[i];
}

void vrefselect_1i16u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_short *d, *x, *y;
    cl_ushort *m;
    d = (cl_short*) dest;
    x = (cl_short*) src1;
    y = (cl_short*) src2;
    m = (cl_ushort*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000U) ? y[i] : x[i];
}

void vrefselect_1u16u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ushort *d, *x, *y, *m;
    d = (cl_ushort*) dest;
    x = (cl_ushort*) src1;
    y = (cl_ushort*) src2;
    m = (cl_ushort*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000U) ? y[i] : x[i];
}

void vrefselect_1i32u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_int *d, *x, *y;
    cl_uint *m;
    d = (cl_int*) dest;
    x = (cl_int*) src1;
    y = (cl_int*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_1u32u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uint *d, *x, *y, *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_1i64u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_long *d, *x, *y;
    cl_ulong *m;
    d = (cl_long*) dest;
    x = (cl_long*) src1;
    y = (cl_long*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

void vrefselect_1u64u(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y, *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

void vrefselect_ffi(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uint *d, *x, *y;
    cl_int *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_int*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_ffu(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_uint *d, *x, *y;
    cl_uint *m;
    d = (cl_uint*) dest;
    x = (cl_uint*) src1;
    y = (cl_uint*) src2;
    m = (cl_uint*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_ddi(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y;
    cl_long *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_long*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_ddu(void *dest, void *src1, void *src2, void *cmp, size_t count) {
    size_t i;
    cl_ulong *d, *x, *y;
    cl_ulong *m;
    d = (cl_ulong*) dest;
    x = (cl_ulong*) src1;
    y = (cl_ulong*) src2;
    m = (cl_ulong*) cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

// Define refSelects
Select refSelects[kTypeCount][2] =  {
    { refselect_1u8u,  refselect_1u8  }, // cl_uchar
    { refselect_1i8u,  refselect_1i8  }, // char
    { refselect_1u16u, refselect_1u16 }, // ushort
    { refselect_1i16u, refselect_1i16 }, // short
    { refselect_1u32u, refselect_1u32 }, // uint
    { refselect_1i32u, refselect_1i32 }, // int
    { refselect_ffu,   refselect_ffi  }, // float
    { refselect_1u64u, refselect_1u64 }, // ulong
    { refselect_1i64u, refselect_1i64 }, // long
    { refselect_ddu,   refselect_ddi }   // double
};

// Define vrefSelects (vector refSelects)
Select vrefSelects[kTypeCount][2] =  {
    { vrefselect_1u8u,  vrefselect_1u8  }, // cl_uchar
    { vrefselect_1i8u,  vrefselect_1i8  }, // char
    { vrefselect_1u16u, vrefselect_1u16 }, // ushort
    { vrefselect_1i16u, vrefselect_1i16 }, // short
    { vrefselect_1u32u, vrefselect_1u32 }, // uint
    { vrefselect_1i32u, vrefselect_1i32 }, // int
    { vrefselect_ffu,   vrefselect_ffi  }, // float
    { vrefselect_1u64u, vrefselect_1u64 }, // ulong
    { vrefselect_1i64u, vrefselect_1i64 }, // long
    { vrefselect_ddu,   vrefselect_ddi  }     // double
};


//-----------------------------------------
// Check functions
//-----------------------------------------
size_t check_uchar(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_uchar *t = (const cl_uchar *) test;
    const cl_uchar *c = (const cl_uchar *) correct;
    size_t i;

    for(i = 0; i < count; i++)
        if (t[i] != c[i]) {
            log_error("\n(check_uchar) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%2.2x vs 0x%2.2x\n", vector_size, i, count, c[i], t[i]);
            return i + 1;
        }

    return 0;
}

size_t check_char(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_char *t = (const cl_char *) test;
    const cl_char *c = (const cl_char *) correct;
    size_t i;


    for( i = 0; i < count; i++ )
        if( t[i] != c[i] ) {
            log_error("\n(check_char) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%2.2x vs 0x%2.2x\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_ushort(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_ushort *t = (const cl_ushort *) test;
    const cl_ushort *c = (const cl_ushort *) correct;
    size_t i;


    for( i = 0; i < count; i++ )
        if(t[i] != c[i]) {
            log_error("\n(check_ushort) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%4.4x vs 0x%4.4x\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_short(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_short *t = (const cl_short *) test;
    const cl_short *c = (const cl_short *) correct;
    size_t i;


    for (i = 0; i < count; i++)
        if(t[i] != c[i]) {
            log_error("\n(check_short) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_uint(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_uint *t = (const cl_uint *) test;
    const cl_uint *c = (const cl_uint *) correct;
    size_t i;



    for (i = 0; i < count; i++)
        if(t[i] != c[i]) {
            log_error("\n(check_uint) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_int(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_int *t = (const cl_int *) test;
    const cl_int *c = (const cl_int *) correct;
    size_t i;


    for(i = 0; i < count; i++)
        if( t[i] != c[i] ) {

            log_error("\n(check_int) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i, count, c[i], t[i]);
            log_error("\n(check_int) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i+1, count,c[i+1], t[i+1]);
            log_error("\n(check_int) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i+2, count,c[i+2], t[i+2]);
            log_error("\n(check_int) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i+3, count,c[i+3], t[i+3]);
            if(i) {
                log_error("\n(check_int) Error for vector size %ld found just after 0x%8.8lx:  "
                          "*0x%8.8x vs 0x%8.8x\n", vector_size, i-1, c[i-1], t[i-1]);
            }
            return i + 1;
        }

    return 0;
}

size_t check_ulong(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_ulong *t = (const cl_ulong *) test;
    const cl_ulong *c = (const cl_ulong *) correct;
    size_t i;


    for( i = 0; i < count; i++ )
        if( t[i] != c[i] ) {
            log_error("\n(check_ulong) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%16.16llx vs 0x%16.16llx\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_long(void *test, void *correct, size_t count, size_t vector_size) {
    const cl_long *t = (const cl_long *) test;
    const cl_long *c = (const cl_long *) correct;
    size_t i;


    for(i = 0; i < count; i++ )
        if(t[i] != c[i]) {
            log_error("\n(check_long) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%16.16llx vs 0x%16.16llx\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_float( void *test, void *correct, size_t count, size_t vector_size ) {
    const cl_uint *t = (const cl_uint *) test;
    const cl_uint *c = (const cl_uint *) correct;
    size_t i;


    for( i = 0; i < count; i++ )
        /* Allow nans to be binary different */
        if ((t[i] != c[i]) && !(isnan(((float *)correct)[i]) && isnan(((float *)test)[i]))) {
            log_error("\n(check_float) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%8.8x vs 0x%8.8x\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

size_t check_double( void *test, void *correct, size_t count, size_t vector_size ) {
    const cl_ulong *t = (const cl_ulong *) test;
    const cl_ulong *c = (const cl_ulong *) correct;
    size_t i;



    for( i = 0; i < count; i++ )
        /* Allow nans to be binary different */
        if ((t[i] != c[i]) && !(isnan(((double *)correct)[i]) && isnan(((double *)test)[i]))) {
            log_error("\n(check_double) Error for vector size %ld found at 0x%8.8lx (of 0x%8.8lx):  "
                      "*0x%16.16llx vs 0x%16.16llx\n", vector_size, i, count, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

CheckResults checkResults[kTypeCount] = {
    check_uchar, check_char, check_ushort, check_short, check_uint,
    check_int, check_float, check_ulong, check_long, check_double };

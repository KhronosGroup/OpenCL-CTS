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
#include "harness/errorHelpers.h"

#include <stdio.h>
#include <cinttypes>
#include "test_select.h"


//-----------------------------------------
// Definitions and initializations
//-----------------------------------------


const char *type_name[kTypeCount] = { "uchar", "char", "ushort", "short",
                                      "half",  "uint", "int",    "float",
                                      "ulong", "long", "double" };

const size_t type_size[kTypeCount] = {
    sizeof(cl_uchar), sizeof(cl_char), sizeof(cl_ushort), sizeof(cl_short),
    sizeof(cl_half),  sizeof(cl_uint), sizeof(cl_int),    sizeof(cl_float),
    sizeof(cl_ulong), sizeof(cl_long), sizeof(cl_double)
};

const Type ctype[kTypeCount][2] = {
    { kuchar, kchar }, // uchar
    { kuchar, kchar }, // char
    { kushort, kshort }, // ushort
    { kushort, kshort }, // short
    { kushort, kshort }, // half
    { kuint, kint }, // uint
    { kuint, kint }, // int
    { kuint, kint }, // float
    { kulong, klong }, // ulong
    { kulong, klong }, // long
    { kulong, klong } // double
};


//-----------------------------------------
// Reference functions
//-----------------------------------------

void refselect_1i8(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_char *const d = (cl_char *)dest;
    const cl_char *const x = (cl_char *)src1;
    const cl_char *const y = (cl_char *)src2;
    const cl_char *const m = (cl_char *)cmp;
    for (i=0; i < count; ++i) {
        d[i] = m[i] ? y[i] : x[i];
    }
}

void refselect_1u8(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uchar *const d = (cl_uchar *)dest;
    const cl_uchar *const x = (cl_uchar *)src1;
    const cl_uchar *const y = (cl_uchar *)src2;
    const cl_char *const m = (cl_char *)cmp;
    for (i=0; i < count; ++i) {
        d[i] = m[i] ? y[i] : x[i];
    }
}

void refselect_1i16(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_short *const d = (cl_short *)dest;
    const cl_short *const x = (cl_short *)src1;
    const cl_short *const y = (cl_short *)src2;
    const cl_short *const m = (cl_short *)cmp;

    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u16(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_short *const m = (cl_short *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i32(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_int *const d = (cl_int *)dest;
    const cl_int *const x = (cl_int *)src1;
    const cl_int *const y = (cl_int *)src2;
    const cl_int *const m = (cl_int *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u32(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_int *const m = (cl_int *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i64(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u64(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i8u(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_char *const d = (cl_char *)dest;
    const cl_char *const x = (cl_char *)src1;
    const cl_char *const y = (cl_char *)src2;
    const cl_uchar *const m = (cl_uchar *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u8u(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uchar *const d = (cl_uchar *)dest;
    const cl_uchar *const x = (cl_uchar *)src1;
    const cl_uchar *const y = (cl_uchar *)src2;
    const cl_uchar *const m = (cl_uchar *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i16u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_short *const d = (cl_short *)dest;
    const cl_short *const x = (cl_short *)src1;
    const cl_short *const y = (cl_short *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u16u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i32u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_int *const d = (cl_int *)dest;
    const cl_int *const x = (cl_int *)src1;
    const cl_int *const y = (cl_int *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u32u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1i64u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_1u64u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_hhi(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_short *const d = (cl_short *)dest;
    const cl_short *const x = (cl_short *)src1;
    const cl_short *const y = (cl_short *)src2;
    const cl_short *const m = (cl_short *)cmp;
    for (i = 0; i < count; ++i) d[i] = m[i] ? y[i] : x[i];
}

void refselect_hhu(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i = 0; i < count; ++i) d[i] = m[i] ? y[i] : x[i];
}

void refselect_ffi(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_int *const d = (cl_int *)dest;
    const cl_int *const x = (cl_int *)src1;
    const cl_int *const y = (cl_int *)src2;
    const cl_int *const m = (cl_int *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ffu(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ddi(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void refselect_ddu(void *const dest, const void *const src1,
                   const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = m[i] ? y[i] : x[i];
}

void vrefselect_1i8(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_char *const d = (cl_char *)dest;
    const cl_char *const x = (cl_char *)src1;
    const cl_char *const y = (cl_char *)src2;
    const cl_char *const m = (cl_char *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80) ? y[i] : x[i];
}

void vrefselect_1u8(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uchar *const d = (cl_uchar *)dest;
    const cl_uchar *const x = (cl_uchar *)src1;
    const cl_uchar *const y = (cl_uchar *)src2;
    const cl_char *const m = (cl_char *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80) ? y[i] : x[i];
}

void vrefselect_1i16(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_short *const d = (cl_short *)dest;
    const cl_short *const x = (cl_short *)src1;
    const cl_short *const y = (cl_short *)src2;
    const cl_short *const m = (cl_short *)cmp;

    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000) ? y[i] : x[i];
}

void vrefselect_1u16(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_short *const m = (cl_short *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000) ? y[i] : x[i];
}

void vrefselect_1i32(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_int *const d = (cl_int *)dest;
    const cl_int *const x = (cl_int *)src1;
    const cl_int *const y = (cl_int *)src2;
    const cl_int *const m = (cl_int *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_1u32(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_int *const m = (cl_int *)cmp;

    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_1i64(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_1u64(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_1i8u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_char *const d = (cl_char *)dest;
    const cl_char *const x = (cl_char *)src1;
    const cl_char *const y = (cl_char *)src2;
    const cl_uchar *const m = (cl_uchar *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80U) ? y[i] : x[i];
}

void vrefselect_1u8u(void *const dest, const void *const src1,
                     const void *const src2, const void *const cmp,
                     size_t count)
{
    size_t i;
    cl_uchar *const d = (cl_uchar *)dest;
    const cl_uchar *const x = (cl_uchar *)src1;
    const cl_uchar *const y = (cl_uchar *)src2;
    const cl_uchar *const m = (cl_uchar *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80U) ? y[i] : x[i];
}

void vrefselect_1i16u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_short *const d = (cl_short *)dest;
    const cl_short *const x = (cl_short *)src1;
    const cl_short *const y = (cl_short *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000U) ? y[i] : x[i];
}

void vrefselect_1u16u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000U) ? y[i] : x[i];
}

void vrefselect_1i32u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_int *const d = (cl_int *)dest;
    const cl_int *const x = (cl_int *)src1;
    const cl_int *const y = (cl_int *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_1u32u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_1i64u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_long *const d = (cl_long *)dest;
    const cl_long *const x = (cl_long *)src1;
    const cl_long *const y = (cl_long *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

void vrefselect_1u64u(void *const dest, const void *const src1,
                      const void *const src2, const void *const cmp,
                      size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

void vrefselect_hhi(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_short *const m = (cl_short *)cmp;
    for (i = 0; i < count; ++i) d[i] = (m[i] & 0x8000) ? y[i] : x[i];
}

void vrefselect_hhu(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ushort *const d = (cl_ushort *)dest;
    const cl_ushort *const x = (cl_ushort *)src1;
    const cl_ushort *const y = (cl_ushort *)src2;
    const cl_ushort *const m = (cl_ushort *)cmp;
    for (i = 0; i < count; ++i) d[i] = (m[i] & 0x8000U) ? y[i] : x[i];
}

void vrefselect_ffi(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_int *const m = (cl_int *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000) ? y[i] : x[i];
}

void vrefselect_ffu(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_uint *const d = (cl_uint *)dest;
    const cl_uint *const x = (cl_uint *)src1;
    const cl_uint *const y = (cl_uint *)src2;
    const cl_uint *const m = (cl_uint *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x80000000U) ? y[i] : x[i];
}

void vrefselect_ddi(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_long *const m = (cl_long *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000LL) ? y[i] : x[i];
}

void vrefselect_ddu(void *const dest, const void *const src1,
                    const void *const src2, const void *const cmp, size_t count)
{
    size_t i;
    cl_ulong *const d = (cl_ulong *)dest;
    const cl_ulong *const x = (cl_ulong *)src1;
    const cl_ulong *const y = (cl_ulong *)src2;
    const cl_ulong *const m = (cl_ulong *)cmp;
    for (i=0; i < count; ++i)
        d[i] = (m[i] & 0x8000000000000000ULL) ? y[i] : x[i];
}

// Define refSelects
Select refSelects[kTypeCount][2] = {
    { refselect_1u8u, refselect_1u8 }, // cl_uchar
    { refselect_1i8u, refselect_1i8 }, // char
    { refselect_1u16u, refselect_1u16 }, // ushort
    { refselect_1i16u, refselect_1i16 }, // short
    { refselect_hhu, refselect_hhi }, // half
    { refselect_1u32u, refselect_1u32 }, // uint
    { refselect_1i32u, refselect_1i32 }, // int
    { refselect_ffu, refselect_ffi }, // float
    { refselect_1u64u, refselect_1u64 }, // ulong
    { refselect_1i64u, refselect_1i64 }, // long
    { refselect_ddu, refselect_ddi } // double
};

// Define vrefSelects (vector refSelects)
Select vrefSelects[kTypeCount][2] = {
    { vrefselect_1u8u, vrefselect_1u8 }, // cl_uchar
    { vrefselect_1i8u, vrefselect_1i8 }, // char
    { vrefselect_1u16u, vrefselect_1u16 }, // ushort
    { vrefselect_1i16u, vrefselect_1i16 }, // short
    { vrefselect_hhu, vrefselect_hhi }, // half
    { vrefselect_1u32u, vrefselect_1u32 }, // uint
    { vrefselect_1i32u, vrefselect_1i32 }, // int
    { vrefselect_ffu, vrefselect_ffi }, // float
    { vrefselect_1u64u, vrefselect_1u64 }, // ulong
    { vrefselect_1i64u, vrefselect_1i64 }, // long
    { vrefselect_ddu, vrefselect_ddi } // double
};


//-----------------------------------------
// Check functions
//-----------------------------------------
size_t check_uchar(const void *const test, const void *const correct,
                   size_t count, size_t vector_size)
{
    const cl_uchar *const t = (const cl_uchar *)test;
    const cl_uchar *const c = (const cl_uchar *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_uchar) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%2.2x vs 0x%2.2x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }
    return 0;
}

size_t check_char(const void *const test, const void *const correct,
                  size_t count, size_t vector_size)
{
    const cl_char *const t = (const cl_char *)test;
    const cl_char *const c = (const cl_char *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_char) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%2.2x vs 0x%2.2x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_ushort(const void *const test, const void *const correct,
                    size_t count, size_t vector_size)
{
    const cl_ushort *const t = (const cl_ushort *)test;
    const cl_ushort *const c = (const cl_ushort *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_ushort) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%4.4x vs 0x%4.4x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_short(const void *const test, const void *const correct,
                   size_t count, size_t vector_size)
{
    const cl_short *const t = (const cl_short *)test;
    const cl_short *const c = (const cl_short *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_short) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%8.8x vs 0x%8.8x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_uint(const void *const test, const void *const correct,
                  size_t count, size_t vector_size)
{
    const cl_uint *const t = (const cl_uint *)test;
    const cl_uint *const c = (const cl_uint *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_uint) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%8.8x vs 0x%8.8x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_int(const void *const test, const void *const correct,
                 size_t count, size_t vector_size)
{
    const cl_int *const t = (const cl_int *)test;
    const cl_int *const c = (const cl_int *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {

                log_error("\n(check_int) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%8.8x vs 0x%8.8x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_ulong(const void *const test, const void *const correct,
                   size_t count, size_t vector_size)
{
    const cl_ulong *const t = (const cl_ulong *)test;
    const cl_ulong *const c = (const cl_ulong *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_ulong) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%16.16" PRIx64 " vs 0x%16.16" PRIx64 "\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_long(const void *const test, const void *const correct,
                  size_t count, size_t vector_size)
{
    const cl_long *const t = (const cl_long *)test;
    const cl_long *const c = (const cl_long *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++)
            if (t[i] != c[i])
            {
                log_error("\n(check_long) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%16.16" PRIx64 " vs 0x%16.16" PRIx64 "\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_half(const void *const test, const void *const correct,
                  size_t count, size_t vector_size)
{
    const cl_ushort *const t = (const cl_ushort *)test;
    const cl_ushort *const c = (const cl_ushort *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++) /* Allow nans to be binary different */
            if ((t[i] != c[i])
                && !(isnan(((cl_half *)correct)[i])
                     && isnan(((cl_half *)test)[i])))
            {
                log_error("\n(check_half) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%4.4x vs 0x%4.4x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_float(const void *const test, const void *const correct,
                   size_t count, size_t vector_size)
{
    const cl_uint *const t = (const cl_uint *)test;
    const cl_uint *const c = (const cl_uint *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++) /* Allow nans to be binary different */
            if ((t[i] != c[i])
                && !(isnan(((float *)correct)[i]) && isnan(((float *)test)[i])))
            {
                log_error("\n(check_float) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%8.8x vs 0x%8.8x\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

size_t check_double(const void *const test, const void *const correct,
                    size_t count, size_t vector_size)
{
    const cl_ulong *const t = (const cl_ulong *)test;
    const cl_ulong *const c = (const cl_ulong *)correct;
    size_t i;

    if (memcmp(t, c, count * sizeof(c[0])) != 0)
    {
        for (i = 0; i < count; i++) /* Allow nans to be binary different */
            if ((t[i] != c[i])
                && !(isnan(((double *)correct)[i])
                     && isnan(((double *)test)[i])))
            {
                log_error("\n(check_double) Error for vector size %zu found at "
                          "0x%8.8zx (of 0x%8.8zx):  "
                          "*0x%16.16" PRIx64 " vs 0x%16.16" PRIx64 "\n",
                          vector_size, i, count, c[i], t[i]);
                return i + 1;
            }
    }

    return 0;
}

CheckResults checkResults[kTypeCount] = {
    check_uchar, check_char, check_ushort, check_short,
    check_half,  check_uint, check_int,    check_float,
    check_ulong, check_long, check_double
};

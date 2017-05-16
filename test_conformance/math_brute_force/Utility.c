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
#include "Utility.h"

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif

void MulD(double *rhi, double *rlo, double u, double v)
{
    const double c = 134217729.0; // 1+2^27
    double up, u1, u2, vp, v1, v2;

    up = u*c;
    u1 = (u - up) + up;
    u2 = u - u1;

    vp = v*c;
    v1 = (v - vp) + vp;
    v2 = v - v1;

    double rh = u*v;
    double rl = (((u1*v1 - rh) + (u1*v2)) + (u2*v1)) + (u2*v2);

    *rhi = rh;
    *rlo = rl;
}

void AddD(double *rhi, double *rlo, double a, double b)
{
    double zhi, zlo;
    zhi = a + b;
    if(fabs(a) > fabs(b)) {
        zlo = zhi - a;
        zlo = b - zlo;
    }
    else {
        zlo = zhi - b;
        zlo = a - zlo;
    }

    *rhi = zhi;
    *rlo = zlo;
}

void MulDD(double *rhi, double *rlo, double xh, double xl, double yh, double yl)
{
    double mh, ml;
    double c = 134217729.0;
    double up, u1, u2, vp, v1, v2;

    up = xh*c;
    u1 = (xh - up) + up;
    u2 = xh - u1;

    vp = yh*c;
    v1 = (yh - vp) + vp;
    v2 = yh - v1;

    mh = xh*yh;
    ml = (((u1*v1 - mh) + (u1*v2)) + (u2*v1)) + (u2*v2);
    ml += xh*yl + xl*yh;

    *rhi = mh + ml;
    *rlo = (mh - (*rhi)) + ml;
}

void AddDD(double *rhi, double *rlo, double xh, double xl, double yh, double yl)
{
    double r, s;
    r = xh + yh;
    s = (fabs(xh) > fabs(yh)) ? (xh - r + yh + yl + xl) : (yh - r + xh + xl + yl);
    *rhi = r + s;
    *rlo = (r - (*rhi)) + s;
}

void DivideDD(double *chi, double *clo, double a, double b)
{
    *chi = a / b;
    double rhi, rlo;
    MulD(&rhi, &rlo, *chi, b);
    AddDD(&rhi, &rlo, -rhi, -rlo, a, 0.0);
    *clo = rhi / b;
}

// These functions comapre two floats/doubles. Since some platforms may choose to
// flush denormals to zeros before comparison, comparison like a < b may give wrong
// result in "certain cases" where we do need correct compasion result when operands
// are denormals .... these functions comapre floats/doubles using signed integer/long int
// rep. In other cases, when flushing to zeros is fine, these should not be used.
// Also these doesn't check for nans and assume nans are handled separately as special edge case
// by the caller which calls these functions
// return 0 if both are equal, 1 if x > y and -1 if x < y.

inline
int compareFloats(float x, float y)
{
    int32f_t a, b;

    a.f = x;
    b.f = y;

    if( a.i & 0x80000000 )
        a.i = 0x80000000 - a.i;
    if( b.i & 0x80000000 )
        b.i = 0x80000000 - b.i;

    if( a.i == b.i )
        return 0;

    return a.i < b.i ? -1 : 1;
}

inline
int compareDoubles(double x, double y)
{
    int64d_t a, b;

    a.d = x;
    b.d = y;

    if( a.l & 0x8000000000000000LL )
        a.l = 0x8000000000000000LL - a.l;
    if( b.l & 0x8000000000000000LL )
        b.l = 0x8000000000000000LL - b.l;

    if( a.l == b.l )
        return 0;

    return a.l < b.l ? -1 : 1;
}

void logFunctionInfo(const char *fname, unsigned int float_size, unsigned int isFastRelaxed)
{
    char const *fpSizeStr = NULL;
    char const *fpFastRelaxedStr = "";
    switch (float_size) {
    case sizeof(cl_double):
        fpSizeStr = "fp64";
        break;
    case sizeof(cl_float):
        fpSizeStr = "fp32";
        break;
    case sizeof(cl_half):
        fpSizeStr = "fp16";
        break;
    }
    if (isFastRelaxed) {
        fpFastRelaxedStr = "rlx";
    }
    vlog("%15s %4s %4s",fname, fpSizeStr, fpFastRelaxedStr);
}


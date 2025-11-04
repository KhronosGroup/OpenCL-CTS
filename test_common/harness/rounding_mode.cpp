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
#include "rounding_mode.h"

#if (defined(__arm__) || defined(__aarch64__))
#define FPSCR_FZ (1 << 24) // Flush-To-Zero mode
#define FPSCR_ROUND_MASK (3 << 22) // Rounding mode:

#define _ARM_FE_FTZ 0x1000000
#define _ARM_FE_NFTZ 0x0
#if defined(__aarch64__)
#define _FPU_GETCW(cw) __asm__("MRS %0,FPCR" : "=r"(cw))
#define _FPU_SETCW(cw) __asm__("MSR FPCR,%0" : : "ri"(cw))
#else
#define _FPU_GETCW(cw) __asm__("VMRS %0,FPSCR" : "=r"(cw))
#define _FPU_SETCW(cw) __asm__("VMSR FPSCR,%0" : : "ri"(cw))
#endif
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#define _ARM_FE_TONEAREST 0x0
#define _ARM_FE_UPWARD 0x400000
#define _ARM_FE_DOWNWARD 0x800000
#define _ARM_FE_TOWARDZERO 0xc00000
RoundingMode set_round(RoundingMode r, Type outType)
{
    static const int flt_rounds[kRoundingModeCount] = {
        _ARM_FE_TONEAREST, _ARM_FE_TONEAREST, _ARM_FE_UPWARD, _ARM_FE_DOWNWARD,
        _ARM_FE_TOWARDZERO
    };
    static const int int_rounds[kRoundingModeCount] = {
        _ARM_FE_TOWARDZERO, _ARM_FE_TONEAREST, _ARM_FE_UPWARD, _ARM_FE_DOWNWARD,
        _ARM_FE_TOWARDZERO
    };
    const int *p = int_rounds;
    if (outType == kfloat || outType == kdouble) p = flt_rounds;

    int64_t fpscr = 0;
    RoundingMode oldRound = get_round();

    _FPU_GETCW(fpscr);
    _FPU_SETCW(p[r] | (fpscr & ~FPSCR_ROUND_MASK));

    return oldRound;
}

RoundingMode get_round(void)
{
    int64_t fpscr;
    int oldRound;

    _FPU_GETCW(fpscr);
    oldRound = (fpscr & FPSCR_ROUND_MASK);

    switch (oldRound)
    {
        case _ARM_FE_TONEAREST: return kRoundToNearestEven;
        case _ARM_FE_UPWARD: return kRoundUp;
        case _ARM_FE_DOWNWARD: return kRoundDown;
        case _ARM_FE_TOWARDZERO: return kRoundTowardZero;
    }

    return kDefaultRoundingMode;
}

#elif !(defined(_WIN32) && defined(_MSC_VER))
RoundingMode set_round(RoundingMode r, Type outType)
{
    static const int flt_rounds[kRoundingModeCount] = {
        FE_TONEAREST, FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO
    };
    static const int int_rounds[kRoundingModeCount] = {
        FE_TOWARDZERO, FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO
    };
    const int *p = int_rounds;
    if (outType == kfloat || outType == kdouble) p = flt_rounds;
    int oldRound = fegetround();
    fesetround(p[r]);

    switch (oldRound)
    {
        case FE_TONEAREST: return kRoundToNearestEven;
        case FE_UPWARD: return kRoundUp;
        case FE_DOWNWARD: return kRoundDown;
        case FE_TOWARDZERO: return kRoundTowardZero;
        default: abort(); // ??!
    }
    return kDefaultRoundingMode; // never happens
}

RoundingMode get_round(void)
{
    int oldRound = fegetround();

    switch (oldRound)
    {
        case FE_TONEAREST: return kRoundToNearestEven;
        case FE_UPWARD: return kRoundUp;
        case FE_DOWNWARD: return kRoundDown;
        case FE_TOWARDZERO: return kRoundTowardZero;
    }

    return kDefaultRoundingMode;
}

#else
RoundingMode set_round(RoundingMode r, Type outType)
{
    static const int flt_rounds[kRoundingModeCount] = { _RC_NEAR, _RC_NEAR,
                                                        _RC_UP, _RC_DOWN,
                                                        _RC_CHOP };
    static const int int_rounds[kRoundingModeCount] = { _RC_CHOP, _RC_NEAR,
                                                        _RC_UP, _RC_DOWN,
                                                        _RC_CHOP };
    const int *p =
        (outType == kfloat || outType == kdouble) ? flt_rounds : int_rounds;
    unsigned int oldRound;

    int err = _controlfp_s(&oldRound, 0, 0); // get rounding mode into oldRound
    if (err)
    {
        vlog_error("\t\tERROR: -- cannot get rounding mode in %s:%d\n",
                   __FILE__, __LINE__);
        return kDefaultRoundingMode; // what else never happens
    }

    oldRound &= _MCW_RC;

    RoundingMode old = (oldRound == _RC_NEAR)
        ? kRoundToNearestEven
        : (oldRound == _RC_UP) ? kRoundUp
                               : (oldRound == _RC_DOWN)
                ? kRoundDown
                : (oldRound == _RC_CHOP) ? kRoundTowardZero
                                         : kDefaultRoundingMode;

    _controlfp_s(&oldRound, p[r], _MCW_RC); // setting new rounding mode
    return old; // returning old rounding mode
}

RoundingMode get_round(void)
{
    unsigned int oldRound;

    int err = _controlfp_s(&oldRound, 0, 0); // get rounding mode into oldRound
    oldRound &= _MCW_RC;
    return (oldRound == _RC_NEAR)
        ? kRoundToNearestEven
        : (oldRound == _RC_UP) ? kRoundUp
                               : (oldRound == _RC_DOWN)
                ? kRoundDown
                : (oldRound == _RC_CHOP) ? kRoundTowardZero
                                         : kDefaultRoundingMode;
}

#endif

//
// FlushToZero() sets the host processor into ftz mode.  It is intended to have
// a remote effect on the behavior of the code in basic_test_conversions.c. Some
// host processors may not support this mode, which case you'll need to do some
// clamping in software by testing against FLT_MIN or DBL_MIN in that file.
//
// Note: IEEE-754 says conversions are basic operations.  As such they do *NOT*
// have the behavior in section 7.5.3 of the OpenCL spec. They *ALWAYS* flush to
// zero for subnormal inputs or outputs when FTZ mode is on like other basic
// operators do (e.g. add, subtract, multiply, divide, etc.)
//
// Configuring hardware to FTZ mode varies by platform.
// CAUTION: Some C implementations may also fail to behave properly in this
// mode.
//
//  On PowerPC, it is done by setting the FPSCR into non-IEEE mode.
//  On Intel, you can do this by turning on the FZ and DAZ bits in the MXCSR --
//  provided that SSE/SSE2
//          is used for floating point computation! If your OS uses x87, you'll
//          need to figure out how to turn that off for the conversions code in
//          basic_test_conversions.c so that they flush to zero properly.
//          Otherwise, you'll need to add appropriate software clamping to
//          basic_test_conversions.c in which case, these function are at
//          liberty to do nothing.
//
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)               \
    || defined(_M_X64)
#include <xmmintrin.h>
#elif defined(__PPC__)
#include <fpu_control.h>
#elif defined(__mips__)
#include "mips/m32c1.h"
#endif
void *FlushToZero(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)               \
    || defined(_M_X64)
    union {
        unsigned int i;
        void *p;
    } u = { _mm_getcsr() };
    _mm_setcsr(u.i | 0x8040);
    return u.p;
#elif defined(__arm__) || defined(__aarch64__) // Clang
    int64_t fpscr;
    _FPU_GETCW(fpscr);
    _FPU_SETCW(fpscr | FPSCR_FZ);
    return NULL;
#elif defined(_M_ARM64) // Visual Studio
    uint64_t fpscr;
    fpscr = _ReadStatusReg(ARM64_FPSR);
    _WriteStatusReg(ARM64_FPCR, fpscr | (1U << 24));
    return NULL;
#elif defined(__PPC__)
    fpu_control_t flags = 0;
    _FPU_GETCW(flags);
    flags |= _FPU_MASK_NI;
    _FPU_SETCW(flags);
    return NULL;
#elif defined(__mips__)
    fpa_bissr(FPA_CSR_FS);
    return NULL;
#else
#error Unknown arch
#endif
#else
#error  Please configure FlushToZero and UnFlushToZero to behave properly on this operating system.
#endif
}

// Undo the effects of FlushToZero above, restoring the host to default
// behavior, using the information passed in p.
void UnFlushToZero(void *p)
{
#if defined(__APPLE__) || defined(__linux__) || defined(_WIN32)
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)               \
    || defined(_M_X64)
    union {
        void *p;
        unsigned int i;
    } u = { p };
    _mm_setcsr(u.i);
#elif defined(__arm__) || defined(__aarch64__) // Clang
    int64_t fpscr;
    _FPU_GETCW(fpscr);
    _FPU_SETCW(fpscr & ~FPSCR_FZ);
#elif defined(_M_ARM64) // Visual Studio
    uint64_t fpscr;
    fpscr = _ReadStatusReg(ARM64_FPSR);
    _WriteStatusReg(ARM64_FPCR, fpscr & ~(1U << 24));
#elif defined(__PPC__)
    fpu_control_t flags = 0;
    _FPU_GETCW(flags);
    flags &= ~_FPU_MASK_NI;
    _FPU_SETCW(flags);
#elif defined(__mips__)
    fpa_bicsr(FPA_CSR_FS);
#else
#error Unknown arch
#endif
#else
#error  Please configure FlushToZero and UnFlushToZero to behave properly on this operating system.
#endif
}

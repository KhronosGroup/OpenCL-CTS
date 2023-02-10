//
// Copyright (c) 2017,2021 The Khronos Group Inc.
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
#include "imageHelpers.h"
#include <limits.h>
#include <assert.h>
#if defined(__APPLE__)
#include <sys/mman.h>
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
#include <malloc.h>
#endif
#include <algorithm>
#include <cinttypes>
#include <iterator>
#if !defined(_WIN32)
#include <cmath>
#endif
#ifdef __SSE2__
#include <x86intrin.h>
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 12)                \
    || (defined(__clang__) && !defined(__apple_build_version__)                \
        && __clang_major__ < 8)
// Add missing intrinsics that aren't in ancient compilers, but are used below
#ifdef __clang__
#define NODEBUG nodebug
#else
#define NODEBUG artificial
#endif
static inline __attribute__((always_inline, NODEBUG)) __m128i
_mm_loadu_si16(const void *p)
{
    struct _loadu_si16
    {
        short v;
    } __attribute__((packed, may_alias));
    short u = ((const struct _loadu_si16 *)p)->v;
    return _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, u);
}

static inline __attribute__((always_inline, NODEBUG)) __m128i
_mm_loadu_si32(const void *p)
{
    struct _loadu_si32
    {
        int v;
    } __attribute__((packed, may_alias));
    int u = ((const struct _loadu_si32 *)p)->v;
    return _mm_set_epi32(0, 0, 0, u);
}
#undef NODEBUG
#endif
#endif

RoundingMode gFloatToHalfRoundingMode = kDefaultRoundingMode;

cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
bool gTestRounding = false;
double sRGBmap(float fc)
{
    double c = (double)fc;

#if !defined(_WIN32)
    if (std::isnan(c)) c = 0.0;
#else
    if (_isnan(c)) c = 0.0;
#endif

    if (c > 1.0)
        c = 1.0;
    else if (c < 0.0)
        c = 0.0;
    else if (c < 0.0031308)
        c = 12.92 * c;
    else
        c = (1055.0 / 1000.0) * pow(c, 5.0 / 12.0) - (55.0 / 1000.0);

    return c * 255.0;
}

double sRGBunmap(float fc)
{
    double c = (double)fc;
    double result;

    if (c <= 0.04045)
        result = c / 12.92;
    else
        result = pow((c + 0.055) / 1.055, 2.4);

    return result;
}

// Precalculated table of linear encodings of sRGB values
float gSRGBTbl[] = {
    0x0.000000p+00f, 0x1.3e4569p-12f, 0x1.3e4569p-11f, 0x1.dd681cp-11f,
    0x1.3e4569p-10f, 0x1.8dd6c2p-10f, 0x1.dd681cp-10f, 0x1.167cbbp-09f,
    0x1.3e4569p-09f, 0x1.660e15p-09f, 0x1.8dd6c2p-09f, 0x1.b6a31cp-09f,
    0x1.e1e31ep-09f, 0x1.07c38cp-08f, 0x1.1fcc2cp-08f, 0x1.390ffbp-08f,
    0x1.53936ep-08f, 0x1.6f5adfp-08f, 0x1.8c6a96p-08f, 0x1.aac6c3p-08f,
    0x1.ca7383p-08f, 0x1.eb74e3p-08f, 0x1.06e76cp-07f, 0x1.18c2a6p-07f,
    0x1.2b4e0ap-07f, 0x1.3e8b7cp-07f, 0x1.527cd7p-07f, 0x1.6723efp-07f,
    0x1.7c8293p-07f, 0x1.929a89p-07f, 0x1.a96d92p-07f, 0x1.c0fd67p-07f,
    0x1.d94bc1p-07f, 0x1.f25a47p-07f, 0x1.061553p-06f, 0x1.135f40p-06f,
    0x1.210bbap-06f, 0x1.2f1b8ep-06f, 0x1.3d8f85p-06f, 0x1.4c6868p-06f,
    0x1.5ba6fcp-06f, 0x1.6b4c05p-06f, 0x1.7b5843p-06f, 0x1.8bcc76p-06f,
    0x1.9ca95ap-06f, 0x1.adefabp-06f, 0x1.bfa021p-06f, 0x1.d1bb75p-06f,
    0x1.e4425ap-06f, 0x1.f73586p-06f, 0x1.054ad5p-05f, 0x1.0f31bbp-05f,
    0x1.194fccp-05f, 0x1.23a55fp-05f, 0x1.2e32c9p-05f, 0x1.38f860p-05f,
    0x1.43f678p-05f, 0x1.4f2d63p-05f, 0x1.5a9d76p-05f, 0x1.664701p-05f,
    0x1.722a57p-05f, 0x1.7e47c8p-05f, 0x1.8a9fa3p-05f, 0x1.973239p-05f,
    0x1.a3ffdcp-05f, 0x1.b108d3p-05f, 0x1.be4d6fp-05f, 0x1.cbcdfdp-05f,
    0x1.d98ac9p-05f, 0x1.e78420p-05f, 0x1.f5ba4cp-05f, 0x1.0216ccp-04f,
    0x1.096f28p-04f, 0x1.10e65ep-04f, 0x1.187c92p-04f, 0x1.2031eap-04f,
    0x1.280689p-04f, 0x1.2ffa93p-04f, 0x1.380e2bp-04f, 0x1.404176p-04f,
    0x1.489496p-04f, 0x1.5107aep-04f, 0x1.599ae0p-04f, 0x1.624e50p-04f,
    0x1.6b2220p-04f, 0x1.741672p-04f, 0x1.7d2b67p-04f, 0x1.866121p-04f,
    0x1.8fb7c1p-04f, 0x1.992f6ap-04f, 0x1.a2c83cp-04f, 0x1.ac8257p-04f,
    0x1.b65dddp-04f, 0x1.c05aeep-04f, 0x1.ca79aap-04f, 0x1.d4ba31p-04f,
    0x1.df1ca4p-04f, 0x1.e9a122p-04f, 0x1.f447cap-04f, 0x1.ff10bdp-04f,
    0x1.04fe0dp-03f, 0x1.0a84ffp-03f, 0x1.101d45p-03f, 0x1.15c6eep-03f,
    0x1.1b8209p-03f, 0x1.214ea6p-03f, 0x1.272cd4p-03f, 0x1.2d1ca2p-03f,
    0x1.331e1fp-03f, 0x1.393159p-03f, 0x1.3f5660p-03f, 0x1.458d43p-03f,
    0x1.4bd60fp-03f, 0x1.5230d4p-03f, 0x1.589da1p-03f, 0x1.5f1c83p-03f,
    0x1.65ad8ap-03f, 0x1.6c50c3p-03f, 0x1.73063ep-03f, 0x1.79ce07p-03f,
    0x1.80a82ep-03f, 0x1.8794c0p-03f, 0x1.8e93cbp-03f, 0x1.95a55ep-03f,
    0x1.9cc987p-03f, 0x1.a40052p-03f, 0x1.ab49cfp-03f, 0x1.b2a60ap-03f,
    0x1.ba1516p-03f, 0x1.c196f7p-03f, 0x1.c92bc1p-03f, 0x1.d0d380p-03f,
    0x1.d88e41p-03f, 0x1.e05c12p-03f, 0x1.e83d00p-03f, 0x1.f0311ap-03f,
    0x1.f8386ap-03f, 0x1.002980p-02f, 0x1.044074p-02f, 0x1.086118p-02f,
    0x1.0c8b72p-02f, 0x1.10bf88p-02f, 0x1.14fd61p-02f, 0x1.194504p-02f,
    0x1.1d9677p-02f, 0x1.21f1c0p-02f, 0x1.2656e6p-02f, 0x1.2ac5efp-02f,
    0x1.2f3ee1p-02f, 0x1.33c1c3p-02f, 0x1.384e9ap-02f, 0x1.3ce56ep-02f,
    0x1.418644p-02f, 0x1.463122p-02f, 0x1.4ae610p-02f, 0x1.4fa512p-02f,
    0x1.546e2fp-02f, 0x1.59416dp-02f, 0x1.5e1ed2p-02f, 0x1.630665p-02f,
    0x1.67f82bp-02f, 0x1.6cf42bp-02f, 0x1.71fa69p-02f, 0x1.770aedp-02f,
    0x1.7c25bdp-02f, 0x1.814addp-02f, 0x1.867a55p-02f, 0x1.8bb42ap-02f,
    0x1.90f862p-02f, 0x1.964703p-02f, 0x1.9ba012p-02f, 0x1.a10396p-02f,
    0x1.a67194p-02f, 0x1.abea12p-02f, 0x1.b16d16p-02f, 0x1.b6faa6p-02f,
    0x1.bc92c7p-02f, 0x1.c2357fp-02f, 0x1.c7e2d4p-02f, 0x1.cd9acbp-02f,
    0x1.d35d6bp-02f, 0x1.d92ab8p-02f, 0x1.df02b9p-02f, 0x1.e4e572p-02f,
    0x1.ead2ebp-02f, 0x1.f0cb27p-02f, 0x1.f6ce2dp-02f, 0x1.fcdc02p-02f,
    0x1.017a56p-01f, 0x1.048c18p-01f, 0x1.07a34bp-01f, 0x1.0abfefp-01f,
    0x1.0de209p-01f, 0x1.11099cp-01f, 0x1.1436a9p-01f, 0x1.176933p-01f,
    0x1.1aa13ep-01f, 0x1.1ddecbp-01f, 0x1.2121dfp-01f, 0x1.246a7ap-01f,
    0x1.27b8a0p-01f, 0x1.2b0c54p-01f, 0x1.2e6598p-01f, 0x1.31c46fp-01f,
    0x1.3528dcp-01f, 0x1.3892e1p-01f, 0x1.3c0280p-01f, 0x1.3f77bdp-01f,
    0x1.42f29ap-01f, 0x1.46731ap-01f, 0x1.49f93ep-01f, 0x1.4d850bp-01f,
    0x1.511682p-01f, 0x1.54ada5p-01f, 0x1.584a79p-01f, 0x1.5becfep-01f,
    0x1.5f9538p-01f, 0x1.634329p-01f, 0x1.66f6d4p-01f, 0x1.6ab03bp-01f,
    0x1.6e6f61p-01f, 0x1.723449p-01f, 0x1.75fef4p-01f, 0x1.79cf65p-01f,
    0x1.7da59fp-01f, 0x1.8181a5p-01f, 0x1.856378p-01f, 0x1.894b1cp-01f,
    0x1.8d3892p-01f, 0x1.912bdep-01f, 0x1.952501p-01f, 0x1.9923ffp-01f,
    0x1.9d28d9p-01f, 0x1.a13392p-01f, 0x1.a5442dp-01f, 0x1.a95aacp-01f,
    0x1.ad7711p-01f, 0x1.b1995fp-01f, 0x1.b5c198p-01f, 0x1.b9efbep-01f,
    0x1.be23d5p-01f, 0x1.c25ddep-01f, 0x1.c69ddcp-01f, 0x1.cae3d1p-01f,
    0x1.cf2fc0p-01f, 0x1.d381abp-01f, 0x1.d7d994p-01f, 0x1.dc377ep-01f,
    0x1.e09b6bp-01f, 0x1.e5055dp-01f, 0x1.e97557p-01f, 0x1.edeb5bp-01f,
    0x1.f2676cp-01f, 0x1.f6e98bp-01f, 0x1.fb71bcp-01f, 0x1.000000p+00f,
};


float sRGBunmap(cl_uchar ic) { return gSRGBTbl[ic]; }

#ifdef __SSE2__
__m128 sRGBunmap(const __m128i ic)
{
    static const float recip_255 = 1.0f / 255.0f;
#ifdef __AVX2__
    __m128 fc = _mm_i32gather_ps(gSRGBTbl, _mm_cvtepu8_epi32(ic), 4);
    // only RGB need to be converted for sRGBA
    return _mm_insert_ps(
        fc,
        _mm_mul_ss(_mm_cvtsi32_ss(_mm_undefined_ps(), _mm_extract_epi8(ic, 3)),
                   _mm_set_ss(recip_255)),
        _MM_MK_INSERTPS_NDX(0, 3, 0));
#else
    // With no gather support, we'll need to load the four components
    // separately...
    uint32_t pixel = _mm_cvtsi128_si32(ic);
    return _mm_set_ps((float)(pixel >> 24) * recip_255,
                      sRGBunmap((cl_uchar)((pixel >> 16) & 0xFF)),
                      sRGBunmap((cl_uchar)((pixel >> 8) & 0xFF)),
                      sRGBunmap((cl_uchar)(pixel & 0xFF)));
#endif
}


#ifdef __SSE4_1__
#define SELECT_I(cond, a, b) _mm_blendv_epi8(b, a, cond)
#define TEST_ANY_ZERO(v) !_mm_test_all_ones(v)
#define TEST_NONZERO(v) !_mm_test_all_zeros(v, v)
#define TEST_ZERO(v) _mm_test_all_zeros(v, v)
#elif defined(__SSE2__)
// n.b. "ANDNOT" is ~A & B, not A & ~B!!
#define SELECT_I(cond, a, b)                                                   \
    _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(cond, b))
#ifdef __x86_64
#define TEST_NONZERO(v)                                                        \
    (_mm_cvtsi128_si64(v)                                                      \
     || _mm_cvtsi128_si64(_mm_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3))))
#else
#define TEST_NONZERO(v)                                                        \
    (_mm_cvtsi128_si32(v)                                                      \
     || _mm_cvtsi128_si32(                                                     \
         _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 1))                         \
         || _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 2))))  \
     || _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 3))))
#endif
#define TEST_ZERO(v) (!TEST_NONZERO(v))
// The int64 extraction trick won't work here... :/
#define TEST_ANY_ZERO(v)                                                       \
    (!_mm_cvtsi128_si32(v)                                                     \
     || !_mm_cvtsi128_si32(                                                    \
         _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 1))                         \
         || !_mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 2)))) \
     || !_mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 3))))
#endif

#ifdef __GNUC__
#define __forceinline __attribute__((always_inline))
#endif

static inline __m128i __forceinline _mm_setmone_si128(void)
{
    __m128i junk = _mm_undefined_si128();
    return _mm_cmpeq_epi32(junk, junk);
}

static inline __m128 cl_half_to_float(__m128i h)
{
#ifdef __F16C__
    return _mm_cvtph_ps(h);
#else
    // Type-punning to get direct access to underlying bits
    union {
        __m128 f;
        __m128i i;
    } f32;

    __m128i zero = _mm_setzero_si128();
    __m128i negOne = _mm_setmone_si128();
    __m128i one = _mm_srli_epi32(negOne, 31);
    __m128i h_exp_mask = _mm_srli_epi16(negOne, CL_HALF_MANT_DIG); // = 0x1f

    // Extract sign bit
    __m128i sign =
        _mm_slli_epi32(_mm_unpacklo_epi16(_mm_srli_epi16(h, 15), zero), 31);

    // Extract FP16 exponent and mantissa
    __m128i h_exp =
        _mm_and_si128(_mm_srli_epi16(h, CL_HALF_MANT_DIG - 1), h_exp_mask);
    __m128i h_mant = _mm_and_si128(h, _mm_srli_epi16(negOne, 6) /* 0x3ff */);

    // Remove FP16 exponent bias and convert to int32
    __m128i exp = _mm_sub_epi16(
        h_exp,
        _mm_srli_epi16(negOne, CL_HALF_MANT_DIG + 1) /* CL_HALF_MAX_EXP - 1 */);
#ifdef __SSE4_1__
    exp = _mm_cvtepi16_epi32(exp);
#else
    exp = _mm_unpacklo_epi16(exp, _mm_cmpgt_epi16(zero, exp));
#endif

    // Add FP32 exponent bias
    __m128i f_exp = _mm_add_epi32(
        exp,
        _mm_srli_epi32(negOne, CL_FLT_MANT_DIG + 1) /* CL_FLT_MAX_EXP - 1 */);

    // Convert mantissa to the 32-bit form
    __m128i f_mant = _mm_slli_epi32(_mm_unpacklo_epi16(h_mant, zero),
                                    CL_FLT_MANT_DIG - CL_HALF_MANT_DIG);

    // Note that due to the way SIMD works, we can't have branches--we have to
    // compute all the possible values.

    // Check for NaN / infinity
    __m128i inf_mask = _mm_cmpeq_epi16(h_exp, h_exp_mask);
    inf_mask = _mm_unpacklo_epi16(inf_mask, inf_mask);
    __m128i mant_zero_mask = _mm_cmpeq_epi32(f_mant, zero);
    // n.b. "ANDNOT" is ~A & B, not A & ~B!!
    __m128i nan_mask = _mm_andnot_si128(mant_zero_mask, inf_mask);

    // NaN -> propagate mantissa and silence it
    __m128i f_mant_nan =
        _mm_or_si128(f_mant, _mm_slli_epi32(one, 22) /* 0x400000 */);
    // Infinity -> zero mantissa
    f_mant = SELECT_I(nan_mask, f_mant_nan, f_mant);
    f_exp = SELECT_I(inf_mask,
                     _mm_srli_epi32(negOne, CL_FLT_MANT_DIG) /* 0xff */, f_exp);

    // Check for zero / denormal
    __m128i exp_zero_mask = _mm_cmpeq_epi16(h_exp, zero);
    exp_zero_mask = _mm_unpacklo_epi16(exp_zero_mask, exp_zero_mask);
    __m128i zero_mask = _mm_and_si128(mant_zero_mask, exp_zero_mask);
    // n.b. "ANDNOT" is ~A & B, not A & ~B!!
    __m128i denorm_mask = _mm_andnot_si128(mant_zero_mask, exp_zero_mask);

    if (TEST_NONZERO(denorm_mask))
    {
        // Denormal -> normalize it
        // - Shift mantissa to make most-significant 1 implicit
        // - Adjust exponent accordingly
        __m128i f_mant_mask =
            _mm_srli_epi32(negOne, 32 - (CL_FLT_MANT_DIG - 1));
#if defined(__AVX512VL__) && defined(__AVX512DQ__)
        // We'll probably never get here, since most CPUs that support AVX-512
        // also support F16C. n.b. No +1 yet for the implicit 1 before the radix
        // point--we really do want to shift at least one place
        __m128i shift =
            _mm_and_si128(_mm_sub_epi32(_mm_lzcnt_epi32(f_mant),
                                        _mm_set1_epi32(32 - CL_FLT_MANT_DIG)),
                          denorm_mask);
        f_mant = _mm_sllv_epi32(f_mant, shift);
        shift = _mm_sub_epi32(shift, _mm_and_si128(one, denorm_mask));
#else
        // No packed leading-zero count until AVX-512... gotta do this the hard
        // way
        __m128i shift = zero;
        __m128i shift_mask = _mm_cmpgt_epi32(
            f_mant, _mm_srli_epi32(negOne, 16) /* 0x0000FFFF */);
        __m128i f_mant_shift =
            SELECT_I(shift_mask, f_mant, _mm_slli_epi32(f_mant, 16));
        // n.b. "ANDNOT" is ~A & B, not A & ~B!!
        shift = _mm_add_epi32(
            shift,
            _mm_andnot_si128(shift_mask, _mm_slli_epi32(one, 4) /* 16 */));
        // Starting from here, we also need to check that mant >= 0, because
        // PCMPGT does a signed comparison; unsigned comparisons weren't added
        // until AVX-512, which also already has a faster way to count leading
        // zeroes anyway
        shift_mask = _mm_or_si128(
            _mm_cmplt_epi32(f_mant_shift, zero),
            _mm_cmpgt_epi32(f_mant_shift,
                            _mm_srli_epi32(negOne, 8) /* 0x00FFFFFF */));
        f_mant_shift =
            SELECT_I(shift_mask, f_mant_shift, _mm_slli_epi32(f_mant_shift, 8));
        shift = _mm_add_epi32(
            shift,
            _mm_andnot_si128(shift_mask, _mm_slli_epi32(one, 3) /* 8 */));
        shift_mask = _mm_or_si128(
            _mm_cmplt_epi32(f_mant_shift, zero),
            _mm_cmpgt_epi32(f_mant_shift,
                            _mm_srli_epi32(negOne, 4) /* 0x0FFFFFFF */));
        f_mant_shift =
            SELECT_I(shift_mask, f_mant_shift, _mm_slli_epi32(f_mant_shift, 4));
        shift = _mm_add_epi32(
            shift,
            _mm_andnot_si128(shift_mask, _mm_slli_epi32(one, 2) /* 4 */));
        shift_mask = _mm_or_si128(
            _mm_cmplt_epi32(f_mant_shift, zero),
            _mm_cmpgt_epi32(f_mant_shift,
                            _mm_srli_epi32(negOne, 2) /* 0x3FFFFFFF */));
        f_mant_shift =
            SELECT_I(shift_mask, f_mant_shift, _mm_slli_epi32(f_mant_shift, 2));
        shift = _mm_add_epi32(
            shift,
            _mm_andnot_si128(shift_mask, _mm_slli_epi32(one, 1) /* 2 */));
        shift_mask = _mm_or_si128(
            _mm_cmplt_epi32(f_mant_shift, zero),
            _mm_cmpgt_epi32(f_mant_shift,
                            _mm_srli_epi32(negOne, 1) /* 0x7FFFFFFF */));
        f_mant_shift =
            SELECT_I(shift_mask, f_mant_shift, _mm_slli_epi32(f_mant_shift, 1));
        shift = _mm_add_epi32(shift, _mm_andnot_si128(shift_mask, one));
        f_mant = SELECT_I(denorm_mask,
                          _mm_srli_epi32(f_mant_shift, 32 - CL_FLT_MANT_DIG),
                          f_mant);
        shift = _mm_and_si128(
            _mm_sub_epi32(shift, _mm_set1_epi32(32 - CL_FLT_MANT_DIG + 1)),
            denorm_mask);
#endif
        f_mant = _mm_and_si128(f_mant, f_mant_mask);
        f_exp = _mm_sub_epi32(f_exp, shift);
    }

    // Zero -> zero exponent
    // n.b. "ANDNOT" is ~A & B, not A & ~B!!
    f_exp = _mm_andnot_si128(zero_mask, f_exp);

    f32.i = _mm_or_si128(
        sign, _mm_or_si128(_mm_slli_epi32(f_exp, CL_FLT_MANT_DIG - 1), f_mant));
    return f32.f;
#endif
}
#endif


uint32_t get_format_type_size(const cl_image_format *format)
{
    return get_channel_data_type_size(format->image_channel_data_type);
}

uint32_t get_channel_data_type_size(cl_channel_type channelType)
{
    switch (channelType)
    {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8: return 1;

        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_HALF_FLOAT:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return sizeof(cl_short);

        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32: return sizeof(cl_int);

        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_SHORT_565_REV:
        case CL_UNORM_SHORT_555_REV:
#endif
            return 2;

#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_8888:
        case CL_UNORM_INT_8888_REV: return 4;
#endif

        case CL_UNORM_INT_101010:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_101010_REV:
#endif
            return 4;

        case CL_FLOAT: return sizeof(cl_float);

        default: return 0;
    }
}

uint32_t get_format_channel_count(const cl_image_format *format)
{
    return get_channel_order_channel_count(format->image_channel_order);
}

uint32_t get_channel_order_channel_count(cl_channel_order order)
{
    switch (order)
    {
        case CL_R:
        case CL_A:
        case CL_Rx:
        case CL_INTENSITY:
        case CL_LUMINANCE:
        case CL_DEPTH:
        case CL_DEPTH_STENCIL: return 1;

        case CL_RG:
        case CL_RA:
        case CL_RGx: return 2;

        case CL_RGB:
        case CL_RGBx:
        case CL_sRGB:
        case CL_sRGBx: return 3;

        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
        case CL_sRGBA:
        case CL_sBGRA:
        case CL_ABGR:
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
#endif
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
#endif
#ifdef CL_ABGR_APPLE
        case CL_ABGR_APPLE:
#endif
            return 4;

        default:
            log_error("%s does not support 0x%x\n", __FUNCTION__, order);
            return 0;
    }
}

cl_channel_type get_channel_type_from_name(const char *name)
{
    struct
    {
        cl_channel_type type;
        const char *name;
    } typeNames[] = { { CL_SNORM_INT8, "CL_SNORM_INT8" },
                      { CL_SNORM_INT16, "CL_SNORM_INT16" },
                      { CL_UNORM_INT8, "CL_UNORM_INT8" },
                      { CL_UNORM_INT16, "CL_UNORM_INT16" },
                      { CL_UNORM_INT24, "CL_UNORM_INT24" },
                      { CL_UNORM_SHORT_565, "CL_UNORM_SHORT_565" },
                      { CL_UNORM_SHORT_555, "CL_UNORM_SHORT_555" },
                      { CL_UNORM_INT_101010, "CL_UNORM_INT_101010" },
                      { CL_SIGNED_INT8, "CL_SIGNED_INT8" },
                      { CL_SIGNED_INT16, "CL_SIGNED_INT16" },
                      { CL_SIGNED_INT32, "CL_SIGNED_INT32" },
                      { CL_UNSIGNED_INT8, "CL_UNSIGNED_INT8" },
                      { CL_UNSIGNED_INT16, "CL_UNSIGNED_INT16" },
                      { CL_UNSIGNED_INT32, "CL_UNSIGNED_INT32" },
                      { CL_HALF_FLOAT, "CL_HALF_FLOAT" },
                      { CL_FLOAT, "CL_FLOAT" },
#ifdef CL_SFIXED14_APPLE
                      { CL_SFIXED14_APPLE, "CL_SFIXED14_APPLE" }
#endif
    };
    for (size_t i = 0; i < sizeof(typeNames) / sizeof(typeNames[0]); i++)
    {
        if (strcmp(typeNames[i].name, name) == 0
            || strcmp(typeNames[i].name + 3, name) == 0)
            return typeNames[i].type;
    }
    return (cl_channel_type)-1;
}

cl_channel_order get_channel_order_from_name(const char *name)
{
    const struct
    {
        cl_channel_order order;
        const char *name;
    } orderNames[] = {
        { CL_R, "CL_R" },
        { CL_A, "CL_A" },
        { CL_Rx, "CL_Rx" },
        { CL_RG, "CL_RG" },
        { CL_RA, "CL_RA" },
        { CL_RGx, "CL_RGx" },
        { CL_RGB, "CL_RGB" },
        { CL_RGBx, "CL_RGBx" },
        { CL_RGBA, "CL_RGBA" },
        { CL_BGRA, "CL_BGRA" },
        { CL_ARGB, "CL_ARGB" },
        { CL_INTENSITY, "CL_INTENSITY" },
        { CL_LUMINANCE, "CL_LUMINANCE" },
        { CL_DEPTH, "CL_DEPTH" },
        { CL_DEPTH_STENCIL, "CL_DEPTH_STENCIL" },
        { CL_sRGB, "CL_sRGB" },
        { CL_sRGBx, "CL_sRGBx" },
        { CL_sRGBA, "CL_sRGBA" },
        { CL_sBGRA, "CL_sBGRA" },
        { CL_ABGR, "CL_ABGR" },
#ifdef CL_1RGB_APPLE
        { CL_1RGB_APPLE, "CL_1RGB_APPLE" },
#endif
#ifdef CL_BGR1_APPLE
        { CL_BGR1_APPLE, "CL_BGR1_APPLE" },
#endif
    };

    for (size_t i = 0; i < sizeof(orderNames) / sizeof(orderNames[0]); i++)
    {
        if (strcmp(orderNames[i].name, name) == 0
            || strcmp(orderNames[i].name + 3, name) == 0)
            return orderNames[i].order;
    }
    return (cl_channel_order)-1;
}


int is_format_signed(const cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_SNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return 1;

        default: return 0;
    }
}

uint32_t get_pixel_size(const cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8: return get_format_channel_count(format);

        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_HALF_FLOAT:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return get_format_channel_count(format) * sizeof(cl_ushort);

        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
            return get_format_channel_count(format) * sizeof(cl_int);

        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_SHORT_565_REV:
        case CL_UNORM_SHORT_555_REV:
#endif
            return 2;

#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_8888:
        case CL_UNORM_INT_8888_REV: return 4;
#endif

        case CL_UNORM_INT_101010:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_101010_REV:
#endif
            return 4;

        case CL_FLOAT:
            return get_format_channel_count(format) * sizeof(cl_float);

        default: return 0;
    }
}

uint32_t next_power_of_two(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

uint32_t get_pixel_alignment(const cl_image_format *format)
{
    return next_power_of_two(get_pixel_size(format));
}

int get_8_bit_image_format(cl_context context, cl_mem_object_type objType,
                           cl_mem_flags flags, size_t channelCount,
                           cl_image_format *outFormat)
{
    cl_image_format formatList[128];
    unsigned int outFormatCount, i;
    int error;


    /* Make sure each image format is supported */
    if ((error = clGetSupportedImageFormats(context, flags, objType, 128,
                                            formatList, &outFormatCount)))
        return error;


    /* Look for one that is an 8-bit format */
    for (i = 0; i < outFormatCount; i++)
    {
        if (formatList[i].image_channel_data_type == CL_SNORM_INT8
            || formatList[i].image_channel_data_type == CL_UNORM_INT8
            || formatList[i].image_channel_data_type == CL_SIGNED_INT8
            || formatList[i].image_channel_data_type == CL_UNSIGNED_INT8)
        {
            if (!channelCount
                || (channelCount
                    && (get_format_channel_count(&formatList[i])
                        == channelCount)))
            {
                *outFormat = formatList[i];
                return 0;
            }
        }
    }

    return -1;
}

int get_32_bit_image_format(cl_context context, cl_mem_object_type objType,
                            cl_mem_flags flags, size_t channelCount,
                            cl_image_format *outFormat)
{
    cl_image_format formatList[128];
    unsigned int outFormatCount, i;
    int error;


    /* Make sure each image format is supported */
    if ((error = clGetSupportedImageFormats(context, flags, objType, 128,
                                            formatList, &outFormatCount)))
        return error;

    /* Look for one that is an 8-bit format */
    for (i = 0; i < outFormatCount; i++)
    {
        if (formatList[i].image_channel_data_type == CL_UNORM_INT_101010
            || formatList[i].image_channel_data_type == CL_FLOAT
            || formatList[i].image_channel_data_type == CL_SIGNED_INT32
            || formatList[i].image_channel_data_type == CL_UNSIGNED_INT32)
        {
            if (!channelCount
                || (channelCount
                    && (get_format_channel_count(&formatList[i])
                        == channelCount)))
            {
                *outFormat = formatList[i];
                return 0;
            }
        }
    }

    return -1;
}

void print_first_pixel_difference_error(size_t where, const char *sourcePixel,
                                        const char *destPixel,
                                        image_descriptor *imageInfo, size_t y,
                                        size_t thirdDim)
{
    size_t pixel_size = get_pixel_size(imageInfo->format);

    log_error("ERROR: Scanline %d did not verify for image size %d,%d,%d "
              "pitch %d (extra %d bytes)\n",
              (int)y, (int)imageInfo->width, (int)imageInfo->height,
              (int)thirdDim, (int)imageInfo->rowPitch,
              (int)imageInfo->rowPitch
                  - (int)imageInfo->width * (int)pixel_size);
    log_error("Failed at column: %zu   ", where);

    switch (pixel_size)
    {
        case 1:
            log_error("*0x%2.2x vs. 0x%2.2x\n", ((cl_uchar *)sourcePixel)[0],
                      ((cl_uchar *)destPixel)[0]);
            break;
        case 2:
            log_error("*0x%4.4x vs. 0x%4.4x\n", ((cl_ushort *)sourcePixel)[0],
                      ((cl_ushort *)destPixel)[0]);
            break;
        case 3:
            log_error("*{0x%2.2x, 0x%2.2x, 0x%2.2x} vs. "
                      "{0x%2.2x, 0x%2.2x, 0x%2.2x}\n",
                      ((cl_uchar *)sourcePixel)[0],
                      ((cl_uchar *)sourcePixel)[1],
                      ((cl_uchar *)sourcePixel)[2], ((cl_uchar *)destPixel)[0],
                      ((cl_uchar *)destPixel)[1], ((cl_uchar *)destPixel)[2]);
            break;
        case 4:
            log_error("*0x%8.8x vs. 0x%8.8x\n", ((cl_uint *)sourcePixel)[0],
                      ((cl_uint *)destPixel)[0]);
            break;
        case 6:
            log_error(
                "*{0x%4.4x, 0x%4.4x, 0x%4.4x} vs. "
                "{0x%4.4x, 0x%4.4x, 0x%4.4x}\n",
                ((cl_ushort *)sourcePixel)[0], ((cl_ushort *)sourcePixel)[1],
                ((cl_ushort *)sourcePixel)[2], ((cl_ushort *)destPixel)[0],
                ((cl_ushort *)destPixel)[1], ((cl_ushort *)destPixel)[2]);
            break;
        case 8:
            log_error("*0x%16.16" PRIx64 " vs. 0x%16.16" PRIx64 "\n",
                      ((cl_ulong *)sourcePixel)[0], ((cl_ulong *)destPixel)[0]);
            break;
        case 12:
            log_error("*{0x%8.8x, 0x%8.8x, 0x%8.8x} vs. "
                      "{0x%8.8x, 0x%8.8x, 0x%8.8x}\n",
                      ((cl_uint *)sourcePixel)[0], ((cl_uint *)sourcePixel)[1],
                      ((cl_uint *)sourcePixel)[2], ((cl_uint *)destPixel)[0],
                      ((cl_uint *)destPixel)[1], ((cl_uint *)destPixel)[2]);
            break;
        case 16:
            log_error("*{0x%8.8x, 0x%8.8x, 0x%8.8x, 0x%8.8x} vs. "
                      "{0x%8.8x, 0x%8.8x, 0x%8.8x, 0x%8.8x}\n",
                      ((cl_uint *)sourcePixel)[0], ((cl_uint *)sourcePixel)[1],
                      ((cl_uint *)sourcePixel)[2], ((cl_uint *)sourcePixel)[3],
                      ((cl_uint *)destPixel)[0], ((cl_uint *)destPixel)[1],
                      ((cl_uint *)destPixel)[2], ((cl_uint *)destPixel)[3]);
            break;
        default:
            log_error("Don't know how to print pixel size of %zu\n",
                      pixel_size);
            break;
    }
}

size_t compare_scanlines(const image_descriptor *imageInfo, const char *aPtr,
                         const char *bPtr)
{
    size_t pixel_size = get_pixel_size(imageInfo->format);
    size_t column;

    for (column = 0; column < imageInfo->width; column++)
    {
        switch (imageInfo->format->image_channel_data_type)
        {
            // If the data type is 101010, then ignore bits 31 and 32 when
            // comparing the row
            case CL_UNORM_INT_101010: {
                cl_uint aPixel = *(cl_uint *)aPtr;
                cl_uint bPixel = *(cl_uint *)bPtr;
                if ((aPixel & 0x3fffffff) != (bPixel & 0x3fffffff))
                    return column;
            }
            break;

            // If the data type is 555, ignore bit 15 when comparing the row
            case CL_UNORM_SHORT_555: {
                cl_ushort aPixel = *(cl_ushort *)aPtr;
                cl_ushort bPixel = *(cl_ushort *)bPtr;
                if ((aPixel & 0x7fff) != (bPixel & 0x7fff)) return column;
            }
            break;

            default:
                if (memcmp(aPtr, bPtr, pixel_size) != 0) return column;
                break;
        }

        aPtr += pixel_size;
        bPtr += pixel_size;
    }

    // If we didn't find a difference, return the width of the image
    return column;
}

int random_log_in_range(int minV, int maxV, MTdata d)
{
    double v = log2(((double)genrand_int32(d) / (double)0xffffffff) + 1);
    int iv = (int)((float)(maxV - minV) * v);
    return iv + minV;
}


#ifdef __SSE2__
static inline __m128i vifloorf(__m128 f)
{
#ifdef __SSE4_1__
    return _mm_cvtps_epi32(
        _mm_round_ps(f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
#else
    // No packed rounding until SSE4... do this the old-fashioned way
    unsigned int mxcsr = _mm_getcsr();
    _mm_setcsr(mxcsr & ~_MM_ROUND_MASK | _MM_ROUND_DOWN);
    __m128i i = _mm_cvtps_epi32(f);
    _mm_setcsr(mxcsr);
    return i;
#endif
}

static inline __m128 vfloorf(__m128 f)
{
#ifdef __SSE4_1__
    return _mm_round_ps(f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
#else
    // No packed rounding until SSE4... do this the old-fashioned way
    unsigned int mxcsr = _mm_getcsr();
    _mm_setcsr(mxcsr & ~_MM_ROUND_MASK | _MM_ROUND_DOWN);
    f = _mm_cvtepi32_ps(_mm_cvtps_epi32(f));
    _mm_setcsr(mxcsr);
    return f;
#endif
}

static inline __m128 frac(__m128 a) { return _mm_sub_ps(a, vfloorf(a)); }
#else
static inline float frac(float a) { return a - floorf(a); }
#endif

// Define the addressing functions
#ifdef __SSE2__
typedef __m128i (*AddressFn)(__m128i value, __m128i maxValue);

__m128i NoAddressFn(__m128i value, __m128i maxValue) { return value; }
__m128i RepeatAddressFn(__m128i value, __m128i maxValue)
{
    __m128i minMask = _mm_cmplt_epi32(value, _mm_setzero_si128());
    __m128i maxMask =
        _mm_cmpgt_epi32(value, _mm_add_epi32(maxValue, _mm_setmone_si128()));
    return SELECT_I(minMask, _mm_add_epi32(value, maxValue),
                    SELECT_I(maxMask, _mm_sub_epi32(value, maxValue), value));
}
__m128i MirroredRepeatAddressFn(__m128i value, __m128i maxValue)
{
#ifdef __SSE4_1__
    return _mm_max_epi32(
        _mm_min_epi32(value, _mm_add_epi32(maxValue, _mm_setmone_si128())),
        _mm_setzero_si128());
#else
    __m128i zero = _mm_setzero_si128();
    maxValue = _mm_add_epi32(maxValue, _mm_setmone_si128());
    __m128i minMask = _mm_cmplt_epi32(value, zero);
    __m128i maxMask = _mm_cmpgt_epi32(value, maxValue);
    return SELECT_I(minMask, zero, SELECT_I(maxMask, maxValue, value));
#endif
}
__m128i ClampAddressFn(__m128i value, __m128i maxValue)
{
    __m128i negOne = _mm_cmpgt_epi32(maxValue, _mm_setzero_si128());
#ifdef __SSE4_1__
    return _mm_max_epi32(_mm_min_epi32(value, maxValue), negOne);
#else
    __m128i minMask = _mm_cmplt_epi32(value, negOne);
    __m128i maxMask = _mm_cmpgt_epi32(value, maxValue);
    return SELECT_I(minMask, negOne, SELECT_I(maxMask, maxValue, value));
#endif
}
__m128i ClampToEdgeNearestFn(__m128i value, __m128i maxValue)
{
#ifdef __SSE4_1__
    return _mm_max_epi32(
        _mm_min_epi32(value, _mm_add_epi32(maxValue, _mm_setmone_si128())),
        _mm_setzero_si128());
#else
    __m128i zero = _mm_setzero_si128();
    maxValue = _mm_add_epi32(maxValue, _mm_setmone_si128());
    __m128i minMask = _mm_cmplt_epi32(value, zero);
    __m128i maxMask = _mm_cmpgt_epi32(value, maxValue);
    return SELECT_I(minMask, zero, SELECT_I(maxMask, maxValue, value));
#endif
}
AddressFn ClampToEdgeLinearFn = ClampToEdgeNearestFn;

// Note: normalized coords get repeated in normalized space, not unnormalized
// space! hence the special case here
__m128 RepeatNormalizedAddressFn(__m128 fValue, __m128i maxValue)
{
    return _mm_mul_ps(frac(fValue), _mm_cvtepi32_ps(maxValue));
}

static inline __m128 vrintf(__m128 f)
{
#ifdef __SSE4_1__
    return _mm_round_ps(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    // No packed rounding until SSE4... do this the old-fashioned way
    return _mm_cvtepi32_ps(_mm_cvtps_epi32(f));
#endif
}

static inline __m128 vfabsf(__m128 f)
{
    return _mm_andnot_ps(_mm_castsi128_ps(_mm_slli_epi32(_mm_setmone_si128(),
                                                         31) /* 0x80000000 */),
                         f);
}

__m128 MirroredRepeatNormalizedAddressFn(__m128 fValue, __m128i maxValue)
{
    // Round to nearest multiple of two.
    // Note halfway values flip flop here due to rte, but they both end up
    // pointing the same place at the end of the day.
    __m128 s_prime = vrintf(_mm_mul_ps(fValue, _mm_set1_ps(0.5f)));
    s_prime = _mm_add_ps(s_prime, s_prime);

    // Reduce to [-1, 1], Apply mirroring -> [0, 1]
    s_prime = vfabsf(_mm_sub_ps(fValue, s_prime));

    // un-normalize
    return _mm_mul_ps(s_prime, _mm_cvtepi32_ps(maxValue));
}
#else
typedef int (*AddressFn)(int value, size_t maxValue);

int NoAddressFn(int value, size_t maxValue) { return value; }
int RepeatAddressFn(int value, size_t maxValue)
{
    if (value < 0)
        value += (int)maxValue;
    else if (value >= (int)maxValue)
        value -= (int)maxValue;
    return value;
}
int MirroredRepeatAddressFn(int value, size_t maxValue)
{
    if (value < 0)
        value = 0;
    else if ((size_t)value >= maxValue)
        value = (int)(maxValue - 1);
    return value;
}
int ClampAddressFn(int value, size_t maxValue)
{
    return (value < -1) ? -1
                        : ((value > (cl_long)maxValue) ? (int)maxValue : value);
}
int ClampToEdgeNearestFn(int value, size_t maxValue)
{
    return (value < 0)
        ? 0
        : (((size_t)value > maxValue - 1) ? (int)maxValue - 1 : value);
}
AddressFn ClampToEdgeLinearFn = ClampToEdgeNearestFn;

// Note: normalized coords get repeated in normalized space, not unnormalized
// space! hence the special case here
volatile float gFloatHome;
float RepeatNormalizedAddressFn(float fValue, size_t maxValue)
{
#ifndef _MSC_VER // Use original if not the VS compiler.
    // General computation for repeat
    return frac(fValue) * (float)maxValue; // Reduce to [0, 1.f]
#else // Otherwise, use this instead:
    // Home the subtraction to a float to break up the sequence of x87
    // instructions emitted by the VS compiler.
    gFloatHome = fValue - floorf(fValue);
    return gFloatHome * (float)maxValue;
#endif
}

float MirroredRepeatNormalizedAddressFn(float fValue, size_t maxValue)
{
    // Round to nearest multiple of two.
    // Note halfway values flip flop here due to rte, but they both end up
    // pointing the same place at the end of the day.
    float s_prime = 2.0f * rintf(fValue * 0.5f);

    // Reduce to [-1, 1], Apply mirroring -> [0, 1]
    s_prime = fabsf(fValue - s_prime);

    // un-normalize
    return s_prime * (float)maxValue;
}
#endif

struct AddressingTable
{
    AddressingTable()
    {
        static_assert(CL_ADDRESS_MIRRORED_REPEAT - CL_ADDRESS_NONE < 6, "");
        static_assert(CL_FILTER_NEAREST - CL_FILTER_LINEAR < 2, "");

        mTable[CL_ADDRESS_NONE - CL_ADDRESS_NONE]
              [CL_FILTER_NEAREST - CL_FILTER_NEAREST] = NoAddressFn;
        mTable[CL_ADDRESS_NONE - CL_ADDRESS_NONE]
              [CL_FILTER_LINEAR - CL_FILTER_NEAREST] = NoAddressFn;
        mTable[CL_ADDRESS_REPEAT - CL_ADDRESS_NONE]
              [CL_FILTER_NEAREST - CL_FILTER_NEAREST] = RepeatAddressFn;
        mTable[CL_ADDRESS_REPEAT - CL_ADDRESS_NONE]
              [CL_FILTER_LINEAR - CL_FILTER_NEAREST] = RepeatAddressFn;
        mTable[CL_ADDRESS_CLAMP_TO_EDGE - CL_ADDRESS_NONE]
              [CL_FILTER_NEAREST - CL_FILTER_NEAREST] = ClampToEdgeNearestFn;
        mTable[CL_ADDRESS_CLAMP_TO_EDGE - CL_ADDRESS_NONE]
              [CL_FILTER_LINEAR - CL_FILTER_NEAREST] = ClampToEdgeLinearFn;
        mTable[CL_ADDRESS_CLAMP - CL_ADDRESS_NONE]
              [CL_FILTER_NEAREST - CL_FILTER_NEAREST] = ClampAddressFn;
        mTable[CL_ADDRESS_CLAMP - CL_ADDRESS_NONE]
              [CL_FILTER_LINEAR - CL_FILTER_NEAREST] = ClampAddressFn;
        mTable[CL_ADDRESS_MIRRORED_REPEAT - CL_ADDRESS_NONE]
              [CL_FILTER_NEAREST - CL_FILTER_NEAREST] = MirroredRepeatAddressFn;
        mTable[CL_ADDRESS_MIRRORED_REPEAT - CL_ADDRESS_NONE]
              [CL_FILTER_LINEAR - CL_FILTER_NEAREST] = MirroredRepeatAddressFn;
    }

    AddressFn operator[](image_sampler_data *sampler)
    {
        return mTable[(int)sampler->addressing_mode - CL_ADDRESS_NONE]
                     [(int)sampler->filter_mode - CL_FILTER_NEAREST];
    }

    AddressFn mTable[6][2];
};

static AddressingTable sAddressingTable;

bool is_sRGBA_order(cl_channel_order image_channel_order)
{
    switch (image_channel_order)
    {
        case CL_sRGB:
        case CL_sRGBx:
        case CL_sRGBA:
        case CL_sBGRA: return true;
        default: return false;
    }
}

// Format helpers

int has_alpha(const cl_image_format *format)
{
    switch (format->image_channel_order)
    {
        case CL_R: return 0;
        case CL_A: return 1;
        case CL_Rx: return 0;
        case CL_RG: return 0;
        case CL_RA: return 1;
        case CL_RGx: return 0;
        case CL_RGB:
        case CL_sRGB: return 0;
        case CL_RGBx:
        case CL_sRGBx: return 0;
        case CL_RGBA: return 1;
        case CL_BGRA: return 1;
        case CL_ARGB: return 1;
        case CL_ABGR: return 1;
        case CL_INTENSITY: return 1;
        case CL_LUMINANCE: return 0;
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE: return 1;
#endif
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE: return 1;
#endif
        case CL_sRGBA:
        case CL_sBGRA: return 1;
        case CL_DEPTH: return 0;
        default:
            log_error("Invalid image channel order: %d\n",
                      format->image_channel_order);
            return 0;
    }
}

#define PRINT_MAX_SIZE_LOGIC 0

#define SWAP(_a, _b)                                                           \
    do                                                                         \
    {                                                                          \
        _a ^= _b;                                                              \
        _b ^= _a;                                                              \
        _a ^= _b;                                                              \
    } while (0)

void get_max_sizes(
    size_t *numberOfSizes, const int maxNumberOfSizes, size_t sizes[][3],
    size_t maxWidth, size_t maxHeight, size_t maxDepth, size_t maxArraySize,
    const cl_ulong maxIndividualAllocSize, // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    const cl_ulong maxTotalAllocSize, // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_mem_object_type image_type, const cl_image_format *format,
    int usingMaxPixelSizeBuffer)
{

    bool is3D = (image_type == CL_MEM_OBJECT_IMAGE3D);
    bool isArray = (image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY
                    || image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY);

    // Validate we have a reasonable max depth for 3D
    if (is3D && maxDepth < 2)
    {
        log_error("ERROR: Requesting max image sizes for 3D images when max "
                  "depth is < 2.\n");
        *numberOfSizes = 0;
        return;
    }
    // Validate we have a reasonable max array size for 1D & 2D image arrays
    if (isArray && maxArraySize < 2)
    {
        log_error("ERROR: Requesting max image sizes for an image array when "
                  "max array size is < 1.\n");
        *numberOfSizes = 0;
        return;
    }

    // Reduce the maximum because we are trying to test the max image
    // dimensions, not the memory allocation
    cl_ulong adjustedMaxTotalAllocSize = maxTotalAllocSize / 4;
    cl_ulong adjustedMaxIndividualAllocSize = maxIndividualAllocSize / 4;
    log_info("Note: max individual allocation adjusted down from %gMB to %gMB "
             "and max total allocation adjusted down from %gMB to %gMB.\n",
             maxIndividualAllocSize / (1024.0 * 1024.0),
             adjustedMaxIndividualAllocSize / (1024.0 * 1024.0),
             maxTotalAllocSize / (1024.0 * 1024.0),
             adjustedMaxTotalAllocSize / (1024.0 * 1024.0));

    // Cap our max allocation to 1.0GB.
    // FIXME -- why?  In the interest of not taking a long time?  We should
    // still test this stuff...
    if (adjustedMaxTotalAllocSize > (cl_ulong)1024 * 1024 * 1024)
    {
        adjustedMaxTotalAllocSize = (cl_ulong)1024 * 1024 * 1024;
        log_info("Limiting max total allocation size to %gMB (down from %gMB) "
                 "for test.\n",
                 adjustedMaxTotalAllocSize / (1024.0 * 1024.0),
                 maxTotalAllocSize / (1024.0 * 1024.0));
    }

    cl_ulong maxAllocSize = adjustedMaxIndividualAllocSize;
    if (adjustedMaxTotalAllocSize < adjustedMaxIndividualAllocSize * 2)
        maxAllocSize = adjustedMaxTotalAllocSize / 2;

    size_t raw_pixel_size = get_pixel_size(format);
    // If the test will be creating input (src) buffer of type int4 or float4,
    // number of pixels will be governed by sizeof(int4 or float4) and not
    // sizeof(dest fomat) Also if pixel size is 12 bytes i.e. RGB or RGBx, we
    // adjust it to 16 bytes as GPUs has no concept of 3 channel images. GPUs
    // expand these to four channel RGBA.
    if (usingMaxPixelSizeBuffer || raw_pixel_size == 12) raw_pixel_size = 16;
    size_t max_pixels = (size_t)maxAllocSize / raw_pixel_size;

    log_info("Maximums: [%zu x %zu x %zu], raw pixel size %zu bytes, "
             "per-allocation limit %gMB.\n",
             maxWidth, maxHeight, isArray ? maxArraySize : maxDepth,
             raw_pixel_size, (maxAllocSize / (1024.0 * 1024.0)));

    // Keep track of the maximum sizes for each dimension
    size_t maximum_sizes[] = { maxWidth, maxHeight, maxDepth };

    switch (image_type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            maximum_sizes[1] = maxArraySize;
            maximum_sizes[2] = 1;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            maximum_sizes[2] = maxArraySize;
            break;
    }


        // Given one fixed sized dimension, this code finds one or two other
        // dimensions, both with very small size, such that the size does not
        // exceed the maximum passed to this function

#if defined(__x86_64) || defined(__arm64__) || defined(__ppc64__)
    size_t other_sizes[] = { 2, 3, 5, 6, 7, 9, 10, 11, 13, 15 };
#else
    size_t other_sizes[] = { 2, 3, 5, 6, 7, 9, 11, 13 };
#endif

    static size_t other_size = 0;
    enum
    {
        num_other_sizes = sizeof(other_sizes) / sizeof(size_t)
    };

    (*numberOfSizes) = 0;

    if (image_type == CL_MEM_OBJECT_IMAGE1D)
    {

        size_t M = maximum_sizes[0];

        // Store the size
        sizes[(*numberOfSizes)][0] = M;
        sizes[(*numberOfSizes)][1] = 1;
        sizes[(*numberOfSizes)][2] = 1;
        ++(*numberOfSizes);
    }

    else if (image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY
             || image_type == CL_MEM_OBJECT_IMAGE2D)
    {

        for (int fixed_dim = 0; fixed_dim < 2; ++fixed_dim)
        {

            // Determine the size of the fixed dimension
            size_t M = maximum_sizes[fixed_dim];
            size_t A = max_pixels;

            int x0_dim = !fixed_dim;
            size_t x0 = static_cast<size_t>(
                fmin(fmin(other_sizes[(other_size++) % num_other_sizes], A / M),
                     maximum_sizes[x0_dim]));

            // Store the size
            sizes[(*numberOfSizes)][fixed_dim] = M;
            sizes[(*numberOfSizes)][x0_dim] = x0;
            sizes[(*numberOfSizes)][2] = 1;
            ++(*numberOfSizes);
        }
    }

    else if (image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY
             || image_type == CL_MEM_OBJECT_IMAGE3D)
    {

        // Iterate over dimensions, finding sizes for the non-fixed dimension
        for (int fixed_dim = 0; fixed_dim < 3; ++fixed_dim)
        {

            // Determine the size of the fixed dimension
            size_t M = maximum_sizes[fixed_dim];
            size_t A = max_pixels;

            // Find two other dimensions, x0 and x1
            int x0_dim = (fixed_dim == 0) ? 1 : 0;
            int x1_dim = (fixed_dim == 2) ? 1 : 2;

            // Choose two other sizes for these dimensions
            size_t x0 = static_cast<size_t>(
                fmin(fmin(A / M, maximum_sizes[x0_dim]),
                     other_sizes[(other_size++) % num_other_sizes]));
            // GPUs have certain restrictions on minimum width (row alignment)
            // of images which has given us issues testing small widths in this
            // test (say we set width to 3 for testing, and compute size based
            // on this width and decide it fits within vram ... but GPU driver
            // decides that, due to row alignment requirements, it has to use
            // width of 16 which doesnt fit in vram). For this purpose we are
            // not testing width < 16 for this test.
            if (x0_dim == 0 && x0 < 16) x0 = 16;
            size_t x1 = static_cast<size_t>(
                fmin(fmin(A / M / x0, maximum_sizes[x1_dim]),
                     other_sizes[(other_size++) % num_other_sizes]));

            // Valid image sizes cannot be below 1. Due to the workaround for
            // the xo_dim where x0 is overidden to 16 there might not be enough
            // space left for x1 dimension. This could be a fractional 0.x size
            // that when cast to integer would result in a value 0. In these
            // cases we clamp the size to a minimum of 1.
            if (x1 < 1) x1 = 1;

            // M and x0 cannot be '0' as they derive from clDeviceInfo calls
            assert(x0 > 0 && M > 0);

            // Store the size
            sizes[(*numberOfSizes)][fixed_dim] = M;
            sizes[(*numberOfSizes)][x0_dim] = x0;
            sizes[(*numberOfSizes)][x1_dim] = x1;
            ++(*numberOfSizes);
        }
    }

    // Log the results
    for (int j = 0; j < (int)(*numberOfSizes); j++)
    {
        switch (image_type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                log_info(" size[%d] = [%zu] (%g MB image)\n", j, sizes[j][0],
                         raw_pixel_size * sizes[j][0] * sizes[j][1]
                             * sizes[j][2] / (1024.0 * 1024.0));
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            case CL_MEM_OBJECT_IMAGE2D:
                log_info(" size[%d] = [%zu %zu] (%g MB image)\n", j,
                         sizes[j][0], sizes[j][1],
                         raw_pixel_size * sizes[j][0] * sizes[j][1]
                             * sizes[j][2] / (1024.0 * 1024.0));
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            case CL_MEM_OBJECT_IMAGE3D:
                log_info(" size[%d] = [%zu %zu %zu] (%g MB image)\n", j,
                         sizes[j][0], sizes[j][1], sizes[j][2],
                         raw_pixel_size * sizes[j][0] * sizes[j][1]
                             * sizes[j][2] / (1024.0 * 1024.0));
                break;
        }
    }
}

float get_max_absolute_error(const cl_image_format *format,
                             image_sampler_data *sampler)
{
    if (sampler->filter_mode == CL_FILTER_NEAREST) return 0.0f;

    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8: return 1.0f / 127.0f;
        case CL_UNORM_INT8: return 1.0f / 255.0f;
        case CL_UNORM_INT16: return 1.0f / 65535.0f;
        case CL_SNORM_INT16: return 1.0f / 32767.0f;
        case CL_FLOAT: return CL_FLT_MIN;
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: return 0x1.0p-14f;
#endif
        case CL_UNORM_SHORT_555:
        case CL_UNORM_SHORT_565: return 1.0f / 31.0f;
        default: return 0.0f;
    }
}

float get_max_relative_error(const cl_image_format *format,
                             image_sampler_data *sampler, int is3D,
                             int isLinearFilter)
{
    float maxError = 0.0f;
    float sampleCount = 1.0f;
    if (isLinearFilter) sampleCount = is3D ? 8.0f : 4.0f;

    // Note that the ULP is defined here as the unit in the last place of the
    // maximum magnitude sample used for filtering.

    // Section 8.3
    switch (format->image_channel_data_type)
    {
        // The spec allows 2 ulps of error for normalized formats
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_UNORM_INT_101010:
            // Maximum sampling error for round to zero normalization based on
            // multiplication by reciprocal (using reciprocal generated in
            // round to +inf mode, so that 1.0 matches spec)
            maxError = 2 * FLT_EPSILON * sampleCount;
            break;

            // If the implementation supports these formats then it will have to
            // allow rounding error here too, because not all 32-bit ints are
            // exactly representable in float
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32: maxError = 1 * FLT_EPSILON; break;
    }


    // Section 8.2
    if (sampler->addressing_mode == CL_ADDRESS_REPEAT
        || sampler->addressing_mode == CL_ADDRESS_MIRRORED_REPEAT
        || sampler->filter_mode != CL_FILTER_NEAREST
        || sampler->normalized_coords)
#if defined(__APPLE__)
    {
        if (sampler->filter_mode != CL_FILTER_NEAREST)
        {
            // The maximum
            if (gDeviceType == CL_DEVICE_TYPE_GPU)
                // Some GPUs ain't so accurate
                maxError += MAKE_HEX_FLOAT(0x1.0p-4f, 0x1L, -4);
            else
                // The standard method of 2d linear filtering delivers 4.0 ulps
                // of error in round to nearest (8 in rtz).
                maxError += 4.0f * FLT_EPSILON;
        }
        else
            // normalized coordinates will introduce some error into the
            // fractional part of the address, affecting results
            maxError += 4.0f * FLT_EPSILON;
    }
#else
    {
#if !defined(_WIN32)
#warning Implementations will likely wish to pick a max allowable sampling error policy here that is better than the spec
#endif
        // The spec allows linear filters to return any result most of the time.
        // That's fine for implementations but a problem for testing. After all
        // users aren't going to like garbage images.  We have "picked a number"
        // here that we are going to attempt to conform to. Implementations are
        // free to pick another number, like infinity, if they like.
        // We picked a number for you, to provide /some/ sanity
        maxError = MAKE_HEX_FLOAT(0x1.0p-7f, 0x1L, -7);
        // ...but this is what the spec allows:
        // maxError = INFINITY;
        // Please feel free to pick any positive number. (NaN wont work.)
    }
#endif

    // The error calculation itself can introduce error
    maxError += FLT_EPSILON * 2;

    return maxError;
}

size_t get_format_max_int(const cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8:
        case CL_SIGNED_INT8: return 127;
        case CL_UNORM_INT8:
        case CL_UNSIGNED_INT8: return 255;

        case CL_SNORM_INT16:
        case CL_SIGNED_INT16: return 32767;

        case CL_UNORM_INT16:
        case CL_UNSIGNED_INT16: return 65535;

        case CL_SIGNED_INT32: return 2147483647L;

        case CL_UNSIGNED_INT32: return 4294967295LL;

        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555: return 31;

        case CL_UNORM_INT_101010: return 1023;

        case CL_HALF_FLOAT: return 1 << 10;

#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: return 16384;
#endif
        default: return 0;
    }
}

int get_format_min_int(const cl_image_format *format)
{
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8:
        case CL_SIGNED_INT8: return -128;
        case CL_UNORM_INT8:
        case CL_UNSIGNED_INT8: return 0;

        case CL_SNORM_INT16:
        case CL_SIGNED_INT16: return -32768;

        case CL_UNORM_INT16:
        case CL_UNSIGNED_INT16: return 0;

        case CL_SIGNED_INT32: return -2147483648LL;

        case CL_UNSIGNED_INT32: return 0;

        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
        case CL_UNORM_INT_101010: return 0;

        case CL_HALF_FLOAT: return -(1 << 10);

#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: return -16384;
#endif

        default: return 0;
    }
}

cl_half convert_float_to_half(float f)
{
    switch (gFloatToHalfRoundingMode)
    {
        case kRoundToNearestEven: return cl_half_from_float(f, CL_HALF_RTE);
        case kRoundTowardZero: return cl_half_from_float(f, CL_HALF_RTZ);
        default:
            log_error("ERROR: Test internal error -- unhandled or unknown "
                      "float->half rounding mode.\n");
            exit(-1);
            return 0xffff;
    }
}

cl_ulong get_image_size(image_descriptor const *imageInfo)
{
    cl_ulong imageSize;

    // Assumes rowPitch and slicePitch are always correctly defined
    if (/*gTestMipmaps*/ imageInfo->num_mip_levels > 1)
    {
        imageSize = (size_t)compute_mipmapped_image_size(*imageInfo);
    }
    else
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D: imageSize = imageInfo->rowPitch; break;
            case CL_MEM_OBJECT_IMAGE2D:
                imageSize = imageInfo->height * imageInfo->rowPitch;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                imageSize = imageInfo->depth * imageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                imageSize = imageInfo->arraySize * imageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                imageSize = imageInfo->arraySize * imageInfo->slicePitch;
                break;
            default:
                log_error("ERROR: Cannot identify image type %x\n",
                          imageInfo->type);
                abort();
        }
    }
    return imageSize;
}

// Calculate image size in megabytes (strictly, mebibytes). Result is rounded
// up.
cl_ulong get_image_size_mb(image_descriptor const *imageInfo)
{
    cl_ulong imageSize = get_image_size(imageInfo);
    cl_ulong mb = imageSize / (1024 * 1024);
    if (imageSize % (1024 * 1024) > 0)
    {
        mb += 1;
    }
    return mb;
}


uint64_t gRoundingStartValue = 0;


void escape_inf_nan_values(char *data, size_t allocSize)
{
    // filter values with 8 not-quite-highest bits
    unsigned int *intPtr = (unsigned int *)data;
    for (size_t i = 0; i<allocSize>> 2; i++)
    {
        if ((intPtr[i] & 0x7F800000) == 0x7F800000) intPtr[i] ^= 0x40000000;
    }

    // Ditto with half floats (16-bit numbers with the 5 not-quite-highest bits
    // = 0x7C00 are special)
    unsigned short *shortPtr = (unsigned short *)data;
    for (size_t i = 0; i<allocSize>> 1; i++)
    {
        if ((shortPtr[i] & 0x7C00) == 0x7C00) shortPtr[i] ^= 0x4000;
    }
}

char *generate_random_image_data(image_descriptor *imageInfo,
                                 BufferOwningPtr<char> &P, MTdata d)
{
    size_t allocSize = static_cast<size_t>(get_image_size(imageInfo));
    size_t pixelRowBytes = imageInfo->width * get_pixel_size(imageInfo->format);
    size_t i;

    if (imageInfo->num_mip_levels > 1)
        allocSize =
            static_cast<size_t>(compute_mipmapped_image_size(*imageInfo));

#if defined(__APPLE__)
    char *data = NULL;
    if (gDeviceType == CL_DEVICE_TYPE_CPU)
    {
        size_t mapSize = ((allocSize + 4095L) & -4096L) + 8192;

        void *map = mmap(0, mapSize, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, 0, 0);
        intptr_t data_end = (intptr_t)map + mapSize - 4096;
        data = (char *)(data_end - (intptr_t)allocSize);

        mprotect(map, 4096, PROT_NONE);
        mprotect((void *)((char *)map + mapSize - 4096), 4096, PROT_NONE);
        P.reset(data, map, mapSize, allocSize);
    }
    else
    {
        data = (char *)malloc(allocSize);
        P.reset(data, NULL, 0, allocSize);
    }
#else
    P.reset(NULL); // Free already allocated memory first, then try to allocate
                   // new block.
    char *data =
        (char *)align_malloc(allocSize, get_pixel_alignment(imageInfo->format));
    P.reset(data, NULL, 0, allocSize, true);
#endif

    if (data == NULL)
    {
        log_error("ERROR: Unable to malloc %zu bytes for "
                  "generate_random_image_data\n",
                  allocSize);
        return 0;
    }

    if (gTestRounding)
    {
        // Special case: fill with a ramp from 0 to the size of the type
        size_t typeSize = get_format_type_size(imageInfo->format);
        switch (typeSize)
        {
            case 1: {
                char *ptr = data;
                for (i = 0; i < allocSize; i++)
                    ptr[i] = (cl_char)(i + gRoundingStartValue);
            }
            break;
            case 2: {
                cl_short *ptr = (cl_short *)data;
                for (i = 0; i < allocSize / 2; i++)
                    ptr[i] = (cl_short)(i + gRoundingStartValue);
            }
            break;
            case 4: {
                cl_int *ptr = (cl_int *)data;
                for (i = 0; i < allocSize / 4; i++)
                    ptr[i] = (cl_int)(i + gRoundingStartValue);
            }
            break;
        }

        // Note: inf or nan float values would cause problems, although we don't
        // know this will actually be a float, so we just know what to look for
        escape_inf_nan_values(data, allocSize);
        return data;
    }

    // Otherwise, we should be able to just fill with random bits no matter what
    cl_uint *p = (cl_uint *)data;
    for (i = 0; i + 4 <= allocSize; i += 4) p[i / 4] = genrand_int32(d);

    for (; i < allocSize; i++) data[i] = genrand_int32(d);

    // Note: inf or nan float values would cause problems, although we don't
    // know this will actually be a float, so we just know what to look for
    escape_inf_nan_values(data, allocSize);

    if (/*!gTestMipmaps*/ imageInfo->num_mip_levels < 2)
    {
        // Fill unused edges with -1, NaN for float
        if (imageInfo->rowPitch > pixelRowBytes)
        {
            size_t height = 0;

            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE2D:
                case CL_MEM_OBJECT_IMAGE3D:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                    height = imageInfo->height;
                    break;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                    height = imageInfo->arraySize;
                    break;
            }

            // Fill in the row padding regions
            for (i = 0; i < height; i++)
            {
                size_t offset = i * imageInfo->rowPitch + pixelRowBytes;
                size_t length = imageInfo->rowPitch - pixelRowBytes;
                memset(data + offset, 0xff, length);
            }
        }

        // Fill in the slice padding regions, if necessary:

        size_t slice_dimension = imageInfo->height;
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        {
            slice_dimension = imageInfo->arraySize;
        }

        if (imageInfo->slicePitch > slice_dimension * imageInfo->rowPitch)
        {
            size_t depth = 0;
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE2D:
                case CL_MEM_OBJECT_IMAGE3D: depth = imageInfo->depth; break;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                    depth = imageInfo->arraySize;
                    break;
            }

            for (i = 0; i < depth; i++)
            {
                size_t offset = i * imageInfo->slicePitch
                    + slice_dimension * imageInfo->rowPitch;
                size_t length = imageInfo->slicePitch
                    - slice_dimension * imageInfo->rowPitch;
                memset(data + offset, 0xff, length);
            }
        }
    }

    return data;
}

#ifdef __SSE2__
#define CLAMP_FLOAT_V(v)                                                       \
    (_mm_max_ps(_mm_min_ps(v, _mm_set1_ps(1.f)), _mm_set1_ps(-1.f)))
#define CLAMP_FLOAT(v)                                                         \
    (_mm_max_ss(_mm_min_ss(v, _mm_set_ss(1.f)), _mm_set_ss(-1.f)))

#ifdef __SSE4_1__
#define SET_ALPHA_1(v)                                                         \
    (_mm_insert_ps(v, _mm_set_ss(1.f), _MM_MK_INSERTPS_NDX(0, 3, 0)))
#define SELECT_F(cond, a, b) _mm_blendv_ps(b, a, cond)
#define EXTRACT_I(v, i) _mm_extract_epi32(v, i)
#ifdef __x86_64
#define EXTRACT_I64(v, i) _mm_extract_epi64(v, i)
#endif
#else
#define SET_ALPHA_1(v) (_mm_movelh_ps(v, _mm_set_ps(0.f, 0.f, 1.f, 0.f)))
// n.b. "ANDNOT" is ~A & B, not A & ~B!!
#define SELECT_F(cond, a, b)                                                   \
    _mm_or_ps(_mm_and_ps(cond, a), _mm_andnot_ps(cond, b))
#define EXTRACT_I(v, i)                                                        \
    _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, i)))
#ifdef __x86_64
#define EXTRACT_I64(v, i)                                                      \
    _mm_cvtsi128_si64(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 2 * i + 1, 2 * i)))
#endif
#endif

static __m128 read_image_pixel_float(void *imageData,
                                     image_descriptor *imageInfo, __m128i coord,
                                     int lod)
{
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth;
    size_t slice_pitch_lod = 0, row_pitch_lod = 0;

    if (imageInfo->num_mip_levels > 1)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                depth_lod =
                    (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                height_lod =
                    (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            default:
                width_lod =
                    (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        }
        row_pitch_lod = width_lod * get_pixel_size(imageInfo->format);
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            slice_pitch_lod = row_pitch_lod;
        else if (imageInfo->type == CL_MEM_OBJECT_IMAGE3D
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            slice_pitch_lod = row_pitch_lod * height_lod;
    }
    else
    {
        row_pitch_lod = imageInfo->rowPitch;
        slice_pitch_lod = imageInfo->slicePitch;
    }

    __m128i extent = _mm_set_epi32(
        0,
        imageInfo->arraySize != 0 ? (int)imageInfo->arraySize : (int)depth_lod,
        (int)height_lod, (int)width_lod);
    __m128i zero = _mm_setzero_si128();
    __m128i minMask = _mm_cmplt_epi32(coord, zero);
    __m128i maxMask =
        _mm_cmpgt_epi32(coord, _mm_add_epi32(extent, _mm_setmone_si128()));
    __m128i extentMask = _mm_cmpeq_epi32(extent, zero);
    __m128i boundsMask =
        _mm_or_si128(_mm_andnot_si128(extentMask, maxMask), minMask);
    if (TEST_NONZERO(boundsMask))
        return has_alpha(imageInfo->format) ? _mm_setzero_ps()
                                            : _mm_set_ps(1.f, 0.f, 0.f, 0.f);

    const cl_image_format *format = imageInfo->format;

    __m128 tempData;
    // Predeclare a bunch of reciprocal constants so GCC doesn't use expensive
    // divisions to compute them in our code
    static const float recip_31 = 1.0f / 31.0f;
    static const float recip_63 = 1.0f / 63.0f;
    static const float recip_127 = 1.0f / 127.0f;
    static const float recip_255 = 1.0f / 255.0f;
    static const float recip_1023 = 1.0f / 1023.0f;
    static const float recip_32767 = 1.0f / 32767.0f;
    static const float recip_65535 = 1.0f / 65535.0f;

    // Advance to the right spot
    char *ptr = (char *)imageData;
    size_t pixelSize = get_pixel_size(format);
    __m128i pitch_lod = _mm_set_epi32(0, (int)slice_pitch_lod,
                                      (int)row_pitch_lod, (int)pixelSize);

    __m128i offsetA = _mm_mul_epu32(coord, pitch_lod);
    __m128i offsetB =
        _mm_mul_epu32(_mm_shuffle_epi32(coord, _MM_SHUFFLE(2, 3, 0, 1)),
                      _mm_shuffle_epi32(pitch_lod, _MM_SHUFFLE(2, 3, 0, 1)));
#ifdef __x86_64
    ptr += EXTRACT_I64(offsetB, 0) + EXTRACT_I64(offsetA, 1)
        + EXTRACT_I64(offsetA, 0);
#else
    // Using PHADDD doesn't gain us much...
    ptr +=
        EXTRACT_I(offsetB, 0) + EXTRACT_I(offsetA, 2) + EXTRACT_I(offsetA, 0);
#endif

    // OpenCL only supports reading floats from certain formats
    size_t channelCount = get_format_channel_count(format);
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData = SET_ALPHA_1(CLAMP_FLOAT(
                        _mm_mul_ss(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]),
                                   _mm_set_ss(recip_127))));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr), 0x7F00, 1);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr),
                                             0x7F00 | dPtr[2], 1);
                    break;
                case 4: pixel = _mm_loadu_si32(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(pixel));
#else
                __m128i signMask;
                signMask = _mm_cmpgt_epi8(_mm_setzero_si128(), pixel);
                pixel = _mm_unpacklo_epi8(pixel, signMask);
                signMask = _mm_unpacklo_epi8(signMask, signMask);
                tempData = _mm_cvtepi32_ps(_mm_unpacklo_epi16(pixel, signMask));
#endif
                tempData =
                    CLAMP_FLOAT_V(_mm_mul_ps(tempData, _mm_set1_ps(recip_127)));
            }
            break;
        }

        case CL_UNORM_INT8: {
            unsigned char *dPtr = (unsigned char *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    if ((is_sRGBA_order(
                            imageInfo->format->image_channel_order)))
                        tempData = SET_ALPHA_1(_mm_set_ss(sRGBunmap(dPtr[0])));
                    else
                        tempData = SET_ALPHA_1(_mm_mul_ss(
                            _mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]),
                            _mm_set_ss(recip_255)));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr), 0xFF00, 1);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr),
                                             0xFF00 | dPtr[2], 1);
                    break;
                case 4: pixel = _mm_loadu_si32(ptr);
#ifdef CL_1RGB_APPLE
                    if (format->image_channel_order == CL_1RGB_APPLE)
#ifdef __SSE4_1__
                        pixel = _mm_insert_epi8(pixel, 0xFF, 0);
#else
                        pixel =
                            _mm_or_si128(pixel,
                                         _mm_bsrli_si128(_mm_setmone_si128(),
                                                         15) /* 0x000000FF */);
#endif
#endif
#ifdef CL_BGR1_APPLE
                    if (format->image_channel_order == CL_BGR1_APPLE)
#ifdef __SSE4_1__
                        pixel = _mm_insert_epi8(pixel, 0xFF, 3);
#else
                        pixel = _mm_or_si128(
                            pixel,
                            _mm_bslli_si128(
                                _mm_bsrli_si128(_mm_setmone_si128(), 15),
                                3) /* 0xFF000000 */);
#endif
#endif
                    break;
            }
            if (channelCount != 1)
            {
                if (is_sRGBA_order(imageInfo->format->image_channel_order))
                    tempData = sRGBunmap(pixel);
                else
                {
#ifdef __SSE4_1__
                    tempData = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pixel));
#else
                    __m128i zero = _mm_setzero_si128();
                    tempData = _mm_cvtepi32_ps(_mm_unpacklo_epi16(
                        _mm_unpacklo_epi8(pixel, zero), zero));
#endif
                    tempData = _mm_mul_ps(tempData, _mm_set1_ps(recip_255));
                }
            }
            break;
        }

        case CL_SIGNED_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr), 0x0100, 1);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr),
                                             0x0100 | dPtr[2], 1);
                    break;
                case 4: pixel = _mm_loadu_si32(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(pixel));
#else
                __m128i signMask;
                signMask = _mm_cmpgt_epi8(_mm_setzero_si128(), pixel);
                pixel = _mm_unpacklo_epi8(pixel, signMask);
                signMask = _mm_unpacklo_epi8(signMask, signMask);
                tempData = _mm_cvtepi32_ps(_mm_unpacklo_epi16(pixel, signMask));
#endif
            }
            break;
        }

        case CL_UNSIGNED_INT8: {
            cl_uchar *dPtr = (cl_uchar *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr), 0x0100, 1);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(_mm_loadu_si16(ptr),
                                             0x0100 | dPtr[2], 1);
                    break;
                case 4: pixel = _mm_loadu_si32(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(pixel));
#else
                __m128i zero = _mm_setzero_si128();
                tempData = _mm_cvtepi32_ps(
                    _mm_unpacklo_epi16(_mm_unpacklo_epi8(pixel, zero), zero));
#endif
            }
            break;
        }

        case CL_SNORM_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData = SET_ALPHA_1(CLAMP_FLOAT(
                        _mm_mul_ss(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]),
                                   _mm_set_ss(recip_32767))));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si32(ptr), 0x7FFF, 3);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2),
                        0x7FFF, 3);
                    break;
                case 4: pixel = _mm_loadu_si64(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(pixel));
#else
                tempData = _mm_cvtepi32_ps(_mm_unpacklo_epi16(
                    pixel, _mm_cmpgt_epi16(_mm_setzero_si128(), pixel)));
#endif
                tempData = CLAMP_FLOAT_V(
                    _mm_mul_ps(tempData, _mm_set1_ps(recip_32767)));
            }
            break;
        }

        case CL_UNORM_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData = SET_ALPHA_1(
                        _mm_mul_ss(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]),
                                   _mm_set_ss(recip_65535)));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si32(ptr), 0xFFFF, 3);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2),
                        0xFFFF, 3);
                    break;
                case 4: pixel = _mm_loadu_si64(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(pixel));
#else
                tempData = _mm_cvtepi32_ps(
                    _mm_unpacklo_epi16(pixel, _mm_setzero_si128()));
#endif
                tempData = _mm_mul_ps(tempData, _mm_set1_ps(recip_65535));
            }
            break;
        }

        case CL_SIGNED_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si32(ptr), 1, 3);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2), 1,
                        3);
                    break;
                case 4: pixel = _mm_loadu_si64(ptr); break;
            }
            if (channelCount != 1)
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(pixel));
#else
                tempData = _mm_cvtepi32_ps(_mm_unpacklo_epi16(
                    pixel, _mm_cmpgt_epi16(_mm_setzero_si128(), pixel)));
#endif
            break;
        }

        case CL_UNSIGNED_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si32(ptr), 1, 3);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2), 1,
                        3);
                    break;
                case 4: pixel = _mm_loadu_si64(ptr); break;
            }
            if (channelCount != 1)
#ifdef __SSE4_1__
                tempData = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(pixel));
#else
                tempData = _mm_cvtepi32_ps(
                    _mm_unpacklo_epi16(pixel, _mm_setzero_si128()));
#endif
            break;
        }

        case CL_HALF_FLOAT: {
            cl_half *dPtr = (cl_half *)ptr;
            __m128i h;
            switch (channelCount)
            {
                case 1:
#ifdef __F16C__
                    tempData = SET_ALPHA_1(_mm_set_ss(_cvtsh_ss(dPtr[0])));
#else
                    tempData =
                        SET_ALPHA_1(_mm_set_ss(cl_half_to_float(dPtr[0])));
#endif
                    break;
                case 2:
                    h = _mm_insert_epi16(_mm_loadu_si32(ptr), 0x3C00, 3);
                    break;
                case 3:
                    h = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2),
                        0x3C00, 3);
                    break;
                case 4: h = _mm_loadu_si64(ptr); break;
            }

            if (channelCount == 1) break;

            tempData = cl_half_to_float(h);
            break;
        }

        case CL_SIGNED_INT32: {
            cl_int *dPtr = (cl_int *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi32_ss(_mm_setzero_ps(), dPtr[0]));
                    break;
                case 2:
#ifdef __SSE4_1__
                    pixel = _mm_insert_epi32(_mm_loadu_si64(ptr), 1, 3);
#else
                    pixel = _mm_insert_epi16(_mm_loadu_si64(ptr), 1, 6);
#endif
                    break;
                case 3:
#ifdef __SSE4_1__
                    pixel = _mm_insert_epi32(
                        _mm_insert_epi32(_mm_loadu_si64(ptr), dPtr[2], 2), 1,
                        3);
#else
                    pixel = _mm_or_si128(_mm_loadu_si64(ptr),
                                         _mm_set_epi32(1, dPtr[2], 0, 0));
#endif
                    break;
                case 4: pixel = _mm_loadu_si128((__m128i_u *)ptr); break;
            }
            if (channelCount != 1) tempData = _mm_cvtepi32_ps(pixel);
            break;
        }

        case CL_UNSIGNED_INT32: {
            cl_uint *dPtr = (cl_uint *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
#ifdef __x86_64
                    tempData =
                        SET_ALPHA_1(_mm_cvtsi64_ss(_mm_setzero_ps(), dPtr[0]));
#else
                    tempData = SET_ALPHA_1(_mm_set_ss((float)dPtr[0]));
#endif
                    break;
                case 2:
#ifdef __SSE4_1__
                    pixel = _mm_insert_epi32(_mm_loadu_si64(ptr), 1, 3);
#else
                    pixel = _mm_insert_epi16(_mm_loadu_si64(ptr), 1, 6);
#endif
                    break;
                case 3:
#ifdef __SSE4_1__
                    pixel = _mm_insert_epi32(
                        _mm_insert_epi32(_mm_loadu_si64(ptr), dPtr[2], 2), 1,
                        3);
#else
                    pixel = _mm_or_si128(_mm_loadu_si64(ptr),
                                         _mm_set_epi32(1, dPtr[2], 0, 0));
#endif
                    break;
                case 4: pixel = _mm_loadu_si128((__m128i_u *)ptr); break;
            }
            if (channelCount != 1)
            {
                // Unfortunately, no instruction for converting unsigned 32-bit
                // integers to float exists until AVX-512; nor is there an
                // instruction for converting packed 64-bit integers to float
                // until same
#ifdef __AVX512VL__
                tempData = _mm_cvtepu32_ps(pixel);
#elif defined(__SSE4_1__)
                // The following is based on the unoptimized output of GCC for
                // scalars
                __m128i negOne = _mm_setmone_si128();
                if (!_mm_testz_si128(
                        pixel, _mm_slli_epi32(negOne, 31) /* 0x80000000 */))
                {
                    __m128i one = _mm_srli_epi32(negOne, 31); // = 1;
                    __m128i reducedPixel = _mm_or_si128(
                        _mm_srli_epi32(pixel, 1), _mm_and_si128(pixel, one));
                    tempData = _mm_cvtepi32_ps(reducedPixel);
                    tempData = _mm_add_ps(tempData, tempData);
                    tempData = SELECT_F(_mm_castsi128_ps(pixel), tempData,
                                        _mm_cvtepi32_ps(pixel));
                }
                else
                    tempData = _mm_cvtepi32_ps(pixel);
#else
                // Testing contents of vectors is unwieldy without SSE4.1
                __m128i negOne = _mm_setmone_si128();
                __m128i one = _mm_srli_epi32(negOne, 31); // = 1;
                __m128i reducedPixel = _mm_or_si128(_mm_srli_epi32(pixel, 1),
                                                    _mm_and_si128(pixel, one));
                tempData = _mm_cvtepi32_ps(reducedPixel);
                tempData = _mm_add_ps(tempData, tempData);
                __m128 tempData2 = _mm_cvtepi32_ps(pixel);
                __m128 mask = _mm_cmpgt_ps(_mm_setzero_ps(), tempData2);
                tempData = SELECT_F(mask, tempData, tempData2);
#endif
            }
            break;
        }

        case CL_UNORM_SHORT_565: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
#ifdef __AVX2__
            __m128i pixel = _mm_broadcastd_epi32(_mm_loadu_si16(dPtr));
            pixel = _mm_insert_epi32(
                _mm_srlv_epi32(pixel, _mm_set_epi32(0, 0, 5, 11)), 1, 3);
#else
            // Shifts may as well be scalar, since before AVX2 there are no
            // vector shift amounts
            __m128i pixel =
                _mm_set_epi32(1, dPtr[0], dPtr[0] >> 5, dPtr[0] >> 11);
#endif
            pixel = _mm_and_si128(pixel, _mm_set_epi32(1, 0x1f, 0x3f, 0x1f));
            tempData =
                _mm_mul_ps(_mm_cvtepi32_ps(pixel),
                           _mm_set_ps(1.0f, recip_31, recip_63, recip_31));
            break;
        }

        case CL_UNORM_SHORT_555: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
#ifdef __AVX2__
            __m128i pixel = _mm_broadcastd_epi32(_mm_loadu_si16(dPtr));
            pixel = _mm_insert_epi32(
                _mm_srlv_epi32(pixel, _mm_set_epi32(0, 0, 5, 10)), 0x1F, 3);
#else
            __m128i pixel =
                _mm_set_epi32(0x1F, dPtr[0], dPtr[0] >> 5, dPtr[0] >> 10);
#endif
            pixel = _mm_and_si128(pixel, _mm_set1_epi16(0x1F));
            tempData =
                _mm_mul_ps(_mm_cvtepi32_ps(pixel), _mm_set1_ps(recip_31));
            break;
        }

        case CL_UNORM_INT_101010: {
            cl_uint *dPtr = (cl_uint *)ptr;
#ifdef __AVX2__
            __m128i pixel = _mm_broadcastd_epi32(_mm_loadu_si32(dPtr));
            pixel = _mm_insert_epi32(
                _mm_srlv_epi32(pixel, _mm_set_epi32(0, 0, 10, 20)), 0x3ff, 3);
#else
            __m128i pixel =
                _mm_set_epi32(0x3ff, dPtr[0], dPtr[0] >> 10, dPtr[0] >> 20);
#endif
            pixel = _mm_and_si128(pixel, _mm_set1_epi16(0x3ff));
            tempData =
                _mm_mul_ps(_mm_cvtepi32_ps(pixel), _mm_set1_ps(recip_1023));
            break;
        }

        case CL_FLOAT: {
            float *dPtr = (float *)ptr;
            switch (channelCount)
            {
                case 1: tempData = SET_ALPHA_1(_mm_load_ss(dPtr)); break;
                case 2:
                    tempData = _mm_loadl_pi(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f),
                                            (__m64 *)ptr);
                    break;
                case 3:
                    tempData = _mm_loadl_pi(
                        _mm_set_ps(1.0f, dPtr[2], 0.0f, 0.0f), (__m64 *)ptr);
                    break;
                case 4: tempData = _mm_loadu_ps(dPtr); break;
            }
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            __m128i pixel;
            switch (channelCount)
            {
                case 1:
                    tempData = SET_ALPHA_1(
                        _mm_mul_ss(_mm_cvtsi32_ss((int)dPtr[0] - 16384),
                                   _mm_set_ss(0x1.0p-14f)));
                    break;
                case 2:
                    pixel = _mm_insert_epi16(_mm_loadu_si32(ptr),
                                             (1 << 14) + 16384, 3);
                    break;
                case 3:
                    pixel = _mm_insert_epi16(
                        _mm_insert_epi16(_mm_loadu_si32(ptr), dPtr[2], 2),
                        (1 << 14) + 16384, 3);
                    break;
                case 4: pixel = _mm_loadu_si64(ptr); break;
            }
            if (channelCount != 1)
            {
#ifdef __SSE4_1__
                pixel = _mm_cvtepu16_epi32(pixel);
#else
                pixel = _mm_unpacklo_epi16(pixel, _mm_setzero_si128());
#endif
                tempData = _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(pixel, _mm_slli_epi32(_mm_setmone_si128(), 14) /* -16384 */, _mm_set1_ps(0x1.0p-14f));
            }
            break;
        }
#endif
    }

    switch (format->image_channel_order)
    {
        case CL_R:
        case CL_Rx:
        case CL_RG:
        case CL_RGx:
        case CL_RGB:
        case CL_RGBx:
        case CL_sRGB:
        case CL_sRGBx:
        case CL_RGBA:
        case CL_sRGBA:
        case CL_DEPTH:
            /* Already correct */
            return tempData;
        case CL_A:
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(0, 1, 1, 1));
        case CL_RA:
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(1, 2, 2, 0));
        case CL_ARGB:
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
#endif
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(0, 3, 2, 1));
        case CL_ABGR:
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(0, 1, 2, 3));
        case CL_BGRA:
        case CL_sBGRA:
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
#endif
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(3, 0, 1, 2));
        case CL_INTENSITY:
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(0, 0, 0, 0));
        case CL_LUMINANCE:
            return _mm_shuffle_ps(tempData, tempData, _MM_SHUFFLE(3, 0, 0, 0));
        default:
            log_error("Invalid format:");
            print_header(format, true);
            break;
    }
    return tempData;
}

void read_image_pixel_float(void *imageData, image_descriptor *imageInfo, int x,
                            int y, int z, float *outData, int lod)
{
    _mm_storeu_ps(outData,
                  read_image_pixel_float(imageData, imageInfo,
                                         _mm_set_epi32(0, z, y, x), lod));
}

#else

#define CLAMP_FLOAT(v) (fmaxf(fminf(v, 1.f), -1.f))

void read_image_pixel_float(void *imageData, image_descriptor *imageInfo, int x,
                            int y, int z, float *outData, int lod)
{
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth;
    size_t slice_pitch_lod = 0, row_pitch_lod = 0;

    if (imageInfo->num_mip_levels > 1)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                depth_lod =
                    (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                height_lod =
                    (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            default:
                width_lod =
                    (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        }
        row_pitch_lod = width_lod * get_pixel_size(imageInfo->format);
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            slice_pitch_lod = row_pitch_lod;
        else if (imageInfo->type == CL_MEM_OBJECT_IMAGE3D
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            slice_pitch_lod = row_pitch_lod * height_lod;
    }
    else
    {
        row_pitch_lod = imageInfo->rowPitch;
        slice_pitch_lod = imageInfo->slicePitch;
    }
    if (x < 0 || y < 0 || z < 0 || x >= (int)width_lod
        || (height_lod != 0 && y >= (int)height_lod)
        || (depth_lod != 0 && z >= (int)depth_lod)
        || (imageInfo->arraySize != 0 && z >= (int)imageInfo->arraySize))
    {
        outData[0] = outData[1] = outData[2] = outData[3] = 0;
        if (!has_alpha(imageInfo->format)) outData[3] = 1;
        return;
    }

    const cl_image_format *format = imageInfo->format;

    unsigned int i;
    float tempData[4];

    // Advance to the right spot
    char *ptr = (char *)imageData;
    size_t pixelSize = get_pixel_size(format);

    ptr += z * slice_pitch_lod + y * row_pitch_lod + x * pixelSize;

    // OpenCL only supports reading floats from certain formats
    size_t channelCount = get_format_channel_count(format);
    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            for (i = 0; i < channelCount; i++)
                tempData[i] = CLAMP_FLOAT((float)dPtr[i] / 127.0f);
            break;
        }

        case CL_UNORM_INT8: {
            unsigned char *dPtr = (unsigned char *)ptr;
            for (i = 0; i < channelCount; i++)
            {
                if ((is_sRGBA_order(imageInfo->format->image_channel_order))
                    && i < 3) // only RGB need to be converted for sRGBA
                    tempData[i] = sRGBunmap(dPtr[i]);
                else
                    tempData[i] = (float)dPtr[i] / 255.0f;
            }
            break;
        }

        case CL_SIGNED_INT8: {
            cl_char *dPtr = (cl_char *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT8: {
            cl_uchar *dPtr = (cl_uchar *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_SNORM_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            for (i = 0; i < channelCount; i++)
                tempData[i] = CLAMP_FLOAT((float)dPtr[i] / 32767.0f);
            break;
        }

        case CL_UNORM_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for (i = 0; i < channelCount; i++)
                tempData[i] = (float)dPtr[i] / 65535.0f;
            break;
        }

        case CL_SIGNED_INT16: {
            cl_short *dPtr = (cl_short *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT16: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_HALF_FLOAT: {
            cl_half *dPtr = (cl_half *)ptr;
            for (i = 0; i < channelCount; i++)
                tempData[i] = cl_half_to_float(dPtr[i]);
            break;
        }

        case CL_SIGNED_INT32: {
            cl_int *dPtr = (cl_int *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_UNSIGNED_INT32: {
            cl_uint *dPtr = (cl_uint *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }

        case CL_UNORM_SHORT_565: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            tempData[0] = (float)(dPtr[0] >> 11) / (float)31;
            tempData[1] = (float)((dPtr[0] >> 5) & 63) / (float)63;
            tempData[2] = (float)(dPtr[0] & 31) / (float)31;
            break;
        }

        case CL_UNORM_SHORT_555: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            tempData[0] = (float)((dPtr[0] >> 10) & 31) / (float)31;
            tempData[1] = (float)((dPtr[0] >> 5) & 31) / (float)31;
            tempData[2] = (float)(dPtr[0] & 31) / (float)31;
            break;
        }

        case CL_UNORM_INT_101010: {
            cl_uint *dPtr = (cl_uint *)ptr;
            tempData[0] = (float)((dPtr[0] >> 20) & 0x3ff) / (float)1023;
            tempData[1] = (float)((dPtr[0] >> 10) & 0x3ff) / (float)1023;
            tempData[2] = (float)(dPtr[0] & 0x3ff) / (float)1023;
            break;
        }

        case CL_FLOAT: {
            float *dPtr = (float *)ptr;
            for (i = 0; i < channelCount; i++) tempData[i] = (float)dPtr[i];
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for (i = 0; i < channelCount; i++)
                tempData[i] = ((int)dPtr[i] - 16384) * 0x1.0p-14f;
            break;
        }
#endif
    }


    outData[0] = outData[1] = outData[2] = 0;
    outData[3] = 1;

    switch (format->image_channel_order)
    {
        case CL_A: outData[3] = tempData[0]; break;
        case CL_R:
        case CL_Rx: outData[0] = tempData[0]; break;
        case CL_RA:
            outData[0] = tempData[0];
            outData[3] = tempData[1];
            break;
        case CL_RG:
        case CL_RGx:
            outData[0] = tempData[0];
            outData[1] = tempData[1];
            break;
        case CL_RGB:
        case CL_RGBx:
        case CL_sRGB:
        case CL_sRGBx:
            outData[0] = tempData[0];
            outData[1] = tempData[1];
            outData[2] = tempData[2];
            break;
        case CL_RGBA:
            outData[0] = tempData[0];
            outData[1] = tempData[1];
            outData[2] = tempData[2];
            outData[3] = tempData[3];
            break;
        case CL_ARGB:
            outData[0] = tempData[1];
            outData[1] = tempData[2];
            outData[2] = tempData[3];
            outData[3] = tempData[0];
            break;
        case CL_ABGR:
            outData[0] = tempData[3];
            outData[1] = tempData[2];
            outData[2] = tempData[1];
            outData[3] = tempData[0];
            break;
        case CL_BGRA:
        case CL_sBGRA:
            outData[0] = tempData[2];
            outData[1] = tempData[1];
            outData[2] = tempData[0];
            outData[3] = tempData[3];
            break;
        case CL_INTENSITY:
            outData[0] = tempData[0];
            outData[1] = tempData[0];
            outData[2] = tempData[0];
            outData[3] = tempData[0];
            break;
        case CL_LUMINANCE:
            outData[0] = tempData[0];
            outData[1] = tempData[0];
            outData[2] = tempData[0];
            break;
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
            outData[0] = tempData[1];
            outData[1] = tempData[2];
            outData[2] = tempData[3];
            outData[3] = 1.0f;
            break;
#endif
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
            outData[0] = tempData[2];
            outData[1] = tempData[1];
            outData[2] = tempData[0];
            outData[3] = 1.0f;
            break;
#endif
        case CL_sRGBA:
            outData[0] = tempData[0];
            outData[1] = tempData[1];
            outData[2] = tempData[2];
            outData[3] = tempData[3];
            break;
        case CL_DEPTH: outData[0] = tempData[0]; break;
        default:
            log_error("Invalid format:");
            print_header(format, true);
            break;
    }
}
#endif

void read_image_pixel_float(void *imageData, image_descriptor *imageInfo, int x,
                            int y, int z, float *outData)
{
    read_image_pixel_float(imageData, imageInfo, x, y, z, outData, 0);
}

bool get_integer_coords(float x, float y, float z, size_t width, size_t height,
                        size_t depth, image_sampler_data *imageSampler,
                        image_descriptor *imageInfo, int &outX, int &outY,
                        int &outZ)
{
    return get_integer_coords_offset(x, y, z, 0.0f, 0.0f, 0.0f, width, height,
                                     depth, imageSampler, imageInfo, outX, outY,
                                     outZ);
}

bool get_integer_coords_offset(float x, float y, float z, float xAddressOffset,
                               float yAddressOffset, float zAddressOffset,
                               size_t width, size_t height, size_t depth,
                               image_sampler_data *imageSampler,
                               image_descriptor *imageInfo, int &outX,
                               int &outY, int &outZ)
{
    AddressFn adFn = sAddressingTable[imageSampler];

#ifdef __SSE2__
    __m128 coord = _mm_set_ps(0.f, z, y, x);
    __m128 addressOffset =
        _mm_set_ps(0.f, zAddressOffset, yAddressOffset, xAddressOffset);
    __m128i extent = _mm_set_epi32(0, depth, height, width);
    __m128 extentf = _mm_cvtepi32_ps(extent);
    __m128i ref = vifloorf(coord);

    __m128 arrayMask;
    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            arrayMask = _mm_castsi128_ps(
                _mm_bsrli_si128(_mm_setmone_si128(), 12)); // = 0, 0, 0, -1
            break;
        case CL_MEM_OBJECT_IMAGE2D:
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            arrayMask = _mm_castsi128_ps(
                _mm_bsrli_si128(_mm_setmone_si128(), 8)); // = 0, 0, -1, -1
            break;
        default:
            arrayMask = _mm_castsi128_ps(
                _mm_bsrli_si128(_mm_setmone_si128(), 4)); // = 0, -1, -1, -1
    }
    __m128 locMask =
        _mm_andnot_ps(_mm_cmpeq_ps(extentf, _mm_setzero_ps()), arrayMask);
    __m128 offsetMask =
        _mm_andnot_ps(_mm_cmpeq_ps(addressOffset, _mm_setzero_ps()), arrayMask);

    // Handle sampler-directed coordinate normalization + clamping.  Note that
    // the array coordinate for image array types is expected to be
    // unnormalized, and is clamped to 0..arraySize-1.
    if (imageSampler->normalized_coords)
    {
        __m128 minMask, maxMask, temp;
        switch (imageSampler->addressing_mode)
        {
            case CL_ADDRESS_REPEAT:
                coord = SELECT_F(
                    locMask, RepeatNormalizedAddressFn(coord, extent), coord);

                // Add in the offset
                coord = _mm_add_ps(coord, addressOffset);
                // Handle wrapping
                minMask = _mm_andnot_ps(_mm_cmplt_ps(coord, _mm_setzero_ps()),
                                        offsetMask);
                maxMask =
                    _mm_andnot_ps(_mm_cmpgt_ps(coord, extentf), offsetMask);
                coord = SELECT_F(
                    minMask, _mm_add_ps(coord, extentf),
                    SELECT_F(maxMask, _mm_sub_ps(coord, extentf), coord));
                break;

            case CL_ADDRESS_MIRRORED_REPEAT:
                coord = SELECT_F(
                    locMask, MirroredRepeatNormalizedAddressFn(coord, extent),
                    coord);
                temp = _mm_add_ps(coord, addressOffset);
                maxMask = _mm_cmpgt_ps(temp, extentf);
                coord = SELECT_F(
                    offsetMask,
                    vfabsf(SELECT_F(
                        maxMask, _mm_sub_ps(extentf, _mm_sub_ps(temp, extentf)),
                        temp)),
                    coord);
                break;

            default:
                // Also, remultiply to the original coords. This simulates any
                // truncation in the pass to OpenCL
#ifdef __FMA4__
                coord =
                    SELECT_F(arrayMask,
                             _mm_macc_ps(coord, extentf, addressOffset), coord);
#elif defined(__FMA__)
                coord = SELECT_F(arrayMask,
                                 _mm_fmadd_ps(coord, extentf, addressOffset),
                                 coord);
#else
                coord = SELECT_F(
                    arrayMask,
                    _mm_add_ps(_mm_mul_ps(coord, extentf), addressOffset),
                    coord);
#endif
                break;
        }
    }

    // At this point, we're dealing with non-normalized coordinates.

    __m128i out =
        SELECT_I(_mm_castps_si128(locMask), adFn(vifloorf(coord), extent),
                 _mm_cvtps_epi32(coord));
    outX = _mm_cvtsi128_si32(out);

    // 1D and 2D arrays require special care for the index coordinate:

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            outY = static_cast<int>(
                calculate_array_index(y, (float)imageInfo->arraySize - 1.0f));
            outZ = 0; /* don't care! */
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            outY = EXTRACT_I(out, 1);
            outZ = static_cast<int>(
                calculate_array_index(z, (float)imageInfo->arraySize - 1.0f));
            break;
        default:
            // legacy path:
            outY = EXTRACT_I(out, 1);
            outZ = EXTRACT_I(out, 2);
    }

    out = _mm_set_epi32(0, outZ, outY, outX);
    __m128i refEqual = _mm_cmpeq_epi32(ref, out);
    return TEST_ANY_ZERO(refEqual);
#else
    float refX = floorf(x), refY = floorf(y), refZ = floorf(z);

    // Handle sampler-directed coordinate normalization + clamping.  Note that
    // the array coordinate for image array types is expected to be
    // unnormalized, and is clamped to 0..arraySize-1.
    if (imageSampler->normalized_coords)
    {
        switch (imageSampler->addressing_mode)
        {
            case CL_ADDRESS_REPEAT:
                x = RepeatNormalizedAddressFn(x, width);
                if (height != 0)
                {
                    if (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY)
                        y = RepeatNormalizedAddressFn(y, height);
                }
                if (depth != 0)
                {
                    if (imageInfo->type != CL_MEM_OBJECT_IMAGE2D_ARRAY)
                        z = RepeatNormalizedAddressFn(z, depth);
                }

                if (xAddressOffset != 0.0)
                {
                    // Add in the offset
                    x += xAddressOffset;
                    // Handle wrapping
                    if (x > width) x -= (float)width;
                    if (x < 0) x += (float)width;
                }
                if ((yAddressOffset != 0.0)
                    && (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY))
                {
                    // Add in the offset
                    y += yAddressOffset;
                    // Handle wrapping
                    if (y > height) y -= (float)height;
                    if (y < 0) y += (float)height;
                }
                if ((zAddressOffset != 0.0)
                    && (imageInfo->type != CL_MEM_OBJECT_IMAGE2D_ARRAY))
                {
                    // Add in the offset
                    z += zAddressOffset;
                    // Handle wrapping
                    if (z > depth) z -= (float)depth;
                    if (z < 0) z += (float)depth;
                }
                break;

            case CL_ADDRESS_MIRRORED_REPEAT:
                x = MirroredRepeatNormalizedAddressFn(x, width);
                if (height != 0)
                {
                    if (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY)
                        y = MirroredRepeatNormalizedAddressFn(y, height);
                }
                if (depth != 0)
                {
                    if (imageInfo->type != CL_MEM_OBJECT_IMAGE2D_ARRAY)
                        z = MirroredRepeatNormalizedAddressFn(z, depth);
                }

                if (xAddressOffset != 0.0)
                {
                    float temp = x + xAddressOffset;
                    if (temp > (float)width)
                        temp = (float)width - (temp - (float)width);
                    x = fabsf(temp);
                }
                if ((yAddressOffset != 0.0)
                    && (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY))
                {
                    float temp = y + yAddressOffset;
                    if (temp > (float)height)
                        temp = (float)height - (temp - (float)height);
                    y = fabsf(temp);
                }
                if ((zAddressOffset != 0.0)
                    && (imageInfo->type != CL_MEM_OBJECT_IMAGE2D_ARRAY))
                {
                    float temp = z + zAddressOffset;
                    if (temp > (float)depth)
                        temp = (float)depth - (temp - (float)depth);
                    z = fabsf(temp);
                }
                break;

            default:
                // Also, remultiply to the original coords. This simulates any
                // truncation in the pass to OpenCL
                x *= (float)width;
                x += xAddressOffset;

                if (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY)
                {
                    y *= (float)height;
                    y += yAddressOffset;
                }

                if (imageInfo->type != CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    z *= (float)depth;
                    z += zAddressOffset;
                }
                break;
        }
    }

    // At this point, we're dealing with non-normalized coordinates.

    outX = adFn(static_cast<int>(floorf(x)), width);

    // 1D and 2D arrays require special care for the index coordinate:

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            outY = static_cast<int>(
                calculate_array_index(y, (float)imageInfo->arraySize - 1.0f));
            outZ = 0; /* don't care! */
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            outY = adFn(static_cast<int>(floorf(y)), height);
            outZ = static_cast<int>(
                calculate_array_index(z, (float)imageInfo->arraySize - 1.0f));
            break;
        default:
            // legacy path:
            if (height != 0) outY = adFn(static_cast<int>(floorf(y)), height);
            if (depth != 0) outZ = adFn(static_cast<int>(floorf(z)), depth);
    }

    return !((int)refX == outX && (int)refY == outY && (int)refZ == outZ);
#endif
}

#ifdef __SSE2__
static inline __m128 pixelMax(__m128 a, __m128 b)
{
    // n.b. Operands must be reversed, because if either one is NaN, the second
    // operand is returned
    return _mm_max_ps(vfabsf(b), vfabsf(a));
}

static inline int IsFloatSubnormal(__m128 x)
{
    // No fpclass until AVX-512 (what took them so long?!!)
    union {
        __m128 f;
        __m128i u;
    } u;
    u.f = vfabsf(x);
    __m128i negOne = _mm_setmone_si128();
    return TEST_NONZERO(
        _mm_cmplt_epi32(_mm_add_epi32(u.u, negOne),
                        _mm_srli_epi32(negOne, 9) /* 0x007fffff */));
}

// If containsDenorms is NULL, flush denorms to zero
// if containsDenorms is not NULL, record whether there are any denorms
static inline __m128 check_for_denorms(__m128 a, int *containsDenorms)
{
    if (NULL != containsDenorms && IsFloatSubnormal(a)) *containsDenorms = 1;
    return a;
}
#else
static inline void pixelMax(const float a[4], const float b[4], float *results);
static inline void pixelMax(const float a[4], const float b[4], float *results)
{
    for (int i = 0; i < 4; i++) results[i] = errMax(fabsf(a[i]), fabsf(b[i]));
}

// If containsDenorms is NULL, flush denorms to zero
// if containsDenorms is not NULL, record whether there are any denorms
static inline void check_for_denorms(float a[4], int *containsDenorms);
static inline void check_for_denorms(float a[4], int *containsDenorms)
{
    if (NULL == containsDenorms)
    {
        for (int i = 0; i < 4; i++)
        {
            if (IsFloatSubnormal(a[i])) a[i] = copysignf(0.0f, a[i]);
        }
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            if (IsFloatSubnormal(a[i]))
            {
                *containsDenorms = 1;
                break;
            }
        }
    }
}
#endif

inline float calculate_array_index(float coord, float extent)
{
    // from Section 8.4 of the 1.2 Spec 'Selecting an Image from an Image Array'
    //
    // given coordinate 'w' that represents an index:
    // layer_index = clamp( rint(w), 0, image_array_size - 1)

    float ret = rintf(coord);
    ret = ret > extent ? extent : ret;
    ret = ret < 0.0f ? 0.0f : ret;

    return ret;
}

#ifdef __SSE2__
#define EXTRACT_F(v, i)                                                        \
    _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, i)))
#endif

/*
 * Utility function to unnormalized a coordinate given a particular sampler.
 *
 * name     - the name of the coordinate, used for verbose debugging only
 * coord    - the coordinate requiring unnormalization
 * offset   - an addressing offset to be added to the coordinate
 * extent   - the max value for this coordinate (e.g. width for x)
 */
#ifdef __SSE2__

#ifdef __AVX__
#define TEST_NONZERO_F(v) !_mm_testz_ps(v, v)
#else
#define TEST_NONZERO_F(v) TEST_NONZERO(_mm_castps_si128(v))
#endif

static __m128 unnormalize_coordinate(const char *name, __m128 coord,
                                     __m128 offset, __m128 extent,
                                     cl_addressing_mode addressing_mode,
                                     int verbose)
{
    __m128 zero = _mm_setzero_ps();
    __m128 ret = zero;
    __m128 offsetMask = _mm_cmpneq_ps(offset, zero);
    __m128 minMask, maxMask, temp;

    switch (addressing_mode)
    {
        case CL_ADDRESS_REPEAT:
            ret = RepeatNormalizedAddressFn(coord, _mm_cvtps_epi32(extent));

            if (verbose)
            {
                log_info("\tRepeat filter denormalizes %s (%f, %f, %f) to %f, "
                         "%f, %f\n",
                         name, EXTRACT_F(coord, 0), EXTRACT_F(coord, 1),
                         EXTRACT_F(coord, 2), EXTRACT_F(ret, 0),
                         EXTRACT_F(ret, 1), EXTRACT_F(ret, 2));
            }

            // Add in the offset, and handle wrapping.
            ret = _mm_add_ps(ret, offset);
            maxMask = _mm_and_ps(offsetMask, _mm_cmpgt_ps(ret, extent));
            minMask = _mm_and_ps(offsetMask, _mm_cmplt_ps(ret, zero));
            ret = SELECT_F(minMask, _mm_add_ps(ret, extent),
                           SELECT_F(maxMask, _mm_sub_ps(ret, extent), ret));

            if (verbose && TEST_NONZERO_F(offsetMask))
            {
                log_info(
                    "\tAddress offset of %f, %f, %f added to get %f, %f, %f\n",
                    EXTRACT_F(offset, 0), EXTRACT_F(offset, 1),
                    EXTRACT_F(offset, 2), EXTRACT_F(ret, 0), EXTRACT_F(ret, 1),
                    EXTRACT_F(ret, 2));
            }
            break;

        case CL_ADDRESS_MIRRORED_REPEAT:
            ret = MirroredRepeatNormalizedAddressFn(coord,
                                                    _mm_cvtps_epi32(extent));

            if (verbose)
            {
                log_info("\tMirrored repeat filter denormalizes %s (%f, %f, "
                         "%f) to %f, %f, %f\n",
                         name, EXTRACT_F(coord, 0), EXTRACT_F(coord, 1),
                         EXTRACT_F(coord, 2), EXTRACT_F(ret, 0),
                         EXTRACT_F(ret, 1), EXTRACT_F(ret, 2));
            }

            temp = _mm_add_ps(ret, offset);
            maxMask = _mm_cmpgt_ps(temp, extent);
            ret = SELECT_F(
                offsetMask,
                vfabsf(SELECT_F(maxMask,
                                _mm_sub_ps(extent, _mm_sub_ps(temp, extent)),
                                temp)),
                ret);

            if (verbose && TEST_NONZERO_F(offsetMask))
            {
                log_info(
                    "\tAddress offset of %f, %f, %f added to get %f, %f, %f\n",
                    EXTRACT_F(offset, 0), EXTRACT_F(offset, 1),
                    EXTRACT_F(offset, 2), EXTRACT_F(ret, 0), EXTRACT_F(ret, 1),
                    EXTRACT_F(ret, 2));
            }
            break;

        default:

            if (verbose)
            {
                ret = _mm_mul_ps(coord, extent);
                log_info("\tFilter denormalizes %s to %f, %f, %f (<%f, %f, %f> "
                         "* <%f, %f, %f>)\n",
                         name, EXTRACT_F(ret, 0), EXTRACT_F(ret, 1),
                         EXTRACT_F(ret, 2), EXTRACT_F(coord, 0),
                         EXTRACT_F(coord, 1), EXTRACT_F(coord, 2),
                         EXTRACT_F(extent, 0), EXTRACT_F(extent, 1),
                         EXTRACT_F(extent, 2));
                if (TEST_NONZERO_F(offsetMask))
                {
                    ret = _mm_add_ps(ret, offset);
                    log_info("\tAddress offset of %f, %f, %f added to get %f, "
                             "%f, %f\n",
                             EXTRACT_F(offset, 0), EXTRACT_F(offset, 1),
                             EXTRACT_F(offset, 2), EXTRACT_F(ret, 0),
                             EXTRACT_F(ret, 1), EXTRACT_F(ret, 2));
                }
            }
            else
            {
#ifdef __FMA4__
                ret = _mm_macc_ps(coord, extent, offset);
#elif defined(__FMA__)
                ret = _mm_fmadd_ps(coord, extent, offset);
#else
                ret = _mm_add_ps(_mm_mul_ps(coord, extent), offset);
#endif
            }
    }

    return ret;
}
#else
static float unnormalize_coordinate(const char *name, float coord, float offset,
                                    float extent,
                                    cl_addressing_mode addressing_mode,
                                    int verbose)
{
    float ret = 0.0f;

    switch (addressing_mode)
    {
        case CL_ADDRESS_REPEAT:
            ret = RepeatNormalizedAddressFn(coord, static_cast<size_t>(extent));

            if (verbose)
            {
                log_info("\tRepeat filter denormalizes %s (%f) to %f\n", name,
                         coord, ret);
            }

            if (offset != 0.0)
            {
                // Add in the offset, and handle wrapping.
                ret += offset;
                if (ret > extent) ret -= extent;
                if (ret < 0.0) ret += extent;
            }

            if (verbose && offset != 0.0f)
            {
                log_info("\tAddress offset of %f added to get %f\n", offset,
                         ret);
            }
            break;

        case CL_ADDRESS_MIRRORED_REPEAT:
            ret = MirroredRepeatNormalizedAddressFn(
                coord, static_cast<size_t>(extent));

            if (verbose)
            {
                log_info(
                    "\tMirrored repeat filter denormalizes %s (%f) to %f\n",
                    name, coord, ret);
            }

            if (offset != 0.0)
            {
                float temp = ret + offset;
                if (temp > extent) temp = extent - (temp - extent);
                ret = fabsf(temp);
            }

            if (verbose && offset != 0.0f)
            {
                log_info("\tAddress offset of %f added to get %f\n", offset,
                         ret);
            }
            break;

        default:

            ret = coord * extent;

            if (verbose)
            {
                log_info("\tFilter denormalizes %s to %f (%f * %f)\n", name,
                         ret, coord, extent);
            }

            ret += offset;

            if (verbose && offset != 0.0f)
            {
                log_info("\tAddress offset of %f added to get %f\n", offset,
                         ret);
            }
    }

    return ret;
}
#endif

FloatPixel
sample_image_pixel_float(void *imageData, image_descriptor *imageInfo, float x,
                         float y, float z, image_sampler_data *imageSampler,
                         float *outData, int verbose, int *containsDenorms)
{
    return sample_image_pixel_float_offset(imageData, imageInfo, x, y, z, 0.0f,
                                           0.0f, 0.0f, imageSampler, outData,
                                           verbose, containsDenorms);
}

// returns max pixel value of the pixels touched
FloatPixel sample_image_pixel_float(void *imageData,
                                    image_descriptor *imageInfo, float x,
                                    float y, float z,
                                    image_sampler_data *imageSampler,
                                    float *outData, int verbose,
                                    int *containsDenorms, int lod)
{
    return sample_image_pixel_float_offset(imageData, imageInfo, x, y, z, 0.0f,
                                           0.0f, 0.0f, imageSampler, outData,
                                           verbose, containsDenorms, lod);
}
#ifdef __SSE2__
FloatPixel sample_image_pixel_float_offset(
    void *imageData, image_descriptor *imageInfo, float x, float y, float z,
    float xAddressOffset, float yAddressOffset, float zAddressOffset,
    image_sampler_data *imageSampler, float *outData, int verbose,
    int *containsDenorms, int lod)
{
    AddressFn adFn = sAddressingTable[imageSampler];
    FloatPixel returnVal;
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth;
    size_t slice_pitch_lod = 0, row_pitch_lod = 0;

    if (imageInfo->num_mip_levels > 1)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                depth_lod =
                    (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                height_lod =
                    (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            default:
                width_lod =
                    (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        }
        row_pitch_lod = width_lod * get_pixel_size(imageInfo->format);
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            slice_pitch_lod = row_pitch_lod;
        else if (imageInfo->type == CL_MEM_OBJECT_IMAGE3D
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            slice_pitch_lod = row_pitch_lod * height_lod;
    }
    else
    {
        slice_pitch_lod = imageInfo->slicePitch;
        row_pitch_lod = imageInfo->rowPitch;
    }

    if (containsDenorms) *containsDenorms = 0;

    __m128 coord, addressOffset, extentf;
    __m128i extent, addressMask;
    switch (imageInfo->type)
    {

            // The image array types require special care:

        case CL_MEM_OBJECT_IMAGE1D:
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            coord = _mm_set_ss(x);
            addressOffset = _mm_set_ss(xAddressOffset);
            extent = _mm_set_epi32(0, 1, 1, width_lod);
            addressMask =
                _mm_bsrli_si128(_mm_setmone_si128(), 12); // = 0, 0, 0, -1
            break;

        case CL_MEM_OBJECT_IMAGE2D:
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            coord = _mm_set_ps(0.f, 0.f, y, x);
            addressOffset =
                _mm_set_ps(0.f, 0.f, yAddressOffset, xAddressOffset);
            extent = _mm_set_epi32(0, 1, height_lod, width_lod);
            addressMask =
                _mm_bsrli_si128(_mm_setmone_si128(), 8); // = 0, 0, -1, -1
            break;

            // Everybody else:

        default:
            coord = _mm_set_ps(0.f, z, y, x);
            addressOffset =
                _mm_set_ps(0.f, zAddressOffset, yAddressOffset, xAddressOffset);
            extent = _mm_set_epi32(0, depth_lod, height_lod, width_lod);
            addressMask =
                _mm_bsrli_si128(_mm_setmone_si128(), 4); // = 0, -1, -1, -1
    }
    extentf = _mm_cvtepi32_ps(extent);

    if (imageSampler->normalized_coords)
    {
        // We need to unnormalize our coordinates differently depending on
        // the image type, but 'x' is always processed the same way.

        const char *name = NULL;
        switch (imageInfo->type)
        {

            case CL_MEM_OBJECT_IMAGE1D:
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            case CL_MEM_OBJECT_IMAGE1D_ARRAY: name = "x"; break;

            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY: name = "x, y"; break;

            default: name = "x, y, z";
        }

        coord = unnormalize_coordinate(name, coord, addressOffset, extentf,
                                       imageSampler->addressing_mode, verbose);
    }
    else if (verbose)
    {

        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_info("Starting coordinate: %f, array index %f\n", x, y);
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_info("Starting coordinate: %f, %f, array index %f\n", x, y,
                         z);
                break;
            case CL_MEM_OBJECT_IMAGE1D:
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                log_info("Starting coordinate: %f\b", x);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_info("Starting coordinate: %f, %f\n", x, y);
                break;
            case CL_MEM_OBJECT_IMAGE3D:
            default: log_info("Starting coordinate: %f, %f, %f\n", x, y, z);
        }
    }

    // At this point, we have unnormalized coordinates.

    if (imageSampler->filter_mode == CL_FILTER_NEAREST)
    {
        __m128i icoord;
        int arrayIndex;

        // We apply the addressing function to the now-unnormalized
        // coordinates.  Note that the array cases again require special
        // care, per section 8.4 in the OpenCL 1.2 Specification.

        icoord = _mm_and_si128(addressMask, adFn(vifloorf(coord), extent));

        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                arrayIndex = static_cast<int>(calculate_array_index(
                    y, (float)(imageInfo->arraySize - 1)));
                if (verbose)
                    log_info("\tArray index %f evaluates to %d\n", y,
                             arrayIndex);
#ifdef __SSE4_1__
                icoord = _mm_insert_epi32(icoord, arrayIndex, 1);
#else
                icoord = _mm_insert_epi16(
                    _mm_insert_epi16(icoord, (short)arrayIndex, 2),
                    (short)(arrayIndex >> 16), 3);
#endif
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                arrayIndex = static_cast<int>(calculate_array_index(
                    z, (float)(imageInfo->arraySize - 1)));
                if (verbose)
                    log_info("\tArray index %f evaluates to %d\n", z,
                             arrayIndex);
#ifdef __SSE4_1__
                icoord = _mm_insert_epi32(icoord, arrayIndex, 2);
#else
                icoord = _mm_insert_epi16(
                    _mm_insert_epi16(icoord, (short)arrayIndex, 4),
                    (short)(arrayIndex >> 16), 5);
#endif
                break;
            default: break;
        }

        if (verbose)
        {
            if (depth_lod)
                log_info(
                    "\tReference integer coords calculated: { %d, %d, %d }\n",
                    EXTRACT_I(icoord, 0), EXTRACT_I(icoord, 1),
                    EXTRACT_I(icoord, 2));
            else
                log_info("\tReference integer coords calculated: { %d, %d }\n",
                         EXTRACT_I(icoord, 0), EXTRACT_I(icoord, 1));
        }

        // SSE has an FTZ mode that will be useful here
        unsigned int mxcsr = 0;
        if (NULL == containsDenorms)
        {
            mxcsr = _mm_getcsr();
            _mm_setcsr(mxcsr & ~_MM_FLUSH_ZERO_MASK | _MM_FLUSH_ZERO_ON);
        }
        __m128 outPixel = check_for_denorms(
            read_image_pixel_float(imageData, imageInfo, icoord, lod),
            containsDenorms);
        if (NULL == containsDenorms) _mm_setcsr(mxcsr);
        _mm_storeu_ps(outData, outPixel);
        _mm_storeu_ps(returnVal.p, vfabsf(outPixel));
        return returnVal;
    }
    else
    {
        // Linear filtering cases.

        // Image arrays can use 2D filtering, but require us to walk into the
        // image a certain number of slices before reading.

        if (depth_lod == 0 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY
            || imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        {
            float array_index = 0;

            size_t layer_offset = 0;

            if (imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            {
                array_index =
                    calculate_array_index(z, (float)(imageInfo->arraySize - 1));
                layer_offset = slice_pitch_lod * (size_t)array_index;
            }
            else if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            {
                array_index =
                    calculate_array_index(y, (float)(imageInfo->arraySize - 1));
                layer_offset = slice_pitch_lod * (size_t)array_index;
            }

            __m128i icoord =
                vifloorf(_mm_sub_ps(coord, _mm_set_ps(0.f, 0.f, 0.5f, 0.5f)));
            __m128i coord00 = _mm_and_si128(addressMask, adFn(icoord, extent));
            __m128i coord11 = _mm_and_si128(
                addressMask,
                adFn(_mm_sub_epi32(icoord, _mm_setmone_si128()), extent));

            if (verbose)
            {
                log_info("\tActual integer coords used (i = floor(x-.5)): i0:{ "
                         "%d, %d } and i1:{ %d, %d }\n",
                         EXTRACT_I(coord00, 0), EXTRACT_I(coord00, 1),
                         EXTRACT_I(coord11, 0), EXTRACT_I(coord11, 1));
                log_info("\tArray coordinate is %f\n", array_index);
            }

            // Walk to beginning of the 'correct' slice, if needed.
            char *imgPtr = ((char *)imageData) + layer_offset;

            // flush subnormal results to zero if necessary
            // SSE has an FTZ mode that will be useful here
            unsigned int mxcsr;
            if (NULL == containsDenorms)
            {
                mxcsr = _mm_getcsr();
                _mm_setcsr(mxcsr & ~_MM_FLUSH_ZERO_MASK | _MM_FLUSH_ZERO_ON);
            }

            // Make coordinate vectors for pixels 01 and 10
#ifdef __AVX2__
            __m128i coord01 = _mm_blend_epi32(coord00, coord11, 0x01);
            __m128i coord10 = _mm_blend_epi32(coord00, coord11, 0x02);
#elif defined(__SSE4_1__)
            __m128i coord01 = _mm_blend_epi16(coord00, coord11, 0x03);
            __m128i coord10 = _mm_blend_epi16(coord00, coord11, 0x0C);
#else
            __m128i coordMask =
                _mm_bsrli_si128(_mm_setmone_si128(), 8); // = 0, 0, -1, -1
            __m128i interleavedCoord =
                _mm_unpacklo_epi32(coord00, coord11); // y2 y1 x2 x1
            __m128i coord01 = _mm_and_si128(
                coordMask,
                _mm_shuffle_epi32(interleavedCoord, _MM_SHUFFLE(3, 0, 2, 1)));
            __m128i coord10 = _mm_and_si128(
                coordMask,
                _mm_shuffle_epi32(interleavedCoord, _MM_SHUFFLE(2, 1, 3, 0)));
#endif

            __m128 upLeft, upRight, lowLeft, lowRight;
            __m128 maxUp, maxLow;
            upLeft = check_for_denorms(
                read_image_pixel_float(imgPtr, imageInfo, coord00, lod),
                containsDenorms);
            upRight = check_for_denorms(
                read_image_pixel_float(imgPtr, imageInfo, coord01, lod),
                containsDenorms);
            maxUp = pixelMax(upLeft, upRight);
            lowLeft = check_for_denorms(
                read_image_pixel_float(imgPtr, imageInfo, coord10, lod),
                containsDenorms);
            lowRight = check_for_denorms(
                read_image_pixel_float(imgPtr, imageInfo, coord11, lod),
                containsDenorms);
            maxLow = pixelMax(lowLeft, lowRight);
            _mm_storeu_ps(returnVal.p, pixelMax(maxUp, maxLow));

            if (verbose)
            {
                if (NULL == containsDenorms)
                    log_info("\tSampled pixels (rgba order, denorms flushed to "
                             "zero):\n");
                else
                    log_info("\tSampled pixels (rgba order):\n");
                log_info("\t\tp00: %f, %f, %f, %f\n", EXTRACT_F(upLeft, 0),
                         EXTRACT_F(upLeft, 1), EXTRACT_F(upLeft, 2),
                         EXTRACT_F(upLeft, 3));
                log_info("\t\tp01: %f, %f, %f, %f\n", EXTRACT_F(upRight, 0),
                         EXTRACT_F(upRight, 1), EXTRACT_F(upRight, 2),
                         EXTRACT_F(upRight, 3));
                log_info("\t\tp10: %f, %f, %f, %f\n", EXTRACT_F(lowLeft, 0),
                         EXTRACT_F(lowLeft, 1), EXTRACT_F(lowLeft, 2),
                         EXTRACT_F(lowLeft, 3));
                log_info("\t\tp11: %f, %f, %f, %f\n", EXTRACT_F(lowRight, 0),
                         EXTRACT_F(lowRight, 1), EXTRACT_F(lowRight, 2),
                         EXTRACT_F(lowRight, 3));
            }

            __m128 fracCoord = frac(_mm_sub_ps(coord, _mm_set1_ps(0.5f)));
            if (verbose)
                log_info("\tfrac( x - 0.5f ) = %f,  frac( y - 0.5f ) = %f\n",
                         EXTRACT_F(fracCoord, 0), EXTRACT_F(fracCoord, 1));

#ifdef __AVX__
            __m256d alphaBeta = _mm256_cvtps_pd(fracCoord); // x x b a
            alphaBeta = _mm256_insertf128_pd(
                alphaBeta,
                _mm_sub_pd(_mm_set1_pd(1.0), _mm256_castpd256_pd128(alphaBeta)),
                1); // 1-b 1-a b a
            //   1-b       1-a       b         a
            //   1-a        b        a        1-b (2 1 0 3)
            //(1-a)(1-b) (1-a)b     a*b      a(1-b)
            //    00        10      11         01
#ifdef __AVX2__
            __m256d weights = _mm256_mul_pd(
                alphaBeta,
                _mm256_permute4x64_pd(alphaBeta, _MM_SHUFFLE(2, 1, 0, 3)));
            __m256d weight01 =
                _mm256_broadcastsd_pd(_mm256_castpd256_pd128(weights));
            __m256d weight11 =
                _mm256_permute4x64_pd(weights, _MM_SHUFFLE(1, 1, 1, 1));
            __m256d weight10 =
                _mm256_permute4x64_pd(weights, _MM_SHUFFLE(2, 2, 2, 2));
            __m256d weight00 =
                _mm256_permute4x64_pd(weights, _MM_SHUFFLE(3, 3, 3, 3));
#else
            // This is now more complicated...
            // Swap the two halves of the vector (1 0 3 2):
            //  0x01 = 0b00000001
            //             ~~  ~~
            //       +-----+    |
            //   +---*----------+
            //   |   +
            //   v   v
            //  +-+ +-+
            //  3 2 1 0
            // ...then shuffle the elements as follows:
            //  0x05 = 0b00000101
            //    +----------+|||
            //    | +---------+||
            //    | |     +----+|
            //    | |     | +---+
            //    v v     v v
            //  3 2 1 0 1 0 3 2
            __m256d weights = _mm256_mul_pd(
                alphaBeta,
                _mm256_shuffle_pd(
                    alphaBeta,
                    _mm256_permute2f128_pd(alphaBeta, alphaBeta, 0x01), 0x05));
            // Duplicate the even and odd elements...
            __m256d weight1001 = _mm256_movedup_pd(weights);
            __m256d weight0011 = _mm256_permute_pd(weights, 0x0F);
            // ...then duplicate the low and high halves of the results
            __m256d weight01 =
                _mm256_permute2f128_pd(_mm256_undefined_pd(), weight1001, 0x22);
            __m256d weight11 =
                _mm256_permute2f128_pd(_mm256_undefined_pd(), weight0011, 0x22);
            __m256d weight10 =
                _mm256_permute2f128_pd(_mm256_undefined_pd(), weight1001, 0x33);
            __m256d weight00 =
                _mm256_permute2f128_pd(_mm256_undefined_pd(), weight0011, 0x33);
#endif
#ifdef __FMA4__
            // Doing it this way instead of using a chain of FMAs avoids stalls
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_macc_pd(
                        _mm256_cvtps_pd(upLeft), weight00,
                        _mm256_mul_pd(_mm256_cvtps_pd(upRight), weight01)),
                    _mm256_macc_pd(
                        _mm256_cvtps_pd(lowLeft), weight10,
                        _mm256_mul_pd(_mm256_cvtps_pd(lowRight), weight11)))));
#elif defined(__FMA__)
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_fmadd_pd(
                        _mm256_cvtps_pd(upLeft), weight00,
                        _mm256_mul_pd(_mm256_cvtps_pd(upRight), weight01)),
                    _mm256_fmadd_pd(
                        _mm256_cvtps_pd(lowLeft), weight10,
                        _mm256_mul_pd(_mm256_cvtps_pd(lowRight), weight11)))));
#else
            // No VDPPD for 256-bit vectors... :/
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_mul_pd(_mm256_cvtps_pd(upLeft), weight00),
                        _mm256_mul_pd(_mm256_cvtps_pd(upRight), weight01)),
                    _mm256_add_pd(
                        _mm256_mul_pd(_mm256_cvtps_pd(lowLeft), weight10),
                        _mm256_mul_pd(_mm256_cvtps_pd(lowRight), weight11)))));
#endif
#else
            __m128d alphaBeta = _mm_cvtps_pd(fracCoord); // b a
            __m128d invAlphaBeta =
                _mm_sub_pd(_mm_set1_pd(1.0), alphaBeta); // 1-b 1-a
            __m128d weights[2];

            // <1-a b> * <1-b 1-a> = <(1-a)(1-b) (1-a)b> <00 10>
            weights[0] = _mm_mul_pd(
                _mm_shuffle_pd(invAlphaBeta, alphaBeta, _MM_SHUFFLE2(1, 0)),
                invAlphaBeta);
            // <b a> * <a 1-b> = <a*b a(1-b)> <11 01>
            weights[1] = _mm_mul_pd(
                alphaBeta,
                _mm_shuffle_pd(invAlphaBeta, alphaBeta, _MM_SHUFFLE2(0, 1)));

            __m128d upLeftL = _mm_cvtps_pd(upLeft);
            __m128d upLeftH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upLeft));
            __m128d upRightL = _mm_cvtps_pd(upRight);
            __m128d upRightH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upRight));
            __m128d lowLeftL = _mm_cvtps_pd(lowLeft);
            __m128d lowLeftH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowLeft));
            __m128d lowRightL = _mm_cvtps_pd(lowRight);
            __m128d lowRightH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowRight));
#ifdef __SSE4_1__
            // In the immediate bytes, the high nibble determines which
            // multiplies take place--in this case, both of them--and the low
            // nibble determines which lines receive the sum--bit 0 for the low
            // lane, bit 1 for the high.
            __m128d rg = _mm_or_pd(
                _mm_dp_pd(weights[0], _mm_unpacklo_pd(lowLeftL, upLeftL), 0x31),
                _mm_dp_pd(weights[0], _mm_unpackhi_pd(lowLeftL, upLeftL),
                          0x32));
            __m128d ba = _mm_or_pd(
                _mm_dp_pd(weights[0], _mm_unpacklo_pd(lowLeftH, upLeftH), 0x31),
                _mm_dp_pd(weights[0], _mm_unpackhi_pd(lowLeftH, upLeftH),
                          0x32));
            rg = _mm_add_pd(
                rg,
                _mm_or_pd(_mm_dp_pd(weights[1],
                                    _mm_unpacklo_pd(upRightL, lowRightL), 0x31),
                          _mm_dp_pd(weights[1],
                                    _mm_unpackhi_pd(upRightL, lowRightL),
                                    0x32)));
            ba = _mm_add_pd(
                ba,
                _mm_or_pd(_mm_dp_pd(weights[1],
                                    _mm_unpacklo_pd(upRightH, lowRightH), 0x31),
                          _mm_dp_pd(weights[1],
                                    _mm_unpackhi_pd(upRightH, lowRightH),
                                    0x32)));
#else
#ifdef __SSE3__
            __m128d weight10 = _mm_movedup_pd(weights[0]);
            __m128d weight01 = _mm_movedup_pd(weights[1]);
#else
            __m128d weight10 = _mm_unpacklo_pd(weights[0], weights[0]);
            __m128d weight01 = _mm_unpacklo_pd(weights[1], weights[1]);
#endif
            __m128d rg = _mm_add_pd(_mm_mul_pd(weight01, upRightL),
                                    _mm_mul_pd(weight10, lowLeftL));
            __m128d ba = _mm_add_pd(_mm_mul_pd(weight01, upRightH),
                                    _mm_mul_pd(weight10, lowLeftH));
            __m128d weight00 = _mm_unpackhi_pd(weights[0], weights[0]);
            __m128d weight11 = _mm_unpackhi_pd(weights[1], weights[1]);
            rg = _mm_add_pd(rg,
                            _mm_add_pd(_mm_mul_pd(weight00, upLeftL),
                                       _mm_mul_pd(weight11, lowRightL)));
            ba = _mm_add_pd(ba,
                            _mm_add_pd(_mm_mul_pd(weight00, upLeftH),
                                       _mm_mul_pd(weight11, lowRightH)));
#endif
            _mm_storeu_ps(outData,
                          _mm_movelh_ps(_mm_cvtpd_ps(rg), _mm_cvtpd_ps(ba)));
#endif
            if (NULL == containsDenorms) _mm_setcsr(mxcsr);
        }
        else
        {
            // 3D linear filtering
            __m128i icoord =
                vifloorf(_mm_sub_ps(coord, _mm_set_ps(0.f, 0.5f, 0.5f, 0.5f)));
            __m128i coord000 = _mm_and_si128(addressMask, adFn(icoord, extent));
            __m128i coord111 = _mm_and_si128(
                addressMask,
                adFn(_mm_sub_epi32(icoord, _mm_setmone_si128()), extent));

            if (verbose)
                log_info("\tActual integer coords used (i = floor(x-.5)): "
                         "i0:{%d, %d, %d} and i1:{%d, %d, %d}\n",
                         EXTRACT_I(coord000, 0), EXTRACT_I(coord000, 1),
                         EXTRACT_I(coord000, 2), EXTRACT_I(coord111, 0),
                         EXTRACT_I(coord111, 1), EXTRACT_I(coord111, 2));

            // flush subnormal results to zero if necessary
            // SSE has an FTZ mode that will be useful here
            unsigned int mxcsr;
            if (NULL == containsDenorms)
            {
                mxcsr = _mm_getcsr();
                _mm_setcsr(mxcsr & ~_MM_FLUSH_ZERO_MASK | _MM_FLUSH_ZERO_ON);
            }

#ifdef __AVX2__
            __m128i coord001 = _mm_blend_epi32(coord000, coord111, 0x01);
            __m128i coord010 = _mm_blend_epi32(coord000, coord111, 0x02);
            __m128i coord011 = _mm_blend_epi32(coord000, coord111, 0x03);
            __m128i coord100 = _mm_blend_epi32(coord000, coord111, 0x04);
            __m128i coord101 = _mm_blend_epi32(coord000, coord111, 0x05);
            __m128i coord110 = _mm_blend_epi32(coord000, coord111, 0x06);
#elif defined(__SSE4_1__)
            __m128i coord001 = _mm_blend_epi16(coord000, coord111, 0x03);
            __m128i coord010 = _mm_blend_epi16(coord000, coord111, 0x0C);
            __m128i coord011 = _mm_blend_epi16(coord000, coord111, 0x0F);
            __m128i coord100 = _mm_blend_epi16(coord000, coord111, 0x30);
            __m128i coord101 = _mm_blend_epi16(coord000, coord111, 0x33);
            __m128i coord110 = _mm_blend_epi16(coord000, coord111, 0x3C);
#else
            // XXX This is horrible without PBLEND...
            __m128i negOne = _mm_setmone_si128();
            __m128i coordMask = _mm_bsrli_si128(negOne, 8); // = 0, 0, -1, -1
            __m128i coord011 = SELECT_I(coordMask, coord000, coord111);
            coordMask = _mm_bsrli_si128(coordMask, 4); // = 0, 0, 0, -1
            __m128i coord001 = SELECT_I(coordMask, coord000, coord111);
            coordMask = _mm_slli_epi64(coordMask, 32); // = 0, 0, -1, 0
            __m128i coord010 = SELECT_I(coordMask, coord000, coord111);
            coordMask = _mm_srli_epi64(negOne, 32); // = 0, -1, 0, -1
            __m128i coord101 = SELECT_I(coordMask, coord000, coord111);
            coordMask = _mm_bslli_si128(coordMask, 8); // = 0, -1, 0, 0
            __m128i coord100 = SELECT_I(coordMask, coord000, coord111);
            coordMask = _mm_bslli_si128(_mm_bsrli_si128(negOne, 4),
                                        4); // = 0, -1, -1, 0
            __m128i coord110 = SELECT_I(coordMask, coord000, coord111);
#endif

            __m128 upLeftA, upRightA, lowLeftA, lowRightA;
            __m128 upLeftB, upRightB, lowLeftB, lowRightB;
            __m128 pixelMaxA, pixelMaxB, pixelMaxC;
            upLeftA = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord000, lod),
                containsDenorms);
            upRightA = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord001, lod),
                containsDenorms);
            pixelMaxA = pixelMax(upLeftA, upRightA);
            lowLeftA = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord010, lod),
                containsDenorms);
            lowRightA = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord011, lod),
                containsDenorms);
            pixelMaxB = pixelMax(lowLeftA, lowRightA);
            pixelMaxC = pixelMax(pixelMaxA, pixelMaxB);
            upLeftB = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord100, lod),
                containsDenorms);
            upRightB = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord101, lod),
                containsDenorms);
            pixelMaxA = pixelMax(upLeftB, upRightB);
            lowLeftB = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord110, lod),
                containsDenorms);
            lowRightB = check_for_denorms(
                read_image_pixel_float(imageData, imageInfo, coord111, lod),
                containsDenorms);
            pixelMaxB = pixelMax(lowLeftB, lowRightB);
            pixelMaxA = pixelMax(pixelMaxA, pixelMaxB);
            _mm_storeu_ps(returnVal.p, pixelMax(pixelMaxA, pixelMaxC));

            if (verbose)
            {
                if (NULL == containsDenorms)
                    log_info("\tSampled pixels (rgba order, denorms flushed to "
                             "zero):\n");
                else
                    log_info("\tSampled pixels (rgba order):\n");
                log_info("\t\tp000: %f, %f, %f, %f\n", EXTRACT_F(upLeftA, 0),
                         EXTRACT_F(upLeftA, 1), EXTRACT_F(upLeftA, 2),
                         EXTRACT_F(upLeftA, 3));
                log_info("\t\tp001: %f, %f, %f, %f\n", EXTRACT_F(upRightA, 0),
                         EXTRACT_F(upRightA, 1), EXTRACT_F(upRightA, 2),
                         EXTRACT_F(upRightA, 3));
                log_info("\t\tp010: %f, %f, %f, %f\n", EXTRACT_F(lowLeftA, 0),
                         EXTRACT_F(lowLeftA, 1), EXTRACT_F(lowLeftA, 2),
                         EXTRACT_F(lowLeftA, 3));
                log_info("\t\tp011: %f, %f, %f, %f\n\n",
                         EXTRACT_F(lowRightA, 0), EXTRACT_F(lowRightA, 1),
                         EXTRACT_F(lowRightA, 2), EXTRACT_F(lowRightA, 3));
                log_info("\t\tp100: %f, %f, %f, %f\n", EXTRACT_F(upLeftB, 0),
                         EXTRACT_F(upLeftB, 1), EXTRACT_F(upLeftB, 2),
                         EXTRACT_F(upLeftB, 3));
                log_info("\t\tp101: %f, %f, %f, %f\n", EXTRACT_F(upRightB, 0),
                         EXTRACT_F(upRightB, 1), EXTRACT_F(upRightB, 2),
                         EXTRACT_F(upRightB, 3));
                log_info("\t\tp110: %f, %f, %f, %f\n", EXTRACT_F(lowLeftB, 0),
                         EXTRACT_F(lowLeftB, 1), EXTRACT_F(lowLeftB, 2),
                         EXTRACT_F(lowLeftB, 3));
                log_info("\t\tp111: %f, %f, %f, %f\n", EXTRACT_F(lowRightB, 0),
                         EXTRACT_F(lowRightB, 1), EXTRACT_F(lowRightB, 2),
                         EXTRACT_F(lowRightB, 3));
            }

            __m128 fracCoord =
                frac(_mm_sub_ps(coord, _mm_set_ps(0.f, 0.5f, 0.5f, 0.5f)));
            if (verbose)
                log_info("\tfrac( x - 0.5f ) = %f,  frac( y - 0.5f ) = %f, "
                         "frac( z - 0.5f ) = %f\n",
                         EXTRACT_F(fracCoord, 0), EXTRACT_F(fracCoord, 1),
                         EXTRACT_F(fracCoord, 2));

#ifdef __AVX__
            __m256d alphaBetaGamma = _mm256_cvtps_pd(fracCoord); // x g b a
            __m256d invABG = _mm256_sub_pd(_mm256_set1_pd(1.0),
                                           alphaBetaGamma); // x 1-g 1-b 1-a
            __m256d alphaBeta =
                _mm256_permute2f128_pd(alphaBetaGamma, invABG, 0x20);
            __m256d weights[2][2][2];
            //     1-g          1-g        1-g         1-g
            //     1-b          1-a         b           a
            //     1-a           b          a          1-b
            // a(1-b)(1-g)  (1-a)b(1-g)  a*b(1-g)  a(1-b)(1-g)
            //     000          010        011         001
            //      g           g       g       g
            //     1-b         1-a      b       a
            //     1-a          b       a      1-b
            // (1-a)(1-b)g  (1-a)b*g  a*b*g  a(1-b)g
            //     100         110     111     101
#ifdef __AVX2__
            __m256d invGamma =
                _mm256_permute4x64_pd(invABG, _MM_SHUFFLE(2, 2, 2, 2));
            weights[1][0][0] = _mm256_mul_pd(
                alphaBeta,
                _mm256_permute4x64_pd(alphaBeta, _MM_SHUFFLE(2, 1, 0, 3)));
            weights[0][0][0] = _mm256_mul_pd(weights[1][0][0], invGamma);
            __m256d gamma =
                _mm256_permute4x64_pd(alphaBetaGamma, _MM_SHUFFLE(2, 2, 2, 2));
            weights[1][0][0] = _mm256_mul_pd(weights[1][0][0], gamma);
            weights[0][0][1] =
                _mm256_broadcastsd_pd(_mm256_castpd256_pd128(weights[0][0][0]));
            weights[0][1][1] = _mm256_permute4x64_pd(weights[0][0][0],
                                                     _MM_SHUFFLE(1, 1, 1, 1));
            weights[1][0][1] =
                _mm256_broadcastsd_pd(_mm256_castpd256_pd128(weights[1][0][0]));
            weights[1][1][1] = _mm256_permute4x64_pd(weights[1][0][0],
                                                     _MM_SHUFFLE(1, 1, 1, 1));
            weights[0][1][0] = _mm256_permute4x64_pd(weights[0][0][0],
                                                     _MM_SHUFFLE(2, 2, 2, 2));
            weights[0][0][0] = _mm256_permute4x64_pd(weights[0][0][0],
                                                     _MM_SHUFFLE(3, 3, 3, 3));
            weights[1][1][0] = _mm256_permute4x64_pd(weights[1][0][0],
                                                     _MM_SHUFFLE(2, 2, 2, 2));
            weights[1][0][0] = _mm256_permute4x64_pd(weights[1][0][0],
                                                     _MM_SHUFFLE(3, 3, 3, 3));
#else
            // Much like before, we must permute the elements of alphaBeta to
            // get them in the form we need
            __m256d invGamma = _mm256_permute2f128(
                _mm256_undefined_pd(), _mm256_movedup_pd(invABG), 0x33);
            weights[1][0][0] = _mm_mul_pd(
                alphaBeta,
                _mm256_shuffle_pd(
                    alphaBeta,
                    _mm256_permute2f128_pd(alphaBeta, alphaBeta, 0x01), 0x05));
            weights[0][0][0] = _mm_mul_pd(weights[1], invGamma);
            __m256d gamma = _mm256_permute2f128(
                _mm256_undefined_pd(), _mm256_movedup_pd(alphaBetaGamma), 0x33);
            weights[1][0][0] = _mm_mul_pd(weights[1], gamma);
            weights[0][1][0] = _mm256_movedup_pd(weights[0][0][0]);
            weights[0][0][0] = _mm256_permute_pd(weights[0][0][0], 0x0F);
            weights[1][1][0] = _mm256_movedup_pd(weights[1][0][0]);
            weights[1][0][0] = _mm256_permute_pd(weights[1][0][0], 0x0F);
            weights[0][0][1] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[0][1][0], 0x22);
            weights[0][1][1] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[0][0][0], 0x22);
            weights[1][0][1] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[1][1][0], 0x22);
            weights[1][1][1] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[1][0][0], 0x22);
            weights[0][1][0] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[0][1][0], 0x33);
            weights[0][0][0] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[0][0][0], 0x33);
            weights[1][1][0] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[1][1][0], 0x33);
            weights[1][0][0] = _mm256_permute2f128_pd(_mm256_undefined_pd(),
                                                      weights[1][0][0], 0x33);
#endif
#ifdef __FMA4__
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_macc_pd(_mm256_cvtps_pd(upLeftA),
                                       weights[0][0][0],
                                       _mm256_mul_pd(_mm256_cvtps_pd(upRightA),
                                                     weights[0][0][1])),
                        _mm256_macc_pd(_mm256_cvtps_pd(lowLeftA),
                                       weights[0][1][0],
                                       _mm256_mul_pd(_mm256_cvtps_pd(lowRightA),
                                                     weights[0][1][1]))),
                    _mm256_add_pd(
                        _mm256_macc_pd(_mm256_cvtps_pd(upLeftB),
                                       weights[1][0][0],
                                       _mm256_mul_pd(_mm256_cvtps_pd(upRightB),
                                                     weights[1][0][1])),
                        _mm256_macc_pd(_mm256_cvtps_pd(lowLeftB),
                                       weights[1][1][0],
                                       _mm256_mul_pd(_mm256_cvtps_pd(lowRightB),
                                                     weights[1][1][1]))))));
#elif defined(__FMA__)
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_fmadd_pd(_mm256_cvtps_pd(upLeftA),
                                        weights[0][0][0],
                                        _mm256_mul_pd(_mm256_cvtps_pd(upRightA),
                                                      weights[0][0][1])),
                        _mm256_fmadd_pd(
                            _mm256_cvtps_pd(lowLeftA), weights[0][1][0],
                            _mm256_mul_pd(_mm256_cvtps_pd(lowRightA),
                                          weights[0][1][1]))),
                    _mm256_add_pd(
                        _mm256_fmadd_pd(_mm256_cvtps_pd(upLeftB),
                                        weights[1][0][0],
                                        _mm256_mul_pd(_mm256_cvtps_pd(upRightB),
                                                      weights[1][0][1])),
                        _mm256_fmadd_pd(
                            _mm256_cvtps_pd(lowLeftB), weights[1][1][0],
                            _mm256_mul_pd(_mm256_cvtps_pd(lowRightB),
                                          weights[1][1][1]))))));
#else
            _mm_storeu_ps(
                outData,
                _mm256_cvtpd_ps(_mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(upLeftA),
                                                    weights[0][0][0]),
                                      _mm256_mul_pd(_mm256_cvtps_pd(upRightA),
                                                    weights[0][0][1])),
                        _mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(lowLeftA),
                                                    weights[0][1][0]),
                                      _mm256_mul_pd(_mm256_cvtps_pd(lowRightA),
                                                    weights[0][1][1]))),
                    _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(upLeftB),
                                                    weights[1][0][0]),
                                      _mm256_mul_pd(_mm256_cvtps_pd(upRightB),
                                                    weights[1][0][1])),
                        _mm256_add_pd(_mm256_mul_pd(_mm256_cvtps_pd(lowLeftB),
                                                    weights[1][1][0]),
                                      _mm256_mul_pd(_mm256_cvtps_pd(lowRightB),
                                                    weights[1][1][1]))))));
#endif
#else
            __m128d alphaBeta = _mm_cvtps_pd(fracCoord); // b a
            __m128d invAlphaBeta =
                _mm_sub_pd(_mm_set1_pd(1.0), alphaBeta); // 1-b 1-a
            __m128d gamma =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), fracCoord));
#ifdef __SSE3__
            gamma = _mm_movedup_pd(gamma);
#else
            gamma = _mm_unpacklo_pd(gamma, gamma);
#endif
            __m128d invGamma = _mm_sub_pd(_mm_set1_pd(1.0), gamma);
            __m128d weights[4];

            // <1-a b> * <1-b 1-a> = <(1-a)(1-b) (1-a)b> <00 10>
            weights[0] = _mm_mul_pd(
                _mm_shuffle_pd(invAlphaBeta, alphaBeta, _MM_SHUFFLE2(1, 0)),
                invAlphaBeta);
            // <b a> * <a 1-b> = <a*b a(1-b)> <11 01>
            weights[1] = _mm_mul_pd(
                alphaBeta,
                _mm_shuffle_pd(invAlphaBeta, alphaBeta, _MM_SHUFFLE2(0, 1)));
            weights[2] = _mm_mul_pd(weights[0], gamma);
            weights[3] = _mm_mul_pd(weights[1], gamma);
            weights[0] = _mm_mul_pd(weights[0], invGamma);
            weights[1] = _mm_mul_pd(weights[1], invGamma);

            __m128d upLeftAL = _mm_cvtps_pd(upLeftA);
            __m128d upLeftAH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upLeftA));
            __m128d upLeftBL = _mm_cvtps_pd(upLeftB);
            __m128d upLeftBH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upLeftB));
            __m128d upRightAL = _mm_cvtps_pd(upRightA);
            __m128d upRightAH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upRightA));
            __m128d upRightBL = _mm_cvtps_pd(upRightB);
            __m128d upRightBH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), upRightB));
            __m128d lowLeftAL = _mm_cvtps_pd(lowLeftA);
            __m128d lowLeftAH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowLeftA));
            __m128d lowLeftBL = _mm_cvtps_pd(lowLeftB);
            __m128d lowLeftBH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowLeftB));
            __m128d lowRightAL = _mm_cvtps_pd(lowRightA);
            __m128d lowRightAH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowRightA));
            __m128d lowRightBL = _mm_cvtps_pd(lowRightB);
            __m128d lowRightBH =
                _mm_cvtps_pd(_mm_movehl_ps(_mm_undefined_ps(), lowRightB));
#ifdef __SSE4_1__
            __m128d rg = _mm_or_pd(
                _mm_dp_pd(weights[0], _mm_unpacklo_pd(lowLeftAL, upLeftAL),
                          0x31),
                _mm_dp_pd(weights[0], _mm_unpackhi_pd(lowLeftAL, upLeftAL),
                          0x32));
            __m128d ba = _mm_or_pd(
                _mm_dp_pd(weights[0], _mm_unpacklo_pd(lowLeftAH, upLeftAH),
                          0x31),
                _mm_dp_pd(weights[0], _mm_unpackhi_pd(lowLeftAH, upLeftAH),
                          0x32));
            rg = _mm_add_pd(
                rg,
                _mm_or_pd(
                    _mm_dp_pd(weights[1],
                              _mm_unpacklo_pd(upRightAL, lowRightAL), 0x31),
                    _mm_dp_pd(weights[1],
                              _mm_unpackhi_pd(upRightAL, lowRightAL), 0x32)));
            ba = _mm_add_pd(
                ba,
                _mm_or_pd(
                    _mm_dp_pd(weights[1],
                              _mm_unpacklo_pd(upRightAH, lowRightAH), 0x31),
                    _mm_dp_pd(weights[1],
                              _mm_unpackhi_pd(upRightAH, lowRightAH), 0x32)));
            rg = _mm_add_pd(
                rg,
                _mm_or_pd(_mm_dp_pd(weights[2],
                                    _mm_unpacklo_pd(lowLeftBL, upLeftBL), 0x31),
                          _mm_dp_pd(weights[2],
                                    _mm_unpackhi_pd(lowLeftBL, upLeftBL),
                                    0x32)));
            ba = _mm_add_pd(
                ba,
                _mm_or_pd(_mm_dp_pd(weights[2],
                                    _mm_unpacklo_pd(lowLeftBH, upLeftBH), 0x31),
                          _mm_dp_pd(weights[2],
                                    _mm_unpackhi_pd(lowLeftBH, upLeftBH),
                                    0x32)));
            rg = _mm_add_pd(
                rg,
                _mm_or_pd(
                    _mm_dp_pd(weights[3],
                              _mm_unpacklo_pd(upRightBL, lowRightBL), 0x31),
                    _mm_dp_pd(weights[3],
                              _mm_unpackhi_pd(upRightBL, lowRightBL), 0x32)));
            ba = _mm_add_pd(
                ba,
                _mm_or_pd(
                    _mm_dp_pd(weights[3],
                              _mm_unpacklo_pd(upRightBH, lowRightBH), 0x31),
                    _mm_dp_pd(weights[3],
                              _mm_unpackhi_pd(upRightBH, lowRightBH), 0x32)));
#else
#ifdef __SSE3__
            __m128d weight010 = _mm_movedup_pd(weights[0]);
            __m128d weight001 = _mm_movedup_pd(weights[1]);
#else
            __m128d weight010 = _mm_unpacklo_pd(weights[0], weights[0]);
            __m128d weight001 = _mm_unpacklo_pd(weights[1], weights[1]);
#endif
            __m128d rg = _mm_add_pd(_mm_mul_pd(weight001, upRightAL),
                                    _mm_mul_pd(weight010, lowLeftAL));
            __m128d ba = _mm_add_pd(_mm_mul_pd(weight001, upRightAH),
                                    _mm_mul_pd(weight010, lowLeftAH));
            __m128d weight000 = _mm_unpackhi_pd(weights[0], weights[0]);
            __m128d weight011 = _mm_unpackhi_pd(weights[1], weights[1]);
            rg = _mm_add_pd(rg,
                            _mm_add_pd(_mm_mul_pd(weight000, upLeftAL),
                                       _mm_mul_pd(weight011, lowRightAL)));
            ba = _mm_add_pd(ba,
                            _mm_add_pd(_mm_mul_pd(weight000, upLeftAH),
                                       _mm_mul_pd(weight011, lowRightAH)));
#ifdef __SSE3__
            __m128d weight110 = _mm_movedup_pd(weights[2]);
            __m128d weight101 = _mm_movedup_pd(weights[3]);
#else
            __m128d weight110 = _mm_unpacklo_pd(weights[2], weights[2]);
            __m128d weight101 = _mm_unpacklo_pd(weights[3], weights[3]);
#endif
            rg = _mm_add_pd(rg,
                            _mm_add_pd(_mm_mul_pd(weight101, upRightBL),
                                       _mm_mul_pd(weight110, lowLeftBL)));
            ba = _mm_add_pd(rg,
                            _mm_add_pd(_mm_mul_pd(weight101, upRightBH),
                                       _mm_mul_pd(weight110, lowLeftBH)));
            __m128d weight100 = _mm_unpackhi_pd(weights[2], weights[2]);
            __m128d weight111 = _mm_unpackhi_pd(weights[3], weights[3]);
            rg = _mm_add_pd(rg,
                            _mm_add_pd(_mm_mul_pd(weight100, upLeftBL),
                                       _mm_mul_pd(weight111, lowRightBL)));
            ba = _mm_add_pd(ba,
                            _mm_add_pd(_mm_mul_pd(weight100, upLeftBH),
                                       _mm_mul_pd(weight111, lowRightBH)));
#endif
            _mm_storeu_ps(outData,
                          _mm_movelh_ps(_mm_cvtpd_ps(rg), _mm_cvtpd_ps(ba)));
#endif
            if (NULL == containsDenorms) _mm_setcsr(mxcsr);
        }

        return returnVal;
    }
}
#else
FloatPixel sample_image_pixel_float_offset(
    void *imageData, image_descriptor *imageInfo, float x, float y, float z,
    float xAddressOffset, float yAddressOffset, float zAddressOffset,
    image_sampler_data *imageSampler, float *outData, int verbose,
    int *containsDenorms, int lod)
{
    AddressFn adFn = sAddressingTable[imageSampler];
    FloatPixel returnVal;
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth;
    size_t slice_pitch_lod = 0, row_pitch_lod = 0;

    if (imageInfo->num_mip_levels > 1)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                depth_lod =
                    (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                height_lod =
                    (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            default:
                width_lod =
                    (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        }
        row_pitch_lod = width_lod * get_pixel_size(imageInfo->format);
        if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            slice_pitch_lod = row_pitch_lod;
        else if (imageInfo->type == CL_MEM_OBJECT_IMAGE3D
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            slice_pitch_lod = row_pitch_lod * height_lod;
    }
    else
    {
        slice_pitch_lod = imageInfo->slicePitch;
        row_pitch_lod = imageInfo->rowPitch;
    }

    if (containsDenorms) *containsDenorms = 0;

    if (imageSampler->normalized_coords)
    {

        // We need to unnormalize our coordinates differently depending on
        // the image type, but 'x' is always processed the same way.
        x = unnormalize_coordinate("x", x, xAddressOffset, (float)width_lod,
                                   imageSampler->addressing_mode, verbose);

        switch (imageInfo->type)
        {

                // The image array types require special care:

            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                z = 0; // don't care -- unused for 1D arrays
                break;

            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                y = unnormalize_coordinate(
                    "y", y, yAddressOffset, (float)height_lod,
                    imageSampler->addressing_mode, verbose);
                break;

                // Everybody else:

            default:
                y = unnormalize_coordinate(
                    "y", y, yAddressOffset, (float)height_lod,
                    imageSampler->addressing_mode, verbose);
                z = unnormalize_coordinate(
                    "z", z, zAddressOffset, (float)depth_lod,
                    imageSampler->addressing_mode, verbose);
        }
    }
    else if (verbose)
    {

        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_info("Starting coordinate: %f, array index %f\n", x, y);
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_info("Starting coordinate: %f, %f, array index %f\n", x, y,
                         z);
                break;
            case CL_MEM_OBJECT_IMAGE1D:
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                log_info("Starting coordinate: %f\n", x);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_info("Starting coordinate: %f, %f\n", x, y);
                break;
            case CL_MEM_OBJECT_IMAGE3D:
            default: log_info("Starting coordinate: %f, %f, %f\n", x, y, z);
        }
    }

    // At this point, we have unnormalized coordinates.

    if (imageSampler->filter_mode == CL_FILTER_NEAREST)
    {
        int ix, iy, iz;

        // We apply the addressing function to the now-unnormalized
        // coordinates.  Note that the array cases again require special
        // care, per section 8.4 in the OpenCL 1.2 Specification.

        ix = adFn(static_cast<int>(floorf(x)), width_lod);

        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                iy = static_cast<int>(calculate_array_index(
                    y, (float)(imageInfo->arraySize - 1)));
                iz = 0;
                if (verbose)
                {
                    log_info("\tArray index %f evaluates to %d\n", y, iy);
                }
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                iy = adFn(static_cast<int>(floorf(y)), height_lod);
                iz = static_cast<int>(calculate_array_index(
                    z, (float)(imageInfo->arraySize - 1)));
                if (verbose)
                {
                    log_info("\tArray index %f evaluates to %d\n", z, iz);
                }
                break;
            default:
                iy = adFn(static_cast<int>(floorf(y)), height_lod);
                if (depth_lod != 0)
                    iz = adFn(static_cast<int>(floorf(z)), depth_lod);
                else
                    iz = 0;
        }

        if (verbose)
        {
            if (iz)
                log_info(
                    "\tReference integer coords calculated: { %d, %d, %d }\n",
                    ix, iy, iz);
            else
                log_info("\tReference integer coords calculated: { %d, %d }\n",
                         ix, iy);
        }

        read_image_pixel_float(imageData, imageInfo, ix, iy, iz, outData, lod);
        check_for_denorms(outData, containsDenorms);
        for (int i = 0; i < 4; i++) returnVal.p[i] = fabsf(outData[i]);
        return returnVal;
    }
    else
    {
        // Linear filtering cases.

        size_t width = width_lod, height = height_lod, depth = depth_lod;

        // Image arrays can use 2D filtering, but require us to walk into the
        // image a certain number of slices before reading.

        if (depth == 0 || imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY
            || imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        {
            float array_index = 0;

            size_t layer_offset = 0;

            if (imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
            {
                array_index =
                    calculate_array_index(z, (float)(imageInfo->arraySize - 1));
                layer_offset = slice_pitch_lod * (size_t)array_index;
            }
            else if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
            {
                array_index =
                    calculate_array_index(y, (float)(imageInfo->arraySize - 1));
                layer_offset = slice_pitch_lod * (size_t)array_index;

                // Set up y and height so that the filtering below is correct
                // 1D filtering on a single slice.
                height = 1;
            }

            int x1 = adFn(static_cast<int>(floorf(x - 0.5f)), width);
            int y1 = 0;
            int x2 = adFn(static_cast<int>(floorf(x - 0.5f) + 1), width);
            int y2 = 0;
            if ((imageInfo->type != CL_MEM_OBJECT_IMAGE1D)
                && (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_ARRAY)
                && (imageInfo->type != CL_MEM_OBJECT_IMAGE1D_BUFFER))
            {
                y1 = adFn(static_cast<int>(floorf(y - 0.5f)), height);
                y2 = adFn(static_cast<int>(floorf(y - 0.5f) + 1), height);
            }
            else
            {
                y = 0.5f;
            }

            if (verbose)
            {
                log_info("\tActual integer coords used (i = floor(x-.5)): i0:{ "
                         "%d, %d } and i1:{ %d, %d }\n",
                         x1, y1, x2, y2);
                log_info("\tArray coordinate is %f\n", array_index);
            }

            // Walk to beginning of the 'correct' slice, if needed.
            char *imgPtr = ((char *)imageData) + layer_offset;

            float upLeft[4], upRight[4], lowLeft[4], lowRight[4];
            float maxUp[4], maxLow[4];
            read_image_pixel_float(imgPtr, imageInfo, x1, y1, 0, upLeft, lod);
            read_image_pixel_float(imgPtr, imageInfo, x2, y1, 0, upRight, lod);
            check_for_denorms(upLeft, containsDenorms);
            check_for_denorms(upRight, containsDenorms);
            pixelMax(upLeft, upRight, maxUp);
            read_image_pixel_float(imgPtr, imageInfo, x1, y2, 0, lowLeft, lod);
            read_image_pixel_float(imgPtr, imageInfo, x2, y2, 0, lowRight, lod);
            check_for_denorms(lowLeft, containsDenorms);
            check_for_denorms(lowRight, containsDenorms);
            pixelMax(lowLeft, lowRight, maxLow);
            pixelMax(maxUp, maxLow, returnVal.p);

            if (verbose)
            {
                if (NULL == containsDenorms)
                    log_info("\tSampled pixels (rgba order, denorms flushed to "
                             "zero):\n");
                else
                    log_info("\tSampled pixels (rgba order):\n");
                log_info("\t\tp00: %f, %f, %f, %f\n", upLeft[0], upLeft[1],
                         upLeft[2], upLeft[3]);
                log_info("\t\tp01: %f, %f, %f, %f\n", upRight[0], upRight[1],
                         upRight[2], upRight[3]);
                log_info("\t\tp10: %f, %f, %f, %f\n", lowLeft[0], lowLeft[1],
                         lowLeft[2], lowLeft[3]);
                log_info("\t\tp11: %f, %f, %f, %f\n", lowRight[0], lowRight[1],
                         lowRight[2], lowRight[3]);
            }

            bool printMe = false;
            if (x1 <= 0 || x2 <= 0 || x1 >= (int)width - 1
                || x2 >= (int)width - 1)
                printMe = true;
            if (y1 <= 0 || y2 <= 0 || y1 >= (int)height - 1
                || y2 >= (int)height - 1)
                printMe = true;

            double weights[2][2];

            weights[0][0] = weights[0][1] = 1.0 - frac(x - 0.5f);
            weights[1][0] = weights[1][1] = frac(x - 0.5f);
            weights[0][0] *= 1.0 - frac(y - 0.5f);
            weights[1][0] *= 1.0 - frac(y - 0.5f);
            weights[0][1] *= frac(y - 0.5f);
            weights[1][1] *= frac(y - 0.5f);

            if (verbose)
                log_info("\tfrac( x - 0.5f ) = %f,  frac( y - 0.5f ) = %f\n",
                         frac(x - 0.5f), frac(y - 0.5f));

            for (int i = 0; i < 3; i++)
            {
                outData[i] = (float)((upLeft[i] * weights[0][0])
                                     + (upRight[i] * weights[1][0])
                                     + (lowLeft[i] * weights[0][1])
                                     + (lowRight[i] * weights[1][1]));
                // flush subnormal results to zero if necessary
                if (NULL == containsDenorms && fabs(outData[i]) < FLT_MIN)
                    outData[i] = copysignf(0.0f, outData[i]);
            }
            outData[3] = (float)((upLeft[3] * weights[0][0])
                                 + (upRight[3] * weights[1][0])
                                 + (lowLeft[3] * weights[0][1])
                                 + (lowRight[3] * weights[1][1]));
            // flush subnormal results to zero if necessary
            if (NULL == containsDenorms && fabs(outData[3]) < FLT_MIN)
                outData[3] = copysignf(0.0f, outData[3]);
        }
        else
        {
            // 3D linear filtering
            int x1 = adFn(static_cast<int>(floorf(x - 0.5f)), width_lod);
            int y1 = adFn(static_cast<int>(floorf(y - 0.5f)), height_lod);
            int z1 = adFn(static_cast<int>(floorf(z - 0.5f)), depth_lod);
            int x2 = adFn(static_cast<int>(floorf(x - 0.5f) + 1), width_lod);
            int y2 = adFn(static_cast<int>(floorf(y - 0.5f) + 1), height_lod);
            int z2 = adFn(static_cast<int>(floorf(z - 0.5f) + 1), depth_lod);

            if (verbose)
                log_info("\tActual integer coords used (i = floor(x-.5)): "
                         "i0:{%d, %d, %d} and i1:{%d, %d, %d}\n",
                         x1, y1, z1, x2, y2, z2);

            float upLeftA[4], upRightA[4], lowLeftA[4], lowRightA[4];
            float upLeftB[4], upRightB[4], lowLeftB[4], lowRightB[4];
            float pixelMaxA[4], pixelMaxB[4];
            read_image_pixel_float(imageData, imageInfo, x1, y1, z1, upLeftA,
                                   lod);
            read_image_pixel_float(imageData, imageInfo, x2, y1, z1, upRightA,
                                   lod);
            check_for_denorms(upLeftA, containsDenorms);
            check_for_denorms(upRightA, containsDenorms);
            pixelMax(upLeftA, upRightA, pixelMaxA);
            read_image_pixel_float(imageData, imageInfo, x1, y2, z1, lowLeftA,
                                   lod);
            read_image_pixel_float(imageData, imageInfo, x2, y2, z1, lowRightA,
                                   lod);
            check_for_denorms(lowLeftA, containsDenorms);
            check_for_denorms(lowRightA, containsDenorms);
            pixelMax(lowLeftA, lowRightA, pixelMaxB);
            pixelMax(pixelMaxA, pixelMaxB, returnVal.p);
            read_image_pixel_float(imageData, imageInfo, x1, y1, z2, upLeftB,
                                   lod);
            read_image_pixel_float(imageData, imageInfo, x2, y1, z2, upRightB,
                                   lod);
            check_for_denorms(upLeftB, containsDenorms);
            check_for_denorms(upRightB, containsDenorms);
            pixelMax(upLeftB, upRightB, pixelMaxA);
            read_image_pixel_float(imageData, imageInfo, x1, y2, z2, lowLeftB,
                                   lod);
            read_image_pixel_float(imageData, imageInfo, x2, y2, z2, lowRightB,
                                   lod);
            check_for_denorms(lowLeftB, containsDenorms);
            check_for_denorms(lowRightB, containsDenorms);
            pixelMax(lowLeftB, lowRightB, pixelMaxB);
            pixelMax(pixelMaxA, pixelMaxB, pixelMaxA);
            pixelMax(pixelMaxA, returnVal.p, returnVal.p);

            if (verbose)
            {
                if (NULL == containsDenorms)
                    log_info("\tSampled pixels (rgba order, denorms flushed to "
                             "zero):\n");
                else
                    log_info("\tSampled pixels (rgba order):\n");
                log_info("\t\tp000: %f, %f, %f, %f\n", upLeftA[0], upLeftA[1],
                         upLeftA[2], upLeftA[3]);
                log_info("\t\tp001: %f, %f, %f, %f\n", upRightA[0], upRightA[1],
                         upRightA[2], upRightA[3]);
                log_info("\t\tp010: %f, %f, %f, %f\n", lowLeftA[0], lowLeftA[1],
                         lowLeftA[2], lowLeftA[3]);
                log_info("\t\tp011: %f, %f, %f, %f\n\n", lowRightA[0],
                         lowRightA[1], lowRightA[2], lowRightA[3]);
                log_info("\t\tp100: %f, %f, %f, %f\n", upLeftB[0], upLeftB[1],
                         upLeftB[2], upLeftB[3]);
                log_info("\t\tp101: %f, %f, %f, %f\n", upRightB[0], upRightB[1],
                         upRightB[2], upRightB[3]);
                log_info("\t\tp110: %f, %f, %f, %f\n", lowLeftB[0], lowLeftB[1],
                         lowLeftB[2], lowLeftB[3]);
                log_info("\t\tp111: %f, %f, %f, %f\n", lowRightB[0],
                         lowRightB[1], lowRightB[2], lowRightB[3]);
            }

            double weights[2][2][2];

            float a = frac(x - 0.5f), b = frac(y - 0.5f), c = frac(z - 0.5f);
            weights[0][0][0] = weights[0][1][0] = weights[0][0][1] =
                weights[0][1][1] = 1.f - a;
            weights[1][0][0] = weights[1][1][0] = weights[1][0][1] =
                weights[1][1][1] = a;
            weights[0][0][0] *= 1.f - b;
            weights[1][0][0] *= 1.f - b;
            weights[0][0][1] *= 1.f - b;
            weights[1][0][1] *= 1.f - b;
            weights[0][1][0] *= b;
            weights[1][1][0] *= b;
            weights[0][1][1] *= b;
            weights[1][1][1] *= b;
            weights[0][0][0] *= 1.f - c;
            weights[0][1][0] *= 1.f - c;
            weights[1][0][0] *= 1.f - c;
            weights[1][1][0] *= 1.f - c;
            weights[0][0][1] *= c;
            weights[0][1][1] *= c;
            weights[1][0][1] *= c;
            weights[1][1][1] *= c;

            if (verbose)
                log_info("\tfrac( x - 0.5f ) = %f,  frac( y - 0.5f ) = %f, "
                         "frac( z - 0.5f ) = %f\n",
                         frac(x - 0.5f), frac(y - 0.5f), frac(z - 0.5f));

            for (int i = 0; i < 3; i++)
            {
                outData[i] = (float)((upLeftA[i] * weights[0][0][0])
                                     + (upRightA[i] * weights[1][0][0])
                                     + (lowLeftA[i] * weights[0][1][0])
                                     + (lowRightA[i] * weights[1][1][0])
                                     + (upLeftB[i] * weights[0][0][1])
                                     + (upRightB[i] * weights[1][0][1])
                                     + (lowLeftB[i] * weights[0][1][1])
                                     + (lowRightB[i] * weights[1][1][1]));
                // flush subnormal results to zero if necessary
                if (NULL == containsDenorms && fabs(outData[i]) < FLT_MIN)
                    outData[i] = copysignf(0.0f, outData[i]);
            }
            outData[3] = (float)((upLeftA[3] * weights[0][0][0])
                                 + (upRightA[3] * weights[1][0][0])
                                 + (lowLeftA[3] * weights[0][1][0])
                                 + (lowRightA[3] * weights[1][1][0])
                                 + (upLeftB[3] * weights[0][0][1])
                                 + (upRightB[3] * weights[1][0][1])
                                 + (lowLeftB[3] * weights[0][1][1])
                                 + (lowRightB[3] * weights[1][1][1]));
            // flush subnormal results to zero if necessary
            if (NULL == containsDenorms && fabs(outData[3]) < FLT_MIN)
                outData[3] = copysignf(0.0f, outData[3]);
        }

        return returnVal;
    }
}
#endif

FloatPixel sample_image_pixel_float_offset(
    void *imageData, image_descriptor *imageInfo, float x, float y, float z,
    float xAddressOffset, float yAddressOffset, float zAddressOffset,
    image_sampler_data *imageSampler, float *outData, int verbose,
    int *containsDenorms)
{
    return sample_image_pixel_float_offset(
        imageData, imageInfo, x, y, z, xAddressOffset, yAddressOffset,
        zAddressOffset, imageSampler, outData, verbose, containsDenorms, 0);
}


int debug_find_vector_in_image(void *imagePtr, image_descriptor *imageInfo,
                               void *vectorToFind, size_t vectorSize, int *outX,
                               int *outY, int *outZ, size_t lod)
{
    int foundCount = 0;
    char *iPtr = (char *)imagePtr;
    size_t width;
    size_t depth;
    size_t height;
    size_t row_pitch;
    size_t slice_pitch;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            width = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height = 1;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            width = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height = 1;
            depth = imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            width = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height =
                (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            width = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height =
                (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            depth = imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            width = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height =
                (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            depth = (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;
            break;
    }

    row_pitch = width * get_pixel_size(imageInfo->format);
    slice_pitch = row_pitch * height;

    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                if (memcmp(iPtr, vectorToFind, vectorSize) == 0)
                {
                    if (foundCount == 0)
                    {
                        *outX = (int)x;
                        if (outY != NULL) *outY = (int)y;
                        if (outZ != NULL) *outZ = (int)z;
                    }
                    foundCount++;
                }
                iPtr += vectorSize;
            }
            iPtr += row_pitch - (width * vectorSize);
        }
        iPtr += slice_pitch - (height * row_pitch);
    }
    return foundCount;
}

int debug_find_pixel_in_image(void *imagePtr, image_descriptor *imageInfo,
                              unsigned int *valuesToFind, int *outX, int *outY,
                              int *outZ, int lod)
{
    char vectorToFind[4 * 4];
    size_t vectorSize = get_format_channel_count(imageInfo->format);


    if (imageInfo->format->image_channel_data_type == CL_UNSIGNED_INT8)
    {
        unsigned char *p = (unsigned char *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (unsigned char)valuesToFind[i];
    }
    else if (imageInfo->format->image_channel_data_type == CL_UNSIGNED_INT16)
    {
        unsigned short *p = (unsigned short *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (unsigned short)valuesToFind[i];
        vectorSize *= 2;
    }
    else if (imageInfo->format->image_channel_data_type == CL_UNSIGNED_INT32)
    {
        unsigned int *p = (unsigned int *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (unsigned int)valuesToFind[i];
        vectorSize *= 4;
    }
    else
    {
        log_info("WARNING: Unable to search for debug pixel: invalid image "
                 "format\n");
        return false;
    }
    return debug_find_vector_in_image(imagePtr, imageInfo, vectorToFind,
                                      vectorSize, outX, outY, outZ, lod);
}

int debug_find_pixel_in_image(void *imagePtr, image_descriptor *imageInfo,
                              int *valuesToFind, int *outX, int *outY,
                              int *outZ, int lod)
{
    char vectorToFind[4 * 4];
    size_t vectorSize = get_format_channel_count(imageInfo->format);

    if (imageInfo->format->image_channel_data_type == CL_SIGNED_INT8)
    {
        char *p = (char *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (char)valuesToFind[i];
    }
    else if (imageInfo->format->image_channel_data_type == CL_SIGNED_INT16)
    {
        short *p = (short *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (short)valuesToFind[i];
        vectorSize *= 2;
    }
    else if (imageInfo->format->image_channel_data_type == CL_SIGNED_INT32)
    {
        int *p = (int *)vectorToFind;
        for (unsigned int i = 0; i < vectorSize; i++)
            p[i] = (int)valuesToFind[i];
        vectorSize *= 4;
    }
    else
    {
        log_info("WARNING: Unable to search for debug pixel: invalid image "
                 "format\n");
        return false;
    }
    return debug_find_vector_in_image(imagePtr, imageInfo, vectorToFind,
                                      vectorSize, outX, outY, outZ, lod);
}

int debug_find_pixel_in_image(void *imagePtr, image_descriptor *imageInfo,
                              float *valuesToFind, int *outX, int *outY,
                              int *outZ, int lod)
{
    char vectorToFind[4 * 4];
    float swizzled[4];
    memcpy(swizzled, valuesToFind, sizeof(swizzled));
    size_t vectorSize = get_pixel_size(imageInfo->format);
    pack_image_pixel(swizzled, imageInfo->format, vectorToFind);
    return debug_find_vector_in_image(imagePtr, imageInfo, vectorToFind,
                                      vectorSize, outX, outY, outZ, lod);
}

template <class T>
void swizzle_vector_for_image(T *srcVector, const cl_image_format *imageFormat)
{
    T temp;
    switch (imageFormat->image_channel_order)
    {
        case CL_A: srcVector[0] = srcVector[3]; break;
        case CL_R:
        case CL_Rx:
        case CL_RG:
        case CL_RGx:
        case CL_RGB:
        case CL_RGBx:
        case CL_RGBA:
        case CL_sRGB:
        case CL_sRGBx:
        case CL_sRGBA: break;
        case CL_RA: srcVector[1] = srcVector[3]; break;
        case CL_ARGB:
            temp = srcVector[3];
            srcVector[3] = srcVector[2];
            srcVector[2] = srcVector[1];
            srcVector[1] = srcVector[0];
            srcVector[0] = temp;
            break;
        case CL_ABGR:
            temp = srcVector[3];
            srcVector[3] = srcVector[0];
            srcVector[0] = temp;
            temp = srcVector[2];
            srcVector[2] = srcVector[1];
            srcVector[1] = temp;
            break;
        case CL_BGRA:
        case CL_sBGRA:
            temp = srcVector[0];
            srcVector[0] = srcVector[2];
            srcVector[2] = temp;
            break;
        case CL_INTENSITY:
            srcVector[3] = srcVector[0];
            srcVector[2] = srcVector[0];
            srcVector[1] = srcVector[0];
            break;
        case CL_LUMINANCE:
            srcVector[2] = srcVector[0];
            srcVector[1] = srcVector[0];
            break;
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
            temp = srcVector[3];
            srcVector[3] = srcVector[2];
            srcVector[2] = srcVector[1];
            srcVector[1] = srcVector[0];
            srcVector[0] = temp;
            break;
#endif
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
            temp = srcVector[0];
            srcVector[0] = srcVector[2];
            srcVector[2] = temp;
            break;
#endif
    }
}

#define SATURATE(v, min, max) (v < min ? min : (v > max ? max : v))

void pack_image_pixel(unsigned int *srcVector,
                      const cl_image_format *imageFormat, void *outData)
{
    swizzle_vector_for_image<unsigned int>(srcVector, imageFormat);
    size_t channelCount = get_format_channel_count(imageFormat);

    switch (imageFormat->image_channel_data_type)
    {
        case CL_UNSIGNED_INT8: {
            unsigned char *ptr = (unsigned char *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (unsigned char)SATURATE(srcVector[i], 0, 255);
            break;
        }
        case CL_UNSIGNED_INT16: {
            unsigned short *ptr = (unsigned short *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (unsigned short)SATURATE(srcVector[i], 0, 65535);
            break;
        }
        case CL_UNSIGNED_INT32: {
            unsigned int *ptr = (unsigned int *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (unsigned int)srcVector[i];
            break;
        }
        default: break;
    }
}

void pack_image_pixel(int *srcVector, const cl_image_format *imageFormat,
                      void *outData)
{
    swizzle_vector_for_image<int>(srcVector, imageFormat);
    size_t chanelCount = get_format_channel_count(imageFormat);

    switch (imageFormat->image_channel_data_type)
    {
        case CL_SIGNED_INT8: {
            char *ptr = (char *)outData;
            for (unsigned int i = 0; i < chanelCount; i++)
                ptr[i] = (char)SATURATE(srcVector[i], -128, 127);
            break;
        }
        case CL_SIGNED_INT16: {
            short *ptr = (short *)outData;
            for (unsigned int i = 0; i < chanelCount; i++)
                ptr[i] = (short)SATURATE(srcVector[i], -32768, 32767);
            break;
        }
        case CL_SIGNED_INT32: {
            int *ptr = (int *)outData;
            for (unsigned int i = 0; i < chanelCount; i++)
                ptr[i] = (int)srcVector[i];
            break;
        }
        default: break;
    }
}

cl_int round_to_even(float v)
{
    // clamp overflow
    if (v >= -(float)CL_INT_MIN) return CL_INT_MAX;
    if (v <= (float)CL_INT_MIN) return CL_INT_MIN;

    // round fractional values to integer value
    if (fabsf(v) < MAKE_HEX_FLOAT(0x1.0p23f, 0x1L, 23))
    {
        static const float magic[2] = { MAKE_HEX_FLOAT(0x1.0p23f, 0x1L, 23),
                                        MAKE_HEX_FLOAT(-0x1.0p23f, -0x1L, 23) };
        float magicVal = magic[v < 0.0f];
        v += magicVal;
        v -= magicVal;
    }

    return (cl_int)v;
}

void pack_image_pixel(float *srcVector, const cl_image_format *imageFormat,
                      void *outData)
{
    swizzle_vector_for_image<float>(srcVector, imageFormat);
    size_t channelCount = get_format_channel_count(imageFormat);
    switch (imageFormat->image_channel_data_type)
    {
        case CL_HALF_FLOAT: {
            cl_half *ptr = (cl_half *)outData;

            switch (gFloatToHalfRoundingMode)
            {
                case kRoundToNearestEven:
                    for (unsigned int i = 0; i < channelCount; i++)
                        ptr[i] = cl_half_from_float(srcVector[i], CL_HALF_RTE);
                    break;
                case kRoundTowardZero:
                    for (unsigned int i = 0; i < channelCount; i++)
                        ptr[i] = cl_half_from_float(srcVector[i], CL_HALF_RTZ);
                    break;
                default:
                    log_error("ERROR: Test internal error -- unhandled or "
                              "unknown float->half rounding mode.\n");
                    exit(-1);
                    break;
            }
            break;
        }

        case CL_FLOAT: {
            cl_float *ptr = (cl_float *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = srcVector[i];
            break;
        }

        case CL_SNORM_INT8: {
            cl_char *ptr = (cl_char *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] =
                    (cl_char)NORMALIZE_SIGNED(srcVector[i], -127.0f, 127.f);
            break;
        }
        case CL_SNORM_INT16: {
            cl_short *ptr = (cl_short *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] =
                    (short)NORMALIZE_SIGNED(srcVector[i], -32767.f, 32767.f);
            break;
        }
        case CL_UNORM_INT8: {
            cl_uchar *ptr = (cl_uchar *)outData;
            if (is_sRGBA_order(imageFormat->image_channel_order))
            {
                ptr[0] = (unsigned char)(sRGBmap(srcVector[0]) + 0.5);
                ptr[1] = (unsigned char)(sRGBmap(srcVector[1]) + 0.5);
                ptr[2] = (unsigned char)(sRGBmap(srcVector[2]) + 0.5);
                if (channelCount == 4)
                    ptr[3] = (unsigned char)NORMALIZE(srcVector[3], 255.f);
            }
            else
            {
                for (unsigned int i = 0; i < channelCount; i++)
                    ptr[i] = (unsigned char)NORMALIZE(srcVector[i], 255.f);
            }
#ifdef CL_1RGB_APPLE
            if (imageFormat->image_channel_order == CL_1RGB_APPLE)
                ptr[0] = 255.0f;
#endif
#ifdef CL_BGR1_APPLE
            if (imageFormat->image_channel_order == CL_BGR1_APPLE)
                ptr[3] = 255.0f;
#endif
            break;
        }
        case CL_UNORM_INT16: {
            cl_ushort *ptr = (cl_ushort *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (unsigned short)NORMALIZE(srcVector[i], 65535.f);
            break;
        }
        case CL_UNORM_SHORT_555: {
            cl_ushort *ptr = (cl_ushort *)outData;
            ptr[0] =
                (((unsigned short)NORMALIZE(srcVector[0], 31.f) & 31) << 10)
                | (((unsigned short)NORMALIZE(srcVector[1], 31.f) & 31) << 5)
                | (((unsigned short)NORMALIZE(srcVector[2], 31.f) & 31) << 0);
            break;
        }
        case CL_UNORM_SHORT_565: {
            cl_ushort *ptr = (cl_ushort *)outData;
            ptr[0] =
                (((unsigned short)NORMALIZE(srcVector[0], 31.f) & 31) << 11)
                | (((unsigned short)NORMALIZE(srcVector[1], 63.f) & 63) << 5)
                | (((unsigned short)NORMALIZE(srcVector[2], 31.f) & 31) << 0);
            break;
        }
        case CL_UNORM_INT_101010: {
            cl_uint *ptr = (cl_uint *)outData;
            ptr[0] =
                (((unsigned int)NORMALIZE(srcVector[0], 1023.f) & 1023) << 20)
                | (((unsigned int)NORMALIZE(srcVector[1], 1023.f) & 1023) << 10)
                | (((unsigned int)NORMALIZE(srcVector[2], 1023.f) & 1023) << 0);
            break;
        }
        case CL_SIGNED_INT8: {
            cl_char *ptr = (cl_char *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] =
                    (cl_char)CONVERT_INT(srcVector[i], -127.0f, 127.f, 127);
            break;
        }
        case CL_SIGNED_INT16: {
            cl_short *ptr = (cl_short *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] =
                    (short)CONVERT_INT(srcVector[i], -32767.f, 32767.f, 32767);
            break;
        }
        case CL_SIGNED_INT32: {
            cl_int *ptr = (cl_int *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = round_to_even(srcVector[i]);
            break;
        }
        case CL_UNSIGNED_INT8: {
            cl_uchar *ptr = (cl_uchar *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] =
                    (cl_uchar)CONVERT_UINT(srcVector[i], 255.f, CL_UCHAR_MAX);
            break;
        }
        case CL_UNSIGNED_INT16: {
            cl_ushort *ptr = (cl_ushort *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (cl_ushort)CONVERT_UINT(srcVector[i], 32767.f,
                                                 CL_USHRT_MAX);
            break;
        }
        case CL_UNSIGNED_INT32: {
            cl_uint *ptr = (cl_uint *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
                ptr[i] = (cl_uint)CONVERT_UINT(
                    srcVector[i],
                    MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffe, 31 - 23),
                    CL_UINT_MAX);
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: {
            cl_ushort *ptr = (cl_ushort *)outData;
            for (unsigned int i = 0; i < channelCount; i++)
            {
                cl_float f = fmaxf(srcVector[i], -1.0f);
                f = fminf(f, 3.0f);
                cl_int d = rintf(f * 0x1.0p14f);
                d += 16384;
                if (d > CL_USHRT_MAX) d = CL_USHRT_MAX;
                ptr[i] = d;
            }
            break;
        }
#endif
        default:
            log_error("INTERNAL ERROR: unknown format (%d)\n",
                      imageFormat->image_channel_data_type);
            exit(-1);
            break;
    }
}

void pack_image_pixel_error(const float *srcVector,
                            const cl_image_format *imageFormat,
                            const void *results, float *errors)
{
    size_t channelCount = get_format_channel_count(imageFormat);
    switch (imageFormat->image_channel_data_type)
    {
        case CL_HALF_FLOAT: {
            const cl_half *ptr = (const cl_half *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = Ulp_Error_Half(ptr[i], srcVector[i]);

            break;
        }

        case CL_FLOAT: {
            const cl_ushort *ptr = (const cl_ushort *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = Ulp_Error(ptr[i], srcVector[i]);

            break;
        }

        case CL_SNORM_INT8: {
            const cl_char *ptr = (const cl_char *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i]
                    - NORMALIZE_SIGNED_UNROUNDED(srcVector[i], -127.0f, 127.f);

            break;
        }
        case CL_SNORM_INT16: {
            const cl_short *ptr = (const cl_short *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i]
                    - NORMALIZE_SIGNED_UNROUNDED(srcVector[i], -32767.f,
                                                 32767.f);

            break;
        }
        case CL_UNORM_INT8: {
            const cl_uchar *ptr = (const cl_uchar *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i] - NORMALIZE_UNROUNDED(srcVector[i], 255.f);

            break;
        }
        case CL_UNORM_INT16: {
            const cl_ushort *ptr = (const cl_ushort *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i] - NORMALIZE_UNROUNDED(srcVector[i], 65535.f);

            break;
        }
        case CL_UNORM_SHORT_555: {
            const cl_ushort *ptr = (const cl_ushort *)results;

            errors[0] =
                ((ptr[0] >> 10) & 31) - NORMALIZE_UNROUNDED(srcVector[0], 31.f);
            errors[1] =
                ((ptr[0] >> 5) & 31) - NORMALIZE_UNROUNDED(srcVector[1], 31.f);
            errors[2] =
                ((ptr[0] >> 0) & 31) - NORMALIZE_UNROUNDED(srcVector[2], 31.f);

            break;
        }
        case CL_UNORM_SHORT_565: {
            const cl_ushort *ptr = (const cl_ushort *)results;

            errors[0] =
                ((ptr[0] >> 11) & 31) - NORMALIZE_UNROUNDED(srcVector[0], 31.f);
            errors[1] =
                ((ptr[0] >> 5) & 63) - NORMALIZE_UNROUNDED(srcVector[1], 63.f);
            errors[2] =
                ((ptr[0] >> 0) & 31) - NORMALIZE_UNROUNDED(srcVector[2], 31.f);

            break;
        }
        case CL_UNORM_INT_101010: {
            const cl_uint *ptr = (const cl_uint *)results;

            errors[0] = ((ptr[0] >> 20) & 1023)
                - NORMALIZE_UNROUNDED(srcVector[0], 1023.f);
            errors[1] = ((ptr[0] >> 10) & 1023)
                - NORMALIZE_UNROUNDED(srcVector[1], 1023.f);
            errors[2] = ((ptr[0] >> 0) & 1023)
                - NORMALIZE_UNROUNDED(srcVector[2], 1023.f);

            break;
        }
        case CL_SIGNED_INT8: {
            const cl_char *ptr = (const cl_char *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] =
                    ptr[i] - CONVERT_INT(srcVector[i], -127.0f, 127.f, 127);

            break;
        }
        case CL_SIGNED_INT16: {
            const cl_short *ptr = (const cl_short *)results;
            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i]
                    - CONVERT_INT(srcVector[i], -32767.f, 32767.f, 32767);
            break;
        }
        case CL_SIGNED_INT32: {
            const cl_int *ptr = (const cl_int *)results;
            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = (cl_float)((cl_long)ptr[i]
                                       - (cl_long)round_to_even(srcVector[i]));
            break;
        }
        case CL_UNSIGNED_INT8: {
            const cl_uchar *ptr = (const cl_uchar *)results;
            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = static_cast<float>(
                    (cl_int)ptr[i]
                    - (cl_int)CONVERT_UINT(srcVector[i], 255.f, CL_UCHAR_MAX));
            break;
        }
        case CL_UNSIGNED_INT16: {
            const cl_ushort *ptr = (const cl_ushort *)results;
            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = static_cast<float>(
                    (cl_int)ptr[i]
                    - (cl_int)CONVERT_UINT(srcVector[i], 32767.f,
                                           CL_USHRT_MAX));
            break;
        }
        case CL_UNSIGNED_INT32: {
            const cl_uint *ptr = (const cl_uint *)results;
            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = (cl_float)(
                    (cl_long)ptr[i]
                    - (cl_long)CONVERT_UINT(
                        srcVector[i],
                        MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffe, 31 - 23),
                        CL_UINT_MAX));
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE: {
            const cl_ushort *ptr = (const cl_ushort *)results;

            for (unsigned int i = 0; i < channelCount; i++)
                errors[i] = ptr[i]
                    - NORMALIZE_SIGNED_UNROUNDED(((int)srcVector[i] - 16384),
                                                 -16384.f, 49151.f);

            break;
        }
#endif
        default:
            log_error("INTERNAL ERROR: unknown format (%d)\n",
                      imageFormat->image_channel_data_type);
            exit(-1);
            break;
    }
}


//
//  Autodetect which rounding mode is used for image writes to CL_HALF_FLOAT
//  This should be called lazily before attempting to verify image writes,
//  otherwise an error will occur.
//
int DetectFloatToHalfRoundingMode(
    cl_command_queue q) // Returns CL_SUCCESS on success
{
    cl_int err = CL_SUCCESS;

    if (gFloatToHalfRoundingMode == kDefaultRoundingMode)
    {
        // Some numbers near 0.5f, that we look at to see how the values are
        // rounded.
        static const cl_uint inData[4 * 4] = {
            0x3f000fffU, 0x3f001000U, 0x3f001001U, 0U,
            0x3f001fffU, 0x3f002000U, 0x3f002001U, 0U,
            0x3f002fffU, 0x3f003000U, 0x3f003001U, 0U,
            0x3f003fffU, 0x3f004000U, 0x3f004001U, 0U
        };
        static const size_t count = sizeof(inData) / (4 * sizeof(inData[0]));
        const float *inp = (const float *)inData;
        cl_context context = NULL;

        // Create an input buffer
        err = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(context),
                                    &context, NULL);
        if (err)
        {
            log_error("Error:  could not get context from command queue in "
                      "DetectFloatToHalfRoundingMode  (%d)",
                      err);
            return err;
        }

        cl_mem inBuf = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
                                          | CL_MEM_ALLOC_HOST_PTR,
                                      sizeof(inData), (void *)inData, &err);
        if (NULL == inBuf || err)
        {
            log_error("Error:  could not create input buffer in "
                      "DetectFloatToHalfRoundingMode  (err: %d)",
                      err);
            return err;
        }

        // Create a small output image
        cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
        cl_mem outImage = create_image_2d(context, CL_MEM_WRITE_ONLY, &fmt,
                                          count, 1, 0, NULL, &err);
        if (NULL == outImage || err)
        {
            log_error("Error:  could not create half float out image in "
                      "DetectFloatToHalfRoundingMode  (err: %d)",
                      err);
            clReleaseMemObject(inBuf);
            return err;
        }

        // Create our program, and a kernel
        const char *kernelSource[1] = {
            "kernel void detect_round( global float4 *in, write_only image2d_t "
            "out )\n"
            "{\n"
            "   write_imagef( out, (int2)(get_global_id(0),0), "
            "in[get_global_id(0)] );\n"
            "}\n"
        };

        clProgramWrapper program;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &program, &kernel, 1,
                                          kernelSource, "detect_round");

        if (NULL == program || err)
        {
            log_error("Error:  could not create program in "
                      "DetectFloatToHalfRoundingMode (err: %d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        cl_device_id device = NULL;
        err = clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(device), &device,
                                    NULL);
        if (err)
        {
            log_error("Error:  could not get device from command queue in "
                      "DetectFloatToHalfRoundingMode  (%d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf);
        if (err)
        {
            log_error("Error: could not set argument 0 of kernel in "
                      "DetectFloatToHalfRoundingMode (%d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outImage);
        if (err)
        {
            log_error("Error: could not set argument 1 of kernel in "
                      "DetectFloatToHalfRoundingMode (%d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        // Run the kernel
        size_t global_work_size = count;
        err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global_work_size,
                                     NULL, 0, NULL, NULL);
        if (err)
        {
            log_error("Error: could not enqueue kernel in "
                      "DetectFloatToHalfRoundingMode (%d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        // read the results
        cl_half outBuf[count * 4];
        memset(outBuf, -1, sizeof(outBuf));
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { count, 1, 1 };
        err = clEnqueueReadImage(q, outImage, CL_TRUE, origin, region, 0, 0,
                                 outBuf, 0, NULL, NULL);
        if (err)
        {
            log_error("Error: could not read output image in "
                      "DetectFloatToHalfRoundingMode (%d)",
                      err);
            clReleaseMemObject(inBuf);
            clReleaseMemObject(outImage);
            return err;
        }

        // Generate our list of reference results
        cl_half rte_ref[count * 4];
        cl_half rtz_ref[count * 4];
        for (size_t i = 0; i < 4 * count; i++)
        {
            rte_ref[i] = cl_half_from_float(inp[i], CL_HALF_RTE);
            rtz_ref[i] = cl_half_from_float(inp[i], CL_HALF_RTZ);
        }

        // Verify that we got something in either rtz or rte mode
        if (0 == memcmp(rte_ref, outBuf, sizeof(rte_ref)))
        {
            log_info("Autodetected float->half rounding mode to be rte\n");
            gFloatToHalfRoundingMode = kRoundToNearestEven;
        }
        else if (0 == memcmp(rtz_ref, outBuf, sizeof(rtz_ref)))
        {
            log_info("Autodetected float->half rounding mode to be rtz\n");
            gFloatToHalfRoundingMode = kRoundTowardZero;
        }
        else
        {
            log_error("ERROR: float to half conversions proceed with invalid "
                      "rounding mode!\n");
            log_info("\nfor:");
            for (size_t i = 0; i < count; i++)
                log_info(" {%a, %a, %a, %a},", inp[4 * i], inp[4 * i + 1],
                         inp[4 * i + 2], inp[4 * i + 3]);
            log_info("\ngot:");
            for (size_t i = 0; i < count; i++)
                log_info(" {0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x},",
                         outBuf[4 * i], outBuf[4 * i + 1], outBuf[4 * i + 2],
                         outBuf[4 * i + 3]);
            log_info("\nrte:");
            for (size_t i = 0; i < count; i++)
                log_info(" {0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x},",
                         rte_ref[4 * i], rte_ref[4 * i + 1], rte_ref[4 * i + 2],
                         rte_ref[4 * i + 3]);
            log_info("\nrtz:");
            for (size_t i = 0; i < count; i++)
                log_info(" {0x%4.4x, 0x%4.4x, 0x%4.4x, 0x%4.4x},",
                         rtz_ref[4 * i], rtz_ref[4 * i + 1], rtz_ref[4 * i + 2],
                         rtz_ref[4 * i + 3]);
            log_info("\n");
            err = -1;
            gFloatToHalfRoundingMode = kRoundingModeCount; // illegal value
        }

        // clean up
        clReleaseMemObject(inBuf);
        clReleaseMemObject(outImage);
        return err;
    }

    // Make sure that the rounding mode was successfully detected, if we checked
    // earlier
    if (gFloatToHalfRoundingMode != kRoundToNearestEven
        && gFloatToHalfRoundingMode != kRoundTowardZero)
        return -2;

    return err;
}

char *create_random_image_data(ExplicitType dataType,
                               image_descriptor *imageInfo,
                               BufferOwningPtr<char> &P, MTdata d,
                               bool image2DFromBuffer)
{
    size_t allocSize, numPixels;
    if (/*gTestMipmaps*/ imageInfo->num_mip_levels > 1)
    {
        allocSize = (size_t)(compute_mipmapped_image_size(*imageInfo) * 4
                             * get_explicit_type_size(dataType))
            / get_pixel_size(imageInfo->format);
        numPixels = allocSize / (get_explicit_type_size(dataType) * 4);
    }
    else
    {
        numPixels = (image2DFromBuffer ? imageInfo->rowPitch : imageInfo->width)
            * imageInfo->height * (imageInfo->depth ? imageInfo->depth : 1)
            * (imageInfo->arraySize ? imageInfo->arraySize : 1);
        allocSize = numPixels * 4 * get_explicit_type_size(dataType);
    }

#if 0 // DEBUG
    {
      fprintf(stderr,"--- create_random_image_data:\n");
      fprintf(stderr,"allocSize = %zu\n",allocSize);
      fprintf(stderr,"numPixels = %zu\n",numPixels);
      fprintf(stderr,"width = %zu\n",imageInfo->width);
      fprintf(stderr,"height = %zu\n",imageInfo->height);
      fprintf(stderr,"depth = %zu\n",imageInfo->depth);
      fprintf(stderr,"rowPitch = %zu\n",imageInfo->rowPitch);
      fprintf(stderr,"slicePitch = %zu\n",imageInfo->slicePitch);
      fprintf(stderr,"arraySize = %zu\n",imageInfo->arraySize);
      fprintf(stderr,"explicit_type_size = %zu\n",get_explicit_type_size(dataType));
    }
#endif

#if defined(__APPLE__)
    char *data = NULL;
    if (gDeviceType == CL_DEVICE_TYPE_CPU)
    {
        size_t mapSize =
            ((allocSize + 4095L) & -4096L) + 8192; // alloc two extra pages.

        void *map = mmap(0, mapSize, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, 0, 0);
        if (map == MAP_FAILED)
        {
            perror("create_random_image_data: mmap");
            log_error("%s:%d: mmap failed, mapSize = %zu\n", __FILE__, __LINE__,
                      mapSize);
        }
        intptr_t data_end = (intptr_t)map + mapSize - 4096;
        data = (char *)(data_end - (intptr_t)allocSize);

        mprotect(map, 4096, PROT_NONE);
        mprotect((void *)((char *)map + mapSize - 4096), 4096, PROT_NONE);
        P.reset(data, map, mapSize);
    }
    else
    {
        data = (char *)malloc(allocSize);
        P.reset(data);
    }
#else
    char *data =
        (char *)align_malloc(allocSize, get_pixel_alignment(imageInfo->format));
    P.reset(data, NULL, 0, allocSize, true);
#endif

    if (data == NULL)
    {
        log_error(
            "ERROR: Unable to malloc %zu bytes for create_random_image_data\n",
            allocSize);
        return NULL;
    }

    switch (dataType)
    {
        case kFloat: {
            float *inputValues = (float *)data;
            switch (imageInfo->format->image_channel_data_type)
            {
                case CL_HALF_FLOAT: {
                    // Generate data that is (mostly) inside the range of a half
                    // float const float HALF_MIN = 5.96046448e-08f;
                    const float HALF_MAX = 65504.0f;

                    size_t i = 0;
                    inputValues[i++] = 0.f;
                    inputValues[i++] = 1.f;
                    inputValues[i++] = -1.f;
                    inputValues[i++] = 2.f;
                    for (; i < numPixels * 4; i++)
                        inputValues[i] = get_random_float(-HALF_MAX - 2.f,
                                                          HALF_MAX + 2.f, d);
                }
                break;
#ifdef CL_SFIXED14_APPLE
                case CL_SFIXED14_APPLE: {
                    size_t i = 0;
                    if (numPixels * 4 >= 8)
                    {
                        inputValues[i++] = INFINITY;
                        inputValues[i++] = 0x1.0p14f;
                        inputValues[i++] = 0x1.0p31f;
                        inputValues[i++] = 0x1.0p32f;
                        inputValues[i++] = -INFINITY;
                        inputValues[i++] = -0x1.0p14f;
                        inputValues[i++] = -0x1.0p31f;
                        inputValues[i++] = -0x1.1p31f;
                    }
                    for (; i < numPixels * 4; i++)
                        inputValues[i] = get_random_float(-1.1f, 3.1f, d);
                }
                break;
#endif
                case CL_FLOAT: {
                    size_t i = 0;
                    inputValues[i++] = INFINITY;
                    inputValues[i++] = -INFINITY;
                    inputValues[i++] = 0.0f;
                    inputValues[i++] = 0.0f;
                    cl_uint *p = (cl_uint *)data;
                    for (; i < numPixels * 4; i++) p[i] = genrand_int32(d);
                }
                break;

                default:
                    size_t i = 0;
                    if (numPixels * 4 >= 36)
                    {
                        inputValues[i++] = 0.0f;
                        inputValues[i++] = 0.5f;
                        inputValues[i++] = 31.5f;
                        inputValues[i++] = 32.0f;
                        inputValues[i++] = 127.5f;
                        inputValues[i++] = 128.0f;
                        inputValues[i++] = 255.5f;
                        inputValues[i++] = 256.0f;
                        inputValues[i++] = 1023.5f;
                        inputValues[i++] = 1024.0f;
                        inputValues[i++] = 32767.5f;
                        inputValues[i++] = 32768.0f;
                        inputValues[i++] = 65535.5f;
                        inputValues[i++] = 65536.0f;
                        inputValues[i++] = 2147483648.0f;
                        inputValues[i++] = 4294967296.0f;
                        inputValues[i++] = MAKE_HEX_FLOAT(0x1.0p63f, 1, 63);
                        inputValues[i++] = MAKE_HEX_FLOAT(0x1.0p64f, 1, 64);
                        inputValues[i++] = -0.0f;
                        inputValues[i++] = -0.5f;
                        inputValues[i++] = -31.5f;
                        inputValues[i++] = -32.0f;
                        inputValues[i++] = -127.5f;
                        inputValues[i++] = -128.0f;
                        inputValues[i++] = -255.5f;
                        inputValues[i++] = -256.0f;
                        inputValues[i++] = -1023.5f;
                        inputValues[i++] = -1024.0f;
                        inputValues[i++] = -32767.5f;
                        inputValues[i++] = -32768.0f;
                        inputValues[i++] = -65535.5f;
                        inputValues[i++] = -65536.0f;
                        inputValues[i++] = -2147483648.0f;
                        inputValues[i++] = -4294967296.0f;
                        inputValues[i++] = -MAKE_HEX_FLOAT(0x1.0p63f, 1, 63);
                        inputValues[i++] = -MAKE_HEX_FLOAT(0x1.0p64f, 1, 64);
                    }
                    if (is_format_signed(imageInfo->format))
                    {
                        for (; i < numPixels * 4; i++)
                            inputValues[i] = get_random_float(-1.1f, 1.1f, d);
                    }
                    else
                    {
                        for (; i < numPixels * 4; i++)
                            inputValues[i] = get_random_float(-0.1f, 1.1f, d);
                    }
                    break;
            }
            break;
        }

        case kInt: {
            int *imageData = (int *)data;

            // We want to generate ints (mostly) in range of the target format
            int formatMin = get_format_min_int(imageInfo->format);
            size_t formatMax = get_format_max_int(imageInfo->format);
            if (formatMin == 0)
            {
                // Unsigned values, but we are only an int, so cap the actual
                // max at the max of signed ints
                if (formatMax > 2147483647L) formatMax = 2147483647L;
            }
            // If the final format is small enough, give us a bit of room for
            // out-of-range values to test
            if (formatMax < 2147483647L) formatMax += 2;
            if (formatMin > -2147483648LL) formatMin -= 2;

            // Now gen
            for (size_t i = 0; i < numPixels * 4; i++)
            {
                imageData[i] = random_in_range(formatMin, (int)formatMax, d);
            }
            break;
        }

        case kUInt:
        case kUnsignedInt: {
            unsigned int *imageData = (unsigned int *)data;

            // We want to generate ints (mostly) in range of the target format
            int formatMin = get_format_min_int(imageInfo->format);
            size_t formatMax = get_format_max_int(imageInfo->format);
            if (formatMin < 0) formatMin = 0;
            // If the final format is small enough, give us a bit of room for
            // out-of-range values to test
            if (formatMax < 4294967295LL) formatMax += 2;

            // Now gen
            for (size_t i = 0; i < numPixels * 4; i++)
            {
                imageData[i] = random_in_range(formatMin, (int)formatMax, d);
            }
            break;
        }
        default:
            // Unsupported source format
            delete[] data;
            return NULL;
    }

    return data;
}

/*
    deprecated
bool clamp_image_coord( image_sampler_data *imageSampler, float value, size_t
max, int &outValue )
{
    int v = (int)value;

    switch(imageSampler->addressing_mode)
    {
        case CL_ADDRESS_REPEAT:
            outValue = v;
            while( v < 0 )
                v += (int)max;
            while( v >= (int)max )
                v -= (int)max;
            if( v != outValue )
            {
                outValue = v;
                return true;
            }
            return false;

        case CL_ADDRESS_MIRRORED_REPEAT:
            log_info( "ERROR: unimplemented for CL_ADDRESS_MIRRORED_REPEAT. Do
we ever use this? exit(-1);

        default:
            if( v < 0 )
            {
                outValue = 0;
                return true;
            }
            if( v >= (int)max )
            {
                outValue = (int)max - 1;
                return true;
            }
            outValue = v;
            return false;
    }

}
*/

void get_sampler_kernel_code(image_sampler_data *imageSampler, char *outLine)
{
    const char *normalized;
    const char *addressMode;
    const char *filterMode;

    if (imageSampler->addressing_mode == CL_ADDRESS_CLAMP)
        addressMode = "CLK_ADDRESS_CLAMP";
    else if (imageSampler->addressing_mode == CL_ADDRESS_CLAMP_TO_EDGE)
        addressMode = "CLK_ADDRESS_CLAMP_TO_EDGE";
    else if (imageSampler->addressing_mode == CL_ADDRESS_REPEAT)
        addressMode = "CLK_ADDRESS_REPEAT";
    else if (imageSampler->addressing_mode == CL_ADDRESS_MIRRORED_REPEAT)
        addressMode = "CLK_ADDRESS_MIRRORED_REPEAT";
    else if (imageSampler->addressing_mode == CL_ADDRESS_NONE)
        addressMode = "CLK_ADDRESS_NONE";
    else
    {
        log_error("**Error: Unknown addressing mode! Aborting...\n");
        abort();
    }

    if (imageSampler->normalized_coords)
        normalized = "CLK_NORMALIZED_COORDS_TRUE";
    else
        normalized = "CLK_NORMALIZED_COORDS_FALSE";

    if (imageSampler->filter_mode == CL_FILTER_LINEAR)
        filterMode = "CLK_FILTER_LINEAR";
    else
        filterMode = "CLK_FILTER_NEAREST";

    sprintf(outLine, "    const sampler_t imageSampler = %s | %s | %s;\n",
            addressMode, filterMode, normalized);
}

void copy_image_data(image_descriptor *srcImageInfo,
                     image_descriptor *dstImageInfo, void *imageValues,
                     void *destImageValues, const size_t sourcePos[],
                     const size_t destPos[], const size_t regionSize[])
{
    //  assert( srcImageInfo->format == dstImageInfo->format );

    size_t src_mip_level_offset = 0, dst_mip_level_offset = 0;
    size_t sourcePos_lod[3], destPos_lod[3], src_lod, dst_lod;
    size_t src_row_pitch_lod, src_slice_pitch_lod;
    size_t dst_row_pitch_lod, dst_slice_pitch_lod;

    size_t pixelSize = get_pixel_size(srcImageInfo->format);

    sourcePos_lod[0] = sourcePos[0];
    sourcePos_lod[1] = sourcePos[1];
    sourcePos_lod[2] = sourcePos[2];
    destPos_lod[0] = destPos[0];
    destPos_lod[1] = destPos[1];
    destPos_lod[2] = destPos[2];
    src_row_pitch_lod = srcImageInfo->rowPitch;
    dst_row_pitch_lod = dstImageInfo->rowPitch;
    src_slice_pitch_lod = srcImageInfo->slicePitch;
    dst_slice_pitch_lod = dstImageInfo->slicePitch;

    if (srcImageInfo->num_mip_levels > 1)
    {
        size_t src_width_lod = 1 /*srcImageInfo->width*/;
        size_t src_height_lod = 1 /*srcImageInfo->height*/;
        size_t src_depth_lod = 1 /*srcImageInfo->depth*/;

        switch (srcImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                src_lod = sourcePos[1];
                sourcePos_lod[1] = sourcePos_lod[2] = 0;
                src_width_lod = (srcImageInfo->width >> src_lod)
                    ? (srcImageInfo->width >> src_lod)
                    : 1;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            case CL_MEM_OBJECT_IMAGE2D:
                src_lod = sourcePos[2];
                sourcePos_lod[1] = sourcePos[1];
                sourcePos_lod[2] = 0;
                src_width_lod = (srcImageInfo->width >> src_lod)
                    ? (srcImageInfo->width >> src_lod)
                    : 1;
                if (srcImageInfo->type == CL_MEM_OBJECT_IMAGE2D)
                    src_height_lod = (srcImageInfo->height >> src_lod)
                        ? (srcImageInfo->height >> src_lod)
                        : 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            case CL_MEM_OBJECT_IMAGE3D:
                src_lod = sourcePos[3];
                sourcePos_lod[1] = sourcePos[1];
                sourcePos_lod[2] = sourcePos[2];
                src_width_lod = (srcImageInfo->width >> src_lod)
                    ? (srcImageInfo->width >> src_lod)
                    : 1;
                src_height_lod = (srcImageInfo->height >> src_lod)
                    ? (srcImageInfo->height >> src_lod)
                    : 1;
                if (srcImageInfo->type == CL_MEM_OBJECT_IMAGE3D)
                    src_depth_lod = (srcImageInfo->depth >> src_lod)
                        ? (srcImageInfo->depth >> src_lod)
                        : 1;
                break;
        }
        src_mip_level_offset = compute_mip_level_offset(srcImageInfo, src_lod);
        src_row_pitch_lod =
            src_width_lod * get_pixel_size(srcImageInfo->format);
        src_slice_pitch_lod = src_row_pitch_lod * src_height_lod;
    }

    if (dstImageInfo->num_mip_levels > 1)
    {
        size_t dst_width_lod = 1 /*dstImageInfo->width*/;
        size_t dst_height_lod = 1 /*dstImageInfo->height*/;
        size_t dst_depth_lod = 1 /*dstImageInfo->depth*/;
        switch (dstImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                dst_lod = destPos[1];
                destPos_lod[1] = destPos_lod[2] = 0;
                dst_width_lod = (dstImageInfo->width >> dst_lod)
                    ? (dstImageInfo->width >> dst_lod)
                    : 1;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            case CL_MEM_OBJECT_IMAGE2D:
                dst_lod = destPos[2];
                destPos_lod[1] = destPos[1];
                destPos_lod[2] = 0;
                dst_width_lod = (dstImageInfo->width >> dst_lod)
                    ? (dstImageInfo->width >> dst_lod)
                    : 1;
                if (dstImageInfo->type == CL_MEM_OBJECT_IMAGE2D)
                    dst_height_lod = (dstImageInfo->height >> dst_lod)
                        ? (dstImageInfo->height >> dst_lod)
                        : 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            case CL_MEM_OBJECT_IMAGE3D:
                dst_lod = destPos[3];
                destPos_lod[1] = destPos[1];
                destPos_lod[2] = destPos[2];
                dst_width_lod = (dstImageInfo->width >> dst_lod)
                    ? (dstImageInfo->width >> dst_lod)
                    : 1;
                dst_height_lod = (dstImageInfo->height >> dst_lod)
                    ? (dstImageInfo->height >> dst_lod)
                    : 1;
                if (dstImageInfo->type == CL_MEM_OBJECT_IMAGE3D)
                    dst_depth_lod = (dstImageInfo->depth >> dst_lod)
                        ? (dstImageInfo->depth >> dst_lod)
                        : 1;
                break;
        }
        dst_mip_level_offset = compute_mip_level_offset(dstImageInfo, dst_lod);
        dst_row_pitch_lod =
            dst_width_lod * get_pixel_size(dstImageInfo->format);
        dst_slice_pitch_lod = dst_row_pitch_lod * dst_height_lod;
    }

    // Get initial pointers
    char *sourcePtr = (char *)imageValues
        + sourcePos_lod[2] * src_slice_pitch_lod
        + sourcePos_lod[1] * src_row_pitch_lod + pixelSize * sourcePos_lod[0]
        + src_mip_level_offset;
    char *destPtr = (char *)destImageValues
        + destPos_lod[2] * dst_slice_pitch_lod
        + destPos_lod[1] * dst_row_pitch_lod + pixelSize * destPos_lod[0]
        + dst_mip_level_offset;

    for (size_t z = 0; z < (regionSize[2] > 0 ? regionSize[2] : 1); z++)
    {
        char *rowSourcePtr = sourcePtr;
        char *rowDestPtr = destPtr;
        for (size_t y = 0; y < regionSize[1]; y++)
        {
            memcpy(rowDestPtr, rowSourcePtr, pixelSize * regionSize[0]);
            rowSourcePtr += src_row_pitch_lod;
            rowDestPtr += dst_row_pitch_lod;
        }

        sourcePtr += src_slice_pitch_lod;
        destPtr += dst_slice_pitch_lod;
    }
}

float random_float(float low, float high, MTdata d)
{
    float t = (float)genrand_real1(d);
    return (1.0f - t) * low + t * high;
}

CoordWalker::CoordWalker(void *coords, bool useFloats, size_t vecSize)
{
    if (useFloats)
    {
        mFloatCoords = (cl_float *)coords;
        mIntCoords = NULL;
    }
    else
    {
        mFloatCoords = NULL;
        mIntCoords = (cl_int *)coords;
    }
    mVecSize = vecSize;
}

CoordWalker::~CoordWalker() {}

cl_float CoordWalker::Get(size_t idx, size_t el)
{
    if (mIntCoords != NULL)
        return (cl_float)mIntCoords[idx * mVecSize + el];
    else
        return mFloatCoords[idx * mVecSize + el];
}


void print_read_header(const cl_image_format *format,
                       image_sampler_data *sampler, bool err, int t)
{
    const char *addressMode = NULL;
    const char *normalizedNames[2] = { "UNNORMALIZED", "NORMALIZED" };

    if (sampler->addressing_mode == CL_ADDRESS_CLAMP)
        addressMode = "CL_ADDRESS_CLAMP";
    else if (sampler->addressing_mode == CL_ADDRESS_CLAMP_TO_EDGE)
        addressMode = "CL_ADDRESS_CLAMP_TO_EDGE";
    else if (sampler->addressing_mode == CL_ADDRESS_REPEAT)
        addressMode = "CL_ADDRESS_REPEAT";
    else if (sampler->addressing_mode == CL_ADDRESS_MIRRORED_REPEAT)
        addressMode = "CL_ADDRESS_MIRRORED_REPEAT";
    else
        addressMode = "CL_ADDRESS_NONE";

    if (t)
    {
        if (err)
            log_error("[%-7s %-24s %d] - %s - %s - %s - %s\n",
                      GetChannelOrderName(format->image_channel_order),
                      GetChannelTypeName(format->image_channel_data_type),
                      (int)get_format_channel_count(format),
                      sampler->filter_mode == CL_FILTER_NEAREST
                          ? "CL_FILTER_NEAREST"
                          : "CL_FILTER_LINEAR",
                      addressMode,
                      normalizedNames[sampler->normalized_coords ? 1 : 0],
                      t == 1 ? "TRANSPOSED" : "NON-TRANSPOSED");
        else
            log_info("[%-7s %-24s %d] - %s - %s - %s - %s\n",
                     GetChannelOrderName(format->image_channel_order),
                     GetChannelTypeName(format->image_channel_data_type),
                     (int)get_format_channel_count(format),
                     sampler->filter_mode == CL_FILTER_NEAREST
                         ? "CL_FILTER_NEAREST"
                         : "CL_FILTER_LINEAR",
                     addressMode,
                     normalizedNames[sampler->normalized_coords ? 1 : 0],
                     t == 1 ? "TRANSPOSED" : "NON-TRANSPOSED");
    }
    else
    {
        if (err)
            log_error("[%-7s %-24s %d] - %s - %s - %s\n",
                      GetChannelOrderName(format->image_channel_order),
                      GetChannelTypeName(format->image_channel_data_type),
                      (int)get_format_channel_count(format),
                      sampler->filter_mode == CL_FILTER_NEAREST
                          ? "CL_FILTER_NEAREST"
                          : "CL_FILTER_LINEAR",
                      addressMode,
                      normalizedNames[sampler->normalized_coords ? 1 : 0]);
        else
            log_info("[%-7s %-24s %d] - %s - %s - %s\n",
                     GetChannelOrderName(format->image_channel_order),
                     GetChannelTypeName(format->image_channel_data_type),
                     (int)get_format_channel_count(format),
                     sampler->filter_mode == CL_FILTER_NEAREST
                         ? "CL_FILTER_NEAREST"
                         : "CL_FILTER_LINEAR",
                     addressMode,
                     normalizedNames[sampler->normalized_coords ? 1 : 0]);
    }
}

void print_write_header(const cl_image_format *format, bool err = false)
{
    if (err)
        log_error("[%-7s %-24s %d]\n",
                  GetChannelOrderName(format->image_channel_order),
                  GetChannelTypeName(format->image_channel_data_type),
                  (int)get_format_channel_count(format));
    else
        log_info("[%-7s %-24s %d]\n",
                 GetChannelOrderName(format->image_channel_order),
                 GetChannelTypeName(format->image_channel_data_type),
                 (int)get_format_channel_count(format));
}


void print_header(const cl_image_format *format, bool err = false)
{
    if (err)
    {
        log_error("[%-7s %-24s %d]\n",
                  GetChannelOrderName(format->image_channel_order),
                  GetChannelTypeName(format->image_channel_data_type),
                  (int)get_format_channel_count(format));
    }
    else
    {
        log_info("[%-7s %-24s %d]\n",
                 GetChannelOrderName(format->image_channel_order),
                 GetChannelTypeName(format->image_channel_data_type),
                 (int)get_format_channel_count(format));
    }
}

bool find_format(cl_image_format *formatList, unsigned int numFormats,
                 cl_image_format *formatToFind)
{
    for (unsigned int i = 0; i < numFormats; i++)
    {
        if (formatList[i].image_channel_order
                == formatToFind->image_channel_order
            && formatList[i].image_channel_data_type
                == formatToFind->image_channel_data_type)
            return true;
    }
    return false;
}

void build_required_image_formats(
    cl_mem_flags flags, cl_mem_object_type image_type, cl_device_id device,
    std::vector<cl_image_format> &formatsToSupport)
{
    formatsToSupport.clear();

    // Minimum list of supported image formats for reading or writing (embedded
    // profile)
    static std::vector<cl_image_format> embeddedProfile_readOrWrite{
        // clang-format off
        { CL_RGBA, CL_UNORM_INT8 },
        { CL_RGBA, CL_UNORM_INT16 },
        { CL_RGBA, CL_SIGNED_INT8 },
        { CL_RGBA, CL_SIGNED_INT16 },
        { CL_RGBA, CL_SIGNED_INT32 },
        { CL_RGBA, CL_UNSIGNED_INT8 },
        { CL_RGBA, CL_UNSIGNED_INT16 },
        { CL_RGBA, CL_UNSIGNED_INT32 },
        { CL_RGBA, CL_HALF_FLOAT },
        { CL_RGBA, CL_FLOAT },
        // clang-format on
    };

    // Minimum list of required image formats for reading or writing
    // num_channels, for all image types.
    static std::vector<cl_image_format> fullProfile_readOrWrite{
        // clang-format off
        { CL_RGBA, CL_UNORM_INT8 },
        { CL_RGBA, CL_UNORM_INT16 },
        { CL_RGBA, CL_SIGNED_INT8 },
        { CL_RGBA, CL_SIGNED_INT16 },
        { CL_RGBA, CL_SIGNED_INT32 },
        { CL_RGBA, CL_UNSIGNED_INT8 },
        { CL_RGBA, CL_UNSIGNED_INT16 },
        { CL_RGBA, CL_UNSIGNED_INT32 },
        { CL_RGBA, CL_HALF_FLOAT },
        { CL_RGBA, CL_FLOAT },
        { CL_BGRA, CL_UNORM_INT8 },
        // clang-format on
    };

    // Minimum list of supported image formats for reading or writing
    // (OpenCL 2.0, 2.1, or 2.2), for all image types.
    static std::vector<cl_image_format> fullProfile_2x_readOrWrite{
        // clang-format off
        { CL_R, CL_UNORM_INT8 },
        { CL_R, CL_UNORM_INT16 },
        { CL_R, CL_SNORM_INT8 },
        { CL_R, CL_SNORM_INT16 },
        { CL_R, CL_SIGNED_INT8 },
        { CL_R, CL_SIGNED_INT16 },
        { CL_R, CL_SIGNED_INT32 },
        { CL_R, CL_UNSIGNED_INT8 },
        { CL_R, CL_UNSIGNED_INT16 },
        { CL_R, CL_UNSIGNED_INT32 },
        { CL_R, CL_HALF_FLOAT },
        { CL_R, CL_FLOAT },
        { CL_RG, CL_UNORM_INT8 },
        { CL_RG, CL_UNORM_INT16 },
        { CL_RG, CL_SNORM_INT8 },
        { CL_RG, CL_SNORM_INT16 },
        { CL_RG, CL_SIGNED_INT8 },
        { CL_RG, CL_SIGNED_INT16 },
        { CL_RG, CL_SIGNED_INT32 },
        { CL_RG, CL_UNSIGNED_INT8 },
        { CL_RG, CL_UNSIGNED_INT16 },
        { CL_RG, CL_UNSIGNED_INT32 },
        { CL_RG, CL_HALF_FLOAT },
        { CL_RG, CL_FLOAT },
        { CL_RGBA, CL_UNORM_INT8 },
        { CL_RGBA, CL_UNORM_INT16 },
        { CL_RGBA, CL_SNORM_INT8 },
        { CL_RGBA, CL_SNORM_INT16 },
        { CL_RGBA, CL_SIGNED_INT8 },
        { CL_RGBA, CL_SIGNED_INT16 },
        { CL_RGBA, CL_SIGNED_INT32 },
        { CL_RGBA, CL_UNSIGNED_INT8 },
        { CL_RGBA, CL_UNSIGNED_INT16 },
        { CL_RGBA, CL_UNSIGNED_INT32 },
        { CL_RGBA, CL_HALF_FLOAT },
        { CL_RGBA, CL_FLOAT },
        { CL_BGRA, CL_UNORM_INT8 },
        // clang-format on
    };

    // Conditional addition to the 2x readOrWrite table:
    // Support for the CL_DEPTH image channel order is required only for 2D
    // images and 2D image arrays.
    static std::vector<cl_image_format> fullProfile_2x_readOrWrite_Depth{
        // clang-format off
        { CL_DEPTH, CL_UNORM_INT16 },
        { CL_DEPTH, CL_FLOAT },
        // clang-format on
    };

    // Conditional addition to the 2x readOrWrite table:
    // Support for reading from the CL_sRGBA image channel order is optional for
    // 1D image buffers. Support for writing to the CL_sRGBA image channel order
    // is optional for all image types.
    static std::vector<cl_image_format> fullProfile_2x_readOrWrite_srgb{
        { CL_sRGBA, CL_UNORM_INT8 },
    };

    // Minimum list of required image formats for reading and writing.
    static std::vector<cl_image_format> fullProfile_readAndWrite{
        // clang-format off
        { CL_R, CL_UNORM_INT8 },
        { CL_R, CL_SIGNED_INT8 },
        { CL_R, CL_SIGNED_INT16 },
        { CL_R, CL_SIGNED_INT32 },
        { CL_R, CL_UNSIGNED_INT8 },
        { CL_R, CL_UNSIGNED_INT16 },
        { CL_R, CL_UNSIGNED_INT32 },
        { CL_R, CL_HALF_FLOAT },
        { CL_R, CL_FLOAT },
        { CL_RGBA, CL_UNORM_INT8 },
        { CL_RGBA, CL_SIGNED_INT8 },
        { CL_RGBA, CL_SIGNED_INT16 },
        { CL_RGBA, CL_SIGNED_INT32 },
        { CL_RGBA, CL_UNSIGNED_INT8 },
        { CL_RGBA, CL_UNSIGNED_INT16 },
        { CL_RGBA, CL_UNSIGNED_INT32 },
        { CL_RGBA, CL_HALF_FLOAT },
        { CL_RGBA, CL_FLOAT },
        // clang-format on
    };

    // Embedded profile
    if (gIsEmbedded)
    {
        copy(embeddedProfile_readOrWrite.begin(),
             embeddedProfile_readOrWrite.end(),
             back_inserter(formatsToSupport));
    }
    // Full profile
    else
    {
        Version version = get_device_cl_version(device);
        if (version < Version(2, 0) || version >= Version(3, 0))
        {
            // Full profile, OpenCL 1.2 or 3.0.
            if (flags & CL_MEM_KERNEL_READ_AND_WRITE)
            {
                // Note: assumes that read-write images are supported!
                copy(fullProfile_readAndWrite.begin(),
                     fullProfile_readAndWrite.end(),
                     back_inserter(formatsToSupport));
            }
            else
            {
                copy(fullProfile_readOrWrite.begin(),
                     fullProfile_readOrWrite.end(),
                     back_inserter(formatsToSupport));
            }
        }
        else
        {
            // Full profile, OpenCL 2.0, 2.1, 2.2.
            if (flags & CL_MEM_KERNEL_READ_AND_WRITE)
            {
                copy(fullProfile_readAndWrite.begin(),
                     fullProfile_readAndWrite.end(),
                     back_inserter(formatsToSupport));
            }
            else
            {
                copy(fullProfile_2x_readOrWrite.begin(),
                     fullProfile_2x_readOrWrite.end(),
                     back_inserter(formatsToSupport));

                // Support for the CL_DEPTH image channel order is required only
                // for 2D images and 2D image arrays.
                if (image_type == CL_MEM_OBJECT_IMAGE2D
                    || image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                {
                    copy(fullProfile_2x_readOrWrite_Depth.begin(),
                         fullProfile_2x_readOrWrite_Depth.end(),
                         back_inserter(formatsToSupport));
                }

                // Support for reading from the CL_sRGBA image channel order is
                // optional for 1D image buffers. Support for writing to the
                // CL_sRGBA image channel order is optional for all image types.
                if (image_type != CL_MEM_OBJECT_IMAGE1D_BUFFER
                    && flags == CL_MEM_READ_ONLY)
                {
                    copy(fullProfile_2x_readOrWrite_srgb.begin(),
                         fullProfile_2x_readOrWrite_srgb.end(),
                         back_inserter(formatsToSupport));
                }
            }
        }
    }
}

bool is_image_format_required(cl_image_format format, cl_mem_flags flags,
                              cl_mem_object_type image_type,
                              cl_device_id device)
{
    std::vector<cl_image_format> formatsToSupport;
    build_required_image_formats(flags, image_type, device, formatsToSupport);

    for (auto &formatItr : formatsToSupport)
    {
        if (formatItr.image_channel_order == format.image_channel_order
            && formatItr.image_channel_data_type
                == format.image_channel_data_type)
        {
            return true;
        }
    }

    return false;
}

cl_uint compute_max_mip_levels(size_t width, size_t height, size_t depth)
{
    cl_uint retMaxMipLevels = 0;
    size_t max_dim = 0;

    max_dim = width;
    max_dim = height > max_dim ? height : max_dim;
    max_dim = depth > max_dim ? depth : max_dim;

    while (max_dim)
    {
        retMaxMipLevels++;
        max_dim >>= 1;
    }
    return retMaxMipLevels;
}

cl_ulong compute_mipmapped_image_size(image_descriptor imageInfo)
{
    cl_ulong retSize = 0;
    size_t curr_width, curr_height, curr_depth, curr_array_size;
    curr_width = imageInfo.width;
    curr_height = imageInfo.height;
    curr_depth = imageInfo.depth;
    curr_array_size = imageInfo.arraySize;

    for (int i = 0; i < (int)imageInfo.num_mip_levels; i++)
    {
        switch (imageInfo.type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                retSize += (cl_ulong)curr_width * curr_height * curr_depth
                    * get_pixel_size(imageInfo.format);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                retSize += (cl_ulong)curr_width * curr_height
                    * get_pixel_size(imageInfo.format);
                break;
            case CL_MEM_OBJECT_IMAGE1D:
                retSize +=
                    (cl_ulong)curr_width * get_pixel_size(imageInfo.format);
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                retSize += (cl_ulong)curr_width * curr_array_size
                    * get_pixel_size(imageInfo.format);
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                retSize += (cl_ulong)curr_width * curr_height * curr_array_size
                    * get_pixel_size(imageInfo.format);
                break;
        }

        switch (imageInfo.type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                curr_depth = curr_depth >> 1 ? curr_depth >> 1 : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                curr_height = curr_height >> 1 ? curr_height >> 1 : 1;
            case CL_MEM_OBJECT_IMAGE1D:
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                curr_width = curr_width >> 1 ? curr_width >> 1 : 1;
        }
    }

    return retSize;
}

size_t compute_mip_level_offset(image_descriptor *imageInfo, size_t lod)
{
    size_t retOffset = 0;
    size_t width, height, depth;
    width = imageInfo->width;
    height = imageInfo->height;
    depth = imageInfo->depth;

    for (size_t i = 0; i < lod; i++)
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                retOffset += (size_t)width * height * imageInfo->arraySize
                    * get_pixel_size(imageInfo->format);
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                retOffset += (size_t)width * height * depth
                    * get_pixel_size(imageInfo->format);
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                retOffset += (size_t)width * imageInfo->arraySize
                    * get_pixel_size(imageInfo->format);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                retOffset +=
                    (size_t)width * height * get_pixel_size(imageInfo->format);
                break;
            case CL_MEM_OBJECT_IMAGE1D:
                retOffset += (size_t)width * get_pixel_size(imageInfo->format);
                break;
        }

        // Compute next lod dimensions
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D: depth = (depth >> 1) ? (depth >> 1) : 1;
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                height = (height >> 1) ? (height >> 1) : 1;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            case CL_MEM_OBJECT_IMAGE1D: width = (width >> 1) ? (width >> 1) : 1;
        }
    }
    return retOffset;
}

const char *convert_image_type_to_string(cl_mem_object_type image_type)
{
    switch (image_type)
    {
        case CL_MEM_OBJECT_IMAGE1D: return "1D";
        case CL_MEM_OBJECT_IMAGE2D: return "2D";
        case CL_MEM_OBJECT_IMAGE3D: return "3D";
        case CL_MEM_OBJECT_IMAGE1D_ARRAY: return "1D array";
        case CL_MEM_OBJECT_IMAGE2D_ARRAY: return "2D array";
        case CL_MEM_OBJECT_IMAGE1D_BUFFER: return "1D image buffer";
        default: return "unrecognized object type";
    }
}

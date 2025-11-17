//
// Copyright (c) 2025 The Khronos Group Inc.
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
#ifndef _mathHelpers_h
#define _mathHelpers_h

#if defined(__APPLE__)
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif
#include <cmath>

template <typename T> inline bool isnan_fp(const T &v) { return std::isnan(v); }

template <> inline bool isnan_fp<cl_half>(const cl_half &v)
{
    uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = ((cl_half)v) & 0x3FF;
    return (h_exp == 0x1F && h_mant != 0);
}

#endif // _mathHelpers_h

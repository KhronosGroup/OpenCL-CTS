//
// Copyright (c) 2021-2024 The Khronos Group Inc.
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

#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include "function_list.h"

// float foo(float)
int TestFunc_Float_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double)
int TestFunc_Double_Double(const Func *f, MTdata, bool relaxedMode);

// half foo(half)
int TestFunc_Half_Half(const Func *f, MTdata, bool relaxedMode);

// int foo(float)
int TestFunc_Int_Float(const Func *f, MTdata, bool relaxedMode);

// int foo(double)
int TestFunc_Int_Double(const Func *f, MTdata, bool relaxedMode);

// float foo(uint)
int TestFunc_Float_UInt(const Func *f, MTdata, bool relaxedMode);

// double foo(ulong)
int TestFunc_Double_ULong(const Func *f, MTdata, bool relaxedMode);

// half (Ushort)
int TestFunc_Half_UShort(const Func *f, MTdata, bool relaxedMode);

// Returns {0, 1} for scalar and {0, -1} for vector.
// int foo(float)
int TestMacro_Int_Float(const Func *f, MTdata, bool relaxedMode);

// Returns {0, 1} for scalar and {0, -1} for vector.
// int foo(double)
int TestMacro_Int_Double(const Func *f, MTdata, bool relaxedMode);

// int foo(half,half)
int TestMacro_Int_Half_Half(const Func *f, MTdata, bool relaxedMode);

// int foo(half)
int TestMacro_Int_Half(const Func *f, MTdata, bool relaxedMode);

// int foo(half)
int TestFunc_Int_Half(const Func *f, MTdata, bool relaxedMode);

// float foo(float, float)
int TestFunc_Float_Float_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double, double)
int TestFunc_Double_Double_Double(const Func *f, MTdata, bool relaxedMode);

// Half foo(half, half)
int TestFunc_Half_Half_Half(const Func *f, MTdata, bool relaxedMode);
// Special handling for nextafter.
// Half foo(Half, Half)
int TestFunc_Half_Half_Half_nextafter(const Func *f, MTdata, bool relaxedMode);

// Half foo(Half, Half)
int TestFunc_Half_Half_Half_common(const Func *f, MTdata, int isNextafter,
                                   bool relaxedMode);

// Half foo(Half, int)
int TestFunc_Half_Half_Int(const Func *f, MTdata, bool relaxedMode);


// float op float
int TestFunc_Float_Float_Float_Operator(const Func *f, MTdata,
                                        bool relaxedMode);

// double op double
int TestFunc_Double_Double_Double_Operator(const Func *f, MTdata,
                                           bool relaxedMode);

// half op half
int TestFunc_Half_Half_Half_Operator(const Func *f, MTdata, bool relaxedMode);

// float foo(float, int)
int TestFunc_Float_Float_Int(const Func *f, MTdata, bool relaxedMode);

// double foo(double, int)
int TestFunc_Double_Double_Int(const Func *f, MTdata, bool relaxedMode);

// Returns {0, 1} for scalar and {0, -1} for vector.
// int foo(float, float)
int TestMacro_Int_Float_Float(const Func *f, MTdata, bool relaxedMode);

// Returns {0, 1} for scalar and {0, -1} for vector.
// int foo(double, double)
int TestMacro_Int_Double_Double(const Func *f, MTdata, bool relaxedMode);

// float foo(float, float, float)
int TestFunc_Float_Float_Float_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double, double, double)
int TestFunc_Double_Double_Double_Double(const Func *f, MTdata,
                                         bool relaxedMode);

// half foo(half, half, half)
int TestFunc_Half_Half_Half_Half(const Func *f, MTdata, bool relaxedMode);

// float foo(float, float*)
int TestFunc_Float2_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double, double*)
int TestFunc_Double2_Double(const Func *f, MTdata, bool relaxedMode);

// half foo(half, half*)
int TestFunc_Half2_Half(const Func *f, MTdata, bool relaxedMode);

// float foo(float, int*)
int TestFunc_FloatI_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double, int*)
int TestFunc_DoubleI_Double(const Func *f, MTdata, bool relaxedMode);

// half foo(half, int*)
int TestFunc_HalfI_Half(const Func *f, MTdata d, bool relaxedMode);

// float foo(float, float, int*)
int TestFunc_FloatI_Float_Float(const Func *f, MTdata, bool relaxedMode);

// double foo(double, double, int*)
int TestFunc_DoubleI_Double_Double(const Func *f, MTdata, bool relaxedMode);

// half foo(half, half, int*)
int TestFunc_HalfI_Half_Half(const Func *f, MTdata d, bool relaxedMode);

// Special handling for mad.
// float mad(float, float, float)
int TestFunc_mad_Float(const Func *f, MTdata, bool relaxedMode);

// Special handling for mad.
// double mad(double, double, double)
int TestFunc_mad_Double(const Func *f, MTdata, bool relaxedMode);

// half mad(half, half, half)
int TestFunc_mad_Half(const Func *f, MTdata, bool relaxedMode);

#endif

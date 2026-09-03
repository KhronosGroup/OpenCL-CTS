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
#ifndef _testBase_h
#define _testBase_h

#include "harness/compat.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <optional>

// The number of errors to print out for each test
#define MAX_ERRORS_TO_PRINT 10

extern void fill_test_values(cl_long *outBufferA, cl_long *outBufferB,
                             size_t numElements, MTdata d);

// The "+", "-" and "*" integer tests fill both operands with full range random
// values. OpenCL C section 6.3 applies the C99 usual arithmetic conversions, so
// an operation whose signed result is not representable is undefined; the
// wrap-around guarantee of section 6.3.11 is scoped to atomics. Only three
// cases can actually get there:
//
//   int, long  operands keep their own width, so "+", "-" and "*" all overflow.
//   ushort     a scalar operand is promoted to int, and only the product of two
//              ushorts can leave the int range.
//
// char, uchar and short are promoted to int as well and their results always
// fit; uint and ulong wrap by definition. Those five types are left alone so
// that their full range and corner case coverage is preserved.
enum DefinedArithmeticOp
{
    kDefinedAdd,
    kDefinedSub,
    kDefinedMul
};

// Maps an operator spelling onto the enum above. Returns an empty optional for
// every operator whose operands do not need to be bounded.
std::optional<DefinedArithmeticOp>
get_defined_arithmetic_op(const char *opName);

// Rewrites both operands in place with random values whose combination under
// `op` is representable, and leaves them untouched for the types that cannot
// overflow.
void init_defined_arithmetic_data(ExplicitType type, DefinedArithmeticOp op,
                                  size_t num_elements, void *inputA,
                                  void *inputB, MTdata d);

#endif // _testBase_h




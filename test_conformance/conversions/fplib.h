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
#ifndef CONVERSIONS_FPLIB_H
#define CONVERSIONS_FPLIB_H

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    qcomRTZ = 0,
    qcomRTE,
    qcomRTP,
    qcomRTN,

    qcomRoundingModeCount
}roundingMode;

float qcom_u64_2_f32(uint64_t data, bool sat, roundingMode rnd);
float qcom_s64_2_f32(int64_t data, bool sat, roundingMode rnd);

#endif

//
// Copyright (c) 2021 The Khronos Group Inc.
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

#include "../testBase.h"

#define ABS_ERROR(result, expected) (fabs(expected - result))
#define CLAMP(_val, _min, _max)                                                \
    ((_val) < (_min) ? (_min) : (_val) > (_max) ? (_max) : (_val))

#define MAX_ERR 0.005f
#define MAX_TRIES 1
#define MAX_CLAMPED 1

extern cl_sampler create_sampler(cl_context context, image_sampler_data *sdata, bool test_mipmaps, cl_int *error);
extern void read_image_pixel_float(void *imageData, image_descriptor *imageInfo,
                                   int x, int y, int z, float *outData);

extern bool gExtraValidateInfo;
extern bool gDisableOffsets;
extern bool gUseKernelSamplers;
extern cl_mem_flags gMemFlagsToUse;
extern int gtestTypesToRun;
extern uint64_t gRoundingStartValue;
extern bool gPrintOptions;
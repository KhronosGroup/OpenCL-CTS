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

extern int test_read_image(cl_context context, cl_command_queue queue,
                           cl_kernel kernel, image_descriptor *imageInfo,
                           image_sampler_data *imageSampler,
                           bool useFloatCoords, ExplicitType outputType,
                           MTdata d);

extern bool get_image_dimensions(image_descriptor *imageInfo, size_t &width,
                                 size_t &height, size_t &depth);

template <class T>
int determine_validation_error_offset(
    void *imagePtr, image_descriptor *imageInfo,
    image_sampler_data *imageSampler, T *resultPtr, T *expected, float error,
    float x, float y, float z, float xAddressOffset, float yAddressOffset,
    float zAddressOffset, size_t j, int &numTries, int &numClamped,
    bool printAsFloat, int lod)
{
    bool image_type_3D = ((imageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
                          || (imageInfo->type == CL_MEM_OBJECT_IMAGE3D));
    bool image_type_1D = (imageInfo->type == CL_MEM_OBJECT_IMAGE1D);
    int actualX, actualY, actualZ;
    int found = debug_find_pixel_in_image(imagePtr, imageInfo, resultPtr,
                                          &actualX, &actualY, &actualZ, lod);
    bool clampingErr = false, clamped = false, otherClampingBug = false;
    int clampedX, clampedY, clampedZ;

    size_t imageWidth, imageHeight, imageDepth;
    if (get_image_dimensions(imageInfo, imageWidth, imageHeight, imageDepth))
    {
        log_error("ERROR: invalid image dimensions");
        return TEST_FAIL;
    }

    clamped = get_integer_coords_offset(
        x, !image_type_1D ? y : 0.0f, image_type_3D ? z : 0.0f, xAddressOffset,
        !image_type_1D ? yAddressOffset : 0.0f,
        image_type_3D ? zAddressOffset : 0.0f, imageWidth, imageHeight,
        imageDepth, imageSampler, imageInfo, clampedX, clampedY, clampedZ);

    if (found)
    {
        // Is it a clamping bug?
        if (clamped && clampedX == actualX
            && (clampedY == actualY || image_type_1D)
            && (clampedZ == actualZ || !image_type_3D))
        {
            if ((--numClamped) == 0)
            {
                if (printAsFloat)
                {
                    log_error("Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did "
                              "not validate! Expected (%g,%g,%g,%g), got "
                              "(%g,%g,%g,%g), error of %g\n",
                              j, x, x, y, y, z, z, (float)expected[0],
                              (float)expected[1], (float)expected[2],
                              (float)expected[3], (float)resultPtr[0],
                              (float)resultPtr[1], (float)resultPtr[2],
                              (float)resultPtr[3], error);
                }
                else
                {
                    log_error(
                        "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not "
                        "validate! Expected (%x,%x,%x,%x), got (%x,%x,%x,%x)\n",
                        j, x, x, y, y, z, z, (int)expected[0], (int)expected[1],
                        (int)expected[2], (int)expected[3], (int)resultPtr[0],
                        (int)resultPtr[1], (int)resultPtr[2],
                        (int)resultPtr[3]);
                }
                log_error("ERROR: TEST FAILED: Read is erroneously clamping "
                          "coordinates!\n");

                if (imageSampler->filter_mode != CL_FILTER_LINEAR)
                {
                    log_error(
                        "\tValue really found in image at %d,%d,%d (%s)\n",
                        actualX, actualY, actualZ,
                        (found > 1) ? "NOT unique!!" : "unique");
                }
                log_error("\n");

                return -1;
            }
            clampingErr = true;
            otherClampingBug = true;
        }
    }
    if (clamped && !otherClampingBug)
    {
        // If we are in clamp-to-edge mode and we're getting zeroes, it's
        // possible we're getting border erroneously
        if (resultPtr[0] == 0 && resultPtr[1] == 0 && resultPtr[2] == 0
            && resultPtr[3] == 0)
        {
            if ((--numClamped) == 0)
            {
                if (printAsFloat)
                {
                    log_error("Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did "
                              "not validate! Expected (%g,%g,%g,%g), got "
                              "(%g,%g,%g,%g), error of %g\n",
                              j, x, x, y, y, z, z, (float)expected[0],
                              (float)expected[1], (float)expected[2],
                              (float)expected[3], (float)resultPtr[0],
                              (float)resultPtr[1], (float)resultPtr[2],
                              (float)resultPtr[3], error);
                }
                else
                {
                    log_error(
                        "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not "
                        "validate! Expected (%x,%x,%x,%x), got (%x,%x,%x,%x)\n",
                        j, x, x, y, y, z, z, (int)expected[0], (int)expected[1],
                        (int)expected[2], (int)expected[3], (int)resultPtr[0],
                        (int)resultPtr[1], (int)resultPtr[2],
                        (int)resultPtr[3]);
                }
                log_error("ERROR: TEST FAILED: Clamping is erroneously "
                          "returning border color!\n");
                return -1;
            }
            clampingErr = true;
        }
    }
    if (!clampingErr)
    {
        if (printAsFloat)
        {
            log_error("Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not "
                      "validate!\n\tExpected (%g,%g,%g,%g),\n\t     got "
                      "(%g,%g,%g,%g), error of %g\n",
                      j, x, x, y, y, z, z, (float)expected[0],
                      (float)expected[1], (float)expected[2],
                      (float)expected[3], (float)resultPtr[0],
                      (float)resultPtr[1], (float)resultPtr[2],
                      (float)resultPtr[3], error);
        }
        else
        {
            log_error("Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not "
                      "validate!\n\tExpected (%x,%x,%x,%x),\n\t     got "
                      "(%x,%x,%x,%x)\n",
                      j, x, x, y, y, z, z, (int)expected[0], (int)expected[1],
                      (int)expected[2], (int)expected[3], (int)resultPtr[0],
                      (int)resultPtr[1], (int)resultPtr[2], (int)resultPtr[3]);
        }
        log_error(
            "Integer coords resolve to %d,%d,%d   with img size %d,%d,%d\n",
            clampedX, clampedY, clampedZ, (int)imageWidth, (int)imageHeight,
            (int)imageDepth);

        if (printAsFloat && gExtraValidateInfo)
        {
            log_error("\nNearby values:\n");
            for (int zOff = -1; zOff <= 1; zOff++)
            {
                for (int yOff = -1; yOff <= 1; yOff++)
                {
                    float top[4], real[4], bot[4];
                    read_image_pixel_float(imagePtr, imageInfo, clampedX - 1,
                                           clampedY + yOff, clampedZ + zOff,
                                           top);
                    read_image_pixel_float(imagePtr, imageInfo, clampedX,
                                           clampedY + yOff, clampedZ + zOff,
                                           real);
                    read_image_pixel_float(imagePtr, imageInfo, clampedX + 1,
                                           clampedY + yOff, clampedZ + zOff,
                                           bot);
                    log_error("\t(%g,%g,%g,%g)", top[0], top[1], top[2],
                              top[3]);
                    log_error(" (%g,%g,%g,%g)", real[0], real[1], real[2],
                              real[3]);
                    log_error(" (%g,%g,%g,%g)\n", bot[0], bot[1], bot[2],
                              bot[3]);
                }
            }
        }
        if (imageSampler->filter_mode != CL_FILTER_LINEAR)
        {
            if (found)
                log_error("\tValue really found in image at %d,%d,%d (%s)\n",
                          actualX, actualY, actualZ,
                          (found > 1) ? "NOT unique!!" : "unique");
            else
                log_error("\tValue not actually found in image\n");
        }
        log_error("\n");

        numClamped = -1; // We force the clamped counter to never work
        if ((--numTries) == 0) return -1;
    }
    return 0;
}


extern int filter_rounding_errors(int forceCorrectlyRoundedWrites,
                                  image_descriptor *imageInfo, float *errors);
extern void filter_undefined_bits(image_descriptor *imageInfo, char *resultPtr);

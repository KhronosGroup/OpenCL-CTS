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

#include "test_common.h"


cl_sampler create_sampler(cl_context context, image_sampler_data *sdata, bool test_mipmaps, cl_int *error) {
    cl_sampler sampler = nullptr;
    if (test_mipmaps) {
        cl_sampler_properties properties[] = {
            CL_SAMPLER_NORMALIZED_COORDS, sdata->normalized_coords,
            CL_SAMPLER_ADDRESSING_MODE, sdata->addressing_mode,
            CL_SAMPLER_FILTER_MODE, sdata->filter_mode,
            CL_SAMPLER_MIP_FILTER_MODE, sdata->filter_mode,
            0};
        sampler = clCreateSamplerWithProperties(context, properties, error);
    } else {
        sampler = clCreateSampler(context, sdata->normalized_coords, sdata->addressing_mode, sdata->filter_mode, error);
    }
    return sampler;
}

void InitFloatCoordsCommon(image_descriptor *imageInfo,
                           image_sampler_data *imageSampler, float *xOffsets,
                           float *yOffsets, float *zOffsets, float xfract,
                           float yfract, float zfract, int normalized_coords,
                           MTdata d, int lod)
{
    size_t i = 0;
    if (gDisableOffsets)
    {
        for (size_t z = 0; z < imageInfo->depth; z++)
        {
            for (size_t y = 0; y < imageInfo->height; y++)
            {
                for (size_t x = 0; x < imageInfo->width; x++, i++)
                {
                    xOffsets[i] = (float)(xfract + (double)x);
                    yOffsets[i] = (float)(yfract + (double)y);
                    zOffsets[i] = (float)(zfract + (double)z);
                }
            }
        }
    }
    else
    {
        for (size_t z = 0; z < imageInfo->depth; z++)
        {
            for (size_t y = 0; y < imageInfo->height; y++)
            {
                for (size_t x = 0; x < imageInfo->width; x++, i++)
                {
                    xOffsets[i] =
                        (float)(xfract
                                + (double)((int)x
                                           + random_in_range(-10, 10, d)));
                    yOffsets[i] =
                        (float)(yfract
                                + (double)((int)y
                                           + random_in_range(-10, 10, d)));
                    zOffsets[i] =
                        (float)(zfract
                                + (double)((int)z
                                           + random_in_range(-10, 10, d)));
                }
            }
        }
    }

    if (imageSampler->addressing_mode == CL_ADDRESS_NONE)
    {
        i = 0;
        for (size_t z = 0; z < imageInfo->depth; z++)
        {
            for (size_t y = 0; y < imageInfo->height; y++)
            {
                for (size_t x = 0; x < imageInfo->width; x++, i++)
                {
                    xOffsets[i] = (float)CLAMP((double)xOffsets[i], 0.0,
                                               (double)imageInfo->width - 1.0);
                    yOffsets[i] = (float)CLAMP((double)yOffsets[i], 0.0,
                                               (double)imageInfo->height - 1.0);
                    zOffsets[i] = (float)CLAMP((double)zOffsets[i], 0.0,
                                               (double)imageInfo->depth - 1.0);
                }
            }
        }
    }

    if (normalized_coords || gTestMipmaps)
    {
        i = 0;
        if (lod == 0)
        {
            for (size_t z = 0; z < imageInfo->depth; z++)
            {
                for (size_t y = 0; y < imageInfo->height; y++)
                {
                    for (size_t x = 0; x < imageInfo->width; x++, i++)
                    {
                        xOffsets[i] = (float)((double)xOffsets[i]
                                              / (double)imageInfo->width);
                        yOffsets[i] = (float)((double)yOffsets[i]
                                              / (double)imageInfo->height);
                        zOffsets[i] = (float)((double)zOffsets[i]
                                              / (double)imageInfo->depth);
                    }
                }
            }
        }
        else if (gTestMipmaps)
        {
            size_t width_lod, height_lod, depth_lod;

            width_lod =
                (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
            height_lod =
                (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
            depth_lod =
                (imageInfo->depth >> lod) ? (imageInfo->depth >> lod) : 1;

            for (size_t z = 0; z < depth_lod; z++)
            {
                for (size_t y = 0; y < height_lod; y++)
                {
                    for (size_t x = 0; x < width_lod; x++, i++)
                    {
                        xOffsets[i] =
                            (float)((double)xOffsets[i] / (double)width_lod);
                        yOffsets[i] =
                            (float)((double)yOffsets[i] / (double)height_lod);
                        zOffsets[i] =
                            (float)((double)zOffsets[i] / (double)depth_lod);
                    }
                }
            }
        }
    }
}

int test_read_image(cl_context context, cl_command_queue queue,
                    cl_kernel kernel, image_descriptor *imageInfo,
                    image_sampler_data *imageSampler, bool useFloatCoords,
                    ExplicitType outputType, MTdata d)
{
    int error;
    size_t threads[3];
    static int initHalf = 0;

    cl_mem_flags image_read_write_flags = CL_MEM_READ_ONLY;

    clMemWrapper xOffsets, yOffsets, zOffsets, results;
    clSamplerWrapper actualSampler;
    BufferOwningPtr<char> maxImageUseHostPtrBackingStore;

    // Create offset data
    BufferOwningPtr<cl_float> xOffsetValues(
        malloc(sizeof(cl_float) * imageInfo->width * imageInfo->height
               * imageInfo->depth));
    BufferOwningPtr<cl_float> yOffsetValues(
        malloc(sizeof(cl_float) * imageInfo->width * imageInfo->height
               * imageInfo->depth));
    BufferOwningPtr<cl_float> zOffsetValues(
        malloc(sizeof(cl_float) * imageInfo->width * imageInfo->height
               * imageInfo->depth));

    if (imageInfo->format->image_channel_data_type == CL_HALF_FLOAT)
        if (DetectFloatToHalfRoundingMode(queue)) return 1;

    BufferOwningPtr<char> imageValues;
    generate_random_image_data(imageInfo, imageValues, d);

    // Construct testing sources
    clProtectedImage protImage;
    clMemWrapper unprotImage;
    cl_mem image;

    if (gtestTypesToRun & kReadTests)
    {
        image_read_write_flags = CL_MEM_READ_ONLY;
    }
    else
    {
        image_read_write_flags = CL_MEM_READ_WRITE;
    }

    if (gMemFlagsToUse == CL_MEM_USE_HOST_PTR)
    {
        // clProtectedImage uses USE_HOST_PTR, so just rely on that for the
        // testing (via Ian) Do not use protected images for max image size test
        // since it rounds the row size to a page size
        if (gTestMaxImages)
        {
            generate_random_image_data(imageInfo,
                                       maxImageUseHostPtrBackingStore, d);
            unprotImage = create_image_3d(
                context, image_read_write_flags | CL_MEM_USE_HOST_PTR,
                imageInfo->format, imageInfo->width, imageInfo->height,
                imageInfo->depth, (gEnablePitch ? imageInfo->rowPitch : 0),
                (gEnablePitch ? imageInfo->slicePitch : 0),
                maxImageUseHostPtrBackingStore, &error);
        }
        else
        {
            error = protImage.Create(context, image_read_write_flags,
                                     imageInfo->format, imageInfo->width,
                                     imageInfo->height, imageInfo->depth);
        }
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create 3D image of size %d x %d x %d "
                      "(pitch %d, %d ) (%s)",
                      (int)imageInfo->width, (int)imageInfo->height,
                      (int)imageInfo->depth, (int)imageInfo->rowPitch,
                      (int)imageInfo->slicePitch, IGetErrorString(error));
            return error;
        }
        if (gTestMaxImages)
            image = (cl_mem)unprotImage;
        else
            image = (cl_mem)protImage;
    }
    else if (gMemFlagsToUse == CL_MEM_COPY_HOST_PTR)
    {
        // Don't use clEnqueueWriteImage; just use copy host ptr to get the data
        // in
        unprotImage = create_image_3d(
            context, image_read_write_flags | CL_MEM_COPY_HOST_PTR,
            imageInfo->format, imageInfo->width, imageInfo->height,
            imageInfo->depth, (gEnablePitch ? imageInfo->rowPitch : 0),
            (gEnablePitch ? imageInfo->slicePitch : 0), imageValues, &error);
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create 3D image of size %d x %d x %d "
                      "(pitch %d, %d ) (%s)",
                      (int)imageInfo->width, (int)imageInfo->height,
                      (int)imageInfo->depth, (int)imageInfo->rowPitch,
                      (int)imageInfo->slicePitch, IGetErrorString(error));
            return error;
        }
        image = unprotImage;
    }
    else // Either CL_MEM_ALLOC_HOST_PTR or none
    {
        // Note: if ALLOC_HOST_PTR is used, the driver allocates memory that can
        // be accessed by the host, but otherwise it works just as if no flag is
        // specified, so we just do the same thing either way
        if (!gTestMipmaps)
        {
            unprotImage = create_image_3d(
                context, image_read_write_flags | gMemFlagsToUse,
                imageInfo->format, imageInfo->width, imageInfo->height,
                imageInfo->depth, (gEnablePitch ? imageInfo->rowPitch : 0),
                (gEnablePitch ? imageInfo->slicePitch : 0), imageValues,
                &error);
            if (error != CL_SUCCESS)
            {
                log_error("ERROR: Unable to create 3D image of size %d x %d x "
                          "%d (pitch %d, %d ) (%s)",
                          (int)imageInfo->width, (int)imageInfo->height,
                          (int)imageInfo->depth, (int)imageInfo->rowPitch,
                          (int)imageInfo->slicePitch, IGetErrorString(error));
                return error;
            }
            image = unprotImage;
        }
        else
        {
            cl_image_desc image_desc = { 0 };
            image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
            image_desc.image_width = imageInfo->width;
            image_desc.image_height = imageInfo->height;
            image_desc.image_depth = imageInfo->depth;
            image_desc.num_mip_levels = imageInfo->num_mip_levels;


            unprotImage =
                clCreateImage(context, image_read_write_flags,
                              imageInfo->format, &image_desc, NULL, &error);
            if (error != CL_SUCCESS)
            {
                log_error("ERROR: Unable to create %d level mipmapped 3D image "
                          "of size %d x %d x %d (pitch %d, %d ) (%s)",
                          (int)imageInfo->num_mip_levels, (int)imageInfo->width,
                          (int)imageInfo->height, (int)imageInfo->depth,
                          (int)imageInfo->rowPitch, (int)imageInfo->slicePitch,
                          IGetErrorString(error));
                return error;
            }
            image = unprotImage;
        }
    }

    if (gMemFlagsToUse != CL_MEM_COPY_HOST_PTR)
    {
        size_t origin[4] = { 0, 0, 0, 0 };
        size_t region[3] = { imageInfo->width, imageInfo->height,
                             imageInfo->depth };

        if (gDebugTrace) log_info(" - Writing image...\n");

        if (!gTestMipmaps)
        {

            error =
                clEnqueueWriteImage(queue, image, CL_TRUE, origin, region,
                                    gEnablePitch ? imageInfo->rowPitch : 0,
                                    gEnablePitch ? imageInfo->slicePitch : 0,
                                    imageValues, 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                log_error("ERROR: Unable to write to 3D image of size %d x %d "
                          "x %d \n",
                          (int)imageInfo->width, (int)imageInfo->height,
                          (int)imageInfo->depth);
                return error;
            }
        }
        else
        {
            int nextLevelOffset = 0;

            for (int i = 0; i < imageInfo->num_mip_levels; i++)
            {
                origin[3] = i;
                error = clEnqueueWriteImage(
                    queue, image, CL_TRUE, origin, region,
                    /*gEnablePitch ? imageInfo->rowPitch :*/ 0,
                    /*gEnablePitch ? imageInfo->slicePitch :*/ 0,
                    ((char *)imageValues + nextLevelOffset), 0, NULL, NULL);
                if (error != CL_SUCCESS)
                {
                    log_error("ERROR: Unable to write to %d level mipmapped 3D "
                              "image of size %d x %d x %d\n",
                              (int)imageInfo->num_mip_levels,
                              (int)imageInfo->width, (int)imageInfo->height,
                              (int)imageInfo->depth);
                    return error;
                }
                nextLevelOffset += region[0] * region[1] * region[2]
                    * get_pixel_size(imageInfo->format);
                // Subsequent mip level dimensions keep halving
                region[0] = region[0] >> 1 ? region[0] >> 1 : 1;
                region[1] = region[1] >> 1 ? region[1] >> 1 : 1;
                region[2] = region[2] >> 1 ? region[2] >> 1 : 1;
            }
        }
    }

    xOffsets = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * imageInfo->width
                                  * imageInfo->height * imageInfo->depth,
                              xOffsetValues, &error);
    test_error(error, "Unable to create x offset buffer");
    yOffsets = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * imageInfo->width
                                  * imageInfo->height * imageInfo->depth,
                              yOffsetValues, &error);
    test_error(error, "Unable to create y offset buffer");
    zOffsets = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * imageInfo->width
                                  * imageInfo->height * imageInfo->depth,
                              zOffsetValues, &error);
    test_error(error, "Unable to create y offset buffer");
    results =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       get_explicit_type_size(outputType) * 4 * imageInfo->width
                           * imageInfo->height * imageInfo->depth,
                       NULL, &error);
    test_error(error, "Unable to create result buffer");

    // Create sampler to use
    actualSampler = create_sampler(context, imageSampler, gTestMipmaps, &error);
    test_error(error, "Unable to create image sampler");

    // Set arguments
    int idx = 0;
    error = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &image);
    test_error(error, "Unable to set kernel arguments");
    if (!gUseKernelSamplers)
    {
        error =
            clSetKernelArg(kernel, idx++, sizeof(cl_sampler), &actualSampler);
        test_error(error, "Unable to set kernel arguments");
    }
    error = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &xOffsets);
    test_error(error, "Unable to set kernel arguments");
    error = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &yOffsets);
    test_error(error, "Unable to set kernel arguments");
    error = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &zOffsets);
    test_error(error, "Unable to set kernel arguments");
    error = clSetKernelArg(kernel, idx++, sizeof(cl_mem), &results);
    test_error(error, "Unable to set kernel arguments");

    const float float_offsets[] = { 0.0f,
                                    MAKE_HEX_FLOAT(0x1.0p-30f, 0x1L, -30),
                                    0.25f,
                                    0.3f,
                                    0.5f - FLT_EPSILON / 4.0f,
                                    0.5f,
                                    0.9f,
                                    1.0f - FLT_EPSILON / 2 };
    int float_offset_count = sizeof(float_offsets) / sizeof(float_offsets[0]);
    int numTries = MAX_TRIES, numClamped = MAX_CLAMPED;
    int loopCount = 2 * float_offset_count;
    if (!useFloatCoords) loopCount = 1;
    if (gTestMaxImages)
    {
        loopCount = 1;
        log_info("Testing each size only once with pixel offsets of %g for max "
                 "sized images.\n",
                 float_offsets[0]);
    }

    // Get the maximum absolute error for this format
    double formatAbsoluteError =
        get_max_absolute_error(imageInfo->format, imageSampler);
    if (gDebugTrace)
        log_info("\tformatAbsoluteError is %e\n", formatAbsoluteError);

    if (0 == initHalf
        && imageInfo->format->image_channel_data_type == CL_HALF_FLOAT)
    {
        initHalf = CL_SUCCESS == DetectFloatToHalfRoundingMode(queue);
        if (initHalf)
        {
            log_info("Half rounding mode successfully detected.\n");
        }
    }

    int nextLevelOffset = 0;
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height,
           depth_lod = imageInfo->depth;

    // Loop over all mipmap levels, if we are testing mipmapped images.
    for (int lod = 0; (gTestMipmaps && lod < imageInfo->num_mip_levels)
         || (!gTestMipmaps && lod < 1);
         lod++)
    {
        size_t resultValuesSize = width_lod * height_lod * depth_lod
            * get_explicit_type_size(outputType) * 4;
        BufferOwningPtr<char> resultValues(malloc(resultValuesSize));
        float lod_float = (float)lod;
        if (gTestMipmaps)
        {
            // Set the lod kernel arg
            if (gDebugTrace) log_info(" - Working at mip level %d\n", lod);
            error = clSetKernelArg(kernel, idx, sizeof(float), &lod_float);
            test_error(error, "Unable to set kernel arguments");
        }

        for (int q = 0; q < loopCount; q++)
        {
            float offset = float_offsets[q % float_offset_count];

            // Init the coordinates
            InitFloatCoordsCommon(imageInfo, imageSampler, xOffsetValues,
                                  yOffsetValues, zOffsetValues,
                                  q >= float_offset_count ? -offset : offset,
                                  q >= float_offset_count ? offset : -offset,
                                  q >= float_offset_count ? -offset : offset,
                                  imageSampler->normalized_coords, d, lod);

            error =
                clEnqueueWriteBuffer(queue, xOffsets, CL_TRUE, 0,
                                     sizeof(cl_float) * imageInfo->height
                                         * imageInfo->width * imageInfo->depth,
                                     xOffsetValues, 0, NULL, NULL);
            test_error(error, "Unable to write x offsets");
            error =
                clEnqueueWriteBuffer(queue, yOffsets, CL_TRUE, 0,
                                     sizeof(cl_float) * imageInfo->height
                                         * imageInfo->width * imageInfo->depth,
                                     yOffsetValues, 0, NULL, NULL);
            test_error(error, "Unable to write y offsets");
            error =
                clEnqueueWriteBuffer(queue, zOffsets, CL_TRUE, 0,
                                     sizeof(cl_float) * imageInfo->height
                                         * imageInfo->width * imageInfo->depth,
                                     zOffsetValues, 0, NULL, NULL);
            test_error(error, "Unable to write z offsets");


            memset(resultValues, 0xff, resultValuesSize);
            clEnqueueWriteBuffer(queue, results, CL_TRUE, 0, resultValuesSize,
                                 resultValues, 0, NULL, NULL);

            // Figure out thread dimensions
            threads[0] = (size_t)width_lod;
            threads[1] = (size_t)height_lod;
            threads[2] = (size_t)depth_lod;

            // Run the kernel
            error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, threads,
                                           NULL, 0, NULL, NULL);
            test_error(error, "Unable to run kernel");

            // Get results
            error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0,
                                        width_lod * height_lod * depth_lod
                                            * get_explicit_type_size(outputType)
                                            * 4,
                                        resultValues, 0, NULL, NULL);
            test_error(error, "Unable to read results from kernel");
            if (gDebugTrace) log_info("    results read\n");

            // Validate results element by element
            char *imagePtr = (char *)imageValues + nextLevelOffset;
            /*
             * FLOAT output type
             */
            if (is_sRGBA_order(imageInfo->format->image_channel_order)
                && (outputType == kFloat))
            {
                // Validate float results
                float *resultPtr = (float *)(char *)resultValues;
                float expected[4], error = 0.0f;
                float maxErr = get_max_relative_error(
                    imageInfo->format, imageSampler, 1 /*3D*/,
                    CL_FILTER_LINEAR == imageSampler->filter_mode);

                for (size_t z = 0, j = 0; z < depth_lod; z++)
                {
                    for (size_t y = 0; y < height_lod; y++)
                    {
                        for (size_t x = 0; x < width_lod; x++, j++)
                        {
                            // Step 1: go through and see if the results verify
                            // for the pixel For the normalized case on a GPU we
                            // put in offsets to the X, Y and Z to see if we
                            // land on the right pixel. This addresses the
                            // significant inaccuracy in GPU normalization in
                            // OpenCL 1.0.
                            int checkOnlyOnePixel = 0;
                            int found_pixel = 0;
                            float offset = NORM_OFFSET;
                            if (!imageSampler->normalized_coords
                                || imageSampler->filter_mode
                                    != CL_FILTER_NEAREST
                                || NORM_OFFSET == 0
#if defined(__APPLE__)
                                // Apple requires its CPU implementation to do
                                // correctly rounded address arithmetic in all
                                // modes
                                || gDeviceType != CL_DEVICE_TYPE_GPU
#endif
                            )
                                offset = 0.0f; // Loop only once

                            for (float norm_offset_x = -offset;
                                 norm_offset_x <= offset && !found_pixel;
                                 norm_offset_x += NORM_OFFSET)
                            {
                                for (float norm_offset_y = -offset;
                                     norm_offset_y <= offset && !found_pixel;
                                     norm_offset_y += NORM_OFFSET)
                                {
                                    for (float norm_offset_z = -offset;
                                         norm_offset_z <= NORM_OFFSET
                                         && !found_pixel;
                                         norm_offset_z += NORM_OFFSET)
                                    {

                                        int hasDenormals = 0;
                                        FloatPixel maxPixel =
                                            sample_image_pixel_float_offset(
                                                imagePtr, imageInfo,
                                                xOffsetValues[j],
                                                yOffsetValues[j],
                                                zOffsetValues[j], norm_offset_x,
                                                norm_offset_y, norm_offset_z,
                                                imageSampler, expected, 0,
                                                &hasDenormals, lod);

                                        float err1 =
                                            ABS_ERROR(sRGBmap(resultPtr[0]),
                                                      sRGBmap(expected[0]));
                                        float err2 =
                                            ABS_ERROR(sRGBmap(resultPtr[1]),
                                                      sRGBmap(expected[1]));
                                        float err3 =
                                            ABS_ERROR(sRGBmap(resultPtr[2]),
                                                      sRGBmap(expected[2]));
                                        float err4 = ABS_ERROR(resultPtr[3],
                                                               expected[3]);
                                        // Clamp to the minimum absolute error
                                        // for the format
                                        if (err1 > 0
                                            && err1 < formatAbsoluteError)
                                        {
                                            err1 = 0.0f;
                                        }
                                        if (err2 > 0
                                            && err2 < formatAbsoluteError)
                                        {
                                            err2 = 0.0f;
                                        }
                                        if (err3 > 0
                                            && err3 < formatAbsoluteError)
                                        {
                                            err3 = 0.0f;
                                        }
                                        if (err4 > 0
                                            && err4 < formatAbsoluteError)
                                        {
                                            err4 = 0.0f;
                                        }
                                        float maxErr = 0.5;

                                        if (!(err1 <= maxErr)
                                            || !(err2 <= maxErr)
                                            || !(err3 <= maxErr)
                                            || !(err4 <= maxErr))
                                        {
                                            // Try flushing the denormals
                                            if (hasDenormals)
                                            {
                                                // If implementation decide to
                                                // flush subnormals to zero, max
                                                // error needs to be adjusted
                                                maxErr += 4 * FLT_MIN;

                                                maxPixel =
                                                    sample_image_pixel_float_offset(
                                                        imagePtr, imageInfo,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z,
                                                        imageSampler, expected,
                                                        0, NULL, lod);

                                                err1 = ABS_ERROR(
                                                    sRGBmap(resultPtr[0]),
                                                    sRGBmap(expected[0]));
                                                err2 = ABS_ERROR(
                                                    sRGBmap(resultPtr[1]),
                                                    sRGBmap(expected[1]));
                                                err3 = ABS_ERROR(
                                                    sRGBmap(resultPtr[2]),
                                                    sRGBmap(expected[2]));
                                                err4 = ABS_ERROR(resultPtr[3],
                                                                 expected[3]);
                                            }
                                        }

                                        found_pixel = (err1 <= maxErr)
                                            && (err2 <= maxErr)
                                            && (err3 <= maxErr)
                                            && (err4 <= maxErr);
                                    } // norm_offset_z
                                } // norm_offset_y
                            } // norm_offset_x

                            // Step 2: If we did not find a match, then print
                            // out debugging info.
                            if (!found_pixel)
                            {
                                // For the normalized case on a GPU we put in
                                // offsets to the X and Y to see if we land on
                                // the right pixel. This addresses the
                                // significant inaccuracy in GPU normalization
                                // in OpenCL 1.0.
                                checkOnlyOnePixel = 0;
                                int shouldReturn = 0;
                                for (float norm_offset_x = -offset;
                                     norm_offset_x <= offset
                                     && !checkOnlyOnePixel;
                                     norm_offset_x += NORM_OFFSET)
                                {
                                    for (float norm_offset_y = -offset;
                                         norm_offset_y <= offset
                                         && !checkOnlyOnePixel;
                                         norm_offset_y += NORM_OFFSET)
                                    {
                                        for (float norm_offset_z = -offset;
                                             norm_offset_z <= offset
                                             && !checkOnlyOnePixel;
                                             norm_offset_z += NORM_OFFSET)
                                        {

                                            int hasDenormals = 0;
                                            FloatPixel maxPixel =
                                                sample_image_pixel_float_offset(
                                                    imagePtr, imageInfo,
                                                    xOffsetValues[j],
                                                    yOffsetValues[j],
                                                    zOffsetValues[j],
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z, imageSampler,
                                                    expected, 0, &hasDenormals,
                                                    lod);

                                            float err1 =
                                                ABS_ERROR(sRGBmap(resultPtr[0]),
                                                          sRGBmap(expected[0]));
                                            float err2 =
                                                ABS_ERROR(sRGBmap(resultPtr[1]),
                                                          sRGBmap(expected[1]));
                                            float err3 =
                                                ABS_ERROR(sRGBmap(resultPtr[2]),
                                                          sRGBmap(expected[2]));
                                            float err4 = ABS_ERROR(resultPtr[3],
                                                                   expected[3]);
                                            float maxErr = 0.6;

                                            if (!(err1 <= maxErr)
                                                || !(err2 <= maxErr)
                                                || !(err3 <= maxErr)
                                                || !(err4 <= maxErr))
                                            {
                                                // Try flushing the denormals
                                                if (hasDenormals)
                                                {
                                                    // If implementation decide
                                                    // to flush subnormals to
                                                    // zero, max error needs to
                                                    // be adjusted
                                                    maxErr += 4 * FLT_MIN;

                                                    maxPixel =
                                                        sample_image_pixel_float(
                                                            imagePtr, imageInfo,
                                                            xOffsetValues[j],
                                                            yOffsetValues[j],
                                                            zOffsetValues[j],
                                                            imageSampler,
                                                            expected, 0, NULL,
                                                            lod);

                                                    err1 = ABS_ERROR(
                                                        sRGBmap(resultPtr[0]),
                                                        sRGBmap(expected[0]));
                                                    err2 = ABS_ERROR(
                                                        sRGBmap(resultPtr[1]),
                                                        sRGBmap(expected[1]));
                                                    err3 = ABS_ERROR(
                                                        sRGBmap(resultPtr[2]),
                                                        sRGBmap(expected[2]));
                                                    err4 =
                                                        ABS_ERROR(resultPtr[3],
                                                                  expected[3]);
                                                }
                                            }

                                            if (!(err1 <= maxErr)
                                                || !(err2 <= maxErr)
                                                || !(err3 <= maxErr)
                                                || !(err4 <= maxErr))
                                            {
                                                log_error(
                                                    "FAILED norm_offsets: %g , "
                                                    "%g , %g:\n",
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z);

                                                float tempOut[4];
                                                shouldReturn |=
                                                    determine_validation_error_offset<
                                                        float>(
                                                        imagePtr, imageInfo,
                                                        imageSampler, resultPtr,
                                                        expected, error,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z, j,
                                                        numTries, numClamped,
                                                        true, lod);
                                                log_error("Step by step:\n");
                                                FloatPixel temp =
                                                    sample_image_pixel_float_offset(
                                                        imagePtr, imageInfo,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z,
                                                        imageSampler, tempOut,
                                                        1 /*verbose*/,
                                                        &hasDenormals, lod);
                                                log_error(
                                                    "\tulps: %2.2f, %2.2f, "
                                                    "%2.2f, %2.2f  (max "
                                                    "allowed: %2.2f)\n\n",
                                                    Ulp_Error(resultPtr[0],
                                                              expected[0]),
                                                    Ulp_Error(resultPtr[1],
                                                              expected[1]),
                                                    Ulp_Error(resultPtr[2],
                                                              expected[2]),
                                                    Ulp_Error(resultPtr[3],
                                                              expected[3]),
                                                    Ulp_Error(
                                                        MAKE_HEX_FLOAT(
                                                            0x1.000002p0f,
                                                            0x1000002L, -24)
                                                            + maxErr,
                                                        MAKE_HEX_FLOAT(
                                                            0x1.000002p0f,
                                                            0x1000002L, -24)));
                                            }
                                            else
                                            {
                                                log_error(
                                                    "Test error: we should "
                                                    "have detected this "
                                                    "passing above.\n");
                                            }
                                        } // norm_offset_z
                                    } // norm_offset_y
                                } // norm_offset_x
                                if (shouldReturn) return 1;
                            } // if (!found_pixel)

                            resultPtr += 4;
                        }
                    }
                }
            }
            /*
             * FLOAT output type
             */
            else if (outputType == kFloat)
            {
                // Validate float results
                float *resultPtr = (float *)(char *)resultValues;
                float expected[4], error = 0.0f;
                float maxErr = get_max_relative_error(
                    imageInfo->format, imageSampler, 1 /*3D*/,
                    CL_FILTER_LINEAR == imageSampler->filter_mode);

                for (size_t z = 0, j = 0; z < depth_lod; z++)
                {
                    for (size_t y = 0; y < height_lod; y++)
                    {
                        for (size_t x = 0; x < width_lod; x++, j++)
                        {
                            // Step 1: go through and see if the results verify
                            // for the pixel For the normalized case on a GPU we
                            // put in offsets to the X, Y and Z to see if we
                            // land on the right pixel. This addresses the
                            // significant inaccuracy in GPU normalization in
                            // OpenCL 1.0.
                            int checkOnlyOnePixel = 0;
                            int found_pixel = 0;
                            float offset = NORM_OFFSET;
                            if (!imageSampler->normalized_coords
                                || imageSampler->filter_mode
                                    != CL_FILTER_NEAREST
                                || NORM_OFFSET == 0
#if defined(__APPLE__)
                                // Apple requires its CPU implementation to do
                                // correctly rounded address arithmetic in all
                                // modes
                                || gDeviceType != CL_DEVICE_TYPE_GPU
#endif
                            )
                                offset = 0.0f; // Loop only once

                            for (float norm_offset_x = -offset;
                                 norm_offset_x <= offset && !found_pixel;
                                 norm_offset_x += NORM_OFFSET)
                            {
                                for (float norm_offset_y = -offset;
                                     norm_offset_y <= offset && !found_pixel;
                                     norm_offset_y += NORM_OFFSET)
                                {
                                    for (float norm_offset_z = -offset;
                                         norm_offset_z <= NORM_OFFSET
                                         && !found_pixel;
                                         norm_offset_z += NORM_OFFSET)
                                    {

                                        int hasDenormals = 0;
                                        FloatPixel maxPixel =
                                            sample_image_pixel_float_offset(
                                                imagePtr, imageInfo,
                                                xOffsetValues[j],
                                                yOffsetValues[j],
                                                zOffsetValues[j], norm_offset_x,
                                                norm_offset_y, norm_offset_z,
                                                imageSampler, expected, 0,
                                                &hasDenormals, lod);

                                        float err1 = ABS_ERROR(resultPtr[0],
                                                               expected[0]);
                                        float err2 = ABS_ERROR(resultPtr[1],
                                                               expected[1]);
                                        float err3 = ABS_ERROR(resultPtr[2],
                                                               expected[2]);
                                        float err4 = ABS_ERROR(resultPtr[3],
                                                               expected[3]);
                                        // Clamp to the minimum absolute error
                                        // for the format
                                        if (err1 > 0
                                            && err1 < formatAbsoluteError)
                                        {
                                            err1 = 0.0f;
                                        }
                                        if (err2 > 0
                                            && err2 < formatAbsoluteError)
                                        {
                                            err2 = 0.0f;
                                        }
                                        if (err3 > 0
                                            && err3 < formatAbsoluteError)
                                        {
                                            err3 = 0.0f;
                                        }
                                        if (err4 > 0
                                            && err4 < formatAbsoluteError)
                                        {
                                            err4 = 0.0f;
                                        }
                                        float maxErr1 = MAX(
                                            maxErr * maxPixel.p[0], FLT_MIN);
                                        float maxErr2 = MAX(
                                            maxErr * maxPixel.p[1], FLT_MIN);
                                        float maxErr3 = MAX(
                                            maxErr * maxPixel.p[2], FLT_MIN);
                                        float maxErr4 = MAX(
                                            maxErr * maxPixel.p[3], FLT_MIN);

                                        if (!(err1 <= maxErr1)
                                            || !(err2 <= maxErr2)
                                            || !(err3 <= maxErr3)
                                            || !(err4 <= maxErr4))
                                        {
                                            // Try flushing the denormals
                                            if (hasDenormals)
                                            {
                                                // If implementation decide to
                                                // flush subnormals to zero, max
                                                // error needs to be adjusted
                                                maxErr1 += 4 * FLT_MIN;
                                                maxErr2 += 4 * FLT_MIN;
                                                maxErr3 += 4 * FLT_MIN;
                                                maxErr4 += 4 * FLT_MIN;

                                                maxPixel =
                                                    sample_image_pixel_float_offset(
                                                        imagePtr, imageInfo,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z,
                                                        imageSampler, expected,
                                                        0, NULL, lod);

                                                err1 = ABS_ERROR(resultPtr[0],
                                                                 expected[0]);
                                                err2 = ABS_ERROR(resultPtr[1],
                                                                 expected[1]);
                                                err3 = ABS_ERROR(resultPtr[2],
                                                                 expected[2]);
                                                err4 = ABS_ERROR(resultPtr[3],
                                                                 expected[3]);
                                            }
                                        }

                                        found_pixel = (err1 <= maxErr1)
                                            && (err2 <= maxErr2)
                                            && (err3 <= maxErr3)
                                            && (err4 <= maxErr4);
                                    } // norm_offset_z
                                } // norm_offset_y
                            } // norm_offset_x

                            // Step 2: If we did not find a match, then print
                            // out debugging info.
                            if (!found_pixel)
                            {
                                // For the normalized case on a GPU we put in
                                // offsets to the X and Y to see if we land on
                                // the right pixel. This addresses the
                                // significant inaccuracy in GPU normalization
                                // in OpenCL 1.0.
                                checkOnlyOnePixel = 0;
                                int shouldReturn = 0;
                                for (float norm_offset_x = -offset;
                                     norm_offset_x <= offset
                                     && !checkOnlyOnePixel;
                                     norm_offset_x += NORM_OFFSET)
                                {
                                    for (float norm_offset_y = -offset;
                                         norm_offset_y <= offset
                                         && !checkOnlyOnePixel;
                                         norm_offset_y += NORM_OFFSET)
                                    {
                                        for (float norm_offset_z = -offset;
                                             norm_offset_z <= offset
                                             && !checkOnlyOnePixel;
                                             norm_offset_z += NORM_OFFSET)
                                        {

                                            int hasDenormals = 0;
                                            FloatPixel maxPixel =
                                                sample_image_pixel_float_offset(
                                                    imagePtr, imageInfo,
                                                    xOffsetValues[j],
                                                    yOffsetValues[j],
                                                    zOffsetValues[j],
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z, imageSampler,
                                                    expected, 0, &hasDenormals,
                                                    lod);

                                            float err1 = ABS_ERROR(resultPtr[0],
                                                                   expected[0]);
                                            float err2 = ABS_ERROR(resultPtr[1],
                                                                   expected[1]);
                                            float err3 = ABS_ERROR(resultPtr[2],
                                                                   expected[2]);
                                            float err4 = ABS_ERROR(resultPtr[3],
                                                                   expected[3]);
                                            float maxErr1 =
                                                MAX(maxErr * maxPixel.p[0],
                                                    FLT_MIN);
                                            float maxErr2 =
                                                MAX(maxErr * maxPixel.p[1],
                                                    FLT_MIN);
                                            float maxErr3 =
                                                MAX(maxErr * maxPixel.p[2],
                                                    FLT_MIN);
                                            float maxErr4 =
                                                MAX(maxErr * maxPixel.p[3],
                                                    FLT_MIN);


                                            if (!(err1 <= maxErr1)
                                                || !(err2 <= maxErr2)
                                                || !(err3 <= maxErr3)
                                                || !(err4 <= maxErr4))
                                            {
                                                // Try flushing the denormals
                                                if (hasDenormals)
                                                {
                                                    maxErr1 += 4 * FLT_MIN;
                                                    maxErr2 += 4 * FLT_MIN;
                                                    maxErr3 += 4 * FLT_MIN;
                                                    maxErr4 += 4 * FLT_MIN;

                                                    maxPixel =
                                                        sample_image_pixel_float(
                                                            imagePtr, imageInfo,
                                                            xOffsetValues[j],
                                                            yOffsetValues[j],
                                                            zOffsetValues[j],
                                                            imageSampler,
                                                            expected, 0, NULL,
                                                            lod);

                                                    err1 =
                                                        ABS_ERROR(resultPtr[0],
                                                                  expected[0]);
                                                    err2 =
                                                        ABS_ERROR(resultPtr[1],
                                                                  expected[1]);
                                                    err3 =
                                                        ABS_ERROR(resultPtr[2],
                                                                  expected[2]);
                                                    err4 =
                                                        ABS_ERROR(resultPtr[3],
                                                                  expected[3]);
                                                }
                                            }

                                            if (!(err1 <= maxErr1)
                                                || !(err2 <= maxErr2)
                                                || !(err3 <= maxErr3)
                                                || !(err4 <= maxErr4))
                                            {
                                                log_error(
                                                    "FAILED norm_offsets: %g , "
                                                    "%g , %g:\n",
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z);

                                                float tempOut[4];
                                                shouldReturn |=
                                                    determine_validation_error_offset<
                                                        float>(
                                                        imagePtr, imageInfo,
                                                        imageSampler, resultPtr,
                                                        expected, error,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z, j,
                                                        numTries, numClamped,
                                                        true, lod);
                                                log_error("Step by step:\n");
                                                FloatPixel temp =
                                                    sample_image_pixel_float_offset(
                                                        imagePtr, imageInfo,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z,
                                                        imageSampler, tempOut,
                                                        1 /*verbose*/,
                                                        &hasDenormals, lod);
                                                log_error(
                                                    "\tulps: %2.2f, %2.2f, "
                                                    "%2.2f, %2.2f  (max "
                                                    "allowed: %2.2f)\n\n",
                                                    Ulp_Error(resultPtr[0],
                                                              expected[0]),
                                                    Ulp_Error(resultPtr[1],
                                                              expected[1]),
                                                    Ulp_Error(resultPtr[2],
                                                              expected[2]),
                                                    Ulp_Error(resultPtr[3],
                                                              expected[3]),
                                                    Ulp_Error(
                                                        MAKE_HEX_FLOAT(
                                                            0x1.000002p0f,
                                                            0x1000002L, -24)
                                                            + maxErr,
                                                        MAKE_HEX_FLOAT(
                                                            0x1.000002p0f,
                                                            0x1000002L, -24)));
                                            }
                                            else
                                            {
                                                log_error(
                                                    "Test error: we should "
                                                    "have detected this "
                                                    "passing above.\n");
                                            }
                                        } // norm_offset_z
                                    } // norm_offset_y
                                } // norm_offset_x
                                if (shouldReturn) return 1;
                            } // if (!found_pixel)

                            resultPtr += 4;
                        }
                    }
                }
            }
            /*
             * UINT output type
             */
            else if (outputType == kUInt)
            {
                // Validate unsigned integer results
                unsigned int *resultPtr = (unsigned int *)(char *)resultValues;
                unsigned int expected[4];
                float error;
                for (size_t z = 0, j = 0; z < depth_lod; z++)
                {
                    for (size_t y = 0; y < height_lod; y++)
                    {
                        for (size_t x = 0; x < width_lod; x++, j++)
                        {
                            // Step 1: go through and see if the results verify
                            // for the pixel For the normalized case on a GPU we
                            // put in offsets to the X, Y and Z to see if we
                            // land on the right pixel. This addresses the
                            // significant inaccuracy in GPU normalization in
                            // OpenCL 1.0.
                            int checkOnlyOnePixel = 0;
                            int found_pixel = 0;
                            for (float norm_offset_x = -NORM_OFFSET;
                                 norm_offset_x <= NORM_OFFSET && !found_pixel
                                 && !checkOnlyOnePixel;
                                 norm_offset_x += NORM_OFFSET)
                            {
                                for (float norm_offset_y = -NORM_OFFSET;
                                     norm_offset_y <= NORM_OFFSET
                                     && !found_pixel && !checkOnlyOnePixel;
                                     norm_offset_y += NORM_OFFSET)
                                {
                                    for (float norm_offset_z = -NORM_OFFSET;
                                         norm_offset_z <= NORM_OFFSET
                                         && !found_pixel && !checkOnlyOnePixel;
                                         norm_offset_z += NORM_OFFSET)
                                    {

                                        // If we are not on a GPU, or we are not
                                        // normalized, then only test with
                                        // offsets (0.0, 0.0) E.g., test one
                                        // pixel.
                                        if (!imageSampler->normalized_coords
                                            || gDeviceType != CL_DEVICE_TYPE_GPU
                                            || NORM_OFFSET == 0)
                                        {
                                            norm_offset_x = 0.0f;
                                            norm_offset_y = 0.0f;
                                            norm_offset_z = 0.0f;
                                            checkOnlyOnePixel = 1;
                                        }

                                        sample_image_pixel_offset<unsigned int>(
                                            imagePtr, imageInfo,
                                            xOffsetValues[j], yOffsetValues[j],
                                            zOffsetValues[j], norm_offset_x,
                                            norm_offset_y, norm_offset_z,
                                            imageSampler, expected, lod);

                                        error = errMax(
                                            errMax(abs_diff_uint(expected[0],
                                                                 resultPtr[0]),
                                                   abs_diff_uint(expected[1],
                                                                 resultPtr[1])),
                                            errMax(
                                                abs_diff_uint(expected[2],
                                                              resultPtr[2]),
                                                abs_diff_uint(expected[3],
                                                              resultPtr[3])));

                                        if (error < MAX_ERR) found_pixel = 1;
                                    } // norm_offset_z
                                } // norm_offset_y
                            } // norm_offset_x

                            // Step 2: If we did not find a match, then print
                            // out debugging info.
                            if (!found_pixel)
                            {
                                // For the normalized case on a GPU we put in
                                // offsets to the X and Y to see if we land on
                                // the right pixel. This addresses the
                                // significant inaccuracy in GPU normalization
                                // in OpenCL 1.0.
                                checkOnlyOnePixel = 0;
                                int shouldReturn = 0;
                                for (float norm_offset_x = -NORM_OFFSET;
                                     norm_offset_x <= NORM_OFFSET
                                     && !checkOnlyOnePixel;
                                     norm_offset_x += NORM_OFFSET)
                                {
                                    for (float norm_offset_y = -NORM_OFFSET;
                                         norm_offset_y <= NORM_OFFSET
                                         && !checkOnlyOnePixel;
                                         norm_offset_y += NORM_OFFSET)
                                    {
                                        for (float norm_offset_z = -NORM_OFFSET;
                                             norm_offset_z <= NORM_OFFSET
                                             && !checkOnlyOnePixel;
                                             norm_offset_z += NORM_OFFSET)
                                        {

                                            // If we are not on a GPU, or we are
                                            // not normalized, then only test
                                            // with offsets (0.0, 0.0) E.g.,
                                            // test one pixel.
                                            if (!imageSampler->normalized_coords
                                                || gDeviceType
                                                    != CL_DEVICE_TYPE_GPU
                                                || NORM_OFFSET == 0)
                                            {
                                                norm_offset_x = 0.0f;
                                                norm_offset_y = 0.0f;
                                                norm_offset_z = 0.0f;
                                                checkOnlyOnePixel = 1;
                                            }

                                            sample_image_pixel_offset<
                                                unsigned int>(
                                                imagePtr, imageInfo,
                                                xOffsetValues[j],
                                                yOffsetValues[j],
                                                zOffsetValues[j], norm_offset_x,
                                                norm_offset_y, norm_offset_z,
                                                imageSampler, expected, lod);

                                            error = errMax(
                                                errMax(
                                                    abs_diff_uint(expected[0],
                                                                  resultPtr[0]),
                                                    abs_diff_uint(
                                                        expected[1],
                                                        resultPtr[1])),
                                                errMax(
                                                    abs_diff_uint(expected[2],
                                                                  resultPtr[2]),
                                                    abs_diff_uint(
                                                        expected[3],
                                                        resultPtr[3])));

                                            if (error > MAX_ERR)
                                            {
                                                log_error(
                                                    "FAILED norm_offsets: %g , "
                                                    "%g , %g:\n",
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z);
                                                shouldReturn |=
                                                    determine_validation_error_offset<
                                                        unsigned int>(
                                                        imagePtr, imageInfo,
                                                        imageSampler, resultPtr,
                                                        expected, error,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z, j,
                                                        numTries, numClamped,
                                                        false, lod);
                                            }
                                            else
                                            {
                                                log_error(
                                                    "Test error: we should "
                                                    "have detected this "
                                                    "passing above.\n");
                                            }
                                        } // norm_offset_z
                                    } // norm_offset_y
                                } // norm_offset_x
                                if (shouldReturn) return 1;
                            } // if (!found_pixel)

                            resultPtr += 4;
                        }
                    }
                }
            }
            else
            /*
             * INT output type
             */
            {
                // Validate integer results
                int *resultPtr = (int *)(char *)resultValues;
                int expected[4];
                float error;
                for (size_t z = 0, j = 0; z < depth_lod; z++)
                {
                    for (size_t y = 0; y < height_lod; y++)
                    {
                        for (size_t x = 0; x < width_lod; x++, j++)
                        {
                            // Step 1: go through and see if the results verify
                            // for the pixel For the normalized case on a GPU we
                            // put in offsets to the X, Y and Z to see if we
                            // land on the right pixel. This addresses the
                            // significant inaccuracy in GPU normalization in
                            // OpenCL 1.0.
                            int checkOnlyOnePixel = 0;
                            int found_pixel = 0;
                            for (float norm_offset_x = -NORM_OFFSET;
                                 norm_offset_x <= NORM_OFFSET && !found_pixel
                                 && !checkOnlyOnePixel;
                                 norm_offset_x += NORM_OFFSET)
                            {
                                for (float norm_offset_y = -NORM_OFFSET;
                                     norm_offset_y <= NORM_OFFSET
                                     && !found_pixel && !checkOnlyOnePixel;
                                     norm_offset_y += NORM_OFFSET)
                                {
                                    for (float norm_offset_z = -NORM_OFFSET;
                                         norm_offset_z <= NORM_OFFSET
                                         && !found_pixel && !checkOnlyOnePixel;
                                         norm_offset_z += NORM_OFFSET)
                                    {

                                        // If we are not on a GPU, or we are not
                                        // normalized, then only test with
                                        // offsets (0.0, 0.0) E.g., test one
                                        // pixel.
                                        if (!imageSampler->normalized_coords
                                            || gDeviceType != CL_DEVICE_TYPE_GPU
                                            || NORM_OFFSET == 0)
                                        {
                                            norm_offset_x = 0.0f;
                                            norm_offset_y = 0.0f;
                                            norm_offset_z = 0.0f;
                                            checkOnlyOnePixel = 1;
                                        }

                                        sample_image_pixel_offset<int>(
                                            imagePtr, imageInfo,
                                            xOffsetValues[j], yOffsetValues[j],
                                            zOffsetValues[j], norm_offset_x,
                                            norm_offset_y, norm_offset_z,
                                            imageSampler, expected, lod);

                                        error = errMax(
                                            errMax(abs_diff_int(expected[0],
                                                                resultPtr[0]),
                                                   abs_diff_int(expected[1],
                                                                resultPtr[1])),
                                            errMax(abs_diff_int(expected[2],
                                                                resultPtr[2]),
                                                   abs_diff_int(expected[3],
                                                                resultPtr[3])));

                                        if (error < MAX_ERR) found_pixel = 1;
                                    } // norm_offset_z
                                } // norm_offset_y
                            } // norm_offset_x

                            // Step 2: If we did not find a match, then print
                            // out debugging info.
                            if (!found_pixel)
                            {
                                // For the normalized case on a GPU we put in
                                // offsets to the X and Y to see if we land on
                                // the right pixel. This addresses the
                                // significant inaccuracy in GPU normalization
                                // in OpenCL 1.0.
                                checkOnlyOnePixel = 0;
                                int shouldReturn = 0;
                                for (float norm_offset_x = -NORM_OFFSET;
                                     norm_offset_x <= NORM_OFFSET
                                     && !checkOnlyOnePixel;
                                     norm_offset_x += NORM_OFFSET)
                                {
                                    for (float norm_offset_y = -NORM_OFFSET;
                                         norm_offset_y <= NORM_OFFSET
                                         && !checkOnlyOnePixel;
                                         norm_offset_y += NORM_OFFSET)
                                    {
                                        for (float norm_offset_z = -NORM_OFFSET;
                                             norm_offset_z <= NORM_OFFSET
                                             && !checkOnlyOnePixel;
                                             norm_offset_z += NORM_OFFSET)
                                        {

                                            // If we are not on a GPU, or we are
                                            // not normalized, then only test
                                            // with offsets (0.0, 0.0) E.g.,
                                            // test one pixel.
                                            if (!imageSampler->normalized_coords
                                                || gDeviceType
                                                    != CL_DEVICE_TYPE_GPU
                                                || NORM_OFFSET == 0
                                                || NORM_OFFSET == 0
                                                || NORM_OFFSET == 0)
                                            {
                                                norm_offset_x = 0.0f;
                                                norm_offset_y = 0.0f;
                                                norm_offset_z = 0.0f;
                                                checkOnlyOnePixel = 1;
                                            }

                                            sample_image_pixel_offset<int>(
                                                imagePtr, imageInfo,
                                                xOffsetValues[j],
                                                yOffsetValues[j],
                                                zOffsetValues[j], norm_offset_x,
                                                norm_offset_y, norm_offset_z,
                                                imageSampler, expected, lod);

                                            error = errMax(
                                                errMax(
                                                    abs_diff_int(expected[0],
                                                                 resultPtr[0]),
                                                    abs_diff_int(expected[1],
                                                                 resultPtr[1])),
                                                errMax(
                                                    abs_diff_int(expected[2],
                                                                 resultPtr[2]),
                                                    abs_diff_int(
                                                        expected[3],
                                                        resultPtr[3])));

                                            if (error > MAX_ERR)
                                            {
                                                log_error(
                                                    "FAILED norm_offsets: %g , "
                                                    "%g , %g:\n",
                                                    norm_offset_x,
                                                    norm_offset_y,
                                                    norm_offset_z);
                                                shouldReturn |=
                                                    determine_validation_error_offset<
                                                        int>(
                                                        imagePtr, imageInfo,
                                                        imageSampler, resultPtr,
                                                        expected, error,
                                                        xOffsetValues[j],
                                                        yOffsetValues[j],
                                                        zOffsetValues[j],
                                                        norm_offset_x,
                                                        norm_offset_y,
                                                        norm_offset_z, j,
                                                        numTries, numClamped,
                                                        false, lod);
                                            }
                                            else
                                            {
                                                log_error(
                                                    "Test error: we should "
                                                    "have detected this "
                                                    "passing above.\n");
                                            }
                                        } // norm_offset_z
                                    } // norm_offset_y
                                } // norm_offset_x
                                if (shouldReturn) return 1;
                            } // if (!found_pixel)

                            resultPtr += 4;
                        }
                    }
                }
            }
        }
        {
            nextLevelOffset += width_lod * height_lod * depth_lod
                * get_pixel_size(imageInfo->format);
            width_lod = (width_lod >> 1) ? (width_lod >> 1) : 1;
            height_lod = (height_lod >> 1) ? (height_lod >> 1) : 1;
            depth_lod = (depth_lod >> 1) ? (depth_lod >> 1) : 1;
        }
    }

    return numTries != MAX_TRIES || numClamped != MAX_CLAMPED;
}
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
#ifndef test_conformance_clImageHelper_h
#define test_conformance_clImageHelper_h

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include "errorHelpers.h"


  // helper function to replace clCreateImage2D , to make the existing code use
  // the functions of version 1.2 and veriosn 1.1  respectively

  static inline cl_mem create_image_2d  (cl_context context,
                           cl_mem_flags flags,
                           const cl_image_format *image_format,
                           size_t image_width,
                           size_t image_height,
                           size_t image_row_pitch,
                           void *host_ptr,
                           cl_int *errcode_ret)
  {
    cl_mem mImage = NULL;

#ifdef CL_VERSION_1_2
    cl_image_desc image_desc_dest;
    image_desc_dest.image_type = CL_MEM_OBJECT_IMAGE2D;;
    image_desc_dest.image_width = image_width;
    image_desc_dest.image_height = image_height;
    image_desc_dest.image_depth= 0;// not usedfor 2d
    image_desc_dest.image_array_size = 0;// not used for 2d
    image_desc_dest.image_row_pitch = image_row_pitch;
    image_desc_dest.image_slice_pitch = 0;
    image_desc_dest.num_mip_levels = 0;
    image_desc_dest.num_samples = 0;
    image_desc_dest.mem_object = NULL;// no image type of CL_MEM_OBJECT_IMAGE1D_BUFFER in CL_VERSION_1_1, so always is NULL
    mImage = clCreateImage( context, flags, image_format, &image_desc_dest, host_ptr, errcode_ret );
    if (errcode_ret && (*errcode_ret)) {
      // Log an info message and rely on the calling function to produce an error
      // if necessary.
      log_info("clCreateImage failed (%d)\n", *errcode_ret);
    }

#else
    mImage = clCreateImage2D( context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret );
    if (errcode_ret && (*errcode_ret)) {
      // Log an info message and rely on the calling function to produce an error
      // if necessary.
      log_info("clCreateImage2D failed (%d)\n", *errcode_ret);
    }
#endif

    return mImage;
  }

    // helper function to replace clCreateImage2D , to make the existing code use
    // the functions of version 1.2 and veriosn 1.1  respectively

    static inline cl_mem create_image_2d_buffer  (cl_context context,
                                    cl_mem_flags flags,
                                    const cl_image_format *image_format,
                                    size_t image_width,
                                    size_t image_height,
                                    size_t image_row_pitch,
                                    cl_mem buffer,
                                    cl_int *errcode_ret)
    {
        cl_mem mImage = NULL;

        cl_image_desc image_desc_dest;
        image_desc_dest.image_type = CL_MEM_OBJECT_IMAGE2D;;
        image_desc_dest.image_width = image_width;
        image_desc_dest.image_height = image_height;
        image_desc_dest.image_depth= 0;// not usedfor 2d
        image_desc_dest.image_array_size = 0;// not used for 2d
        image_desc_dest.image_row_pitch = image_row_pitch;
        image_desc_dest.image_slice_pitch = 0;
        image_desc_dest.num_mip_levels = 0;
        image_desc_dest.num_samples = 0;
        image_desc_dest.mem_object = buffer;
        mImage = clCreateImage( context, flags, image_format, &image_desc_dest, NULL, errcode_ret );
        if (errcode_ret && (*errcode_ret)) {
            // Log an info message and rely on the calling function to produce an error
            // if necessary.
            log_info("clCreateImage failed (%d)\n", *errcode_ret);
        }

        return mImage;
    }



  static inline cl_mem create_image_3d (cl_context context,
                          cl_mem_flags flags,
                          const cl_image_format *image_format,
                          size_t image_width,
                          size_t image_height,
                          size_t image_depth,
                          size_t image_row_pitch,
                          size_t image_slice_pitch,
                          void *host_ptr,
                          cl_int *errcode_ret)
  {
    cl_mem mImage;

#ifdef CL_VERSION_1_2
    cl_image_desc image_desc;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    image_desc.image_width = image_width;
    image_desc.image_height = image_height;
    image_desc.image_depth = image_depth;
    image_desc.image_array_size = 0;// not used for one image
    image_desc.image_row_pitch = image_row_pitch;
    image_desc.image_slice_pitch = image_slice_pitch;
    image_desc.num_mip_levels = 0;
    image_desc.num_samples = 0;
    image_desc.mem_object = NULL; // no image type of CL_MEM_OBJECT_IMAGE1D_BUFFER in CL_VERSION_1_1, so always is NULL
    mImage = clCreateImage( context,
                           flags,
                           image_format,
                           &image_desc,
                           host_ptr,
                           errcode_ret );
    if (errcode_ret && (*errcode_ret)) {
      // Log an info message and rely on the calling function to produce an error
      // if necessary.
      log_info("clCreateImage failed (%d)\n", *errcode_ret);
    }

#else
    mImage = clCreateImage3D( context,
                             flags, image_format,
                             image_width,
                             image_height,
                             image_depth,
                             image_row_pitch,
                             image_slice_pitch,
                             host_ptr,
                             errcode_ret );
    if (errcode_ret && (*errcode_ret)) {
      // Log an info message and rely on the calling function to produce an error
      // if necessary.
      log_info("clCreateImage3D failed (%d)\n", *errcode_ret);
    }
#endif

    return mImage;
  }

    static inline cl_mem create_image_2d_array (cl_context context,
                                   cl_mem_flags flags,
                                   const cl_image_format *image_format,
                                   size_t image_width,
                                   size_t image_height,
                                   size_t image_array_size,
                                   size_t image_row_pitch,
                                   size_t image_slice_pitch,
                                   void *host_ptr,
                                   cl_int *errcode_ret)
    {
        cl_mem mImage;

        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        image_desc.image_width = image_width;
        image_desc.image_height = image_height;
        image_desc.image_depth = 1;
        image_desc.image_array_size = image_array_size;
        image_desc.image_row_pitch = image_row_pitch;
        image_desc.image_slice_pitch = image_slice_pitch;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.mem_object = NULL;
        mImage = clCreateImage( context,
                               flags,
                               image_format,
                               &image_desc,
                               host_ptr,
                               errcode_ret );
        if (errcode_ret && (*errcode_ret)) {
            // Log an info message and rely on the calling function to produce an error
            // if necessary.
            log_info("clCreateImage failed (%d)\n", *errcode_ret);
        }

        return mImage;
    }

    static inline cl_mem create_image_1d_array (cl_context context,
                                         cl_mem_flags flags,
                                         const cl_image_format *image_format,
                                         size_t image_width,
                                         size_t image_array_size,
                                         size_t image_row_pitch,
                                         size_t image_slice_pitch,
                                         void *host_ptr,
                                         cl_int *errcode_ret)
    {
        cl_mem mImage;

        cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
        image_desc.image_width = image_width;
        image_desc.image_height = 1;
        image_desc.image_depth = 1;
        image_desc.image_array_size = image_array_size;
        image_desc.image_row_pitch = image_row_pitch;
        image_desc.image_slice_pitch = image_slice_pitch;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.mem_object = NULL;
        mImage = clCreateImage( context,
                               flags,
                               image_format,
                               &image_desc,
                               host_ptr,
                               errcode_ret );
        if (errcode_ret && (*errcode_ret)) {
            // Log an info message and rely on the calling function to produce an error
            // if necessary.
            log_info("clCreateImage failed (%d)\n", *errcode_ret);
        }

        return mImage;
    }

    static inline cl_mem create_image_1d (cl_context context,
                                   cl_mem_flags flags,
                                   const cl_image_format *image_format,
                                   size_t image_width,
                                   size_t image_row_pitch,
                                   void *host_ptr,
                                   cl_mem buffer,
                                   cl_int *errcode_ret)
    {
        cl_mem mImage;

        cl_image_desc image_desc;
        image_desc.image_type = buffer ? CL_MEM_OBJECT_IMAGE1D_BUFFER: CL_MEM_OBJECT_IMAGE1D;
        image_desc.image_width = image_width;
        image_desc.image_height = 1;
        image_desc.image_depth = 1;
        image_desc.image_row_pitch = image_row_pitch;
        image_desc.image_slice_pitch = 0;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.mem_object = buffer;
        mImage = clCreateImage( context,
                               flags,
                               image_format,
                               &image_desc,
                               host_ptr,
                               errcode_ret );
        if (errcode_ret && (*errcode_ret)) {
            // Log an info message and rely on the calling function to produce an error
            // if necessary.
            log_info("clCreateImage failed (%d)\n", *errcode_ret);
        }

        return mImage;
    }


#endif

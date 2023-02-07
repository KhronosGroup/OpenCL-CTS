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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

int test_imagearraycopy_single_format(cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format)
{
  cl_uchar    *imgptr, *bufptr;
  clMemWrapper      image, buffer;
  int        img_width = 512;
  int        img_height = 512;
  size_t    elem_size;
  size_t    buffer_size;
  int        i;
  cl_int          err;
  MTdata          d;
  cl_event  copyevent;

  log_info("Testing %s %s\n", GetChannelOrderName(format->image_channel_order), GetChannelTypeName(format->image_channel_data_type));

  image = create_image_2d(context, CL_MEM_READ_WRITE, format, img_width,
                          img_height, 0, NULL, &err);
  test_error(err, "create_image_2d failed");

  err = clGetImageInfo(image, CL_IMAGE_ELEMENT_SIZE, sizeof(size_t), &elem_size, NULL);
  test_error(err, "clGetImageInfo failed");

  buffer_size = sizeof(cl_uchar) * elem_size * img_width * img_height;

  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
  test_error(err, "clCreateBuffer failed");

  d = init_genrand( gRandomSeed );
  imgptr = (cl_uchar*)malloc(buffer_size);
  for (i=0; i<(int)buffer_size; i++) {
     imgptr[i] = (cl_uchar)genrand_int32(d);
  }
  free_mtdata(d); d = NULL;

  size_t origin[3]={0,0,0}, region[3]={img_width,img_height,1};
  err = clEnqueueWriteImage( queue, image, CL_TRUE, origin, region, 0, 0, imgptr, 0, NULL, NULL );
  test_error(err, "clEnqueueWriteBuffer failed");

  err = clEnqueueCopyImageToBuffer( queue, image, buffer, origin, region, 0, 0, NULL, &copyevent );
  test_error(err, "clEnqueueCopyImageToBuffer failed");

  bufptr = (cl_uchar*)malloc(buffer_size);

  err = clEnqueueReadBuffer( queue, buffer, CL_TRUE, 0, buffer_size, bufptr, 1, &copyevent, NULL);
  test_error(err, "clEnqueueReadBuffer failed");

  err = clReleaseEvent(copyevent);
  test_error(err, "clReleaseEvent failed");

  if (memcmp(imgptr, bufptr, buffer_size) != 0) {
    log_error( "ERROR: Results did not validate!\n" );
    unsigned char * inchar = (unsigned char*)imgptr;
    unsigned char * outchar = (unsigned char*)bufptr;
    int failuresPrinted = 0;
    int i;
    for (i=0; i< (int)buffer_size; i+=(int)elem_size) {
        int failed = 0;
        int j;
        for (j=0; j<(int)elem_size; j++)
            if (inchar[i+j] != outchar[i+j])
                failed = 1;
        char values[4096];
        values[0] = 0;
        if (failed) {
            sprintf(values + strlen(values), "%d(0x%x) -> expected [", i, i);
            int j;
            for (j=0; j<(int)elem_size; j++)
                sprintf(values + strlen( values), "0x%02x ", inchar[i+j]);
            sprintf(values + strlen(values), "] != actual [");
            for (j=0; j<(int)elem_size; j++)
                sprintf(values + strlen( values), "0x%02x ", outchar[i+j]);
            sprintf(values + strlen(values), "]");
            log_error("%s\n", values);
            failuresPrinted++;
        }
        if (failuresPrinted > 5) {
            log_error("Not printing further failures...\n");
            break;
        }
    }
    err = -1;
  }

  free(imgptr);
  free(bufptr);

  if (err)
    log_error("IMAGE to ARRAY copy test failed for image_channel_order=0x%lx and image_channel_data_type=0x%lx\n",
              (unsigned long)format->image_channel_order, (unsigned long)format->image_channel_data_type);

  return err;
}

int test_imagearraycopy(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  cl_int          err;
  cl_image_format *formats;
  cl_uint         num_formats;
  cl_uint         i;

  PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

  err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &num_formats);
  test_error(err, "clGetSupportedImageFormats failed");

  formats = (cl_image_format *)malloc(num_formats * sizeof(cl_image_format));

  err = clGetSupportedImageFormats(context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, num_formats, formats, NULL);
  test_error(err, "clGetSupportedImageFormats failed");

  for (i = 0; i < num_formats; i++) {
    err |= test_imagearraycopy_single_format(device, context, queue, &formats[i]);
  }

  free(formats);
  if (err)
    log_error("IMAGE to ARRAY copy test failed\n");
  else
    log_info("IMAGE to ARRAY copy test passed\n");

  return err;
}

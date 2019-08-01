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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

const cl_mem_flags flag_set[] = {
  CL_MEM_ALLOC_HOST_PTR,
  CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
  CL_MEM_USE_HOST_PTR,
  CL_MEM_COPY_HOST_PTR,
  0
};
const char* flag_set_names[] = {
  "CL_MEM_ALLOC_HOST_PTR",
  "CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR",
  "CL_MEM_USE_HOST_PTR",
  "CL_MEM_COPY_HOST_PTR",
  "0"
};

int test_enqueue_map_buffer(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    const size_t bufferSize = 256*256;
    int src_flag_id;
    MTdata d = init_genrand( gRandomSeed );
    cl_char *initialData = (cl_char*)malloc(bufferSize);
    cl_char *finalData = (cl_char*)malloc(bufferSize);

    for (src_flag_id=0; src_flag_id < sizeof(flag_set)/sizeof(flag_set[0]); src_flag_id++)
    {
        clMemWrapper memObject;
        log_info("Testing with cl_mem_flags src: %s\n", flag_set_names[src_flag_id]);

        generate_random_data( kChar, (unsigned int)bufferSize, d, initialData );

        if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
            memObject = clCreateBuffer(context, flag_set[src_flag_id],  bufferSize * sizeof( cl_char ), initialData, &error);
        else
            memObject = clCreateBuffer(context, flag_set[src_flag_id],  bufferSize * sizeof( cl_char ), NULL, &error);
        test_error( error, "Unable to create testing buffer" );

        if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
        {
            error = clEnqueueWriteBuffer(queue, memObject, CL_TRUE, 0, bufferSize * sizeof( cl_char ), initialData, 0, NULL, NULL);
            test_error( error, "clEnqueueWriteBuffer failed");
        }

        for( int i = 0; i < 128; i++ )
        {

          size_t offset = (size_t)random_in_range( 0, (int)bufferSize - 1, d );
          size_t length = (size_t)random_in_range( 1, (int)( bufferSize - offset ), d );

          cl_char *mappedRegion = (cl_char *)clEnqueueMapBuffer( queue, memObject, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                                                offset, length, 0, NULL, NULL, &error );
          if( error != CL_SUCCESS )
          {
            print_error( error, "clEnqueueMapBuffer call failed" );
            log_error( "\tOffset: %d  Length: %d\n", (int)offset, (int)length );
            free( initialData );
            free( finalData );
            free_mtdata(d);
            return -1;
          }

          // Write into the region
          for( size_t j = 0; j < length; j++ )
          {
            cl_char spin = (cl_char)genrand_int32( d );

            // Test read AND write in one swipe
            cl_char value = mappedRegion[ j ];
            value = spin - value;
            mappedRegion[ j ] = value;

            // Also update the initial data array
            value = initialData[ offset + j ];
            value = spin - value;
            initialData[ offset + j ] = value;
          }

          // Unmap
          error = clEnqueueUnmapMemObject( queue, memObject, mappedRegion, 0, NULL, NULL );
          test_error( error, "Unable to unmap buffer" );
        }

        // Final validation: read actual values of buffer and compare against our reference
        error = clEnqueueReadBuffer( queue, memObject, CL_TRUE, 0, sizeof( cl_char ) * bufferSize, finalData, 0, NULL, NULL );
        test_error( error, "Unable to read results" );

        for( size_t q = 0; q < bufferSize; q++ )
        {
            if( initialData[ q ] != finalData[ q ] )
            {
                log_error( "ERROR: Sample %d did not validate! Got %d, expected %d\n", (int)q, (int)finalData[ q ], (int)initialData[ q ] );
                free( initialData );
                free( finalData );
                free_mtdata(d);
                return -1;
            }
        }
    } // cl_mem flags

    free( initialData );
    free( finalData );
    free_mtdata(d);

    return 0;
}

int test_enqueue_map_image(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT32 };
    const size_t imageSize = 256;
    int src_flag_id;
    cl_uint *initialData;
    cl_uint *finalData;
    MTdata  d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    initialData = (cl_uint*)malloc(imageSize * imageSize * 4 *sizeof(cl_uint));
    finalData = (cl_uint*)malloc(imageSize * imageSize * 4 *sizeof(cl_uint));

    if( !is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, &format ) )
    {
        log_error( "ERROR: Test requires basic OpenCL 1.0 format CL_RGBA:CL_UNSIGNED_INT32, which is unsupported by this device!\n" );
        free(initialData);
        free(finalData);
        return -1;
    }

    d = init_genrand( gRandomSeed );
  for (src_flag_id=0; src_flag_id < sizeof(flag_set)/sizeof(flag_set[0]); src_flag_id++) {
    clMemWrapper memObject;
    log_info("Testing with cl_mem_flags src: %s\n", flag_set_names[src_flag_id]);

    generate_random_data( kUInt, (unsigned int)( imageSize * imageSize ), d, initialData );

    if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
      memObject = create_image_2d( context, CL_MEM_READ_WRITE | flag_set[src_flag_id], &format,
                                  imageSize, imageSize, 0, initialData, &error );
    else
      memObject = create_image_2d( context, CL_MEM_READ_WRITE | flag_set[src_flag_id], &format,
                                  imageSize, imageSize, 0, NULL, &error );
    test_error( error, "Unable to create testing buffer" );

    if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR)) {
      size_t write_origin[3]={0,0,0}, write_region[3]={imageSize, imageSize, 1};
      error = clEnqueueWriteImage(queue, memObject, CL_TRUE, write_origin, write_region, NULL, NULL, initialData, 0, NULL, NULL);
      test_error( error, "Unable to write to testing buffer" );
    }

    for( int i = 0; i < 128; i++ )
    {

      size_t offset[3], region[3];
      size_t rowPitch;

      offset[ 0 ] = (size_t)random_in_range( 0, (int)imageSize - 1, d );
      region[ 0 ] = (size_t)random_in_range( 1, (int)( imageSize - offset[ 0 ] - 1), d );
      offset[ 1 ] = (size_t)random_in_range( 0, (int)imageSize - 1, d );
      region[ 1 ] = (size_t)random_in_range( 1, (int)( imageSize - offset[ 1 ] - 1), d );
      offset[ 2 ] = 0;
      region[ 2 ] = 1;
      cl_uint *mappedRegion = (cl_uint *)clEnqueueMapImage( queue, memObject, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                                           offset, region, &rowPitch, NULL, 0, NULL, NULL, &error );
      if( error != CL_SUCCESS )
      {
        print_error( error, "clEnqueueMapImage call failed" );
        log_error( "\tOffset: %d,%d  Region: %d,%d\n", (int)offset[0], (int)offset[1], (int)region[0], (int)region[1] );
        free(initialData);
        free(finalData);
        free_mtdata(d);
        return -1;
      }

      // Write into the region
      cl_uint *mappedPtr = mappedRegion;
      for( size_t y = 0; y < region[ 1 ]; y++ )
      {
        for( size_t x = 0; x < region[ 0 ] * 4; x++ )
        {
          cl_int spin = (cl_int)random_in_range( 16, 1024, d );

          cl_int value;
          // Test read AND write in one swipe
          value = mappedPtr[ ( y * rowPitch/sizeof(cl_uint) ) + x ];
          value = spin - value;
          mappedPtr[ ( y * rowPitch/sizeof(cl_uint) ) + x ] = value;

          // Also update the initial data array
          value = initialData[ ( ( offset[ 1 ] + y ) * imageSize + offset[ 0 ] ) * 4 + x ];
          value = spin - value;
          initialData[ ( ( offset[ 1 ] + y ) * imageSize + offset[ 0 ] ) * 4 + x ] = value;
        }
      }

      // Unmap
      error = clEnqueueUnmapMemObject( queue, memObject, mappedRegion, 0, NULL, NULL );
      test_error( error, "Unable to unmap buffer" );
    }

    // Final validation: read actual values of buffer and compare against our reference
    size_t finalOrigin[3] = { 0, 0, 0 }, finalRegion[3] = { imageSize, imageSize, 1 };
    error = clEnqueueReadImage( queue, memObject, CL_TRUE, finalOrigin, finalRegion, 0, 0, finalData, 0, NULL, NULL );
    test_error( error, "Unable to read results" );

    for( size_t q = 0; q < imageSize * imageSize * 4; q++ )
    {
      if( initialData[ q ] != finalData[ q ] )
      {
        log_error( "ERROR: Sample %d (coord %d,%d) did not validate! Got %d, expected %d\n", (int)q, (int)( ( q / 4 ) % imageSize ), (int)( ( q / 4 ) / imageSize ),
                                    (int)finalData[ q ], (int)initialData[ q ] );
        free(initialData);
        free(finalData);
        free_mtdata(d);
        return -1;
      }
    }
  } // cl_mem_flags

    free(initialData);
    free(finalData);
    free_mtdata(d);
    return 0;
}



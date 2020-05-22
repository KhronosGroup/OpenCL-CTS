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
#ifndef test_conformance_checker_Image_MEM_HOST_WRITE_ONLY_h
#define test_conformance_checker_Image_MEM_HOST_WRITE_ONLY_h

#include "checker_image_mem_host_read_only.hpp"

template < class T> class cImage_check_mem_host_write_only : public cImage_check_mem_host_read_only<T>
{

public:
  cImage_check_mem_host_write_only(cl_device_id deviceID, cl_context context, cl_command_queue queue)
  : cImage_check_mem_host_read_only <T> (deviceID, context, queue)
  {
  }

  ~cImage_check_mem_host_write_only() {};

  clMemWrapper m_Image_2;

  cl_int verify_RW_Image();
  cl_int verify_RW_Image_Mapping();

  cl_int Setup_Test_Environment();
  cl_int update_host_mem_2();

  cl_int verify_data();
};

template < class T >
cl_int cImage_check_mem_host_write_only<T>::Setup_Test_Environment()
{
  int all= this->get_image_elements();

  T vv2 = 0;
  this->host_m_2.Init( all, vv2);
  vv2 = TEST_VALUE;
  this->host_m_0.Init( all, vv2);

  cl_int err = CL_SUCCESS;
  this->m_Image_2 = clCreateImage(this->m_context,
                                  CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &( this-> m_cl_image_format), &(this->m_cl_Image_desc),
                                  this->host_m_2.pData, &err);
  test_error(err, "clCreateImage error");

  return err;
}

// Copy image data from a write_only image to a read_write image and read the
// contents.
template < class T >
cl_int cImage_check_mem_host_write_only< T >::update_host_mem_2()
{
  size_t orig[3] = {0, 0, 0};
  size_t img_region[3] = {0, 0, 0};
  img_region[0] = this->m_cl_Image_desc.image_width;
  img_region[1] = this->m_cl_Image_desc.image_height;
  img_region[2] = this->m_cl_Image_desc.image_depth;

  cl_event event;
  cl_int err = CL_SUCCESS;
  err = clEnqueueCopyImage(this->m_queue,
                           this->m_Image,
                           this->m_Image_2,
                           orig,
                           orig,
                           img_region,
                           0, NULL, &event);
  test_error(err, "clEnqueueCopyImage error");

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  this->host_m_2.Set_to_zero();

  err = clEnqueueReadImage(this->m_queue, this->m_Image_2, this->m_blocking,
                           this->buffer_origin, this->region,
                           this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
                           this->host_m_2.pData, 0, NULL, &event);
  test_error(err, "clEnqueueReadImage error");

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  return err;
}

template < class T >
cl_int cImage_check_mem_host_write_only<T>::verify_data()
{
  cl_int err = CL_SUCCESS;
  if (!this->host_m_1.Equal_rect_from_orig(this->host_m_2, this->buffer_origin,
                                           this->region, this->host_row_pitch,
                                           this->host_slice_pitch)) {
    log_error("Image and host data difference found\n");
    return FAILURE;
  }

  int total = (int)(this->region[0] * this->region[1] * this->region[2]);
  T v = TEST_VALUE;
  int tot = (int)(this->host_m_2.Count(v));
  if(tot != total) {
    log_error("Image data content difference found\n");
    return FAILURE;
  }

  return err;
}

template < class T >
cl_int cImage_check_mem_host_write_only<T>::verify_RW_Image()
{
  cl_int err = CL_SUCCESS;

  this->Init_rect();

  cl_event event;
  size_t img_orig[3] = {0, 0, 0};
  size_t img_region[3] = {0, 0, 0};
  img_region[0] = this->m_cl_Image_desc.image_width;
  img_region[1] = this->m_cl_Image_desc.image_height;
  img_region[2] = this->m_cl_Image_desc.image_depth;

  int color[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  err = clEnqueueFillImage(this->m_queue,
                           this->m_Image,
                           &color,
                           img_orig, img_region,
                           0, NULL, &event); // Fill the buffer with data

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }
  test_error(err, "clEnqueueFillImage error");

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  T v = TEST_VALUE;

  err= clEnqueueWriteImage(this->m_queue, this->m_Image, this->m_blocking,
                           this->buffer_origin, this->region,
                           this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
                           this->host_m_0.pData, 0, NULL, &event);
  test_error(err, "clEnqueueWriteImage error"); // Test writing to buffer

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  update_host_mem_2(); // Read buffer contents into mem_2

  err = this->verify_data(); // Compare the contents of mem_2 and mem_1,
                             // mem_1 is same as mem_0 in setup test environment
  test_error(err, "verify_data error");

  v = 0;
  this->host_m_2.Set_to(v);
  err = clEnqueueReadImage(this->m_queue, this->m_Image, this->m_blocking,
                           this->buffer_origin, this->region,
                           this->buffer_row_pitch_bytes, this->buffer_slice_pitch_bytes,
                           this->host_m_1.pData, 0, NULL, &event);

  if (err == CL_SUCCESS){
    log_error("Calling clEnqueueReadImage on a memory object created with the CL_MEM_HOST_WRITE_ONLY flag should not return CL_SUCCESS\n");
    err = FAILURE;
    return FAILURE;

  } else {
    log_info("Test succeeded\n\n");
    err = CL_SUCCESS;
  }

  /* Qualcomm fix: 12506 Do not wait on invalid event/ no need for syncronization calls after clEnqueueReadImage fails
   *
   * The call to clEnqueueReadImage fails as expected and returns an invalid event on
   * which clWaitForEvents cannot be called. (It will rightly fail with a CL_INVALID_EVENT error)
   * Further, we don't need to do any additional flushes or finishes here since we were in sync
   * before the (failing) call to clEnqueueReadImage

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, " clWaitForEvents error")
  }
  Qualcomm fix: end*/

  return err;
}

template < class T >
cl_int cImage_check_mem_host_write_only<T>::verify_RW_Image_Mapping()
{
  this->Init_rect();

  cl_event event;
  size_t img_orig[3] = {0, 0, 0};
  size_t img_region[3] = {0, 0, 0};
  img_region[0] = this->m_cl_Image_desc.image_width;
  img_region[1] = this->m_cl_Image_desc.image_height;
  img_region[2] = this->m_cl_Image_desc.image_depth;

  int color[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  cl_int err = CL_SUCCESS;


  // Fill image with pattern
  err = clEnqueueFillImage(this->m_queue, this->m_Image,
                           &color, img_orig, img_region,
                           0, NULL, &event);

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  // Map image for writing
  T* dataPtr = (T*) clEnqueueMapImage(this->m_queue, this->m_Image,
                                      this->m_blocking, CL_MAP_WRITE,
                                      this->buffer_origin, this->region,
                                      &(this->buffer_row_pitch_bytes),
                                      &(this->buffer_slice_pitch_bytes),
                                      0, NULL, &event, &err);
  test_error(err, "clEnqueueMapImage CL_MAP_WRITE pointer error");

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  // Verify map pointer
  err = this->verify_mapping_ptr(dataPtr);
  test_error(err, "clEnqueueMapImage CL_MAP_WRITE pointer error");

  // Verify mapped data

  // The verify_data_with_offset method below compares dataPtr against
  // this->host_m_2.pData. The comparison should start at origin {0, 0, 0}.
  update_host_mem_2();

  // Check the content of mem and host_ptr
  size_t offset[3] = {0, 0, 0};
  err = cImage_check_mem_host_read_only<T>::verify_data_with_offset(dataPtr,
                                                                    offset);
  test_error(err, "verify_data error");

  // Unmap memory object
  err = clEnqueueUnmapMemObject(this->m_queue, this->m_Image, dataPtr,
                                0, NULL, &event);
  test_error(err, "clEnqueueUnmapMemObject error");

  if (!this->m_blocking) {
    err = clWaitForEvents(1, &event);
    test_error(err, "clWaitForEvents error");
  }

  err = clReleaseEvent(event);
  test_error(err, "clReleaseEvent error");

  dataPtr = (T*) clEnqueueMapImage(this->m_queue, this->m_Image, this->m_blocking,
                                   CL_MAP_READ,
                                   this->buffer_origin, this->region,
                                   &(this->buffer_row_pitch_bytes),
                                   &(this->buffer_slice_pitch_bytes),
                                   0, NULL, &event, &err);

  if (err == CL_SUCCESS) {
    log_error("Calling clEnqueueMapImage (CL_MAP_READ) on a memory object created with the CL_MEM_HOST_WRITE_ONLY flag should not return CL_SUCCESS\n");
    err = FAILURE;
    return FAILURE;

  } else {
    log_info("Test succeeded\n\n");
    err = CL_SUCCESS;
  }

  return err;
}

#endif

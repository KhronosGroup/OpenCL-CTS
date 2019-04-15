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
#include "common.h"

const char *byte_manipulation_kernels[] = {
  // Each device will write it's id into the bytes that it "owns", ownership is based on round robin (global_id % num_id)
  // num_id is equal to number of SVM devices in the system plus one (for the host code).
  // id is the index (id) of the device that this kernel is executing on.
  // For example, if there are 2 SVM devices and the host; the buffer should look like this after each device and the host write their id's:
  // 0, 1, 2, 0, 1, 2, 0, 1, 2...
  "__kernel void write_owned_locations(__global char* a, uint num_id, uint id)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "   int owner = i % num_id;\n"
  "    if(id == owner) \n"
  "       a[i] = id;\n"  // modify location if it belongs to this device, write id
  "}\n"

  // Verify that a device can see the byte sized updates from the other devices, sum up the device id's and see if they match expected value.
  // Note: this must be called with a reduced NDRange so that neighbor acesses don't go past end of buffer.
  // For example if there are two SVM devices and the host (3 total devices) the buffer should look like this:
  // 0,1,2,0,1,2...
  // and the expected sum at each point is 0+1+2 = 3.
  "__kernel void sum_neighbor_locations(__global char* a, uint num_devices, volatile __global uint* error_count)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    uint expected_sum = (num_devices * (num_devices - 1))/2;\n"
  "    uint sum = 0;\n"
  "    for(uint j=0; j<num_devices; j++) {\n"
  "        sum += a[i + j];\n" // add my neighbors to the right
  "    }\n"
  "    if(sum != expected_sum)\n"
  "        atomic_inc(error_count);\n"
  "}\n"
};



int test_svm_byte_granularity(cl_device_id deviceID, cl_context c, cl_command_queue queue, int num_elements)
{
  clContextWrapper context;
  clProgramWrapper program;
  clKernelWrapper k1,k2;
  clCommandQueueWrapper queues[MAXQ];

  cl_uint     num_devices = 0;
  cl_int      err = CL_SUCCESS;
  cl_int        rval = CL_SUCCESS;

  err = create_cl_objects(deviceID, &byte_manipulation_kernels[0], &context, &program, &queues[0], &num_devices, CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
  if(err == 1) return 0; // no devices capable of requested SVM level, so don't execute but count test as passing.
  if(err < 0) return -1; // fail test.

  cl_uint num_devices_plus_host = num_devices + 1;

  k1 = clCreateKernel(program, "write_owned_locations", &err);
  test_error(err, "clCreateKernel failed");
  k2 = clCreateKernel(program, "sum_neighbor_locations", &err);
  test_error(err, "clCreateKernel failed");


  cl_char *pA = (cl_char*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_char) * num_elements, 0);

  cl_uint **error_counts =  (cl_uint**) malloc(sizeof(void*) * num_devices);

  for(cl_uint i=0; i < num_devices; i++) {
    error_counts[i] = (cl_uint*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_uint), 0);
    *error_counts[i] = 0;
  }
  for(int i=0; i < num_elements; i++) pA[i] = -1;

  err |= clSetKernelArgSVMPointer(k1, 0, pA);
  err |= clSetKernelArg(k1, 1, sizeof(cl_uint), &num_devices_plus_host);
  test_error(err, "clSetKernelArg failed");

  // get all the devices going simultaneously
  size_t element_num = num_elements;
  for(cl_uint d=0; d < num_devices; d++)  // device ids starting at 1.
  {
    err = clSetKernelArg(k1, 2, sizeof(cl_uint), &d);
    test_error(err, "clSetKernelArg failed");
    err = clEnqueueNDRangeKernel(queues[d], k1, 1, NULL, &element_num, NULL, 0, NULL, NULL);
    test_error(err,"clEnqueueNDRangeKernel failed");
  }

  for(cl_uint d=0; d < num_devices; d++) clFlush(queues[d]);

  cl_uint host_id = num_devices;  // host code will take the id above the devices.
  for(int i = (int)num_devices; i < num_elements; i+= num_devices_plus_host) pA[i] = host_id;

  for(cl_uint id = 0; id < num_devices; id++) clFinish(queues[id]);

  // now check that each device can see the byte writes made by the other devices.

  err |= clSetKernelArgSVMPointer(k2, 0, pA);
  err |= clSetKernelArg(k2, 1, sizeof(cl_uint), &num_devices_plus_host);
  test_error(err, "clSetKernelArg failed");

  // adjusted so k2 doesn't read past end of buffer
  size_t adjusted_num_elements = num_elements - num_devices;
  for(cl_uint id = 0; id < num_devices; id++)
  {
    err = clSetKernelArgSVMPointer(k2, 2, error_counts[id]);
    test_error(err, "clSetKernelArg failed");

    err = clEnqueueNDRangeKernel(queues[id], k2, 1, NULL, &adjusted_num_elements, NULL, 0, NULL, NULL);
    test_error(err,"clEnqueueNDRangeKernel failed");
  }

  for(cl_uint id = 0; id < num_devices; id++) clFinish(queues[id]);

  bool failed = false;

  // see if any of the devices found errors
  for(cl_uint i=0; i < num_devices; i++) {
    if(*error_counts[i] > 0)
      failed = true;
  }
  cl_uint expected = (num_devices_plus_host * (num_devices_plus_host - 1))/2;
  // check that host can see the byte writes made by the devices.
  for(cl_uint i = 0; i < num_elements - num_devices_plus_host; i++)
  {
    int sum = 0;
    for(cl_uint j=0; j < num_devices_plus_host; j++) sum += pA[i+j];
    if(sum != expected)
      failed = true;
  }

  clSVMFree(context, pA);
  for(cl_uint i=0; i < num_devices; i++) clSVMFree(context, error_counts[i]);

  if(failed)
    return -1;
  return 0;
}

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
#include "allocation_utils.h"

cl_command_queue reset_queue(cl_context context, cl_device_id device_id, cl_command_queue *queue, int *error)
{
  log_info("Invalid command queue. Releasing and recreating the command queue.\n");
  clReleaseCommandQueue(*queue);
    *queue = clCreateCommandQueue(context, device_id, 0, error);
  return *queue;
}

int check_allocation_error(cl_context context, cl_device_id device_id, int error, cl_command_queue *queue, cl_event *event) {
  //log_info("check_allocation_error context=%p device_id=%p error=%d *queue=%p\n", context, device_id, error, *queue);
  if (error == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST && event != 0)
  {
    // check for errors from clWaitForEvents (e.g after clEnqueueWriteBuffer)
    cl_int eventError;
    error = clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(error), &eventError, 0);
    if (CL_SUCCESS != error)
    {
      log_error("Failed to get event execution status: %s\n", IGetErrorString(error));
      return FAILED_ABORT;
    }
    if (eventError >= 0)
    {
      log_error("Non-negative event execution status after CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: %s\n", IGetErrorString(error));
      return FAILED_ABORT;
    }
    error = eventError;
  }
  if ((error == CL_MEM_OBJECT_ALLOCATION_FAILURE ) || (error == CL_OUT_OF_RESOURCES ) || (error == CL_OUT_OF_HOST_MEMORY) || (error == CL_INVALID_IMAGE_SIZE)) {
    return FAILED_TOO_BIG;
  } else if (error == CL_INVALID_COMMAND_QUEUE) {
    *queue = reset_queue(context, device_id, queue, &error);
    if (CL_SUCCESS != error)
    {
      log_error("Failed to reset command queue after corrupted queue: %s\n", IGetErrorString(error));
      return FAILED_ABORT;
    }
    // Try again with smaller resources.
    return FAILED_TOO_BIG;
  } else if (error != CL_SUCCESS) {
    log_error("Allocation failed with %s.\n", IGetErrorString(error));
    return FAILED_ABORT;
  }
  return SUCCEEDED;
}


double toMB(cl_ulong size_in) {
  return (double)size_in/(1024.0*1024.0);
}

size_t get_actual_allocation_size(cl_mem mem) {
  int error;
  cl_mem_object_type type;
  size_t size, width, height;

  error = clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(type), &type, NULL);
  if (error) {
      print_error(error, "clGetMemObjectInfo failed for CL_MEM_TYPE.");
    return 0;
  }

  if (type == CL_MEM_OBJECT_BUFFER) {
    error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size), &size, NULL);
    if (error) {
      print_error(error, "clGetMemObjectInfo failed for CL_MEM_SIZE.");
      return 0;
    }
    return size;
  } else if (type == CL_MEM_OBJECT_IMAGE2D) {
    error = clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
    if (error) {
      print_error(error, "clGetMemObjectInfo failed for CL_IMAGE_WIDTH.");
      return 0;
    }
    error = clGetImageInfo(mem, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
    if (error) {
      print_error(error, "clGetMemObjectInfo failed for CL_IMAGE_HEIGHT.");
      return 0;
    }
    return width*height*4*sizeof(cl_uint);
  }

  log_error("Invalid CL_MEM_TYPE: %d\n", type);
  return 0;
}



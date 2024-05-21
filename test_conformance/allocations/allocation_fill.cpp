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
#include "allocation_fill.h"

#define BUFFER_CHUNK_SIZE 8 * 1024 * 1024
#define IMAGE_LINES 8

#include "harness/compat.h"

int fill_buffer_with_data(cl_context context, cl_device_id device_id,
                          cl_command_queue *queue, cl_mem mem, size_t size,
                          MTdata d, cl_bool blocking_write)
{
    size_t i, j;
    cl_uint *data;
    int error, result;
    cl_uint checksum_delta = 0;
    cl_event event;

    size_t size_to_use = BUFFER_CHUNK_SIZE;
    if (size_to_use > size) size_to_use = size;

    data = (cl_uint *)malloc(size_to_use);
    if (data == NULL)
    {
        log_error("Failed to malloc host buffer for writing into buffer.\n");
        return FAILED_ABORT;
    }
    for (i = 0; i < size - size_to_use; i += size_to_use)
    {
        // Put values in the data, and keep a checksum as we go along.
        for (j = 0; j < size_to_use / sizeof(cl_uint); j++)
        {
            data[j] = genrand_int32(d);
            checksum_delta += data[j];
        }
        if (blocking_write)
        {
            error = clEnqueueWriteBuffer(*queue, mem, CL_TRUE, i, size_to_use,
                                         data, 0, NULL, NULL);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteBuffer failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }
        }
        else
        {
            error = clEnqueueWriteBuffer(*queue, mem, CL_FALSE, i, size_to_use,
                                         data, 0, NULL, &event);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteBuffer failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }

            error = clWaitForEvents(1, &event);
            result = check_allocation_error(context, device_id, error, queue,
                                            &event);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clWaitForEvents failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseEvent(event);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }

            clReleaseEvent(event);
        }
    }

    // Deal with any leftover bits
    if (i < size)
    {
        // Put values in the data, and keep a checksum as we go along.
        for (j = 0; j < (size - i) / sizeof(cl_uint); j++)
        {
            data[j] = (cl_uint)genrand_int32(d);
            checksum_delta += data[j];
        }

        if (blocking_write)
        {
            error = clEnqueueWriteBuffer(*queue, mem, CL_TRUE, i, size - i,
                                         data, 0, NULL, NULL);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteBuffer failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }
        }
        else
        {
            error = clEnqueueWriteBuffer(*queue, mem, CL_FALSE, i, size - i,
                                         data, 0, NULL, &event);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteBuffer failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }

            error = clWaitForEvents(1, &event);
            result = check_allocation_error(context, device_id, error, queue,
                                            &event);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clWaitForEvents failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseEvent(event);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }

            clReleaseEvent(event);
        }
    }

    free(data);
    // Only update the checksum if this succeeded.
    checksum += checksum_delta;
    return SUCCEEDED;
}


int fill_image_with_data(cl_context context, cl_device_id device_id,
                         cl_command_queue *queue, cl_mem mem, size_t width,
                         size_t height, MTdata d, cl_bool blocking_write)
{
    size_t origin[3], region[3], j;
    int error, result;
    cl_uint *data;
    cl_uint checksum_delta = 0;
    cl_event event;

    size_t image_lines_to_use;
    image_lines_to_use = IMAGE_LINES;
    if (image_lines_to_use > height) image_lines_to_use = height;

    data = (cl_uint *)malloc(width * 4 * sizeof(cl_uint) * image_lines_to_use);
    if (data == NULL)
    {
        log_error("Failed to malloc host buffer for writing into image.\n");
        return FAILED_ABORT;
    }
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = width;
    region[1] = image_lines_to_use;
    region[2] = 1;
    for (origin[1] = 0; origin[1] < height - image_lines_to_use;
         origin[1] += image_lines_to_use)
    {
        // Put values in the data, and keep a checksum as we go along.
        for (j = 0; j < width * 4 * image_lines_to_use; j++)
        {
            data[j] = (cl_uint)genrand_int32(d);
            checksum_delta += data[j];
        }

        if (blocking_write)
        {
            error = clEnqueueWriteImage(*queue, mem, CL_TRUE, origin, region, 0,
                                        0, data, 0, NULL, NULL);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteImage failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }
            result = clFinish(*queue);
            if (result != SUCCEEDED)
            {
                print_error(
                    error,
                    "clFinish failed after successful enqueuing filling "
                    "buffer with data.");
                return result;
            }
        }
        else
        {
            error = clEnqueueWriteImage(*queue, mem, CL_FALSE, origin, region,
                                        0, 0, data, 0, NULL, &event);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteImage failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }

            error = clWaitForEvents(1, &event);
            result = check_allocation_error(context, device_id, error, queue,
                                            &event);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clWaitForEvents failed.");
            }

            if (result != SUCCEEDED)
            {
                clReleaseEvent(event);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }

            clReleaseEvent(event);
        }
    }

    // Deal with any leftover bits
    if (origin[1] < height)
    {
        // Put values in the data, and keep a checksum as we go along.
        for (j = 0; j < width * 4 * (height - origin[1]); j++)
        {
            data[j] = (cl_uint)genrand_int32(d);
            checksum_delta += data[j];
        }

        region[1] = height - origin[1];
        if (blocking_write)
        {
            error = clEnqueueWriteImage(*queue, mem, CL_TRUE, origin, region, 0,
                                        0, data, 0, NULL, NULL);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteImage failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }
        }
        else
        {
            error = clEnqueueWriteImage(*queue, mem, CL_FALSE, origin, region,
                                        0, 0, data, 0, NULL, &event);
            result = check_allocation_error(context, device_id, error, queue);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clEnqueueWriteImage failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseMemObject(mem);
                free(data);
                return result;
            }

            error = clWaitForEvents(1, &event);
            result = check_allocation_error(context, device_id, error, queue,
                                            &event);

            if (result == FAILED_ABORT)
            {
                print_error(error, "clWaitForEvents failed.");
            }

            if (result != SUCCEEDED)
            {
                clFinish(*queue);
                clReleaseEvent(event);
                free(data);
                clReleaseMemObject(mem);
                return result;
            }

            clReleaseEvent(event);
        }
    }

    free(data);
    // Only update the checksum if this succeeded.
    checksum += checksum_delta;
    return SUCCEEDED;
}


int fill_mem_with_data(cl_context context, cl_device_id device_id,
                       cl_command_queue *queue, cl_mem mem, MTdata d,
                       cl_bool blocking_write)
{
    int error;
    cl_mem_object_type type;
    size_t size, width, height;

    error = clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(type), &type, NULL);
    test_error_abort(error, "clGetMemObjectInfo failed for CL_MEM_TYPE.");

    if (type == CL_MEM_OBJECT_BUFFER)
    {
        error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size), &size, NULL);
        test_error_abort(error, "clGetMemObjectInfo failed for CL_MEM_SIZE.");
        return fill_buffer_with_data(context, device_id, queue, mem, size, d,
                                     blocking_write);
    }
    else if (type == CL_MEM_OBJECT_IMAGE2D)
    {
        error =
            clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
        test_error_abort(error, "clGetImageInfo failed for CL_IMAGE_WIDTH.");
        error =
            clGetImageInfo(mem, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
        test_error_abort(error, "clGetImageInfo failed for CL_IMAGE_HEIGHT.");
        return fill_image_with_data(context, device_id, queue, mem, width,
                                    height, d, blocking_write);
    }

    log_error("Invalid CL_MEM_TYPE: %d\n", type);
    return FAILED_ABORT;
}

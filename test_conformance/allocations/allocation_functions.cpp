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
#include "allocation_functions.h"
#include "allocation_fill.h"


static cl_image_format image_format = { CL_RGBA, CL_UNSIGNED_INT32 };

int allocate_buffer(cl_context context, cl_command_queue *queue,
                    cl_device_id device_id, cl_mem *mem,
                    size_t size_to_allocate, cl_bool blocking_write)
{
    int error;
    // log_info("\t\tAttempting to allocate a %gMB array and fill with %s
    // writes.\n", (size_to_allocate/(1024.0*1024.0)), (blocking_write ?
    // "blocking" : "non-blocking"));
    *mem = clCreateBuffer(context, CL_MEM_READ_WRITE, size_to_allocate, NULL,
                          &error);
    return check_allocation_error(context, device_id, error, queue);
}


int find_good_image_size(cl_device_id device_id, size_t size_to_allocate,
                         size_t *width, size_t *height, size_t *max_size)
{
    size_t max_width, max_height, num_pixels, found_width, found_height;
    int error;

    if (checkForImageSupport(device_id))
    {
        log_info("Can not allocate an image on this device because it does not "
                 "support images.");
        return FAILED_ABORT;
    }

    if (size_to_allocate == 0)
    {
        log_error("Trying to allocate a zero sized image.\n");
        return FAILED_ABORT;
    }

    error = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                            sizeof(max_width), &max_width, NULL);
    test_error_abort(error, "clGetDeviceInfo failed.");
    error = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                            sizeof(max_height), &max_height, NULL);
    test_error_abort(error, "clGetDeviceInfo failed.");

    num_pixels = size_to_allocate / (sizeof(cl_uint) * 4);

    // Use a 64-bit variable to avoid overflow in 32-bit architectures
    long long unsigned max_pixels = (long long unsigned)max_width * max_height;

    if (num_pixels > max_pixels)
    {
        if (NULL != max_size)
        {
            *max_size = max_width * max_height * sizeof(cl_uint) * 4;
        }
        return FAILED_TOO_BIG;
    }

    // We want a close-to-square aspect ratio.
    // Note that this implicitly assumes that  max width >= max height
    found_width = (int)sqrt((double)num_pixels);
    if (found_width > max_width)
    {
        found_width = max_width;
    }
    if (found_width == 0) found_width = 1;

    found_height = (size_t)num_pixels / found_width;
    if (found_height > max_height)
    {
        found_height = max_height;
    }
    if (found_height == 0) found_height = 1;

    *width = found_width;
    *height = found_height;

    if (NULL != max_size)
    {
        *max_size = found_width * found_height * sizeof(cl_uint) * 4;
    }

    return SUCCEEDED;
}


int allocate_image2d_read(cl_context context, cl_command_queue *queue,
                          cl_device_id device_id, cl_mem *mem,
                          size_t size_to_allocate, cl_bool blocking_write)
{
    size_t width, height;
    int error;

    error = find_good_image_size(device_id, size_to_allocate, &width, &height,
                                 NULL);
    if (error != SUCCEEDED) return error;

    log_info("\t\tAttempting to allocate a %gMB read-only image (%d x %d) and "
             "fill with %s writes.\n",
             (size_to_allocate / (1024.0 * 1024.0)), (int)width, (int)height,
             (blocking_write ? "blocking" : "non-blocking"));
    *mem = create_image_2d(context, CL_MEM_READ_ONLY, &image_format, width,
                           height, 0, NULL, &error);

    return check_allocation_error(context, device_id, error, queue);
}


int allocate_image2d_write(cl_context context, cl_command_queue *queue,
                           cl_device_id device_id, cl_mem *mem,
                           size_t size_to_allocate, cl_bool blocking_write)
{
    size_t width, height;
    int error;

    error = find_good_image_size(device_id, size_to_allocate, &width, &height,
                                 NULL);
    if (error != SUCCEEDED) return error;

    // log_info("\t\tAttempting to allocate a %gMB write-only image (%d x %d)
    // and fill with %s writes.\n", (size_to_allocate/(1024.0*1024.0)),
    //(int)width, (int)height, (blocking_write ? "blocking" : "non-blocking"));
    *mem = create_image_2d(context, CL_MEM_WRITE_ONLY, &image_format, width,
                           height, 0, NULL, &error);

    return check_allocation_error(context, device_id, error, queue);
}

int do_allocation(cl_context context, cl_command_queue *queue,
                  cl_device_id device_id, size_t size_to_allocate, int type,
                  cl_mem *mem)
{
    if (type == BUFFER)
        return allocate_buffer(context, queue, device_id, mem, size_to_allocate,
                               true);
    if (type == IMAGE_READ)
        return allocate_image2d_read(context, queue, device_id, mem,
                                     size_to_allocate, true);
    if (type == IMAGE_WRITE)
        return allocate_image2d_write(context, queue, device_id, mem,
                                      size_to_allocate, true);
    if (type == BUFFER_NON_BLOCKING)
        return allocate_buffer(context, queue, device_id, mem, size_to_allocate,
                               false);
    if (type == IMAGE_READ_NON_BLOCKING)
        return allocate_image2d_read(context, queue, device_id, mem,
                                     size_to_allocate, false);
    if (type == IMAGE_WRITE_NON_BLOCKING)
        return allocate_image2d_write(context, queue, device_id, mem,
                                      size_to_allocate, false);
    log_error("Invalid allocation type: %d\n", type);
    return FAILED_ABORT;
}


int allocate_size(cl_context context, cl_command_queue *queue,
                  cl_device_id device_id, int multiple_allocations,
                  size_t size_to_allocate, int type, cl_mem mems[],
                  int *number_of_mems, size_t *final_size, int force_fill,
                  MTdata d)
{

    cl_ulong max_individual_allocation_size, global_mem_size;
    int error, result;
    size_t amount_allocated;
    size_t reduction_amount;
    int current_allocation;
    size_t allocation_this_time, actual_allocation;

    // Set the number of mems used to 0 so if we fail to create even a single
    // one we don't end up returning a garbage value
    *number_of_mems = 0;

    max_individual_allocation_size =
        get_device_info_max_mem_alloc_size(device_id);
    global_mem_size = get_device_info_global_mem_size(device_id);

    if (global_mem_size > (cl_ulong)SIZE_MAX)
    {
        global_mem_size = (cl_ulong)SIZE_MAX;
    }

    if (size_to_allocate > global_mem_size)
    {
        log_error("Can not allocate more than the global memory size.\n");
        return FAILED_ABORT;
    }

    amount_allocated = 0;
    current_allocation = 0;

    // If allocating for images, reduce the maximum allocation size to the
    // maximum image size. If we don't do this, then the value of
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE / 4 can be higher than the maximum image
    // size on systems with 16GB or RAM or more. In this case, we succeed in
    // allocating an image but its size is less than
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE / 4 (min_allocation_allowed) and thus we
    // fail the allocation below.
    if (type == IMAGE_READ || type == IMAGE_READ_NON_BLOCKING
        || type == IMAGE_WRITE || type == IMAGE_WRITE_NON_BLOCKING)
    {
        size_t width;
        size_t height;
        size_t max_size;
        error = find_good_image_size(device_id, size_to_allocate, &width,
                                     &height, &max_size);
        if (!(error == SUCCEEDED || error == FAILED_TOO_BIG)) return error;
        if (max_size < max_individual_allocation_size)
            max_individual_allocation_size = max_size;
    }

    reduction_amount = (size_t)max_individual_allocation_size / 16;

    if (type == BUFFER || type == BUFFER_NON_BLOCKING)
        log_info("\tAttempting to allocate a buffer of size %gMB.\n",
                 toMB(size_to_allocate));
    else if (type == IMAGE_READ || type == IMAGE_READ_NON_BLOCKING)
        log_info("\tAttempting to allocate a read-only image of size %gMB.\n",
                 toMB(size_to_allocate));
    else if (type == IMAGE_WRITE || type == IMAGE_WRITE_NON_BLOCKING)
        log_info("\tAttempting to allocate a write-only image of size %gMB.\n",
                 toMB(size_to_allocate));

    //  log_info("\t\t(Reduction size is %gMB per iteration, minimum allowable
    //  individual allocation size is %gMB.)\n",
    //           toMB(reduction_amount), toMB(min_allocation_allowed));
    //  if (force_fill && type != IMAGE_WRITE && type !=
    //  IMAGE_WRITE_NON_BLOCKING) log_info("\t\t(Allocations will be filled with
    //  random data for checksum calculation.)\n");

    // If we are only doing a single allocation, only allow 1
    int max_to_allocate = multiple_allocations ? MAX_NUMBER_TO_ALLOCATE : 1;

    // Make sure that the maximum number of images allocated is constrained by
    // the maximum that may be passed to a kernel
    if (type != BUFFER && type != BUFFER_NON_BLOCKING)
    {
        cl_device_info param_name =
            (type == IMAGE_READ || type == IMAGE_READ_NON_BLOCKING)
            ? CL_DEVICE_MAX_READ_IMAGE_ARGS
            : CL_DEVICE_MAX_WRITE_IMAGE_ARGS;

        cl_uint max_image_args;
        error = clGetDeviceInfo(device_id, param_name, sizeof(max_image_args),
                                &max_image_args, NULL);
        test_error(error,
                   "clGetDeviceInfo failed for CL_DEVICE_MAX IMAGE_ARGS");

        if ((int)max_image_args < max_to_allocate)
        {
            log_info("\t\tMaximum number of images per kernel limited to %d\n",
                     (int)max_image_args);
            max_to_allocate = max_image_args;
        }
    }


    // Try to allocate the requested amount.
    while (amount_allocated != size_to_allocate
           && current_allocation < max_to_allocate)
    {

        // Determine how much more is needed
        allocation_this_time = size_to_allocate - amount_allocated;

        // Bound by the individual allocation size
        if (allocation_this_time > max_individual_allocation_size)
            allocation_this_time = (size_t)max_individual_allocation_size;

        // Allocate the largest object possible
        result = FAILED_TOO_BIG;
        // log_info("\t\tTrying sub-allocation %d at size %gMB.\n",
        // current_allocation, toMB(allocation_this_time));
        while (result == FAILED_TOO_BIG && allocation_this_time != 0)
        {

            // Create the object
            result =
                do_allocation(context, queue, device_id, allocation_this_time,
                              type, &mems[current_allocation]);
            if (result == SUCCEEDED)
            {
                // Allocation succeeded, another memory object was added to the
                // array
                *number_of_mems = (current_allocation + 1);

                // Verify the size is correct to within 1MB.
                actual_allocation =
                    get_actual_allocation_size(mems[current_allocation]);
                if (fabs((double)allocation_this_time
                         - (double)actual_allocation)
                    > 1024.0 * 1024.0)
                {
                    log_error("Allocation not of expected size. Expected %gMB, "
                              "got %gMB.\n",
                              toMB(allocation_this_time),
                              toMB(actual_allocation));
                    return FAILED_ABORT;
                }

                // If we are filling the allocation for verification do so
                if (force_fill)
                {
                    // log_info("\t\t\tWriting random values to object and
                    // calculating checksum.\n");
                    cl_bool blocking_write = true;
                    if (type == BUFFER_NON_BLOCKING
                        || type == IMAGE_READ_NON_BLOCKING
                        || type == IMAGE_WRITE_NON_BLOCKING)
                    {
                        blocking_write = false;
                    }
                    result = fill_mem_with_data(context, device_id, queue,
                                                mems[current_allocation], d,
                                                blocking_write);
                }
            }

            // If creation failed, try to create a smaller object
            if (result == FAILED_TOO_BIG)
            {
                // log_info("\t\t\tAllocation %d failed at size %gMB. Trying
                // smaller.\n", current_allocation, toMB(allocation_this_time));
                if (allocation_this_time > reduction_amount)
                    allocation_this_time -= reduction_amount;
                else if (reduction_amount > 1)
                {
                    reduction_amount /= 2;
                }
                else
                {
                    allocation_this_time = 0;
                }
            }
        }

        if (result == FAILED_ABORT)
        {
            log_error("\t\tAllocation failed.\n");
            return FAILED_ABORT;
        }

        if (!allocation_this_time)
        {
            log_info("\t\tFailed to allocate %gMB across several objects.\n",
                     toMB(size_to_allocate));
            return FAILED_TOO_BIG;
        }

        // Otherwise we succeeded
        if (result != SUCCEEDED)
        {
            log_error("Test logic error.");
            exit(-1);
        }
        amount_allocated += allocation_this_time;

        *final_size = amount_allocated;

        current_allocation++;
    }

    log_info("\t\tSucceeded in allocating %gMB using %d memory objects.\n",
             toMB(amount_allocated), current_allocation);
    return SUCCEEDED;
}

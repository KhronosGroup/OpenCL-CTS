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
#include "allocation_execute.h"
#include "allocation_functions.h"


const char *buffer_kernel_pattern = {
    "__kernel void sample_test(%s __global uint *result, __global %s *array_sizes, uint per_item)\n"
    "{\n"
    "\tint tid = get_global_id(0);\n"
    "\tuint r = 0;\n"
    "\t%s i;\n"
    "\tfor(i=(%s)tid*(%s)per_item; i<(%s)(1+tid)*(%s)per_item; i++) {\n"
    "%s"
    "\t}\n"
    "\tresult[tid] = r;\n"
    "}\n" };

const char *image_kernel_pattern = {
    "__kernel void sample_test(%s __global uint *result)\n"
    "{\n"
    "\tuint4 color;\n"
    "\tcolor = (uint4)(0);\n"
    "%s"
    "\tint x, y;\n"
    "%s"
    "\tresult[get_global_id(0)] += color.x + color.y + color.z + color.w;\n"
    "}\n" };

const char *read_pattern = {
    "\tfor(y=0; y<get_image_height(image%d); y++)\n"
    "\t\tif (y %s get_global_size(0) == get_global_id(0))\n"
    "\t\t\tfor (x=0; x<get_image_width(image%d); x++) {\n"
    "\t\t\t\tcolor += read_imageui(image%d, sampler, (int2)(x,y));\n"
    "\t\t\t}\n"
};

const char *offset_pattern =
"\tconst uint4 offset = (uint4)(0,1,2,3);\n";

const char *sampler_pattern =
"\tconst sampler_t sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n";


const char *write_pattern = {
    "\tfor(y=0; y<get_image_height(image%d); y++)\n"
    "\t\tif (y %s get_global_size(0) == get_global_id(0))\n"
    "\t\t\tfor (x=0; x<get_image_width(image%d); x++) {\n"
    "\t\t\t\tcolor = (uint4)x*(uint4)y+offset;\n"
    "\t\t\t\twrite_imageui(image%d, (int2)(x,y), color);\n"
    "\t\t\t}\n"
    "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"
};


int check_image(cl_command_queue queue, cl_mem mem) {
    int error;
    cl_mem_object_type type;
    size_t width, height;
    size_t origin[3], region[3], x, j;
    cl_uint *data;

    error = clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(type), &type, NULL);
    if (error) {
        print_error(error, "clGetMemObjectInfo failed for CL_MEM_TYPE.");
        return -1;
    }

    switch (type)
    {
        case CL_MEM_OBJECT_BUFFER:
            log_error("Expected image object, not buffer.\n");
            return -1;
        case CL_MEM_OBJECT_IMAGE2D:
            error = clGetImageInfo(mem, CL_IMAGE_WIDTH, sizeof(width), &width,
                                   NULL);
            if (error)
            {
                print_error(error,
                            "clGetMemObjectInfo failed for CL_IMAGE_WIDTH.");
                return -1;
            }
            error = clGetImageInfo(mem, CL_IMAGE_HEIGHT, sizeof(height),
                                   &height, NULL);
            if (error)
            {
                print_error(error,
                            "clGetMemObjectInfo failed for CL_IMAGE_HEIGHT.");
                return -1;
            }
            break;
        default: log_error("unexpected object type"); return -1;
    }


    data = (cl_uint*)malloc(width*4*sizeof(cl_uint));
    if (data == NULL) {
        log_error("Failed to malloc host buffer for writing into image.\n");
        return FAILED_ABORT;
    }
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = width;
    region[1] = 1;
    region[2] = 1;
    for (origin[1] = 0; origin[1] < height; origin[1]++) {
        error = clEnqueueReadImage(queue, mem, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
        if (error) {
            print_error(error, "clEnqueueReadImage failed");
            free(data);
            return error;
        }

        for (x=0; x<width; x++) {
            for (j=0; j<4; j++) {
                if (data[x*4+j] != (cl_uint)(x*origin[1]+j)) {
                    log_error("Pixel %d, %d, component %d, expected %u, got %u.\n",
                              (int)x, (int)origin[1], (int)j, (cl_uint)(x*origin[1]+j), data[x*4+j]);
                    return -1;
                }
            }
        }
    }
    free(data);
    return 0;
}


#define NUM_OF_WORK_ITEMS 8192*2

int execute_kernel(cl_context context, cl_command_queue *queue, cl_device_id device_id, int test, cl_mem mems[], int number_of_mems_used, int verify_checksum) {

    char *argument_string;
    char *access_string;
    char *kernel_string;
    int i, error, result;
    clKernelWrapper kernel;
    clProgramWrapper program;
    clMemWrapper result_mem;
    char *ptr;
    size_t global_dims[3];
    cl_uint per_item;
    cl_uint per_item_uint;
    cl_uint returned_results[NUM_OF_WORK_ITEMS], final_result;
    clEventWrapper event;
    cl_int event_status;

    // Allocate memory for the kernel source
    argument_string = (char*)malloc(sizeof(char)*MAX_NUMBER_TO_ALLOCATE*64);
    access_string = (char*)malloc(sizeof(char)*MAX_NUMBER_TO_ALLOCATE*(strlen(read_pattern)+10));
    kernel_string = (char*)malloc(sizeof(char)*MAX_NUMBER_TO_ALLOCATE*(strlen(read_pattern)+10+64)+1024);
    argument_string[0] = '\0';
    access_string[0] = '\0';
    kernel_string[0] = '\0';

    // Zero the results.
    for (i=0; i<NUM_OF_WORK_ITEMS; i++)
        returned_results[i] = 0;

    // detect if device supports ulong/int64
    //detect whether profile of the device is embedded
    bool support64 = true;
    char profile[1024] = "";
    error = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PROFILE failed\n" );
    if ((NULL != strstr(profile, "EMBEDDED_PROFILE")) &&
        (!is_extension_available(device_id, "cles_khr_int64"))) {
            support64 = false;
    }

    // Build the kernel source
    if (test == BUFFER || test == BUFFER_NON_BLOCKING) {
        for(i=0; i<number_of_mems_used; i++) {
            sprintf(argument_string + strlen(argument_string), " __global uint *buffer%d, ", i);
            sprintf(access_string + strlen( access_string), "\t\tif (i<array_sizes[%d]) r += buffer%d[i];\n", i, i);
        }
        char type[10];
        if (support64) {
            sprintf(type, "ulong");
        }
        else {
            sprintf(type, "uint");
        }
        sprintf(kernel_string, buffer_kernel_pattern, argument_string, type, type, type, type, type, type, access_string);
    }
    else if (test == IMAGE_READ || test == IMAGE_READ_NON_BLOCKING) {
        for(i=0; i<number_of_mems_used; i++) {
            sprintf(argument_string + strlen(argument_string), " read_only image2d_t image%d, ", i);
            sprintf(access_string + strlen(access_string), read_pattern, i, "%", i, i);
        }
        sprintf(kernel_string, image_kernel_pattern, argument_string, sampler_pattern, access_string);
    }
    else if (test == IMAGE_WRITE || test == IMAGE_WRITE_NON_BLOCKING) {
        for(i=0; i<number_of_mems_used; i++) {
            sprintf(argument_string + strlen(argument_string), " write_only image2d_t image%d, ", i);
            sprintf(access_string + strlen( access_string), write_pattern, i, "%", i, i);
        }
        sprintf(kernel_string, image_kernel_pattern, argument_string, offset_pattern, access_string);
    }
    ptr = kernel_string;

    // Create the kernel
    error = create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&ptr, "sample_test" );

    free(argument_string);
    free(access_string);
    free(kernel_string);

    result = check_allocation_error(context, device_id, error, queue);
    if (result != SUCCEEDED) {
        if (result == FAILED_TOO_BIG)
            log_info("\t\tCreate kernel failed: %s.\n", IGetErrorString(error));
        else
            print_error(error, "Create kernel and program failed");
        return result;
    }

    // Set the arguments
    for (i=0; i<number_of_mems_used; i++) {
        error = clSetKernelArg(kernel, i, sizeof(cl_mem), &mems[i]);
        test_error(error, "clSetKernelArg failed");
    }

    // Set the result
    result_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint)*NUM_OF_WORK_ITEMS, &returned_results, &error);
    test_error(error, "clCreateBuffer failed");
    error = clSetKernelArg(kernel, i, sizeof(result_mem), &result_mem);
    test_error(error, "clSetKernelArg failed");

    // Thread dimensions for execution
    global_dims[0] = NUM_OF_WORK_ITEMS; global_dims[1] = 1; global_dims[2] = 1;

    // We have extra arguments for the buffer kernel because we need to pass in the buffer sizes
    cl_ulong *ulSizes = NULL;
    cl_uint  *uiSizes = NULL;
    if (support64) {
        ulSizes = (cl_ulong*)malloc(sizeof(cl_ulong)*number_of_mems_used);
    }
    else {
        uiSizes = (cl_uint*)malloc(sizeof(cl_uint)*number_of_mems_used);
    }
    cl_ulong max_size = 0;
    clMemWrapper buffer_sizes;
    if (test == BUFFER || test == BUFFER_NON_BLOCKING) {
        for (i=0; i<number_of_mems_used; i++) {
            size_t size;
            error = clGetMemObjectInfo(mems[i], CL_MEM_SIZE, sizeof(size), &size, NULL);
            test_error_abort(error, "clGetMemObjectInfo failed for CL_MEM_SIZE.");
            if (support64) {
                ulSizes[i] = size/sizeof(cl_uint);
            }
            else {
                uiSizes[i] = (cl_uint)size/sizeof(cl_uint);
            }
            if (size/sizeof(cl_uint) > max_size)
                max_size = size/sizeof(cl_uint);
        }
        if (support64) {
            buffer_sizes = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong)*number_of_mems_used, ulSizes, &error);
        }
        else {
            buffer_sizes = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_uint)*number_of_mems_used, uiSizes, &error);
        }
        test_error_abort(error, "clCreateBuffer failed");
        error = clSetKernelArg(kernel, number_of_mems_used+1, sizeof(cl_mem), &buffer_sizes);
        test_error(error, "clSetKernelArg failed");
        per_item = (cl_uint)ceil((double)max_size/global_dims[0]);
        if (per_item > CL_UINT_MAX)
            log_error("Size is too large for a uint parameter to the kernel. Expect invalid results.\n");
        per_item_uint = (cl_uint)per_item;
        error = clSetKernelArg(kernel, number_of_mems_used+2, sizeof(per_item_uint), &per_item_uint);
        test_error(error, "clSetKernelArg failed");
    }
    if (ulSizes) {
        free(ulSizes);
    }
    if (uiSizes) {
        free(uiSizes);
    }

    size_t local_dims[3] = {1,1,1};
    error = get_max_common_work_group_size(context, kernel, global_dims[0], &local_dims[0]);
    test_error(error, "get_max_common_work_group_size failed");

    // Execute the kernel
    error = clEnqueueNDRangeKernel(*queue, kernel, 1, NULL, global_dims, local_dims, 0, NULL, &event);
    result = check_allocation_error(context, device_id, error, queue);
    if (result != SUCCEEDED) {
        if (result == FAILED_TOO_BIG)
            log_info("\t\tExecute kernel failed: %s (global dim: %ld, local dim: %ld)\n", IGetErrorString(error), global_dims[0], local_dims[0]);
        else
            print_error(error, "clEnqueueNDRangeKernel failed");
        return result;
    }

    // Finish the test
    error = clFinish(*queue);

    result = check_allocation_error(context, device_id, error, queue);

    if (result != SUCCEEDED) {
        if (result == FAILED_TOO_BIG)
            log_info("\t\tclFinish failed: %s.\n", IGetErrorString(error));
        else
            print_error(error, "clFinish failed");
        return result;
    }

    // Verify that the event from the execution did not have an error
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
    test_error_abort(error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
    if (event_status < 0) {
        result = check_allocation_error(context, device_id, event_status, queue);
        if (result != SUCCEEDED) {
            if (result == FAILED_TOO_BIG)
                log_info("\t\tEvent returned from kernel execution indicates failure: %s.\n", IGetErrorString(event_status));
            else
                print_error(event_status, "clEnqueueNDRangeKernel failed");
            return result;
        }
    }

    // If we are not verifying the checksum return here
    if (!verify_checksum) {
        log_info("Note: Allocations were not initialized so kernel execution can not verify correct results.\n");
        return SUCCEEDED;
    }

    // Verify the checksum.
    // Read back the result
    error = clEnqueueReadBuffer(*queue, result_mem, CL_TRUE, 0, sizeof(cl_uint)*NUM_OF_WORK_ITEMS, &returned_results, 0, NULL, NULL);
    test_error_abort(error, "clEnqueueReadBuffer failed");
    final_result = 0;
    if (test == BUFFER || test == IMAGE_READ || test == BUFFER_NON_BLOCKING || test == IMAGE_READ_NON_BLOCKING) {
        // For buffers or read images we are just looking at the sum of what each thread summed up
        for (i=0; i<NUM_OF_WORK_ITEMS; i++) {
            final_result += returned_results[i];
        }
        if (final_result != checksum) {
            log_error("\t\tChecksum failed to verify. Expected %u got %u.\n", checksum, final_result);
            return FAILED_ABORT;
        }
        log_info("\t\tChecksum verified (%u == %u).\n", checksum, final_result);
    } else {
        // For write images we need to verify the values
        for (i=0; i<number_of_mems_used; i++) {
            if (check_image(*queue, mems[i])) {
                log_error("\t\tImage contents failed to verify for image %d.\n", (int)i);
                return FAILED_ABORT;
            }
        }
        log_info("\t\tImage contents verified.\n");
    }

    // Finish the test
    error = clFinish(*queue);
    result = check_allocation_error(context, device_id, error, queue);
    if (result != SUCCEEDED) {
        if (result == FAILED_TOO_BIG)
            log_info("\t\tclFinish failed: %s.\n", IGetErrorString(error));
        else
            print_error(error, "clFinish failed");
        return result;
    }

    return SUCCEEDED;
}



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

const char *find_targets_kernel[] = {

    "__kernel void find_targets(__global uint* image, uint target, volatile "
    "__global atomic_uint *numTargetsFound, volatile __global atomic_uint "
    "*targetLocations)\n"
    "{\n"
    " size_t i = get_global_id(0);\n"
    " uint index;\n"
    " if(image[i] == target) {\n"
    "   index = atomic_fetch_add_explicit(numTargetsFound, 1u, "
    "memory_order_relaxed, memory_scope_device); \n"
    "   atomic_exchange_explicit(&targetLocations[index], i, "
    "memory_order_relaxed, memory_scope_all_svm_devices); \n"
    " }\n"
    "}\n"
};


void spawnAnalysisTask(int location)
{
  //    printf("found target at location %d\n", location);
}

#define MAX_TARGETS 1024

// Goals: demonstrate use of SVM's atomics to do fine grain synchronization between the device and host.
// Concept: a device kernel is used to search an input image for regions that match a target pattern.
// The device immediately notifies the host when it finds a target (via an atomic operation that works across host and devices).
// The host is then able to spawn a task that further analyzes the target while the device continues searching for more targets.
REGISTER_TEST(svm_fine_grain_sync_buffers)
{
    clContextWrapper contextWrapper = NULL;
    clProgramWrapper program = NULL;
    cl_uint num_devices = 0;
    cl_int err = CL_SUCCESS;
    clCommandQueueWrapper queues[MAXQ];

    err = create_cl_objects(
        device, &find_targets_kernel[0], &contextWrapper, &program, &queues[0],
        &num_devices, CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS);
    context = contextWrapper;
    if (err == 1)
        return 0; // no devices capable of requested SVM level, so don't execute
                  // but count test as passing.
    if (err < 0) return -1; // fail test.

    clKernelWrapper kernel = clCreateKernel(program, "find_targets", &err);
    test_error(err, "clCreateKernel failed");

    size_t num_pixels = num_elements;
    // cl_uint num_pixels = 1024*1024*32;

    cl_uint *pInputImage = (cl_uint *)clSVMAlloc(
        context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        sizeof(cl_uint) * num_pixels, 0);
    cl_uint *pNumTargetsFound = (cl_uint *)clSVMAlloc(
        context,
        CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        sizeof(cl_uint), 0);
    cl_int *pTargetLocations = (cl_int *)clSVMAlloc(
        context,
        CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        sizeof(cl_int) * MAX_TARGETS, 0);

    cl_uint targetDescriptor = 777;
    *pNumTargetsFound = 0;
    cl_uint i;
    for (i = 0; i < MAX_TARGETS; i++) pTargetLocations[i] = -1;
    for (i = 0; i < num_pixels; i++) pInputImage[i] = 0;
    pInputImage[0] = targetDescriptor;
    pInputImage[3] = targetDescriptor;
    pInputImage[num_pixels - 1] = targetDescriptor;

    err |= clSetKernelArgSVMPointer(kernel, 0, pInputImage);
    err |=
        clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *)&targetDescriptor);
    err |= clSetKernelArgSVMPointer(kernel, 2, pNumTargetsFound);
    err |= clSetKernelArgSVMPointer(kernel, 3, pTargetLocations);
    test_error(err, "clSetKernelArg failed");

    cl_event done;
    err = clEnqueueNDRangeKernel(queues[0], kernel, 1, NULL, &num_pixels, NULL,
                                 0, NULL, &done);
    test_error(err, "clEnqueueNDRangeKernel failed");
    clFlush(queues[0]);


    i = 0;
    cl_int status;
    // check for new targets, if found spawn a task to analyze target.
    do
    {
        err = clGetEventInfo(done, CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int), &status, NULL);
        test_error(err, "clGetEventInfo failed");
        if (AtomicLoadExplicit(&pTargetLocations[i], memory_order_relaxed)
            != -1) // -1 indicates slot not used yet.
        {
            spawnAnalysisTask(pTargetLocations[i]);
            i++;
        }
    } while (status != CL_COMPLETE
             || AtomicLoadExplicit(&pTargetLocations[i], memory_order_relaxed)
                 != -1);

    clReleaseEvent(done);
    clSVMFree(context, pInputImage);
    clSVMFree(context, pNumTargetsFound);
    clSVMFree(context, pTargetLocations);

    if (i != 3) return -1;
    return 0;
}

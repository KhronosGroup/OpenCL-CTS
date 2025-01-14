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
#include "harness/mt19937.h"

#include <vector>

#define GLOBAL_SIZE 65536

static const char *sources[] = {
"__kernel void migrate_kernel(__global uint * restrict a, __global uint * restrict b, __global uint * restrict c)\n"
"{\n"
"    size_t i = get_global_id(0);\n"
"    a[i] ^= 0x13579bdf;\n"
"    b[i] ^= 0x2468ace0;\n"
"    c[i] ^= 0x731fec8f;\n"
"}\n"
};

static void
fill_buffer(cl_uint* p, size_t n, MTdata seed)
{
    for (size_t i=0; i<n; ++i)
        p[i] = (cl_uint)genrand_int32(seed);
}

static bool
check(const char* s, cl_uint* a, cl_uint* e, size_t n)
{
    bool ok = true;
    for (size_t i=0; ok && i<n; ++i) {
        if (a[i] != e[i]) {
            log_error("ERROR: %s mismatch at word %u, *%08x vs %08x\n", s, (unsigned int)i, e[i], a[i]);
            ok = false;
        }
    }
    return ok;
}

static int
wait_and_release(const char* s, cl_event* evs, int n)
{
    cl_int error = clWaitForEvents(n, evs);
    if (error == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) {
        for (int i=0; i<n; ++i) {
            cl_int e;
            error = clGetEventInfo(evs[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &e, NULL);
            test_error(error, "clGetEventInfo failed");
            if (e != CL_COMPLETE) {
                log_error("ERROR: %s event %d execution status was %s\n", s, i, IGetErrorString(e));
                return e;
            }
        }
    } else
        test_error(error, "clWaitForEvents failed");

    for (int i=0; i<n; ++i) {
        error = clReleaseEvent(evs[i]);
        test_error(error, "clReleaseEvent failed");
    }

    return 0;
}

REGISTER_TEST(svm_migrate)
{
    std::vector<cl_uint> amem(GLOBAL_SIZE);
    std::vector<cl_uint> bmem(GLOBAL_SIZE);
    std::vector<cl_uint> cmem(GLOBAL_SIZE);
    cl_event evs[20];

    const size_t global_size = GLOBAL_SIZE;

    RandomSeed seed(0);

    clContextWrapper contextWrapper = NULL;
    clCommandQueueWrapper queues[MAXQ];
    cl_uint num_devices = 0;
    clProgramWrapper program;
    cl_int error;

    error = create_cl_objects(device, &sources[0], &contextWrapper, &program,
                              &queues[0], &num_devices,
                              CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    context = contextWrapper;
    if (error) return -1;

    if (num_devices > 1) {
        log_info("  Running on two devices.\n");
    } else {
        // Ensure we have two distinct queues
        cl_device_id did;
        error = clGetCommandQueueInfo(queues[0], CL_QUEUE_DEVICE, sizeof(did), (void *)&did, NULL);
        test_error(error, "clGetCommandQueueInfo failed");

        cl_command_queue_properties cqp;
        error = clGetCommandQueueInfo(queues[0], CL_QUEUE_PROPERTIES, sizeof(cqp), &cqp, NULL);
        test_error(error, "clGetCommandQueueInfo failed");

        cl_queue_properties qp[3] = { CL_QUEUE_PROPERTIES, cqp, 0 };
        queues[1] = clCreateCommandQueueWithProperties(context, did, qp, &error);
        test_error(error, "clCteateCommandQueueWithProperties failed");
    }

    clKernelWrapper kernel = clCreateKernel(program, "migrate_kernel", &error);
    test_error(error, "clCreateKernel failed");

    char* asvm = (char*)clSVMAlloc(context, CL_MEM_READ_WRITE, global_size*sizeof(cl_uint), 16);
    if (asvm == NULL) {
        log_error("ERROR: clSVMAlloc returned NULL at %s:%d\n", __FILE__, __LINE__);
        return -1;
    }

    char* bsvm = (char *)clSVMAlloc(context, CL_MEM_READ_WRITE, global_size*sizeof(cl_uint), 16);
    if (bsvm == NULL) {
        log_error("ERROR: clSVMAlloc returned NULL at %s:%d\n", __FILE__, __LINE__);
        clSVMFree(context, asvm);
        return -1;
    }

    char* csvm = (char *)clSVMAlloc(context, CL_MEM_READ_WRITE, global_size*sizeof(cl_uint), 16);
    if (csvm == NULL) {
        log_error("ERROR: clSVMAlloc returned NULL at %s:%d\n", __FILE__, __LINE__);
        clSVMFree(context, bsvm);
        clSVMFree(context, asvm);
        return -1;
    }

    error = clSetKernelArgSVMPointer(kernel, 0, (void*)asvm);
    test_error(error, "clSetKernelArgSVMPointer failed");

    error = clSetKernelArgSVMPointer(kernel, 1, (void*)bsvm);
    test_error(error, "clSetKernelArgSVMPointer failed");

    error = clSetKernelArgSVMPointer(kernel, 2, (void*)csvm);
    test_error(error, "clSetKernelArgSVMPointer failed");

    // Initialize host copy of data (and result)
    fill_buffer(amem.data(), global_size, seed);
    fill_buffer(bmem.data(), global_size, seed);
    fill_buffer(cmem.data(), global_size, seed);

    // Now we're ready to start
    {
        // First, fill in the data on device0
        cl_uint patt[] = { 0, 0, 0, 0};
        error = clEnqueueSVMMemFill(queues[0], (void *)asvm, patt, sizeof(patt), global_size*sizeof(cl_uint), 0, NULL, &evs[0]);
        test_error(error, "clEnqueueSVMMemFill failed");

        error = clEnqueueSVMMemFill(queues[0], (void *)bsvm, patt, sizeof(patt), global_size*sizeof(cl_uint), 0, NULL, &evs[1]);
        test_error(error, "clEnqueueSVMMemFill failed");

        error = clEnqueueSVMMemFill(queues[0], (void *)csvm, patt, sizeof(patt), global_size*sizeof(cl_uint), 0, NULL, &evs[2]);
        test_error(error, "clEnqueueSVMMemFill failed");
    }

    {
        // Now migrate fully to device 1 and discard the data
        char* ptrs[] = { asvm, bsvm, csvm };
        error = clEnqueueSVMMigrateMem(queues[1], 3, (const void**)ptrs, NULL, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1, &evs[2], &evs[3]);
        test_error(error, "clEnqueueSVMMigrateMem failed");
    }

    {
        // Test host flag
        char *ptrs[] = { asvm+1, bsvm+3, csvm+5 };
        const size_t szs[] = { 1, 1, 0 };
        error = clEnqueueSVMMigrateMem(queues[0], 3, (const void**)ptrs, szs, CL_MIGRATE_MEM_OBJECT_HOST, 1, &evs[3], &evs[4]);
        test_error(error, "clEnqueueSVMMigrateMem failed");
    }

    {
        // Next fill with known data
        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_WRITE, (void*)asvm, global_size*sizeof(cl_uint), 1, &evs[4], &evs[5]);
        test_error(error, "clEnqueueSVMMap failed");

        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_WRITE, (void*)bsvm, global_size*sizeof(cl_uint), 0, NULL, &evs[6]);
        test_error(error, "clEnqueueSVMMap failed");

        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_WRITE, (void*)csvm, global_size*sizeof(cl_uint), 0, NULL, &evs[7]);
        test_error(error, "clEnqueueSVMMap failed");
    }

    error = clFlush(queues[0]);
    test_error(error, "clFlush failed");

    error = clFlush(queues[1]);
    test_error(error, "clFlush failed");

    // Check the event command type for clEnqueueSVMMigrateMem (OpenCL 3.0 and
    // newer)
    Version version = get_device_cl_version(device);
    if (version >= Version(3, 0))
    {
        cl_command_type commandType;
        error = clGetEventInfo(evs[3], CL_EVENT_COMMAND_TYPE,
                               sizeof(commandType), &commandType, NULL);
        test_error(error, "clGetEventInfo failed");
        if (commandType != CL_COMMAND_SVM_MIGRATE_MEM)
        {
            log_error("Invalid command type returned for "
                      "clEnqueueSVMMigrateMem: %X\n",
                      commandType);
            return TEST_FAIL;
        }
    }

    error = wait_and_release("first batch", evs, 8);
    if (error)
        return -1;

    memcpy((void *)asvm, (void *)amem.data(), global_size * sizeof(cl_uint));
    memcpy((void *)bsvm, (void *)bmem.data(), global_size * sizeof(cl_uint));
    memcpy((void *)csvm, (void *)cmem.data(), global_size * sizeof(cl_uint));

    {
        error = clEnqueueSVMUnmap(queues[1], (void *)asvm, 0, NULL, &evs[0]);
        test_error(error, "clEnqueueSVMUnmap failed");

        error = clEnqueueSVMUnmap(queues[1], (void *)bsvm, 0, NULL, &evs[1]);
        test_error(error, "clEnqueueSVMUnmap failed");

        error = clEnqueueSVMUnmap(queues[1], (void *)csvm, 0, NULL, &evs[2]);
        test_error(error, "clEnqueueSVMUnmap failed");
    }


    {
        // Now try some overlapping regions, and operate on the result
        char *ptrs[] = { asvm+100, bsvm+17, csvm+1000, asvm+101, bsvm+19, csvm+1017 };
        const size_t szs[] = { 13, 23, 43, 3, 7, 11 };

        error = clEnqueueSVMMigrateMem(queues[0], 3, (const void**)ptrs, szs, 0, 1, &evs[2], &evs[3]);
        test_error(error, "clEnqueueSVMMigrateMem failed");

        error = clEnqueueNDRangeKernel(queues[0], kernel, 1, NULL, &global_size, NULL, 0, NULL, &evs[4]);
        test_error(error, "clEnqueueNDRangeKernel failed");
    }

    {
        // Now another pair
        char *ptrs[] = { asvm+8, bsvm+17, csvm+31, csvm+83 };
        const size_t szs[] = { 0, 1, 3, 7 };

        error = clEnqueueSVMMigrateMem(queues[1], 4, (const void**)ptrs, szs, 0, 1, &evs[4], &evs[5]);
        test_error(error, "clEnqueueSVMMigrateMem failed");

        error = clEnqueueNDRangeKernel(queues[1], kernel, 1, NULL, &global_size, NULL, 0, NULL, &evs[6]);
        test_error(error, "clEnqueueNDRangeKernel failed");
    }

    {
        // Another pair
        char *ptrs[] = { asvm+64, asvm+128, bsvm+64, bsvm+128, csvm, csvm+64 };
        const size_t szs[] = { 64, 64, 64, 64, 64, 64 };

        error = clEnqueueSVMMigrateMem(queues[0], 6, (const void**)ptrs, szs, 0, 1, &evs[6], &evs[7]);
        test_error(error, "clEnqueueSVMMigrateMem failed");

        error = clEnqueueNDRangeKernel(queues[0], kernel, 1, NULL, &global_size, NULL, 0, NULL, &evs[8]);
        test_error(error, "clEnqueueNDRangeKernel failed");
    }

    {
        // Final pair
        char *ptrs[] = { asvm, asvm, bsvm, csvm, csvm };
        const size_t szs[] = { 0, 1, 0, 1, 0 };

        error = clEnqueueSVMMigrateMem(queues[1], 5, (const void**)ptrs, szs, 0, 1, &evs[8], &evs[9]);
        test_error(error, "clEnqueueSVMMigrateMem failed");

        error = clEnqueueNDRangeKernel(queues[1], kernel, 1, NULL, &global_size, NULL, 0, NULL, &evs[10]);
        test_error(error, "clEnqueueNDRangeKernel failed");
    }

    {
        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_READ, (void*)asvm, global_size*sizeof(cl_uint), 0, NULL, &evs[11]);
        test_error(error, "clEnqueueSVMMap failed");

        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_READ, (void*)bsvm, global_size*sizeof(cl_uint), 0, NULL, &evs[12]);
        test_error(error, "clEnqueueSVMMap failed");

        error = clEnqueueSVMMap(queues[1], CL_FALSE, CL_MAP_READ, (void*)csvm, global_size*sizeof(cl_uint), 0, NULL, &evs[13]);
        test_error(error, "clEnqueueSVMMap failed");
    }

    error = clFlush(queues[0]);
    test_error(error, "clFlush failed");

    error = clFlush(queues[1]);
    test_error(error, "clFlush failed");

    error = wait_and_release("batch 2", evs, 14);
    if (error)
        return -1;

    // Check kernel results
    bool ok = check("memory a", (cl_uint *)asvm, amem.data(), global_size);
    ok &= check("memory b", (cl_uint *)bsvm, bmem.data(), global_size);
    ok &= check("memory c", (cl_uint *)csvm, cmem.data(), global_size);

    {
        void *ptrs[] = { asvm, bsvm, csvm };

        error = clEnqueueSVMUnmap(queues[1], (void *)asvm, 0, NULL, &evs[0]);
        test_error(error, "clEnqueueSVMUnmap failed");

        error = clEnqueueSVMUnmap(queues[1], (void *)bsvm, 0, NULL, &evs[1]);
        test_error(error, "clEnqueueSVMUnmap failed");

        error = clEnqueueSVMUnmap(queues[1], (void *)csvm, 0, NULL, &evs[2]);
        test_error(error, "clEnqueueSVMUnmap failed");

        error = clEnqueueSVMFree(queues[1], 3, ptrs, NULL, NULL, 0, NULL, &evs[3]);
    }

    error = clFlush(queues[1]);
    test_error(error, "clFlush failed");

    error = wait_and_release("batch 3", evs, 4);
    if (error)
        return -1;

    // The wrappers will clean up the rest
    return ok ? 0 : -1;
}


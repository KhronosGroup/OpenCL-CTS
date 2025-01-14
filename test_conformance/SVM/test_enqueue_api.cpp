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
#include <atomic>

#if !defined(_WIN32)
#include <unistd.h>
#endif

typedef struct
{
  std::atomic<cl_uint> status;
  cl_uint num_svm_pointers;
  std::vector<void *> svm_pointers;
} CallbackData;

void generate_data(std::vector<cl_uchar> &data, size_t size, MTdata seed)
{
  cl_uint randomData = genrand_int32(seed);
  cl_uint bitsLeft = 32;

  for( size_t i = 0; i < size; i++ )
  {
    if( 0 == bitsLeft)
    {
      randomData = genrand_int32(seed);
      bitsLeft = 32;
    }
    data[i] = (cl_uchar)( randomData & 255 );
    randomData >>= 8; randomData -= 8;
  }
}

//callback which will be passed to clEnqueueSVMFree command
void CL_CALLBACK callback_svm_free(cl_command_queue queue, cl_uint num_svm_pointers, void * svm_pointers[], void * user_data)
{
  CallbackData *data = (CallbackData *)user_data;
  data->num_svm_pointers = num_svm_pointers;
  data->svm_pointers.resize(num_svm_pointers, 0);

  cl_context context;
  if(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, 0) != CL_SUCCESS)
  {
    log_error("clGetCommandQueueInfo failed in the callback\n");
    return;
  }

  for (size_t i = 0; i < num_svm_pointers; ++i)
  {
    data->svm_pointers[i] = svm_pointers[i];
    clSVMFree(context, svm_pointers[i]);
  }

  data->status.store(1, std::memory_order_release);
}

REGISTER_TEST(svm_enqueue_api)
{
    clContextWrapper contextWrapper = NULL;
    clCommandQueueWrapper queues[MAXQ];
    cl_uint num_devices = 0;
    const size_t elementNum = 1024;
    const size_t numSVMBuffers = 32;
    cl_int error = CL_SUCCESS;
    RandomSeed seed(0);

    error = create_cl_objects(device, NULL, &contextWrapper, NULL, &queues[0],
                              &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    context = contextWrapper;
    if (error) return TEST_FAIL;

    queue = queues[0];

    // all possible sizes of vectors and scalars
    size_t typeSizes[] = {
        sizeof(cl_uchar),   sizeof(cl_uchar2),  sizeof(cl_uchar3),
        sizeof(cl_uchar4),  sizeof(cl_uchar8),  sizeof(cl_uchar16),
        sizeof(cl_ushort),  sizeof(cl_ushort2), sizeof(cl_ushort3),
        sizeof(cl_ushort4), sizeof(cl_ushort8), sizeof(cl_ushort16),
        sizeof(cl_uint),    sizeof(cl_uint2),   sizeof(cl_uint3),
        sizeof(cl_uint4),   sizeof(cl_uint8),   sizeof(cl_uint16),
        sizeof(cl_ulong),   sizeof(cl_ulong2),  sizeof(cl_ulong3),
        sizeof(cl_ulong4),  sizeof(cl_ulong8),  sizeof(cl_ulong16),
    };

    enum allocationTypes
    {
        host,
        svm
    };

    struct TestType
    {
        allocationTypes srcAlloc;
        allocationTypes dstAlloc;
        TestType(allocationTypes type1, allocationTypes type2)
            : srcAlloc(type1), dstAlloc(type2)
        {}
    };

    std::vector<TestType> testTypes;

    testTypes.push_back(TestType(host, host));
    testTypes.push_back(TestType(host, svm));
    testTypes.push_back(TestType(svm, host));
    testTypes.push_back(TestType(svm, svm));

    for (const auto test_case : testTypes)
    {
        log_info("clEnqueueSVMMemcpy case: src_alloc = %s, dst_alloc = %s\n",
                 test_case.srcAlloc == svm ? "svm" : "host",
                 test_case.dstAlloc == svm ? "svm" : "host");
        for (size_t i = 0; i < ARRAY_SIZE(typeSizes); ++i)
        {
            // generate initial data
            std::vector<cl_uchar> fillData0(typeSizes[i]),
                fillData1(typeSizes[i]);
            generate_data(fillData0, typeSizes[i], seed);
            generate_data(fillData1, typeSizes[i], seed);
            size_t data_size = elementNum * typeSizes[i];
            std::vector<cl_uchar> srcHostData(data_size, 0);
            std::vector<cl_uchar> dstHostData(data_size, 0);
            generate_data(srcHostData, srcHostData.size(), seed);
            generate_data(dstHostData, dstHostData.size(), seed);

            cl_uchar *srcBuffer = (cl_uchar *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, data_size, 0);
            cl_uchar *dstBuffer = (cl_uchar *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, data_size, 0);

            clEventWrapper userEvent = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
            clEventWrapper eventMemFillList[2];

            error = clEnqueueSVMMemFill(queue, srcBuffer, &fillData0[0],
                                        typeSizes[i], data_size, 1, &userEvent,
                                        &eventMemFillList[0]);
            test_error(error, "clEnqueueSVMMemFill failed");
            error = clEnqueueSVMMemFill(queue, dstBuffer, &fillData1[0],
                                        typeSizes[i], data_size, 1, &userEvent,
                                        &eventMemFillList[1]);
            test_error(error, "clEnqueueSVMMemFill failed");

            error = clSetUserEventStatus(userEvent, CL_COMPLETE);
            test_error(error, "clSetUserEventStatus failed");

            cl_uchar *src_ptr;
            cl_uchar *dst_ptr;
            if (test_case.srcAlloc == host)
            {
                src_ptr = srcHostData.data();
            }
            else if (test_case.srcAlloc == svm)
            {
                src_ptr = srcBuffer;
            }
            if (test_case.dstAlloc == host)
            {
                dst_ptr = dstHostData.data();
            }
            else if (test_case.dstAlloc == svm)
            {
                dst_ptr = dstBuffer;
            }
            clEventWrapper eventMemcpy;
            error =
                clEnqueueSVMMemcpy(queue, CL_FALSE, dst_ptr, src_ptr, data_size,
                                   2, &eventMemFillList[0], &eventMemcpy);
            test_error(error, "clEnqueueSVMMemcpy failed");

            // coarse grain only supported. Synchronization required using map
            clEventWrapper eventMap[2];

            error = clEnqueueSVMMap(queue, CL_FALSE, CL_MAP_READ, srcBuffer,
                                    data_size, 1, &eventMemcpy, &eventMap[0]);
            test_error(error, "clEnqueueSVMMap srcBuffer failed");

            error = clEnqueueSVMMap(queue, CL_FALSE, CL_MAP_READ, dstBuffer,
                                    data_size, 1, &eventMemcpy, &eventMap[1]);
            test_error(error, "clEnqueueSVMMap dstBuffer failed");

            error = clWaitForEvents(2, &eventMap[0]);
            test_error(error, "clWaitForEvents failed");

            // data verification
            for (size_t j = 0; j < data_size; ++j)
            {
                if (dst_ptr[j] != src_ptr[j])
                {
                    log_error(
                        "Invalid data at index %zu, dst_ptr %d, src_ptr %d\n",
                        j, dst_ptr[j], src_ptr[j]);
                    return TEST_FAIL;
                }
            }
            clEventWrapper eventUnmap[2];
            error =
                clEnqueueSVMUnmap(queue, srcBuffer, 0, nullptr, &eventUnmap[0]);
            test_error(error, "clEnqueueSVMUnmap srcBuffer failed");

            error =
                clEnqueueSVMUnmap(queue, dstBuffer, 0, nullptr, &eventUnmap[1]);
            test_error(error, "clEnqueueSVMUnmap dstBuffer failed");

            error = clEnqueueSVMMemFill(queue, srcBuffer, &fillData1[0],
                                        typeSizes[i], data_size / 2, 0, 0, 0);
            test_error(error, "clEnqueueSVMMemFill failed");

            error = clEnqueueSVMMemFill(queue, dstBuffer + data_size / 2,
                                        &fillData1[0], typeSizes[i],
                                        data_size / 2, 0, 0, 0);
            test_error(error, "clEnqueueSVMMemFill failed");

            error = clEnqueueSVMMemcpy(queue, CL_FALSE, dstBuffer, srcBuffer,
                                       data_size / 2, 0, 0, 0);
            test_error(error, "clEnqueueSVMMemcpy failed");

            error = clEnqueueSVMMemcpy(
                queue, CL_TRUE, dstBuffer + data_size / 2,
                srcBuffer + data_size / 2, data_size / 2, 0, 0, 0);
            test_error(error, "clEnqueueSVMMemcpy failed");

            void *ptrs[] = { (void *)srcBuffer, (void *)dstBuffer };

            clEventWrapper eventFree;
            error = clEnqueueSVMFree(queue, 2, ptrs, 0, 0, 0, 0, &eventFree);
            test_error(error, "clEnqueueSVMFree failed");

            error = clWaitForEvents(1, &eventFree);
            test_error(error, "clWaitForEvents failed");

            // event info verification for new SVM commands
            cl_command_type commandType;
            for (auto &check_event : eventMemFillList)
            {
                error =
                    clGetEventInfo(check_event, CL_EVENT_COMMAND_TYPE,
                                   sizeof(cl_command_type), &commandType, NULL);
                test_error(error, "clGetEventInfo failed");
                if (commandType != CL_COMMAND_SVM_MEMFILL)
                {
                    log_error("Invalid command type returned for "
                              "clEnqueueSVMMemFill\n");
                    return TEST_FAIL;
                }
            }

            error = clGetEventInfo(eventMemcpy, CL_EVENT_COMMAND_TYPE,
                                   sizeof(cl_command_type), &commandType, NULL);
            test_error(error, "clGetEventInfo failed");
            if (commandType != CL_COMMAND_SVM_MEMCPY)
            {
                log_error(
                    "Invalid command type returned for clEnqueueSVMMemcpy\n");
                return TEST_FAIL;
            }
            for (size_t map_id = 0; map_id < ARRAY_SIZE(eventMap); map_id++)
            {
                error =
                    clGetEventInfo(eventMap[map_id], CL_EVENT_COMMAND_TYPE,
                                   sizeof(cl_command_type), &commandType, NULL);
                test_error(error, "clGetEventInfo failed");
                if (commandType != CL_COMMAND_SVM_MAP)
                {
                    log_error(
                        "Invalid command type returned for clEnqueueSVMMap\n");
                    return TEST_FAIL;
                }

                error =
                    clGetEventInfo(eventUnmap[map_id], CL_EVENT_COMMAND_TYPE,
                                   sizeof(cl_command_type), &commandType, NULL);
                test_error(error, "clGetEventInfo failed");
                if (commandType != CL_COMMAND_SVM_UNMAP)
                {
                    log_error("Invalid command type returned for "
                              "clEnqueueSVMUnmap\n");
                    return TEST_FAIL;
                }
            }
            error = clGetEventInfo(eventFree, CL_EVENT_COMMAND_TYPE,
                                   sizeof(cl_command_type), &commandType, NULL);
            test_error(error, "clGetEventInfo failed");
            if (commandType != CL_COMMAND_SVM_FREE)
            {
                log_error(
                    "Invalid command type returned for clEnqueueSVMFree\n");
                return TEST_FAIL;
            }
        }
    }
    std::vector<void *> buffers(numSVMBuffers, 0);
    for (size_t i = 0; i < numSVMBuffers; ++i)
        buffers[i] = clSVMAlloc(context, CL_MEM_READ_WRITE, elementNum, 0);

    // verify if callback is triggered correctly
    CallbackData data;
    data.status = 0;

    error = clEnqueueSVMFree(queue, buffers.size(), &buffers[0],
                             callback_svm_free, &data, 0, 0, 0);
    test_error(error, "clEnqueueSVMFree failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    // wait for the callback
    while (data.status.load(std::memory_order_acquire) == 0)
    {
        usleep(1);
    }

    // check if number of SVM pointers returned in the callback matches with
    // expected
    if (data.num_svm_pointers != buffers.size())
    {
        log_error("Invalid number of SVM pointers returned in the callback, "
                  "expected: %zu, got: %d\n",
                  buffers.size(), data.num_svm_pointers);
        return TEST_FAIL;
    }

    // check if pointers returned in callback are correct
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        if (data.svm_pointers[i] != buffers[i])
        {
            log_error(
                "Invalid SVM pointer returned in the callback, idx: %zu\n", i);
            return TEST_FAIL;
        }
    }

    return 0;
}

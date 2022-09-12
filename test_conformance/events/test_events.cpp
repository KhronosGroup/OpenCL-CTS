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
#include "testBase.h"

#if !defined(_WIN32)
#include "unistd.h" // for "sleep" used in the "while (1)" busy wait loop in
#endif
// test_event_flush

const char *sample_long_test_kernel[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     int  i;\n"
    "\n"
    "    for( i = 0; i < 10000; i++ )\n"
    "    {\n"
    "        dst[tid] = (int)src[tid] * 3;\n"
    "    }\n"
    "\n"
    "}\n"
};

int create_and_execute_kernel(cl_context inContext, cl_command_queue inQueue,
                              cl_program *outProgram, cl_kernel *outKernel,
                              cl_mem *streams, unsigned int lineCount,
                              const char **lines, const char *kernelName,
                              cl_event *outEvent)
{
    size_t threads[1] = { 1000 }, localThreads[1];
    int error;

    if (create_single_kernel_helper(inContext, outProgram, outKernel, lineCount,
                                    lines, kernelName))
    {
        return -1;
    }

    error = get_max_common_work_group_size(inContext, *outKernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    streams[0] = clCreateBuffer(inContext, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 1000, NULL, &error);
    test_error(error, "Creating test array failed");
    streams[1] = clCreateBuffer(inContext, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * 1000, NULL, &error);
    test_error(error, "Creating test array failed");

    /* Set the arguments */
    error = clSetKernelArg(*outKernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set kernel arguments");
    error = clSetKernelArg(*outKernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set kernel arguments");

    error = clEnqueueNDRangeKernel(inQueue, *outKernel, 1, NULL, threads,
                                   localThreads, 0, NULL, outEvent);
    test_error(error, "Unable to execute test kernel");

    return 0;
}

#define SETUP_EVENT(c, q)                                                      \
    clProgramWrapper program;                                                  \
    clKernelWrapper kernel;                                                    \
    clMemWrapper streams[2];                                                   \
    clEventWrapper event;                                                      \
    int error;                                                                 \
    if (create_and_execute_kernel(c, q, &program, &kernel, &streams[0], 1,     \
                                  sample_long_test_kernel, "sample_test",      \
                                  &event))                                     \
        return -1;

#define FINISH_EVENT(_q) clFinish(_q)

const char *IGetStatusString(cl_int status)
{
    static char tempString[128];
    switch (status)
    {
        case CL_COMPLETE: return "CL_COMPLETE";
        case CL_RUNNING: return "CL_RUNNING";
        case CL_QUEUED: return "CL_QUEUED";
        case CL_SUBMITTED: return "CL_SUBMITTED";
        default:
            sprintf(tempString, "<unknown: %d>", (int)status);
            return tempString;
    }
}

/* Note: tests clGetEventStatus and clReleaseEvent (implicitly) */
int test_event_get_execute_status(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    cl_int status;
    SETUP_EVENT(context, queue);

    /* Now wait for it to be done */
    error = clWaitForEvents(1, &event);
    test_error(error, "Unable to wait for event");

    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error,
               "Calling clGetEventStatus to wait for event completion failed");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after event complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    FINISH_EVENT(queue);
    return 0;
}

int test_event_get_info(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    SETUP_EVENT(context, queue);

    /* Verify parameters of clGetEventInfo not already tested by other tests */
    cl_command_queue otherQueue;
    size_t size;

    error = clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE, sizeof(otherQueue),
                           &otherQueue, &size);
    test_error(error, "Unable to get event info!");
    // We can not check if this is the right queue because this is an opaque
    // object.
    if (size != sizeof(queue))
    {
        log_error("ERROR: Returned command queue size does not validate "
                  "(expected %d, got %d)\n",
                  (int)sizeof(queue), (int)size);
        return -1;
    }

    cl_command_type type;
    error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(type), &type,
                           &size);
    test_error(error, "Unable to get event info!");
    if (type != CL_COMMAND_NDRANGE_KERNEL)
    {
        log_error("ERROR: Returned command type does not validate (expected "
                  "%d, got %d)\n",
                  (int)CL_COMMAND_NDRANGE_KERNEL, (int)type);
        return -1;
    }
    if (size != sizeof(type))
    {
        log_error("ERROR: Returned command type size does not validate "
                  "(expected %d, got %d)\n",
                  (int)sizeof(type), (int)size);
        return -1;
    }

    cl_uint count;
    error = clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(count),
                           &count, &size);
    test_error(error, "Unable to get event info for CL_EVENT_REFERENCE_COUNT!");
    if (size != sizeof(count))
    {
        log_error("ERROR: Returned command type size does not validate "
                  "(expected %d, got %d)\n",
                  (int)sizeof(type), (int)size);
        return -1;
    }

    cl_context testCtx;
    error = clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(testCtx), &testCtx,
                           &size);
    test_error(error, "Unable to get event context info!");
    if (size != sizeof(context))
    {
        log_error("ERROR: Returned context size does not validate (expected "
                  "%d, got %d)\n",
                  (int)sizeof(context), (int)size);
        return -1;
    }
    if (testCtx != context)
    {
        log_error(
            "ERROR: Returned context does not match (expected %p, got %p)\n",
            (void *)context, (void *)testCtx);
        return -1;
    }

    FINISH_EVENT(queue);
    return 0;
}

int test_event_get_write_array_status(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    cl_mem stream;
    cl_float testArray[1024 * 32];
    cl_event event;
    int error;
    cl_int status;


    stream = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");

    error = clEnqueueWriteBuffer(queue, stream, CL_FALSE, 0,
                                 sizeof(cl_float) * 1024 * 32,
                                 (void *)testArray, 0, NULL, &event);
    test_error(error, "Unable to set testing kernel data");

    /* Now wait for it to be done */
    error = clWaitForEvents(1, &event);
    test_error(error, "Unable to wait for event");

    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error,
               "Calling clGetEventStatus to wait for event completion failed");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array write complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }


    clReleaseMemObject(stream);
    clReleaseEvent(event);

    return 0;
}

int test_event_get_read_array_status(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    cl_mem stream;
    cl_float testArray[1024 * 32];
    cl_event event;
    int error;
    cl_int status;


    stream = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");

    error = clEnqueueReadBuffer(queue, stream, CL_FALSE, 0,
                                sizeof(cl_float) * 1024 * 32, (void *)testArray,
                                0, NULL, &event);
    test_error(error, "Unable to get testing kernel data");


    /* It should still be running... */
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");

    if (status != CL_RUNNING && status != CL_QUEUED && status != CL_SUBMITTED
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "during array read (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    /* Now wait for it to be done */
    error = clWaitForEvents(1, &event);
    test_error(error, "Unable to wait for event");

    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error,
               "Calling clGetEventStatus to wait for event completion failed");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array read complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }


    clReleaseMemObject(stream);
    clReleaseEvent(event);

    return 0;
}

/* clGetEventStatus not implemented yet */

int test_event_wait_for_execute(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    cl_int status;
    SETUP_EVENT(context, queue);

    /* Now we wait for it to be done, then test the status again */
    error = clWaitForEvents(1, &event);
    test_error(error, "Unable to wait for execute event");

    /* Make sure it worked */
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after event complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    FINISH_EVENT(queue);
    return 0;
}

int test_event_wait_for_array(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_float readArray[1024 * 32];
    cl_float writeArray[1024 * 32];
    cl_event events[2];
    int error;
    cl_int status;


    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");

    error = clEnqueueReadBuffer(queue, streams[0], CL_FALSE, 0,
                                sizeof(cl_float) * 1024 * 32, (void *)readArray,
                                0, NULL, &events[0]);
    test_error(error, "Unable to read testing kernel data");

    error = clEnqueueWriteBuffer(queue, streams[1], CL_FALSE, 0,
                                 sizeof(cl_float) * 1024 * 32,
                                 (void *)writeArray, 0, NULL, &events[1]);
    test_error(error, "Unable to write testing kernel data");

    /* Both should still be running */
    error = clGetEventInfo(events[0], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_RUNNING && status != CL_QUEUED && status != CL_SUBMITTED
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "during array read (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    error = clGetEventInfo(events[1], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_RUNNING && status != CL_QUEUED && status != CL_SUBMITTED
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "during array write (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    /* Now try waiting for both */
    error = clWaitForEvents(2, events);
    test_error(error, "Unable to wait for array events");

    /* Double check status on both */
    error = clGetEventInfo(events[0], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array read complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    error = clGetEventInfo(events[1], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array write complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);

    return 0;
}

int test_event_flush(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    int loopCount = 0;
    cl_int status;
    SETUP_EVENT(context, queue);

    /* Now flush. Note that we can't guarantee this actually lets the op finish,
     * but we can guarantee it's no longer queued */
    error = clFlush(queue);
    test_error(error, "Unable to flush events");

    /* Make sure it worked */
    while (1)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(status), &status, NULL);
        test_error(error, "Calling clGetEventStatus didn't work!");

        if (status != CL_QUEUED) break;

#if !defined(_WIN32)
        sleep(1); // give it some time here.
#else // _WIN32
        Sleep(1000);
#endif
        ++loopCount;
    }

    /*
    CL_QUEUED (command has been enqueued in the command-queue),
    CL_SUBMITTED (enqueued command has been submitted by the host to the device
    associated with the command-queue), CL_RUNNING (device is currently
    executing this command), CL_COMPLETE (the command has completed), or Error
    code given by a negative integer value. (command was abnormally terminated –
    this may be caused by a bad memory access etc.).
    */
    if (status != CL_COMPLETE && status != CL_SUBMITTED && status != CL_RUNNING
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after event flush (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    /* Now wait */
    error = clFinish(queue);
    test_error(error, "Unable to finish events");

    FINISH_EVENT(queue);
    return 0;
}


int test_event_finish_execute(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    cl_int status;
    SETUP_EVENT(context, queue);

    /* Now flush and finish all ops */
    error = clFinish(queue);
    test_error(error, "Unable to finish all events");

    /* Make sure it worked */
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after event complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    FINISH_EVENT(queue);
    return 0;
}

int test_event_finish_array(cl_device_id deviceID, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_float readArray[1024 * 32];
    cl_float writeArray[1024 * 32];
    cl_event events[2];
    int error;
    cl_int status;


    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * 1024 * 32, NULL, &error);
    test_error(error, "Creating test array failed");

    error = clEnqueueReadBuffer(queue, streams[0], CL_FALSE, 0,
                                sizeof(cl_float) * 1024 * 32, (void *)readArray,
                                0, NULL, &events[0]);
    test_error(error, "Unable to read testing kernel data");

    error = clEnqueueWriteBuffer(queue, streams[1], CL_FALSE, 0,
                                 sizeof(cl_float) * 1024 * 32,
                                 (void *)writeArray, 0, NULL, &events[1]);
    test_error(error, "Unable to write testing kernel data");

    /* Both should still be running */
    error = clGetEventInfo(events[0], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_RUNNING && status != CL_QUEUED && status != CL_SUBMITTED
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "during array read (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    error = clGetEventInfo(events[1], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_RUNNING && status != CL_QUEUED && status != CL_SUBMITTED
        && status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "during array write (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    /* Now try finishing all ops */
    error = clFinish(queue);
    test_error(error, "Unable to finish all events");

    /* Double check status on both */
    error = clGetEventInfo(events[0], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array read complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    error = clGetEventInfo(events[1], CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventStatus didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetErrorStatus "
                  "after array write complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);

    return 0;
}


#define NUM_EVENT_RUNS 100

int test_event_release_before_done(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    // Create a kernel to run
    clProgramWrapper program;
    clKernelWrapper kernel[NUM_EVENT_RUNS];
    size_t threads[1] = { 1000 };
    cl_event events[NUM_EVENT_RUNS];
    cl_int status;
    clMemWrapper streams[NUM_EVENT_RUNS][2];
    int error, i;

    // Create a kernel
    if (create_single_kernel_helper(context, &program, &kernel[0], 1,
                                    sample_long_test_kernel, "sample_test"))
    {
        return -1;
    }

    for (i = 1; i < NUM_EVENT_RUNS; i++)
    {
        kernel[i] = clCreateKernel(program, "sample_test", &error);
        test_error(error, "Unable to create kernel");
    }

    error =
        get_max_common_work_group_size(context, kernel[0], 1024, &threads[0]);
    test_error(error, "Unable to get work group size to use");

    // Create a set of streams to use as arguments
    for (i = 0; i < NUM_EVENT_RUNS; i++)
    {
        streams[i][0] =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(cl_float) * threads[0], NULL, &error);
        streams[i][1] =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(cl_int) * threads[0], NULL, &error);
        if ((streams[i][0] == NULL) || (streams[i][1] == NULL))
        {
            log_error("ERROR: Unable to allocate testing streams");
            return -1;
        }
    }

    // Execute the kernels one by one, hopefully making sure they won't be done
    // by the time we get to the end
    for (i = 0; i < NUM_EVENT_RUNS; i++)
    {
        error = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &streams[i][0]);
        error |= clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &streams[i][1]);
        test_error(error, "Unable to set kernel arguments");

        error = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads,
                                       threads, 0, NULL, &events[i]);
        test_error(error, "Unable to execute test kernel");
    }

    // Free all but the last event
    for (i = 0; i < NUM_EVENT_RUNS - 1; i++)
    {
        clReleaseEvent(events[i]);
    }

    // Get status on the last one, then free it
    error = clGetEventInfo(events[NUM_EVENT_RUNS - 1],
                           CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status),
                           &status, NULL);
    test_error(error, "Unable to get event status");

    clReleaseEvent(events[NUM_EVENT_RUNS - 1]);

    // Was the status still-running?
    if (status == CL_COMPLETE)
    {
        log_info("WARNING: Events completed before they could be released, so "
                 "test is a null-op. Increase workload and try again.");
    }
    else if (status == CL_RUNNING || status == CL_QUEUED
             || status == CL_SUBMITTED)
    {
        log_info("Note: Event status was running or queued when released, so "
                 "test was good.\n");
    }

    // If we didn't crash by now, the test succeeded
    clFinish(queue);

    return 0;
}

int test_event_enqueue_marker(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    cl_int status;
    SETUP_EVENT(context, queue);

    /* Now we queue a marker and wait for that, which--since it queues
     * afterwards--should guarantee the execute finishes too */
    clEventWrapper markerEvent;
    // error = clEnqueueMarker( queue, &markerEvent );

#ifdef CL_VERSION_1_2
    error = clEnqueueMarkerWithWaitList(queue, 0, NULL, &markerEvent);
#else
    error = clEnqueueMarker(queue, &markerEvent);
#endif
    test_error(error, "Unable to queue marker");
    /* Now we wait for it to be done, then test the status again */
    error = clWaitForEvents(1, &markerEvent);
    test_error(error, "Unable to wait for marker event");

    /* Check the status of the first event */
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
    test_error(error, "Calling clGetEventInfo didn't work!");
    if (status != CL_COMPLETE)
    {
        log_error("ERROR: Incorrect status returned from clGetEventInfo after "
                  "event complete (%d:%s)\n",
                  status, IGetStatusString(status));
        return -1;
    }

    FINISH_EVENT(queue);
    return 0;
}

#ifdef CL_VERSION_1_2
int test_event_enqueue_marker_with_event_list(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements)
{
    SETUP_EVENT(context, queue);
    cl_event event_list[3] = { NULL, NULL, NULL };

    size_t threads[1] = { 10 }, localThreads[1] = { 1 };
    cl_uint event_count = 2;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[0]);
    test_error(error, " clEnqueueMarkerWithWaitList   1 ");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[1]);
    test_error(error, " clEnqueueMarkerWithWaitList 2");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, " clEnqueueMarkerWithWaitList  3");

    // test the case event returned
    error = clEnqueueMarkerWithWaitList(queue, event_count, event_list,
                                        &event_list[2]);
    test_error(error, " clEnqueueMarkerWithWaitList ");

    error = clReleaseEvent(event_list[0]);
    error |= clReleaseEvent(event_list[1]);
    test_error(error, "clReleaseEvent");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[0]);
    test_error(error, " clEnqueueMarkerWithWaitList   1 -1 ");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[1]);
    test_error(error, " clEnqueueMarkerWithWaitList  2-2");

    // test the case event =NULL,   caused [CL_INVALID_VALUE] : OpenCL Error :
    // clEnqueueMarkerWithWaitList failed: event is a NULL value
    error = clEnqueueMarkerWithWaitList(queue, event_count, event_list, NULL);
    test_error(error, " clEnqueueMarkerWithWaitList ");

    error = clReleaseEvent(event_list[0]);
    error |= clReleaseEvent(event_list[1]);
    error |= clReleaseEvent(event_list[2]);
    test_error(error, "clReleaseEvent");

    FINISH_EVENT(queue);
    return 0;
}

int test_event_enqueue_barrier_with_event_list(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    SETUP_EVENT(context, queue);
    cl_event event_list[3] = { NULL, NULL, NULL };

    size_t threads[1] = { 10 }, localThreads[1] = { 1 };
    cl_uint event_count = 2;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[0]);
    test_error(error, " clEnqueueBarrierWithWaitList   1 ");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[1]);
    test_error(error, " clEnqueueBarrierWithWaitList 2");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, " clEnqueueBarrierWithWaitList  20");

    // test the case event returned
    error = clEnqueueBarrierWithWaitList(queue, event_count, event_list,
                                         &event_list[2]);
    test_error(error, " clEnqueueBarrierWithWaitList ");

    clReleaseEvent(event_list[0]);
    clReleaseEvent(event_list[1]);

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[0]);
    test_error(error, " clEnqueueBarrierWithWaitList   1 ");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, &event_list[1]);
    test_error(error, " clEnqueueBarrierWithWaitList 2");

    // test the case event =NULL,   caused [CL_INVALID_VALUE] : OpenCL Error :
    // clEnqueueMarkerWithWaitList failed: event is a NULL value
    error = clEnqueueBarrierWithWaitList(queue, event_count, event_list, NULL);
    test_error(error, " clEnqueueBarrierWithWaitList ");

    clReleaseEvent(event_list[0]);
    clReleaseEvent(event_list[1]);
    clReleaseEvent(event_list[2]);

    FINISH_EVENT(queue);
    return 0;
}
#endif

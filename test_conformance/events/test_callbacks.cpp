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
#include "action_classes.h"
#include "harness/conversions.h"
#include "harness/ThreadPool.h"

#if !defined(_MSC_VER)
#include <unistd.h>
#endif // !_MSC_VER

extern const char *IGetStatusString(cl_int status);

#define PRINT_OPS 0

// Yes, this is somewhat nasty, in that we're relying on the CPU (the real CPU,
// not the OpenCL device) to be atomic w.r.t. boolean values. Although if it
// isn't, we'll just miss the check on this bool until the next time around, so
// it's not that big of a deal. Ideally, we'd be using a semaphore with a
// trywait on it, but then that introduces the fun issue of what to do on Win32,
// etc. This way is far more portable, and worst case of failure is a slightly
// longer test run.
static bool sCallbackTriggered = false;


#define EVENT_CALLBACK_TYPE_TOTAL 3
static bool sCallbackTriggered_flag[EVENT_CALLBACK_TYPE_TOTAL] = { false, false,
                                                                   false };
cl_int event_callback_types[EVENT_CALLBACK_TYPE_TOTAL] = { CL_SUBMITTED,
                                                           CL_RUNNING,
                                                           CL_COMPLETE };

// Our callback function
/*void CL_CALLBACK single_event_callback_function( cl_event event, cl_int
commandStatus, void * userData )
{
     int i=*static_cast<int *>(userData);
    log_info( "\tEvent callback  %d   triggered\n",  i);
    sCallbackTriggered_flag [ i ] = true;
}*/

/*   use struct as call back para */
typedef struct
{
    cl_int event_type;
    int index;
} CALL_BACK_USER_DATA;

void CL_CALLBACK single_event_callback_function_flags(cl_event event,
                                                      cl_int commandStatus,
                                                      void *userData)
{
    // int i=*static_cast<int *>(userData);
    CALL_BACK_USER_DATA *pdata = static_cast<CALL_BACK_USER_DATA *>(userData);

    log_info("\tEvent callback  %d  of type %d triggered\n", pdata->index,
             pdata->event_type);
    sCallbackTriggered_flag[pdata->index] = true;
}

int test_callback_event_single(cl_device_id device, cl_context context,
                               cl_command_queue queue, Action *actionToTest)
{
    // Note: we don't use the waiting feature here. We just want to verify that
    // we get a callback called when the given event finishes

    cl_int error = actionToTest->Setup(device, context, queue);
    test_error(error, "Unable to set up test action");

    // Set up a user event, which we use as a gate for the second event
    clEventWrapper gateEvent = clCreateUserEvent(context, &error);
    test_error(error, "Unable to set up user gate event");

    // Set up the execution of the action with its actual event
    clEventWrapper actualEvent;
    error = actionToTest->Execute(queue, 1, &gateEvent, &actualEvent);
    test_error(error, "Unable to set up action execution");

    // Set up the callback on the actual event

    /*  use struct as call back para */
    CALL_BACK_USER_DATA user_data[EVENT_CALLBACK_TYPE_TOTAL];
    for (int i = 0; i < EVENT_CALLBACK_TYPE_TOTAL; i++)
    {
        user_data[i].event_type = event_callback_types[i];
        user_data[i].index = i;
        error = clSetEventCallback(actualEvent, event_callback_types[i],
                                   single_event_callback_function_flags,
                                   user_data + i);
    }

    // Now release the user event, which will allow our actual action to run
    error = clSetUserEventStatus(gateEvent, CL_COMPLETE);
    test_error(error, "Unable to trigger gate event");

    // Now we wait for completion. Note that we can actually wait on the event
    // itself, at least at first
    error = clWaitForEvents(1, &actualEvent);
    test_error(error, "Unable to wait for actual test event");

    // Note: we can check our callback now, and it MIGHT have been triggered,
    // but that's not guaranteed
    if (sCallbackTriggered)
    {
        // We're all good, so return success
        return 0;
    }

    // The callback has not yet been called, but that doesn't mean it won't be.
    // So wait for it
    log_info("\tWaiting for callback...");
    fflush(stdout);
    for (int i = 0; i < 10 * 10; i++)
    {
        usleep(100000); // 1/10th second

        int cc = 0;
        for (int k = 0; k < EVENT_CALLBACK_TYPE_TOTAL; k++)
            if (sCallbackTriggered_flag[k])
            {
                cc++;
            }

        if (cc == EVENT_CALLBACK_TYPE_TOTAL)
        {
            log_info("\n");
            return 0;
        }
        log_info(".");
        fflush(stdout);
    }

    // If we got here, we never got the callback
    log_error("\nCallback not called within 10 seconds! (assuming failure)\n");
    return -1;
}

#define TEST_ACTION(name)                                                      \
    {                                                                          \
        name##Action action;                                                   \
        log_info("-- Testing " #name "...\n");                                 \
        if ((error = test_callback_event_single(deviceID, context, queue,      \
                                                &action))                      \
            != CL_SUCCESS)                                                     \
            retVal++;                                                          \
        clFinish(queue);                                                       \
    }

int test_callbacks(cl_device_id deviceID, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    cl_int error;
    int retVal = 0;

    log_info("\n");

    TEST_ACTION(NDRangeKernel)

    TEST_ACTION(ReadBuffer)
    TEST_ACTION(WriteBuffer)
    TEST_ACTION(MapBuffer)
    TEST_ACTION(UnmapBuffer)

    if (checkForImageSupport(deviceID) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
        log_info("\nNote: device does not support images. Skipping remainder "
                 "of callback tests...\n");
    }
    else
    {
        TEST_ACTION(ReadImage2D)
        TEST_ACTION(WriteImage2D)
        TEST_ACTION(CopyImage2Dto2D)
        TEST_ACTION(Copy2DImageToBuffer)
        TEST_ACTION(CopyBufferTo2DImage)
        TEST_ACTION(MapImage)

        if (checkFor3DImageSupport(deviceID) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
            log_info("\nNote: device does not support 3D images. Skipping "
                     "remainder of waitlist tests...\n");
        else
        {
            TEST_ACTION(ReadImage3D)
            TEST_ACTION(WriteImage3D)
            TEST_ACTION(CopyImage2Dto3D)
            TEST_ACTION(CopyImage3Dto2D)
            TEST_ACTION(CopyImage3Dto3D)
            TEST_ACTION(Copy3DImageToBuffer)
            TEST_ACTION(CopyBufferTo3DImage)
        }
    }

    return retVal;
}

#define SIMUTANEOUS_ACTION_TOTAL 18
static bool sSimultaneousFlags[54]; // for 18 actions with 3 callback status
static volatile int sSimultaneousCount;

Action *actions[19] = { 0 };

// Callback for the simultaneous tests
void CL_CALLBACK simultaneous_event_callback_function(cl_event event,
                                                      cl_int commandStatus,
                                                      void *userData)
{
    int eventIndex = (int)(size_t)userData;
    int actionIndex = eventIndex / EVENT_CALLBACK_TYPE_TOTAL;
    int statusIndex = eventIndex % EVENT_CALLBACK_TYPE_TOTAL;
    log_info("\tEvent callback triggered for action %s callback type %s \n",
             actions[actionIndex]->GetName(), IGetStatusString(statusIndex));
    sSimultaneousFlags[actionIndex] = true;
    ThreadPool_AtomicAdd(&sSimultaneousCount, 1);
}

int test_callbacks_simultaneous(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    cl_int error;

    // Unlike the singles test, in this one, we run a bunch of events all at
    // once, to verify that the callbacks do get called once-and-only-once for
    // each event, even if the run out of order or are dependent on each other

    // First, the list of actions to run
    int actionCount = 0, index = 0;

    actions[index++] = new NDRangeKernelAction();
    actions[index++] = new ReadBufferAction();
    actions[index++] = new WriteBufferAction();
    actions[index++] = new MapBufferAction();
    actions[index++] = new UnmapBufferAction();

    if (checkForImageSupport(deviceID) != CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
        actions[index++] = new ReadImage2DAction();
        actions[index++] = new WriteImage2DAction();
        actions[index++] = new CopyImage2Dto2DAction();
        actions[index++] = new Copy2DImageToBufferAction();
        actions[index++] = new CopyBufferTo2DImageAction();
        actions[index++] = new MapImageAction();

        if (checkFor3DImageSupport(deviceID) != CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
            actions[index++] = new ReadImage3DAction();
            actions[index++] = new WriteImage3DAction();
            actions[index++] = new CopyImage2Dto3DAction();
            actions[index++] = new CopyImage3Dto2DAction();
            actions[index++] = new CopyImage3Dto3DAction();
            actions[index++] = new Copy3DImageToBufferAction();
            actions[index++] = new CopyBufferTo3DImageAction();
        }
    }
    actionCount = index;
    actions[index++] = NULL;

    // Now set them all up
    log_info("\tSetting up test events...\n");
    for (index = 0; actions[index] != NULL; index++)
    {
        error = actions[index]->Setup(deviceID, context, queue);
        test_error(error, "Unable to set up test action");
        sSimultaneousFlags[index] = false;
    }
    sSimultaneousCount = 0;

    // Set up the user event to start them all
    clEventWrapper gateEvent = clCreateUserEvent(context, &error);
    test_error(error, "Unable to set up user gate event");

    // Start executing, all tied to the gate event
    // clEventWrapper actionEvents[ 18 ];// current actionCount is 18
    clEventWrapper *actionEvents = new clEventWrapper[actionCount];
    if (actionEvents == NULL)
    {
        log_error(" memory error in test_callbacks_simultaneous  \n");
        for (size_t i = 0; i < (sizeof(actions) / sizeof(actions[0])); ++i)
            if (actions[i]) delete actions[i];
        return -1;
    }

    RandomSeed seed(gRandomSeed);
    for (index = 0; actions[index] != NULL; index++)
    {
        // Randomly choose to wait on the gate, or wait on the previous event
        cl_event *eventPtr = &gateEvent;
        if ((index > 0) && (random_in_range(0, 255, seed) & 1))
            eventPtr = &actionEvents[index - 1];

        error =
            actions[index]->Execute(queue, 1, eventPtr, &actionEvents[index]);
        test_error(error, "Unable to execute test action");


        for (int k = 0; k < EVENT_CALLBACK_TYPE_TOTAL; k++)
        {
            error = clSetEventCallback(
                actionEvents[index], event_callback_types[k],
                simultaneous_event_callback_function,
                (void *)(size_t)(index * EVENT_CALLBACK_TYPE_TOTAL + k));
            test_error(error, "Unable to set event callback function");
        }
    }

    int total_callbacks = actionCount * EVENT_CALLBACK_TYPE_TOTAL;

    // Now release the user event, which will allow our actual action to run
    error = clSetUserEventStatus(gateEvent, CL_COMPLETE);
    test_error(error, "Unable to trigger gate event");

    // Wait on the actual action events now
    log_info("\tWaiting for test completions...\n");
    error = clWaitForEvents(actionCount, &actionEvents[0]);
    test_error(error, "Unable to wait for actual test events");

    // Note: we can check our callback now, and it MIGHT have been triggered,
    // but that's not guaranteed
    int last_count = 0;
    if (((last_count = sSimultaneousCount)) == total_callbacks)
    {
        // We're all good, so return success
        log_info("\t%d of %d callbacks received\n", sSimultaneousCount,
                 total_callbacks);

        if (actionEvents) delete[] actionEvents;
        for (size_t i = 0; i < (sizeof(actions) / sizeof(actions[0])); ++i)
            if (actions[i]) delete actions[i];
        return 0;
    }

    // We haven't gotten (all) of the callbacks, so wait for them
    log_info("\tWe've only received %d of the %d callbacks we expected; "
             "waiting for more...\n",
             last_count, total_callbacks);

    for (int i = 0; i < 10 * 10; i++)
    {
        usleep(100000); // 1/10th second
        if (((last_count = sSimultaneousCount)) == total_callbacks)
        {
            // All of the callbacks were executed
            if (actionEvents) delete[] actionEvents;
            for (size_t i = 0; i < (sizeof(actions) / sizeof(actions[0])); ++i)
                if (actions[i]) delete actions[i];
            return 0;
        }
    }

    // If we got here, some of the callbacks did not occur in time
    log_error("\nError: We only ever received %d of our %d callbacks!\n",
              last_count, total_callbacks);
    log_error("Events that did not receive callbacks:\n");
    for (index = 0; actions[index] != NULL; index++)
    {
        if (!sSimultaneousFlags[index])
            log_error("\t%s\n", actions[index]->GetName());
    }

    if (actionEvents) delete[] actionEvents;
    return -1;
}

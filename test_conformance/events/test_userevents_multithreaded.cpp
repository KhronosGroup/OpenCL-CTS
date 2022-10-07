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

#include <thread>

#if !defined(_MSC_VER)
#include <unistd.h>
#endif // !_MSC_VER

void trigger_user_event(cl_event *event)
{
    usleep(1000000);
    log_info("\tTriggering gate from separate thread...\n");
    clSetUserEventStatus(*event, CL_COMPLETE);
}

int test_userevents_multithreaded(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    cl_int error;


    // Set up a user event to act as a gate
    clEventWrapper gateEvent = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create user gate event");

    // Set up a few actions gated on the user event
    NDRangeKernelAction action1;
    ReadBufferAction action2;
    WriteBufferAction action3;

    clEventWrapper actionEvents[3];
    Action *actions[] = { &action1, &action2, &action3, NULL };

    for (int i = 0; actions[i] != NULL; i++)
    {
        error = actions[i]->Setup(deviceID, context, queue);
        test_error(error, "Unable to set up test action");

        error = actions[i]->Execute(queue, 1, &gateEvent, &actionEvents[i]);
        test_error(error, "Unable to execute test action");
    }

    // Now, instead of releasing the gate, we spawn a separate thread to do so
    log_info("\tStarting trigger thread...\n");
    std::thread thread(trigger_user_event, &gateEvent);

    log_info("\tWaiting for actions...\n");
    error = clWaitForEvents(3, &actionEvents[0]);
    test_error(error, "Unable to wait for action events");

    thread.join();
    log_info("\tActions completed.\n");

    // If we got here without error, we're good
    return 0;
}

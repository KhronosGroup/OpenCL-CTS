//
// Copyright (c) 2020 The Khronos Group Inc.
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

static volatile cl_int sDestructorIndex;

void CL_CALLBACK context_destructor_callback(cl_context context, void *userData)
{
    int *userPtr = (int *)userData;

    // ordering of callbacks is guaranteed, meaning we don't need to do atomic
    // operation here
    *userPtr = ++sDestructorIndex;
}

REGISTER_TEST_VERSION(context_destructor_callback, Version(3, 0))
{
    cl_int error;
    clContextWrapper localContext =
        clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    test_error(error, "Unable to create local context");

    // Set up some variables to catch the order in which callbacks are called
    volatile int callbackOrders[3] = { 0, 0, 0 };
    sDestructorIndex = 0;

    // Set up the callbacks
    error = clSetContextDestructorCallback(
        localContext, context_destructor_callback, (void *)&callbackOrders[0]);
    test_error(error, "Unable to set destructor callback");

    error = clSetContextDestructorCallback(
        localContext, context_destructor_callback, (void *)&callbackOrders[1]);
    test_error(error, "Unable to set destructor callback");

    error = clSetContextDestructorCallback(
        localContext, context_destructor_callback, (void *)&callbackOrders[2]);
    test_error(error, "Unable to set destructor callback");

    // Now release the context, which SHOULD call the callbacks
    localContext.reset();

    // At this point, all three callbacks should have already been called
    int numErrors = 0;
    for (int i = 0; i < 3; i++)
    {
        // Spin waiting for the release to finish.  If you don't call the
        // context_destructor_callback, you will not pass the test.
        log_info("\tWaiting for callback %d...\n", i);
        int wait = 0;
        while (0 == callbackOrders[i])
        {
            usleep(100000); // 1/10th second
            if (++wait >= 10 * 10)
            {
                log_error("\tERROR: Callback %d was not called within 10 "
                          "seconds!  Assuming failure.\n",
                          i + 1);
                numErrors++;
                break;
            }
        }

        if (callbackOrders[i] != 3 - i)
        {
            log_error("\tERROR: Callback %d was called in the wrong order! "
                      "(Was called order %d, should have been order %d)\n",
                      i + 1, callbackOrders[i], 3 - i);
            numErrors++;
        }
    }

    return (numErrors > 0) ? TEST_FAIL : TEST_PASS;
}

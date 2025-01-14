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

void CL_CALLBACK mem_destructor_callback(cl_mem memObject, void *userData)
{
    int *userPtr = (int *)userData;

    // ordering of callbacks is guaranteed, meaning we don't need to do atomic
    // operation here
    *userPtr = ++sDestructorIndex;
}

int test_mem_object_destructor_callback_single(clMemWrapper &memObject)
{
    cl_int error;

    // Set up some variables to catch the order in which callbacks are called
    volatile int callbackOrders[3] = { 0, 0, 0 };
    sDestructorIndex = 0;

    // Set up the callbacks
    error = clSetMemObjectDestructorCallback(memObject, mem_destructor_callback,
                                             (void *)&callbackOrders[0]);
    test_error(error, "Unable to set destructor callback");

    error = clSetMemObjectDestructorCallback(memObject, mem_destructor_callback,
                                             (void *)&callbackOrders[1]);
    test_error(error, "Unable to set destructor callback");

    error = clSetMemObjectDestructorCallback(memObject, mem_destructor_callback,
                                             (void *)&callbackOrders[2]);
    test_error(error, "Unable to set destructor callback");

    // Now release the buffer, which SHOULD call the callbacks
    memObject.reset();

    // At this point, all three callbacks should have already been called
    int numErrors = 0;
    for (int i = 0; i < 3; i++)
    {
        // Spin waiting for the release to finish.  If you don't call the
        // mem_destructor_callback, you will not pass the test.  bugzilla 6316
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

REGISTER_TEST(mem_object_destructor_callback)
{
    clMemWrapper testBuffer, testImage;
    cl_int error;


    // Create a buffer and an image to test callbacks against
    testBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024, NULL, &error);
    test_error(error, "Unable to create testing buffer");

    if (test_mem_object_destructor_callback_single(testBuffer) != TEST_PASS)
    {
        log_error("ERROR: Destructor callbacks for buffer object FAILED\n");
        return TEST_FAIL;
    }

    if (checkForImageSupport(device) == 0)
    {
        cl_image_format imageFormat = { CL_RGBA, CL_SIGNED_INT8 };
        testImage = create_image_2d(context, CL_MEM_READ_ONLY, &imageFormat, 16,
                                    16, 0, NULL, &error);
        test_error(error, "Unable to create testing image");

        if (test_mem_object_destructor_callback_single(testImage) != TEST_PASS)
        {
            log_error("ERROR: Destructor callbacks for image object FAILED\n");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

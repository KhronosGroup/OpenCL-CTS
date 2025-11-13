// Copyright (c) 2022 The Khronos Group Inc.
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
#include "harness/testHarness.h"

int main(int argc, const char *argv[])
{
    // A device may report the required properties of a queue that
    // is compatible with command-buffers via the query
    // CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR. We account
    // for this in the tests themselves, rather than here, where we have a
    // device to query.
    const cl_command_queue_properties queue_properties = 0;
    return runTestHarnessWithCheck(argc, argv,
                                   test_registry::getInstance().num_tests(),
                                   test_registry::getInstance().definitions(),
                                   false, queue_properties, nullptr);
}

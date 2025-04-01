//
// Copyright (c) 2025 The Khronos Group Inc.
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
#include "basic_command_buffer.h"

REGISTER_TEST(multi_flag_creation)
{
    return MakeAndRunTest<MultiFlagCreationTest>(device, context, queue,
                                                 num_elements);
}

REGISTER_TEST(single_ndrange)
{
    return MakeAndRunTest<BasicEnqueueTest>(device, context, queue,
                                            num_elements);
}

REGISTER_TEST(interleaved_enqueue)
{
    return MakeAndRunTest<InterleavedEnqueueTest>(device, context, queue,
                                                  num_elements);
}

REGISTER_TEST(mixed_commands)
{
    return MakeAndRunTest<MixedCommandsTest>(device, context, queue,
                                             num_elements);
}

REGISTER_TEST(explicit_flush)
{
    return MakeAndRunTest<ExplicitFlushTest>(device, context, queue,
                                             num_elements);
}

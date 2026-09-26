//
// Copyright (c) 2024 The Khronos Group Inc.
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

#pragma once

#include "basic_command_buffer.h"
#include <type_traits>

template <class TBase>
struct CommandBufferWithImmutableMemoryObjectsTest : public TBase
{
    using TBase::TBase;

    static_assert(std::is_base_of<BasicCommandBufferTest, TBase>::value,
                  "TBase must be BasicCommandBufferTest or a derived class");

    bool Skip() override
    {
        bool is_immutable_memory_objects_supported = is_extension_available(
            BasicCommandBufferTest::device, "cl_ext_immutable_memory_objects");

        return !is_immutable_memory_objects_supported || TBase::Skip();
    }
};

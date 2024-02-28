//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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

#pragma once

#ifndef _testBase_h
#define _testBase_h

#include "harness/compat.h"
#include "harness/rounding_mode.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

// Ensure max min macros are undefined - compilation issues when using min max from numeric_limits template class
#if defined( max )
    #undef max
#endif

#if defined( min )
    #undef min
#endif

#endif // _testBase_h

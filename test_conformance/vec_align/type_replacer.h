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
#include <stdlib.h>

size_t doReplace(char * dest, size_t destLength, const char * source,
          const char * stringToReplace1,  const char * replaceWith1,
          const char * stringToReplace2, const char * replaceWith2);

size_t doSingleReplace(char * dest, size_t destLength, const char * source,
               const char * stringToReplace, const char * replaceWith);

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
#ifndef MINGW_COMPAT_H
#define MINGW_COMPAT_H

#if defined(__MINGW32__)
char *basename(char *path);
#include <malloc.h>

#if defined(__MINGW64__)
//mingw-w64 doesnot have __mingw_aligned_malloc, instead it has _aligned_malloc
#define __mingw_aligned_malloc _aligned_malloc
#define __mingw_aligned_free _aligned_free
#include <stddef.h>
#endif //(__MINGW64__)

#endif //(__MINGW32__)
#endif // MINGW_COMPAT_H

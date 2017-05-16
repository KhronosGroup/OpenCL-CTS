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
#if defined(__MINGW32__)

#include "mingw_compat.h"
#include <stdio.h>
#include <string.h>

//This function is unavailable on various mingw compilers,
//especially 64 bit so implementing it here
const char *basename_dot=".";
char*
basename(char *path)
{
    char *p = path, *b = NULL;
    int len = strlen(path);

    if (path == NULL) {
        return (char*)basename_dot;
    }

    // Not absolute path on windows
    if (path[1] != ':') {
        return path;
    }

    // Trim trailing path seperators
    if (path[len - 1]  == '\\' ||
        path[len - 1]  == '/' ) {
        len--;
        path[len] = '\0';
    }

    while (len) {
        while((*p != '\\' || *p != '/')  && len) {
            p++;
            len--;
        }
        p++;
        b = p;
     }

     return b;
}

#endif
//
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

#include "hostInfo.h"

/* Read phisicall host memory */
#if defined(unix) || defined(__unix__) || defined(__unix)

#include <sys/sysinfo.h>

cl_ulong get_host_physicall_memory()
{
    cl_ulong physical_memory = 0;
    static struct sysinfo s_info;
    sysinfo(&s_info);
    physical_memory = s_info.totalram;
    return physical_memory;
}

#endif

#if defined(_WIN64) || defined(_WIN64) || defined(_WIN32)

#include <windows.h>

cl_ulong get_host_physicall_memory()
{
    cl_ulong physical_memory = 0;
    GetPhysicallyInstalledSystemMemory(&physical_memory);
    physical_memory *= 1024;
    return physical_memory;
}

#endif
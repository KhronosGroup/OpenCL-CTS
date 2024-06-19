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
#ifndef _extensionHelpers_h
#define _extensionHelpers_h

// Load a specific function that is part of an OpenCL extension
#define GET_PFN(device, fn_name)                                               \
    fn_name##_fn fn_name = reinterpret_cast<fn_name##_fn>(                     \
        clGetExtensionFunctionAddressForPlatform(                              \
            getPlatformFromDevice(device), #fn_name));                         \
    do                                                                         \
    {                                                                          \
        if (!fn_name)                                                          \
        {                                                                      \
            log_error(                                                         \
                "ERROR: Failed to get function pointer for %s at %s:%d\n",     \
                #fn_name, __FILE__, __LINE__);                                 \
            return TEST_FAIL;                                                  \
        }                                                                      \
    } while (false)


#endif // _extensionHelpers_h

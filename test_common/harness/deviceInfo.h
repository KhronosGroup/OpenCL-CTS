//
// Copyright (c) 2017-2019 The Khronos Group Inc.
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
#ifndef _deviceInfo_h
#define _deviceInfo_h

// Configuration
#include "../config.hpp"

#include <string>

#include <CL/opencl.h>

/* Helper to return a string containing device information for the specified
 * device info parameter. */
std::string get_device_info_string(cl_device_id device,
                                   cl_device_info param_name);

/* Determines if an extension is supported by a device. */
int is_extension_available(cl_device_id device, const char *extensionName);

/* Returns the version of the extension the device supports or throws an
 * exception if the extension is not supported by the device. */
cl_version get_extension_version(cl_device_id device,
                                 const char *extensionName);

/* Returns a string containing the supported extensions list for a device. */
std::string get_device_extensions_string(cl_device_id device);

/* Returns a string containing the supported IL version(s) for a device. */
std::string get_device_il_version_string(cl_device_id device);

/* Returns a string containing the supported OpenCL version for a device. */
std::string get_device_version_string(cl_device_id device);

/* Returns a string containing the device name. */
std::string get_device_name(cl_device_id device);

// Returns the maximum size in bytes for Kernel Parameters
size_t get_max_param_size(cl_device_id device);

#endif // _deviceInfo_h

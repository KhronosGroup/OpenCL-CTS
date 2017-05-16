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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_SUPPORTED_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_SUPPORTED_HPP

#include "../common.hpp"

// Returns true if type is supported by device; otherwise - false;
template<class Type>
bool type_supported(cl_device_id device)
{
    (void) device;
    return false;
}

#define ADD_SUPPORTED_TYPE(Type) \
    template<> \
    bool type_supported<Type>(cl_device_id device) \
    { \
        (void) device; \
        return true; \
    }

ADD_SUPPORTED_TYPE(cl_char)
ADD_SUPPORTED_TYPE(cl_uchar)
ADD_SUPPORTED_TYPE(cl_short)
ADD_SUPPORTED_TYPE(cl_ushort)
ADD_SUPPORTED_TYPE(cl_int)
ADD_SUPPORTED_TYPE(cl_uint)

// ulong
template<>
bool type_supported<cl_ulong>(cl_device_id device)
{
    // long types do not have to be supported in EMBEDDED_PROFILE.
    char profile[128];
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), (void *)&profile, NULL);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: clGetDeviceInfo failed with CL_DEVICE_PROFILE\n");
        return false;
    }

    if (std::strcmp(profile, "EMBEDDED_PROFILE") == 0)
        return is_extension_available(device, "cles_khr_int64");

    return true;
}
// long
template<>
bool type_supported<cl_long>(cl_device_id device)
{
    return type_supported<cl_ulong>(device);
}
ADD_SUPPORTED_TYPE(cl_float)
// double
template<>
bool type_supported<cl_double>(cl_device_id device)
{
    return is_extension_available(device, "cl_khr_fp64");
}

#define ADD_SUPPORTED_VEC_TYPE1(Type, n) \
    template<> \
    bool type_supported<Type ## n>(cl_device_id device) \
    { \
        return type_supported<Type>(device); \
    }

#define ADD_SUPPORTED_VEC_TYPE2(Type) \
    ADD_SUPPORTED_VEC_TYPE1(Type, 2) \
    ADD_SUPPORTED_VEC_TYPE1(Type, 4) \
    ADD_SUPPORTED_VEC_TYPE1(Type, 8) \
    ADD_SUPPORTED_VEC_TYPE1(Type, 16)

ADD_SUPPORTED_VEC_TYPE2(cl_char)
ADD_SUPPORTED_VEC_TYPE2(cl_uchar)
ADD_SUPPORTED_VEC_TYPE2(cl_short)
ADD_SUPPORTED_VEC_TYPE2(cl_ushort)
ADD_SUPPORTED_VEC_TYPE2(cl_int)
ADD_SUPPORTED_VEC_TYPE2(cl_uint)
ADD_SUPPORTED_VEC_TYPE2(cl_long)
ADD_SUPPORTED_VEC_TYPE2(cl_ulong)
ADD_SUPPORTED_VEC_TYPE2(cl_float)
// ADD_SUPPORTED_VEC_TYPE2(cl_double)

#undef ADD_SUPPORTED_VEC_TYPE2
#undef ADD_SUPPORTED_VEC_TYPE1
#undef ADD_SUPPORTED_TYPE

#endif // TEST_CONFORMANCE_CLCPP_UTILS_COMMON_TYPE_SUPPORTED_HPP

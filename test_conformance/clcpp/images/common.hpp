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
#ifndef TEST_CONFORMANCE_CLCPP_IMAGES_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_IMAGES_COMMON_HPP

#include <type_traits>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include "../harness/imageHelpers.h"


namespace detail
{

template<cl_channel_type channel_type>
struct channel_info;

template<>
struct channel_info<CL_SIGNED_INT8>
{
    typedef cl_char channel_type;
    typedef cl_int4 element_type;
    static std::string function_suffix() { return "i"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_SIGNED_INT16>
{
    typedef cl_short channel_type;
    typedef cl_int4 element_type;
    static std::string function_suffix() { return "i"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_SIGNED_INT32>
{
    typedef cl_int channel_type;
    typedef cl_int4 element_type;
    static std::string function_suffix() { return "i"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_UNSIGNED_INT8>
{
    typedef cl_uchar channel_type;
    typedef cl_uint4 element_type;
    static std::string function_suffix() { return "ui"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_UNSIGNED_INT16>
{
    typedef cl_ushort channel_type;
    typedef cl_uint4 element_type;
    static std::string function_suffix() { return "ui"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_UNSIGNED_INT32>
{
    typedef cl_uint channel_type;
    typedef cl_uint4 element_type;
    static std::string function_suffix() { return "ui"; }

    channel_type channel_min() { return (std::numeric_limits<channel_type>::min)(); }
    channel_type channel_max() { return (std::numeric_limits<channel_type>::max)(); }
};

template<>
struct channel_info<CL_FLOAT>
{
    typedef cl_float channel_type;
    typedef cl_float4 element_type;
    static std::string function_suffix() { return "f"; }

    channel_type channel_min() { return -1e-3f; }
    channel_type channel_max() { return +1e+3f; }
};

template<cl_mem_object_type image_type>
struct image_info;

template<>
struct image_info<CL_MEM_OBJECT_IMAGE1D>
{
    static std::string image_type_name() { return "image1d"; }
    static std::string coord_accessor() { return "x"; }
};

template<>
struct image_info<CL_MEM_OBJECT_IMAGE2D>
{
    static std::string image_type_name() { return "image2d"; }
    static std::string coord_accessor() { return "xy"; }
};

template<>
struct image_info<CL_MEM_OBJECT_IMAGE3D>
{
    static std::string image_type_name() { return "image3d"; }
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    static std::string coord_accessor() { return "xyzw"; }
#else
    static std::string coord_accessor() { return "xyz"; }
#endif
};

} // namespace

template<cl_mem_object_type ImageType, cl_channel_type ChannelType>
struct image_test_base :
    detail::channel_info<ChannelType>,
    detail::image_info<ImageType>
{ };

// Create image_descriptor (used by harness/imageHelpers functions)
image_descriptor create_image_descriptor(cl_image_desc &image_desc, cl_image_format *image_format)
{
    image_descriptor image_info;
    image_info.width = image_desc.image_width;
    image_info.height = image_desc.image_height;
    image_info.depth = image_desc.image_depth;
    image_info.arraySize = image_desc.image_array_size;
    image_info.rowPitch = image_desc.image_row_pitch;
    image_info.slicePitch = image_desc.image_slice_pitch;
    image_info.format = image_format;
    image_info.buffer = image_desc.mem_object;
    image_info.type = image_desc.image_type;
    image_info.num_mip_levels = image_desc.num_mip_levels;
    return image_info;
}

const std::vector<cl_channel_order> get_channel_orders(cl_device_id device)
{
    // According to "Minimum List of Supported Image Formats" of OpenCL specification:
    return { CL_R, CL_RG, CL_RGBA };
}

bool is_test_supported(cl_device_id device)
{
    // Check for image support
    if (checkForImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
        log_info("SKIPPED: Device does not support images. Skipping test.\n");
        return false;
    }
    return true;
}

// Checks if x is equal to y.
template<class type>
inline bool are_equal(const type& x,
                      const type& y)
{
    for(size_t i = 0; i < vector_size<type>::value; i++)
    {
        if(!(x.s[i] == y.s[i]))
        {
            return false;
        }
    }
    return true;
}

#endif // TEST_CONFORMANCE_CLCPP_IMAGES_COMMON_HPP

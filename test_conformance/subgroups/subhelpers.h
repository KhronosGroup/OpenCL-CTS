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
#ifndef SUBHELPERS_H
#define SUBHELPERS_H

#include "testHarness.h"
#include "kernelHelpers.h"
#include "typeWrappers.h"
#include "imageHelpers.h"

#include <limits>
#include <vector>
#include <type_traits>
#include <bitset>
#include <regex>
#include <map>

extern MTdata gMTdata;
typedef std::bitset<128> bs128;
extern cl_half_rounding_mode g_rounding_mode;

bs128 cl_uint4_to_bs128(cl_uint4 v);
cl_uint4 bs128_to_cl_uint4(bs128 v);
cl_uint4 generate_bit_mask(cl_uint subgroup_local_id,
                           const std::string &mask_type,
                           cl_uint max_sub_group_size);

// limit possible input values to avoid arithmetic rounding/overflow issues.
// for each subgroup values defined different values
// for rest of workitems set 1 shuffle values
void fill_and_shuffle_safe_values(std::vector<cl_ulong> &safe_values,
                                  size_t sb_size);

struct WorkGroupParams
{

    WorkGroupParams(size_t gws, size_t lws, int dm_arg = -1, int cs_arg = -1)
        : global_workgroup_size(gws), local_workgroup_size(lws),
          divergence_mask_arg(dm_arg), cluster_size_arg(cs_arg)
    {
        subgroup_size = 0;
        cluster_size = 0;
        work_items_mask = 0;
        use_core_subgroups = true;
        dynsc = 0;
        load_masks();
    }
    size_t global_workgroup_size;
    size_t local_workgroup_size;
    size_t subgroup_size;
    cl_uint cluster_size;
    bs128 work_items_mask;
    size_t dynsc;
    bool use_core_subgroups;
    std::vector<bs128> all_work_item_masks;
    int divergence_mask_arg;
    int cluster_size_arg;
    void save_kernel_source(const std::string &source, std::string name = "")
    {
        if (name == "")
        {
            name = "default";
        }
        if (kernel_function_name.find(name) != kernel_function_name.end())
        {
            log_info("Kernel definition duplication. Source will be "
                     "overwritten for function name %s\n",
                     name.c_str());
        }
        kernel_function_name[name] = source;
    };
    // return specific defined kernel or default.
    std::string get_kernel_source(std::string name)
    {
        if (kernel_function_name.find(name) == kernel_function_name.end())
        {
            return kernel_function_name["default"];
        }
        return kernel_function_name[name];
    }


private:
    std::map<std::string, std::string> kernel_function_name;
    void load_masks()
    {
        if (divergence_mask_arg != -1)
        {
            // 1 in string will be set 1, 0 will be set 0
            bs128 mask_0xf0f0f0f0("11110000111100001111000011110000"
                                  "11110000111100001111000011110000"
                                  "11110000111100001111000011110000"
                                  "11110000111100001111000011110000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0xf0f0f0f0);
            // 1 in string will be set 0, 0 will be set 1
            bs128 mask_0x0f0f0f0f("11110000111100001111000011110000"
                                  "11110000111100001111000011110000"
                                  "11110000111100001111000011110000"
                                  "11110000111100001111000011110000",
                                  128, '1', '0');
            all_work_item_masks.push_back(mask_0x0f0f0f0f);
            bs128 mask_0x5555aaaa("10101010101010101010101010101010"
                                  "10101010101010101010101010101010"
                                  "10101010101010101010101010101010"
                                  "10101010101010101010101010101010",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0x5555aaaa);
            bs128 mask_0xaaaa5555("10101010101010101010101010101010"
                                  "10101010101010101010101010101010"
                                  "10101010101010101010101010101010"
                                  "10101010101010101010101010101010",
                                  128, '1', '0');
            all_work_item_masks.push_back(mask_0xaaaa5555);
            // 0x0f0ff0f0
            bs128 mask_0x0f0ff0f0("00001111000011111111000011110000"
                                  "00001111000011111111000011110000"
                                  "00001111000011111111000011110000"
                                  "00001111000011111111000011110000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0x0f0ff0f0);
            // 0xff0000ff
            bs128 mask_0xff0000ff("11111111000000000000000011111111"
                                  "11111111000000000000000011111111"
                                  "11111111000000000000000011111111"
                                  "11111111000000000000000011111111",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0xff0000ff);
            // 0xff00ff00
            bs128 mask_0xff00ff00("11111111000000001111111100000000"
                                  "11111111000000001111111100000000"
                                  "11111111000000001111111100000000"
                                  "11111111000000001111111100000000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0xff00ff00);
            // 0x00ffff00
            bs128 mask_0x00ffff00("00000000111111111111111100000000"
                                  "00000000111111111111111100000000"
                                  "00000000111111111111111100000000"
                                  "00000000111111111111111100000000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0x00ffff00);
            // 0x80 1 workitem highest id for 8 subgroup size
            bs128 mask_0x80808080("10000000100000001000000010000000"
                                  "10000000100000001000000010000000"
                                  "10000000100000001000000010000000"
                                  "10000000100000001000000010000000",
                                  128, '0', '1');

            all_work_item_masks.push_back(mask_0x80808080);
            // 0x8000 1 workitem highest id for 16 subgroup size
            bs128 mask_0x80008000("10000000000000001000000000000000"
                                  "10000000000000001000000000000000"
                                  "10000000000000001000000000000000"
                                  "10000000000000001000000000000000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0x80008000);
            // 0x80000000 1 workitem highest id for 32 subgroup size
            bs128 mask_0x80000000("10000000000000000000000000000000"
                                  "10000000000000000000000000000000"
                                  "10000000000000000000000000000000"
                                  "10000000000000000000000000000000",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0x80000000);
            // 0x80000000 00000000 1 workitem highest id for 64 subgroup size
            // 0x80000000 1 workitem highest id for 32 subgroup size
            bs128 mask_0x8000000000000000("10000000000000000000000000000000"
                                          "00000000000000000000000000000000"
                                          "10000000000000000000000000000000"
                                          "00000000000000000000000000000000",
                                          128, '0', '1');

            all_work_item_masks.push_back(mask_0x8000000000000000);
            // 0x80000000 00000000 00000000 00000000 1 workitem highest id for
            // 128 subgroup size
            bs128 mask_0x80000000000000000000000000000000(
                "10000000000000000000000000000000"
                "00000000000000000000000000000000"
                "00000000000000000000000000000000"
                "00000000000000000000000000000000",
                128, '0', '1');
            all_work_item_masks.push_back(
                mask_0x80000000000000000000000000000000);

            bs128 mask_0xffffffff("11111111111111111111111111111111"
                                  "11111111111111111111111111111111"
                                  "11111111111111111111111111111111"
                                  "11111111111111111111111111111111",
                                  128, '0', '1');
            all_work_item_masks.push_back(mask_0xffffffff);
        }
    }
};

enum class SubgroupsBroadcastOp
{
    broadcast,
    broadcast_first,
    non_uniform_broadcast
};

enum class NonUniformVoteOp
{
    elect,
    all,
    any,
    all_equal
};

enum class BallotOp
{
    ballot,
    inverse_ballot,
    ballot_bit_extract,
    ballot_bit_count,
    ballot_inclusive_scan,
    ballot_exclusive_scan,
    ballot_find_lsb,
    ballot_find_msb,
    eq_mask,
    ge_mask,
    gt_mask,
    le_mask,
    lt_mask,
};

enum class ShuffleOp
{
    shuffle,
    shuffle_up,
    shuffle_down,
    shuffle_xor,
    rotate,
    clustered_rotate,
};

enum class ArithmeticOp
{
    add_,
    max_,
    min_,
    mul_,
    and_,
    or_,
    xor_,
    logical_and,
    logical_or,
    logical_xor
};

const char *const operation_names(ArithmeticOp operation);
const char *const operation_names(BallotOp operation);
const char *const operation_names(ShuffleOp operation);
const char *const operation_names(NonUniformVoteOp operation);
const char *const operation_names(SubgroupsBroadcastOp operation);

class subgroupsAPI {
public:
    subgroupsAPI(cl_platform_id platform, bool use_core_subgroups)
    {
        static_assert(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE
                          == CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                      "Enums have to be the same");
        static_assert(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE
                          == CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
                      "Enums have to be the same");
        if (use_core_subgroups)
        {
            _clGetKernelSubGroupInfo_ptr = &clGetKernelSubGroupInfo;
            clGetKernelSubGroupInfo_name = "clGetKernelSubGroupInfo";
        }
        else
        {
            _clGetKernelSubGroupInfo_ptr = (clGetKernelSubGroupInfoKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clGetKernelSubGroupInfoKHR");
            clGetKernelSubGroupInfo_name = "clGetKernelSubGroupInfoKHR";
        }
    }
    clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfo_ptr()
    {
        return _clGetKernelSubGroupInfo_ptr;
    }
    const char *clGetKernelSubGroupInfo_name;

private:
    clGetKernelSubGroupInfoKHR_fn _clGetKernelSubGroupInfo_ptr;
};

// Need to defined custom type for vector size = 3 and half type. This is
// because of 3-component types are otherwise indistinguishable from the
// 4-component types, and because the half type is indistinguishable from some
// other 16-bit type (ushort)
namespace subgroups {
struct cl_char3
{
    ::cl_char3 data;
};
struct cl_uchar3
{
    ::cl_uchar3 data;
};
struct cl_short3
{
    ::cl_short3 data;
};
struct cl_ushort3
{
    ::cl_ushort3 data;
};
struct cl_int3
{
    ::cl_int3 data;
};
struct cl_uint3
{
    ::cl_uint3 data;
};
struct cl_long3
{
    ::cl_long3 data;
};
struct cl_ulong3
{
    ::cl_ulong3 data;
};
struct cl_float3
{
    ::cl_float3 data;
};
struct cl_double3
{
    ::cl_double3 data;
};
struct cl_half
{
    ::cl_half data;
};
struct cl_half2
{
    ::cl_half2 data;
};
struct cl_half3
{
    ::cl_half3 data;
};
struct cl_half4
{
    ::cl_half4 data;
};
struct cl_half8
{
    ::cl_half8 data;
};
struct cl_half16
{
    ::cl_half16 data;
};
}

// Declare operator<< for cl_ types, accessing the .s member.
#define OP_OSTREAM(Ty, VecSize)                                                \
    std::ostream &operator<<(std::ostream &os, const Ty##VecSize &val);

// Declare operator<< for subgroups::cl_ types, accessing the .data member and
// forwarding to operator<< for the cl_ types.
#define OP_OSTREAM_SUBGROUP(Ty, VecSize)                                       \
    std::ostream &operator<<(std::ostream &os, const Ty##VecSize &val);

// Declare operator<< for all vector sizes.
#define OP_OSTREAM_ALL_VEC(Ty)                                                 \
    OP_OSTREAM(Ty, 2)                                                          \
    OP_OSTREAM(Ty, 4)                                                          \
    OP_OSTREAM(Ty, 8)                                                          \
    OP_OSTREAM(Ty, 16)                                                         \
    OP_OSTREAM_SUBGROUP(subgroups::Ty, 3)

OP_OSTREAM_ALL_VEC(cl_char)
OP_OSTREAM_ALL_VEC(cl_uchar)
OP_OSTREAM_ALL_VEC(cl_short)
OP_OSTREAM_ALL_VEC(cl_ushort)
OP_OSTREAM_ALL_VEC(cl_int)
OP_OSTREAM_ALL_VEC(cl_uint)
OP_OSTREAM_ALL_VEC(cl_long)
OP_OSTREAM_ALL_VEC(cl_ulong)
OP_OSTREAM_ALL_VEC(cl_float)
OP_OSTREAM_ALL_VEC(cl_double)
OP_OSTREAM_ALL_VEC(cl_half)
OP_OSTREAM_SUBGROUP(subgroups::cl_half, )
OP_OSTREAM_SUBGROUP(subgroups::cl_half, 2)
OP_OSTREAM_SUBGROUP(subgroups::cl_half, 4)
OP_OSTREAM_SUBGROUP(subgroups::cl_half, 8)
OP_OSTREAM_SUBGROUP(subgroups::cl_half, 16)

#undef OP_OSTREAM
#undef OP_OSTREAM_SUBGROUP
#undef OP_OSTREAM_ALL_VEC

template <typename Ty>
std::string print_expected_obtained(const Ty &expected, const Ty &obtained)
{
    std::ostringstream oss;
    oss << "Expected: " << expected << " Obtained: " << obtained;
    return oss.str();
}

static bool int64_ok(cl_device_id device)
{
    char profile[128];
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile),
                            (void *)&profile, NULL);
    if (error)
    {
        log_info("clGetDeviceInfo failed with CL_DEVICE_PROFILE\n");
        return false;
    }

    if (strcmp(profile, "EMBEDDED_PROFILE") == 0)
        return is_extension_available(device, "cles_khr_int64");

    return true;
}

static bool double_ok(cl_device_id device)
{
    int error;
    cl_device_fp_config c;
    error = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(c),
                            (void *)&c, NULL);
    if (error)
    {
        log_info("clGetDeviceInfo failed with CL_DEVICE_DOUBLE_FP_CONFIG\n");
        return false;
    }
    return c != 0;
}

static bool half_ok(cl_device_id device)
{
    int error;
    cl_device_fp_config c;
    error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(c),
                            (void *)&c, NULL);
    if (error)
    {
        log_info("clGetDeviceInfo failed with CL_DEVICE_HALF_FP_CONFIG\n");
        return false;
    }
    return c != 0;
}

template <typename Ty> struct CommonTypeManager
{

    static const char *name() { return ""; }
    static const char *add_typedef() { return "\n"; }
    typedef std::false_type is_vector_type;
    typedef std::false_type is_sb_vector_size3;
    typedef std::false_type is_sb_vector_type;
    typedef std::false_type is_sb_scalar_type;
    static const bool type_supported(cl_device_id) { return true; }
    static const Ty identify_limits(ArithmeticOp operation)
    {
        switch (operation)
        {
            case ArithmeticOp::add_: return (Ty)0;
            case ArithmeticOp::max_: return (std::numeric_limits<Ty>::min)();
            case ArithmeticOp::min_: return (std::numeric_limits<Ty>::max)();
            case ArithmeticOp::mul_: return (Ty)1;
            case ArithmeticOp::and_: return (Ty)~0;
            case ArithmeticOp::or_: return (Ty)0;
            case ArithmeticOp::xor_: return (Ty)0;
            default: log_error("Unknown operation request\n"); break;
        }
        return 0;
    }
};

template <typename> struct TypeManager;

template <> struct TypeManager<cl_int> : public CommonTypeManager<cl_int>
{
    static const char *name() { return "int"; }
    static const char *add_typedef() { return "typedef int Type;\n"; }
    static cl_int identify_limits(ArithmeticOp operation)
    {
        switch (operation)
        {
            case ArithmeticOp::add_: return (cl_int)0;
            case ArithmeticOp::max_:
                return (std::numeric_limits<cl_int>::min)();
            case ArithmeticOp::min_:
                return (std::numeric_limits<cl_int>::max)();
            case ArithmeticOp::mul_: return (cl_int)1;
            case ArithmeticOp::and_: return (cl_int)~0;
            case ArithmeticOp::or_: return (cl_int)0;
            case ArithmeticOp::xor_: return (cl_int)0;
            case ArithmeticOp::logical_and: return (cl_int)1;
            case ArithmeticOp::logical_or: return (cl_int)0;
            case ArithmeticOp::logical_xor: return (cl_int)0;
            default: log_error("Unknown operation request\n"); break;
        }
        return 0;
    }
};
template <> struct TypeManager<cl_int2> : public CommonTypeManager<cl_int2>
{
    static const char *name() { return "int2"; }
    static const char *add_typedef() { return "typedef int2 Type;\n"; }
    typedef std::true_type is_vector_type;
    using scalar_type = cl_int;
};
template <>
struct TypeManager<subgroups::cl_int3>
    : public CommonTypeManager<subgroups::cl_int3>
{
    static const char *name() { return "int3"; }
    static const char *add_typedef() { return "typedef int3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_int;
};
template <> struct TypeManager<cl_int4> : public CommonTypeManager<cl_int4>
{
    static const char *name() { return "int4"; }
    static const char *add_typedef() { return "typedef int4 Type;\n"; }
    using scalar_type = cl_int;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_int8> : public CommonTypeManager<cl_int8>
{
    static const char *name() { return "int8"; }
    static const char *add_typedef() { return "typedef int8 Type;\n"; }
    using scalar_type = cl_int;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_int16> : public CommonTypeManager<cl_int16>
{
    static const char *name() { return "int16"; }
    static const char *add_typedef() { return "typedef int16 Type;\n"; }
    using scalar_type = cl_int;
    typedef std::true_type is_vector_type;
};
// cl_uint
template <> struct TypeManager<cl_uint> : public CommonTypeManager<cl_uint>
{
    static const char *name() { return "uint"; }
    static const char *add_typedef() { return "typedef uint Type;\n"; }
};
template <> struct TypeManager<cl_uint2> : public CommonTypeManager<cl_uint2>
{
    static const char *name() { return "uint2"; }
    static const char *add_typedef() { return "typedef uint2 Type;\n"; }
    using scalar_type = cl_uint;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_uint3>
    : public CommonTypeManager<subgroups::cl_uint3>
{
    static const char *name() { return "uint3"; }
    static const char *add_typedef() { return "typedef uint3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_uint;
};
template <> struct TypeManager<cl_uint4> : public CommonTypeManager<cl_uint4>
{
    static const char *name() { return "uint4"; }
    static const char *add_typedef() { return "typedef uint4 Type;\n"; }
    using scalar_type = cl_uint;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_uint8> : public CommonTypeManager<cl_uint8>
{
    static const char *name() { return "uint8"; }
    static const char *add_typedef() { return "typedef uint8 Type;\n"; }
    using scalar_type = cl_uint;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_uint16> : public CommonTypeManager<cl_uint16>
{
    static const char *name() { return "uint16"; }
    static const char *add_typedef() { return "typedef uint16 Type;\n"; }
    using scalar_type = cl_uint;
    typedef std::true_type is_vector_type;
};
// cl_short
template <> struct TypeManager<cl_short> : public CommonTypeManager<cl_short>
{
    static const char *name() { return "short"; }
    static const char *add_typedef() { return "typedef short Type;\n"; }
};
template <> struct TypeManager<cl_short2> : public CommonTypeManager<cl_short2>
{
    static const char *name() { return "short2"; }
    static const char *add_typedef() { return "typedef short2 Type;\n"; }
    using scalar_type = cl_short;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_short3>
    : public CommonTypeManager<subgroups::cl_short3>
{
    static const char *name() { return "short3"; }
    static const char *add_typedef() { return "typedef short3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_short;
};
template <> struct TypeManager<cl_short4> : public CommonTypeManager<cl_short4>
{
    static const char *name() { return "short4"; }
    static const char *add_typedef() { return "typedef short4 Type;\n"; }
    using scalar_type = cl_short;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_short8> : public CommonTypeManager<cl_short8>
{
    static const char *name() { return "short8"; }
    static const char *add_typedef() { return "typedef short8 Type;\n"; }
    using scalar_type = cl_short;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<cl_short16> : public CommonTypeManager<cl_short16>
{
    static const char *name() { return "short16"; }
    static const char *add_typedef() { return "typedef short16 Type;\n"; }
    using scalar_type = cl_short;
    typedef std::true_type is_vector_type;
};
// cl_ushort
template <> struct TypeManager<cl_ushort> : public CommonTypeManager<cl_ushort>
{
    static const char *name() { return "ushort"; }
    static const char *add_typedef() { return "typedef ushort Type;\n"; }
};
template <>
struct TypeManager<cl_ushort2> : public CommonTypeManager<cl_ushort2>
{
    static const char *name() { return "ushort2"; }
    static const char *add_typedef() { return "typedef ushort2 Type;\n"; }
    using scalar_type = cl_ushort;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_ushort3>
    : public CommonTypeManager<subgroups::cl_ushort3>
{
    static const char *name() { return "ushort3"; }
    static const char *add_typedef() { return "typedef ushort3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_ushort;
};
template <>
struct TypeManager<cl_ushort4> : public CommonTypeManager<cl_ushort4>
{
    static const char *name() { return "ushort4"; }
    static const char *add_typedef() { return "typedef ushort4 Type;\n"; }
    using scalar_type = cl_ushort;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<cl_ushort8> : public CommonTypeManager<cl_ushort8>
{
    static const char *name() { return "ushort8"; }
    static const char *add_typedef() { return "typedef ushort8 Type;\n"; }
    using scalar_type = cl_ushort;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<cl_ushort16> : public CommonTypeManager<cl_ushort16>
{
    static const char *name() { return "ushort16"; }
    static const char *add_typedef() { return "typedef ushort16 Type;\n"; }
    using scalar_type = cl_ushort;
    typedef std::true_type is_vector_type;
};
// cl_char
template <> struct TypeManager<cl_char> : public CommonTypeManager<cl_char>
{
    static const char *name() { return "char"; }
    static const char *add_typedef() { return "typedef char Type;\n"; }
};
template <> struct TypeManager<cl_char2> : public CommonTypeManager<cl_char2>
{
    static const char *name() { return "char2"; }
    static const char *add_typedef() { return "typedef char2 Type;\n"; }
    using scalar_type = cl_char;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_char3>
    : public CommonTypeManager<subgroups::cl_char3>
{
    static const char *name() { return "char3"; }
    static const char *add_typedef() { return "typedef char3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_char;
};
template <> struct TypeManager<cl_char4> : public CommonTypeManager<cl_char4>
{
    static const char *name() { return "char4"; }
    static const char *add_typedef() { return "typedef char4 Type;\n"; }
    using scalar_type = cl_char;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_char8> : public CommonTypeManager<cl_char8>
{
    static const char *name() { return "char8"; }
    static const char *add_typedef() { return "typedef char8 Type;\n"; }
    using scalar_type = cl_char;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_char16> : public CommonTypeManager<cl_char16>
{
    static const char *name() { return "char16"; }
    static const char *add_typedef() { return "typedef char16 Type;\n"; }
    using scalar_type = cl_char;
    typedef std::true_type is_vector_type;
};
// cl_uchar
template <> struct TypeManager<cl_uchar> : public CommonTypeManager<cl_uchar>
{
    static const char *name() { return "uchar"; }
    static const char *add_typedef() { return "typedef uchar Type;\n"; }
};
template <> struct TypeManager<cl_uchar2> : public CommonTypeManager<cl_uchar2>
{
    static const char *name() { return "uchar2"; }
    static const char *add_typedef() { return "typedef uchar2 Type;\n"; }
    using scalar_type = cl_uchar;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_uchar3>
    : public CommonTypeManager<subgroups::cl_char3>
{
    static const char *name() { return "uchar3"; }
    static const char *add_typedef() { return "typedef uchar3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_uchar;
};
template <> struct TypeManager<cl_uchar4> : public CommonTypeManager<cl_uchar4>
{
    static const char *name() { return "uchar4"; }
    static const char *add_typedef() { return "typedef uchar4 Type;\n"; }
    using scalar_type = cl_uchar;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_uchar8> : public CommonTypeManager<cl_uchar8>
{
    static const char *name() { return "uchar8"; }
    static const char *add_typedef() { return "typedef uchar8 Type;\n"; }
    using scalar_type = cl_uchar;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<cl_uchar16> : public CommonTypeManager<cl_uchar16>
{
    static const char *name() { return "uchar16"; }
    static const char *add_typedef() { return "typedef uchar16 Type;\n"; }
    using scalar_type = cl_uchar;
    typedef std::true_type is_vector_type;
};
// cl_long
template <> struct TypeManager<cl_long> : public CommonTypeManager<cl_long>
{
    static const char *name() { return "long"; }
    static const char *add_typedef() { return "typedef long Type;\n"; }
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_long2> : public CommonTypeManager<cl_long2>
{
    static const char *name() { return "long2"; }
    static const char *add_typedef() { return "typedef long2 Type;\n"; }
    using scalar_type = cl_long;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_long3>
    : public CommonTypeManager<subgroups::cl_long3>
{
    static const char *name() { return "long3"; }
    static const char *add_typedef() { return "typedef long3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_long;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_long4> : public CommonTypeManager<cl_long4>
{
    static const char *name() { return "long4"; }
    static const char *add_typedef() { return "typedef long4 Type;\n"; }
    using scalar_type = cl_long;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_long8> : public CommonTypeManager<cl_long8>
{
    static const char *name() { return "long8"; }
    static const char *add_typedef() { return "typedef long8 Type;\n"; }
    using scalar_type = cl_long;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_long16> : public CommonTypeManager<cl_long16>
{
    static const char *name() { return "long16"; }
    static const char *add_typedef() { return "typedef long16 Type;\n"; }
    using scalar_type = cl_long;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
// cl_ulong
template <> struct TypeManager<cl_ulong> : public CommonTypeManager<cl_ulong>
{
    static const char *name() { return "ulong"; }
    static const char *add_typedef() { return "typedef ulong Type;\n"; }
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_ulong2> : public CommonTypeManager<cl_ulong2>
{
    static const char *name() { return "ulong2"; }
    static const char *add_typedef() { return "typedef ulong2 Type;\n"; }
    using scalar_type = cl_ulong;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_ulong3>
    : public CommonTypeManager<subgroups::cl_ulong3>
{
    static const char *name() { return "ulong3"; }
    static const char *add_typedef() { return "typedef ulong3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_ulong;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_ulong4> : public CommonTypeManager<cl_ulong4>
{
    static const char *name() { return "ulong4"; }
    static const char *add_typedef() { return "typedef ulong4 Type;\n"; }
    using scalar_type = cl_ulong;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <> struct TypeManager<cl_ulong8> : public CommonTypeManager<cl_ulong8>
{
    static const char *name() { return "ulong8"; }
    static const char *add_typedef() { return "typedef ulong8 Type;\n"; }
    using scalar_type = cl_ulong;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};
template <>
struct TypeManager<cl_ulong16> : public CommonTypeManager<cl_ulong16>
{
    static const char *name() { return "ulong16"; }
    static const char *add_typedef() { return "typedef ulong16 Type;\n"; }
    using scalar_type = cl_ulong;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return int64_ok(device);
    }
};

// cl_float
template <> struct TypeManager<cl_float> : public CommonTypeManager<cl_float>
{
    static const char *name() { return "float"; }
    static const char *add_typedef() { return "typedef float Type;\n"; }
    static cl_float identify_limits(ArithmeticOp operation)
    {
        switch (operation)
        {
            case ArithmeticOp::add_: return 0.0f;
            case ArithmeticOp::max_:
                return -std::numeric_limits<float>::infinity();
            case ArithmeticOp::min_:
                return std::numeric_limits<float>::infinity();
            case ArithmeticOp::mul_: return (cl_float)1;
            default: log_error("Unknown operation request\n"); break;
        }
        return 0;
    }
};
template <> struct TypeManager<cl_float2> : public CommonTypeManager<cl_float2>
{
    static const char *name() { return "float2"; }
    static const char *add_typedef() { return "typedef float2 Type;\n"; }
    using scalar_type = cl_float;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<subgroups::cl_float3>
    : public CommonTypeManager<subgroups::cl_float3>
{
    static const char *name() { return "float3"; }
    static const char *add_typedef() { return "typedef float3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_float;
};
template <> struct TypeManager<cl_float4> : public CommonTypeManager<cl_float4>
{
    static const char *name() { return "float4"; }
    static const char *add_typedef() { return "typedef float4 Type;\n"; }
    using scalar_type = cl_float;
    typedef std::true_type is_vector_type;
};
template <> struct TypeManager<cl_float8> : public CommonTypeManager<cl_float8>
{
    static const char *name() { return "float8"; }
    static const char *add_typedef() { return "typedef float8 Type;\n"; }
    using scalar_type = cl_float;
    typedef std::true_type is_vector_type;
};
template <>
struct TypeManager<cl_float16> : public CommonTypeManager<cl_float16>
{
    static const char *name() { return "float16"; }
    static const char *add_typedef() { return "typedef float16 Type;\n"; }
    using scalar_type = cl_float;
    typedef std::true_type is_vector_type;
};

// cl_double
template <> struct TypeManager<cl_double> : public CommonTypeManager<cl_double>
{
    static const char *name() { return "double"; }
    static const char *add_typedef() { return "typedef double Type;\n"; }
    static cl_double identify_limits(ArithmeticOp operation)
    {
        switch (operation)
        {
            case ArithmeticOp::add_: return 0.0;
            case ArithmeticOp::max_:
                return -std::numeric_limits<double>::infinity();
            case ArithmeticOp::min_:
                return std::numeric_limits<double>::infinity();
            case ArithmeticOp::mul_: return (cl_double)1;
            default: log_error("Unknown operation request\n"); break;
        }
        return 0;
    }
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};
template <>
struct TypeManager<cl_double2> : public CommonTypeManager<cl_double2>
{
    static const char *name() { return "double2"; }
    static const char *add_typedef() { return "typedef double2 Type;\n"; }
    using scalar_type = cl_double;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_double3>
    : public CommonTypeManager<subgroups::cl_double3>
{
    static const char *name() { return "double3"; }
    static const char *add_typedef() { return "typedef double3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = cl_double;
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};
template <>
struct TypeManager<cl_double4> : public CommonTypeManager<cl_double4>
{
    static const char *name() { return "double4"; }
    static const char *add_typedef() { return "typedef double4 Type;\n"; }
    using scalar_type = cl_double;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};
template <>
struct TypeManager<cl_double8> : public CommonTypeManager<cl_double8>
{
    static const char *name() { return "double8"; }
    static const char *add_typedef() { return "typedef double8 Type;\n"; }
    using scalar_type = cl_double;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};
template <>
struct TypeManager<cl_double16> : public CommonTypeManager<cl_double16>
{
    static const char *name() { return "double16"; }
    static const char *add_typedef() { return "typedef double16 Type;\n"; }
    using scalar_type = cl_double;
    typedef std::true_type is_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return double_ok(device);
    }
};

// cl_half
template <>
struct TypeManager<subgroups::cl_half>
    : public CommonTypeManager<subgroups::cl_half>
{
    static const char *name() { return "half"; }
    static const char *add_typedef() { return "typedef half Type;\n"; }
    typedef std::true_type is_sb_scalar_type;
    static subgroups::cl_half identify_limits(ArithmeticOp operation)
    {
        switch (operation)
        {
            case ArithmeticOp::add_: return { 0x0000 };
            case ArithmeticOp::max_: return { 0xfc00 };
            case ArithmeticOp::min_: return { 0x7c00 };
            case ArithmeticOp::mul_: return { 0x3c00 };
            default: log_error("Unknown operation request\n"); break;
        }
        return { 0 };
    }
    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_half2>
    : public CommonTypeManager<subgroups::cl_half2>
{
    static const char *name() { return "half2"; }
    static const char *add_typedef() { return "typedef half2 Type;\n"; }
    using scalar_type = subgroups::cl_half;
    typedef std::true_type is_sb_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_half3>
    : public CommonTypeManager<subgroups::cl_half3>
{
    static const char *name() { return "half3"; }
    static const char *add_typedef() { return "typedef half3 Type;\n"; }
    typedef std::true_type is_sb_vector_size3;
    using scalar_type = subgroups::cl_half;

    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_half4>
    : public CommonTypeManager<subgroups::cl_half4>
{
    static const char *name() { return "half4"; }
    static const char *add_typedef() { return "typedef half4 Type;\n"; }
    using scalar_type = subgroups::cl_half;
    typedef std::true_type is_sb_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_half8>
    : public CommonTypeManager<subgroups::cl_half8>
{
    static const char *name() { return "half8"; }
    static const char *add_typedef() { return "typedef half8 Type;\n"; }
    using scalar_type = subgroups::cl_half;
    typedef std::true_type is_sb_vector_type;

    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};
template <>
struct TypeManager<subgroups::cl_half16>
    : public CommonTypeManager<subgroups::cl_half16>
{
    static const char *name() { return "half16"; }
    static const char *add_typedef() { return "typedef half16 Type;\n"; }
    using scalar_type = subgroups::cl_half;
    typedef std::true_type is_sb_vector_type;
    static const bool type_supported(cl_device_id device)
    {
        return half_ok(device);
    }
};

// set scalar value to vector of halfs
template <typename Ty, int N = 0>
typename std::enable_if<TypeManager<Ty>::is_sb_vector_type::value>::type
set_value(Ty &lhs, const cl_ulong &rhs)
{
    const int size = sizeof(Ty) / sizeof(typename TypeManager<Ty>::scalar_type);
    for (auto i = 0; i < size; ++i)
    {
        lhs.data.s[i] = rhs;
    }
}


// set scalar value to vector
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_vector_type::value>::type
set_value(Ty &lhs, const cl_ulong &rhs)
{
    const int size = sizeof(Ty) / sizeof(typename TypeManager<Ty>::scalar_type);
    for (auto i = 0; i < size; ++i)
    {
        lhs.s[i] = rhs;
    }
}

// set vector to vector value
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_vector_type::value>::type
set_value(Ty &lhs, const Ty &rhs)
{
    lhs = rhs;
}

// set scalar value to vector size 3
template <typename Ty, int N = 0>
typename std::enable_if<TypeManager<Ty>::is_sb_vector_size3::value>::type
set_value(Ty &lhs, const cl_ulong &rhs)
{
    for (auto i = 0; i < 3; ++i)
    {
        lhs.data.s[i] = rhs;
    }
}

// set scalar value to scalar
template <typename Ty>
typename std::enable_if<std::is_scalar<Ty>::value>::type
set_value(Ty &lhs, const cl_ulong &rhs)
{
    lhs = static_cast<Ty>(rhs);
}

// set scalar value to half scalar
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_sb_scalar_type::value>::type
set_value(Ty &lhs, const cl_ulong &rhs)
{
    lhs.data = cl_half_from_float(static_cast<cl_float>(rhs), g_rounding_mode);
}

// compare for common vectors
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_vector_type::value, bool>::type
compare(const Ty &lhs, const Ty &rhs)
{
    const int size = sizeof(Ty) / sizeof(typename TypeManager<Ty>::scalar_type);
    for (auto i = 0; i < size; ++i)
    {
        if (lhs.s[i] != rhs.s[i])
        {
            return false;
        }
    }
    return true;
}

// compare for vectors 3
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_sb_vector_size3::value, bool>::type
compare(const Ty &lhs, const Ty &rhs)
{
    for (auto i = 0; i < 3; ++i)
    {
        if (lhs.data.s[i] != rhs.data.s[i])
        {
            return false;
        }
    }
    return true;
}

// compare for half vectors
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_sb_vector_type::value, bool>::type
compare(const Ty &lhs, const Ty &rhs)
{
    const int size = sizeof(Ty) / sizeof(typename TypeManager<Ty>::scalar_type);
    for (auto i = 0; i < size; ++i)
    {
        if (lhs.data.s[i] != rhs.data.s[i])
        {
            return false;
        }
    }
    return true;
}

// compare for scalars
template <typename Ty>
typename std::enable_if<std::is_scalar<Ty>::value, bool>::type
compare(const Ty &lhs, const Ty &rhs)
{
    return lhs == rhs;
}

// compare for scalar halfs
template <typename Ty>
typename std::enable_if<TypeManager<Ty>::is_sb_scalar_type::value, bool>::type
compare(const Ty &lhs, const Ty &rhs)
{
    return lhs.data == rhs.data;
}

template <typename Ty> inline bool compare_ordered(const Ty &lhs, const Ty &rhs)
{
    return lhs == rhs;
}

template <>
inline bool compare_ordered(const subgroups::cl_half &lhs,
                            const subgroups::cl_half &rhs)
{
    return cl_half_to_float(lhs.data) == cl_half_to_float(rhs.data);
}

template <typename Ty>
inline bool compare_ordered(const subgroups::cl_half &lhs, const int &rhs)
{
    return cl_half_to_float(lhs.data) == rhs;
}

template <typename Ty, typename Fns> class KernelExecutor {
public:
    KernelExecutor(cl_context c, cl_command_queue q, cl_kernel k, size_t g,
                   size_t l, Ty *id, size_t is, Ty *mid, Ty *mod, cl_int *md,
                   size_t ms, Ty *od, size_t os, size_t ts = 0)
        : context(c), queue(q), kernel(k), global(g), local(l), idata(id),
          isize(is), mapin_data(mid), mapout_data(mod), mdata(md), msize(ms),
          odata(od), osize(os), tsize(ts)
    {
        has_status = false;
        run_failed = false;
    }
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    size_t global;
    size_t local;
    Ty *idata;
    size_t isize;
    Ty *mapin_data;
    Ty *mapout_data;
    cl_int *mdata;
    size_t msize;
    Ty *odata;
    size_t osize;
    size_t tsize;
    bool run_failed;

private:
    bool has_status;
    test_status status;

public:
    // Run a test kernel to compute the result of a built-in on an input
    int run()
    {
        clMemWrapper in;
        clMemWrapper xy;
        clMemWrapper out;
        clMemWrapper tmp;
        int error;

        in = clCreateBuffer(context, CL_MEM_READ_ONLY, isize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        xy = clCreateBuffer(context, CL_MEM_WRITE_ONLY, msize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, osize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        if (tsize)
        {
            tmp = clCreateBuffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 tsize, NULL, &error);
            test_error(error, "clCreateBuffer failed");
        }

        error = clSetKernelArg(kernel, 0, sizeof(in), (void *)&in);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(xy), (void *)&xy);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 2, sizeof(out), (void *)&out);
        test_error(error, "clSetKernelArg failed");

        if (tsize)
        {
            error = clSetKernelArg(kernel, 3, sizeof(tmp), (void *)&tmp);
            test_error(error, "clSetKernelArg failed");
        }

        error = clEnqueueWriteBuffer(queue, in, CL_FALSE, 0, isize, idata, 0,
                                     NULL, NULL);
        test_error(error, "clEnqueueWriteBuffer failed");

        error = clEnqueueWriteBuffer(queue, xy, CL_FALSE, 0, msize, mdata, 0,
                                     NULL, NULL);
        test_error(error, "clEnqueueWriteBuffer failed");
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                       0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clEnqueueReadBuffer(queue, xy, CL_FALSE, 0, msize, mdata, 0,
                                    NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clEnqueueReadBuffer(queue, out, CL_FALSE, 0, osize, odata, 0,
                                    NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        return error;
    }

private:
    test_status
    run_and_check_with_cluster_size(const WorkGroupParams &test_params)
    {
        cl_int error = run();
        if (error != CL_SUCCESS)
        {
            print_error(error, "Failed to run subgroup test kernel");
            status = TEST_FAIL;
            run_failed = true;
            return status;
        }

        test_status tmp_status =
            Fns::chk(idata, odata, mapin_data, mapout_data, mdata, test_params);

        if (!has_status || tmp_status == TEST_FAIL
            || (tmp_status == TEST_PASS && status != TEST_FAIL))
        {
            status = tmp_status;
            has_status = true;
        }

        return status;
    }

public:
    test_status run_and_check(WorkGroupParams &test_params)
    {
        test_status tmp_status = TEST_SKIPPED_ITSELF;

        if (test_params.cluster_size_arg != -1)
        {
            for (cl_uint cluster_size = 1;
                 cluster_size <= test_params.subgroup_size; cluster_size *= 2)
            {
                test_params.cluster_size = cluster_size;
                cl_int error =
                    clSetKernelArg(kernel, test_params.cluster_size_arg,
                                   sizeof(cl_uint), &cluster_size);
                test_error_fail(error, "Unable to set cluster size");

                tmp_status = run_and_check_with_cluster_size(test_params);

                if (tmp_status == TEST_FAIL) break;
            }
        }
        else
        {
            tmp_status = run_and_check_with_cluster_size(test_params);
        }

        return tmp_status;
    }
};

// Driver for testing a single built in function
template <typename Ty, typename Fns, size_t TSIZE = 0> struct subgroup_test
{
    static test_status run(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements,
                           const char *kname, const char *src,
                           WorkGroupParams test_params)
    {
        size_t tmp;
        cl_int error;
        size_t subgroup_size, num_subgroups;
        size_t global = test_params.global_workgroup_size;
        size_t local = test_params.local_workgroup_size;
        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_platform_id platform;
        std::vector<cl_int> sgmap;
        sgmap.resize(4 * global);
        std::vector<Ty> mapin;
        mapin.resize(local);
        std::vector<Ty> mapout;
        mapout.resize(local);
        std::stringstream kernel_sstr;

        Fns::log_test(test_params, "");

        // Make sure a test of type Ty is supported by the device
        if (!TypeManager<Ty>::type_supported(device))
        {
            log_info("Data type not supported : %s\n", TypeManager<Ty>::name());
            return TEST_SKIPPED_ITSELF;
        }

        if (strstr(TypeManager<Ty>::name(), "double"))
        {
            kernel_sstr << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
        }
        else if (strstr(TypeManager<Ty>::name(), "half"))
        {
            kernel_sstr << "#pragma OPENCL EXTENSION cl_khr_fp16: enable\n";
        }

        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                (void *)&platform, NULL);
        test_error_fail(error, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");
        if (test_params.use_core_subgroups)
        {
            kernel_sstr
                << "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
        }
        kernel_sstr << "#define XY(M,I) M[I].x = get_sub_group_local_id(); "
                       "M[I].y = get_sub_group_id();\n";
        kernel_sstr << TypeManager<Ty>::add_typedef();
        kernel_sstr << src;
        const std::string &kernel_str = kernel_sstr.str();
        const char *kernel_src = kernel_str.c_str();

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &kernel_src, kname);
        if (error != CL_SUCCESS) return TEST_FAIL;

        // Determine some local dimensions to use for the test.
        error = get_max_common_work_group_size(
            context, kernel, test_params.global_workgroup_size, &local);
        test_error_fail(error, "get_max_common_work_group_size failed");

        // Limit it a bit so we have muliple work groups
        // Ideally this will still be large enough to give us multiple
        if (local > test_params.local_workgroup_size)
            local = test_params.local_workgroup_size;


        // Get the sub group info
        subgroupsAPI subgroupsApiSet(platform, test_params.use_core_subgroups);
        clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfo_ptr =
            subgroupsApiSet.clGetKernelSubGroupInfo_ptr();
        if (clGetKernelSubGroupInfo_ptr == NULL)
        {
            log_error("ERROR: %s function not available\n",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }
        error = clGetKernelSubGroupInfo_ptr(
            kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
            sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: %s function error for "
                      "CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE\n",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }

        subgroup_size = tmp;

        error = clGetKernelSubGroupInfo_ptr(
            kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
            sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: %s function error for "
                      "CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE\n",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }

        num_subgroups = tmp;
        // Make sure the number of sub groups is what we expect
        if (num_subgroups != (local + subgroup_size - 1) / subgroup_size)
        {
            log_error("ERROR: unexpected number of subgroups (%zu) returned\n",
                      num_subgroups);
            return TEST_FAIL;
        }

        std::vector<Ty> idata;
        std::vector<Ty> odata;
        size_t input_array_size = global;
        size_t output_array_size = global;
        size_t dynscl = test_params.dynsc;

        if (dynscl != 0)
        {
            input_array_size = global / local * num_subgroups * dynscl;
            output_array_size = global / local * dynscl;
        }

        idata.resize(input_array_size);
        odata.resize(output_array_size);

        if (test_params.divergence_mask_arg != -1)
        {
            cl_uint4 mask_vector;
            mask_vector.x = 0xffffffffU;
            mask_vector.y = 0xffffffffU;
            mask_vector.z = 0xffffffffU;
            mask_vector.w = 0xffffffffU;
            error = clSetKernelArg(kernel, test_params.divergence_mask_arg,
                                   sizeof(cl_uint4), &mask_vector);
            test_error_fail(error, "Unable to set divergence mask argument");
        }

        if (test_params.cluster_size_arg != -1)
        {
            cl_uint dummy_cluster_size = 1;
            error = clSetKernelArg(kernel, test_params.cluster_size_arg,
                                   sizeof(cl_uint), &dummy_cluster_size);
            test_error_fail(error, "Unable to set dummy cluster size");
        }

        KernelExecutor<Ty, Fns> executor(
            context, queue, kernel, global, local, idata.data(),
            input_array_size * sizeof(Ty), mapin.data(), mapout.data(),
            sgmap.data(), global * sizeof(cl_int4), odata.data(),
            output_array_size * sizeof(Ty), TSIZE * sizeof(Ty));

        // Run the kernel once on zeroes to get the map
        memset(idata.data(), 0, input_array_size * sizeof(Ty));
        error = executor.run();
        test_error_fail(error, "Running kernel first time failed");

        // Generate the desired input for the kernel
        test_params.subgroup_size = subgroup_size;
        Fns::gen(idata.data(), mapin.data(), sgmap.data(), test_params);

        test_status status;

        if (test_params.divergence_mask_arg != -1)
        {
            for (auto &mask : test_params.all_work_item_masks)
            {
                test_params.work_items_mask = mask;
                cl_uint4 mask_vector = bs128_to_cl_uint4(mask);
                clSetKernelArg(kernel, test_params.divergence_mask_arg,
                               sizeof(cl_uint4), &mask_vector);

                status = executor.run_and_check(test_params);

                if (status == TEST_FAIL) break;
            }
        }
        else
        {
            status = executor.run_and_check(test_params);
        }
        // Detailed failure and skip messages should be logged by
        // run_and_check.
        if (status == TEST_PASS)
        {
            Fns::log_test(test_params, " passed");
        }
        else if (!executor.run_failed && status == TEST_FAIL)
        {
            test_fail("Data verification failed\n");
        }
        return status;
    }
};

void set_last_workgroup_params(int non_uniform_size, int &number_of_subgroups,
                               int subgroup_size, int &workgroup_size,
                               int &last_subgroup_size);

template <typename Ty>
static void set_randomdata_for_subgroup(Ty *workgroup, int wg_offset,
                                        int current_sbs)
{
    int randomize_data = (int)(genrand_int32(gMTdata) % 3);
    // Initialize data matrix indexed by local id and sub group id
    switch (randomize_data)
    {
        case 0:
            memset(&workgroup[wg_offset], 0, current_sbs * sizeof(Ty));
            break;
        case 1: {
            memset(&workgroup[wg_offset], 0, current_sbs * sizeof(Ty));
            int wi_id = (int)(genrand_int32(gMTdata) % (cl_uint)current_sbs);
            set_value(workgroup[wg_offset + wi_id], 41);
        }
        break;
        case 2:
            memset(&workgroup[wg_offset], 0xff, current_sbs * sizeof(Ty));
            break;
    }
}

struct RunTestForType
{
    RunTestForType(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements,
                   WorkGroupParams test_params)
        : device_(device), context_(context), queue_(queue),
          num_elements_(num_elements), test_params_(test_params)
    {}
    template <typename T, typename U>
    int run_impl(const std::string &function_name)
    {
        int error = TEST_PASS;
        std::string source =
            std::regex_replace(test_params_.get_kernel_source(function_name),
                               std::regex("\\%s"), function_name);
        std::string kernel_name = "test_" + function_name;
        error = subgroup_test<T, U>::run(device_, context_, queue_,
                                         num_elements_, kernel_name.c_str(),
                                         source.c_str(), test_params_);

        // If we return TEST_SKIPPED_ITSELF here, then an entire suite may be
        // reported as having been skipped even if some tests within it
        // passed, as the status codes are erroneously ORed together:
        return error == TEST_FAIL ? TEST_FAIL : TEST_PASS;
    }

private:
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    int num_elements_;
    WorkGroupParams test_params_;
};

#endif

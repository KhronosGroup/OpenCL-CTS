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

#include "subhelpers.h"

#include <random>

// Define operator<< for cl_ types, accessing the .s member.
#define OP_OSTREAM(Ty, VecSize)                                                \
    std::ostream& operator<<(std::ostream& os, const Ty##VecSize& val)         \
    {                                                                          \
        os << +val.s[0]; /* unary plus forces char to be printed as number */  \
        for (unsigned i = 1; i < VecSize; i++)                                 \
        {                                                                      \
            os << ", " << +val.s[i];                                           \
        }                                                                      \
        return os;                                                             \
    }

// Define operator<< for subgroups::cl_ types, accessing the .data member and
// forwarding to operator<< for the cl_ types.
#define OP_OSTREAM_SUBGROUP(Ty, VecSize)                                       \
    std::ostream& operator<<(std::ostream& os, const Ty##VecSize& val)         \
    {                                                                          \
        return os << val.data;                                                 \
    }

// Define operator<< for all vector sizes.
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

bs128 cl_uint4_to_bs128(cl_uint4 v)
{
    return bs128(v.s0) | (bs128(v.s1) << 32) | (bs128(v.s2) << 64)
        | (bs128(v.s3) << 96);
}

cl_uint4 bs128_to_cl_uint4(bs128 v)
{
    bs128 bs128_ffffffff = 0xffffffffU;

    cl_uint4 r;
    r.s0 = ((v >> 0) & bs128_ffffffff).to_ulong();
    r.s1 = ((v >> 32) & bs128_ffffffff).to_ulong();
    r.s2 = ((v >> 64) & bs128_ffffffff).to_ulong();
    r.s3 = ((v >> 96) & bs128_ffffffff).to_ulong();

    return r;
}

cl_uint4 generate_bit_mask(cl_uint subgroup_local_id,
                           const std::string &mask_type,
                           cl_uint max_sub_group_size)
{
    bs128 mask128;
    cl_uint4 mask;
    cl_uint pos = subgroup_local_id;
    if (mask_type == "eq") mask128.set(pos);
    if (mask_type == "le" || mask_type == "lt")
    {
        for (cl_uint i = 0; i <= pos; i++) mask128.set(i);
        if (mask_type == "lt") mask128.reset(pos);
    }
    if (mask_type == "ge" || mask_type == "gt")
    {
        for (cl_uint i = pos; i < max_sub_group_size; i++) mask128.set(i);
        if (mask_type == "gt") mask128.reset(pos);
    }

    // convert std::bitset<128> to uint4
    auto const uint_mask = bs128{ static_cast<unsigned long>(-1) };
    mask.s0 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s1 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s2 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s3 = (mask128 & uint_mask).to_ulong();

    return mask;
}

const char *const operation_names(ArithmeticOp operation)
{
    switch (operation)
    {
        case ArithmeticOp::add_: return "add";
        case ArithmeticOp::max_: return "max";
        case ArithmeticOp::min_: return "min";
        case ArithmeticOp::mul_: return "mul";
        case ArithmeticOp::and_: return "and";
        case ArithmeticOp::or_: return "or";
        case ArithmeticOp::xor_: return "xor";
        case ArithmeticOp::logical_and: return "logical_and";
        case ArithmeticOp::logical_or: return "logical_or";
        case ArithmeticOp::logical_xor: return "logical_xor";
        default: log_error("Unknown operation request\n"); break;
    }
    return "";
}

const char *const operation_names(BallotOp operation)
{
    switch (operation)
    {
        case BallotOp::ballot: return "ballot";
        case BallotOp::inverse_ballot: return "inverse_ballot";
        case BallotOp::ballot_bit_extract: return "bit_extract";
        case BallotOp::ballot_bit_count: return "bit_count";
        case BallotOp::ballot_inclusive_scan: return "inclusive_scan";
        case BallotOp::ballot_exclusive_scan: return "exclusive_scan";
        case BallotOp::ballot_find_lsb: return "find_lsb";
        case BallotOp::ballot_find_msb: return "find_msb";
        case BallotOp::eq_mask: return "eq";
        case BallotOp::ge_mask: return "ge";
        case BallotOp::gt_mask: return "gt";
        case BallotOp::le_mask: return "le";
        case BallotOp::lt_mask: return "lt";
        default: log_error("Unknown operation request\n"); break;
    }
    return "";
}

const char *const operation_names(ShuffleOp operation)
{
    switch (operation)
    {
        case ShuffleOp::shuffle: return "shuffle";
        case ShuffleOp::shuffle_up: return "shuffle_up";
        case ShuffleOp::shuffle_down: return "shuffle_down";
        case ShuffleOp::shuffle_xor: return "shuffle_xor";
        case ShuffleOp::rotate: return "rotate";
        case ShuffleOp::clustered_rotate: return "clustered_rotate";
        default: log_error("Unknown operation request\n"); break;
    }
    return "";
}

const char *const operation_names(NonUniformVoteOp operation)
{
    switch (operation)
    {
        case NonUniformVoteOp::all: return "all";
        case NonUniformVoteOp::all_equal: return "all_equal";
        case NonUniformVoteOp::any: return "any";
        case NonUniformVoteOp::elect: return "elect";
        default: log_error("Unknown operation request\n"); break;
    }
    return "";
}

const char *const operation_names(SubgroupsBroadcastOp operation)
{
    switch (operation)
    {
        case SubgroupsBroadcastOp::broadcast: return "broadcast";
        case SubgroupsBroadcastOp::broadcast_first: return "broadcast_first";
        case SubgroupsBroadcastOp::non_uniform_broadcast:
            return "non_uniform_broadcast";
        default: log_error("Unknown operation request\n"); break;
    }
    return "";
}

void set_last_workgroup_params(int non_uniform_size, int &number_of_subgroups,
                               int subgroup_size, int &workgroup_size,
                               int &last_subgroup_size)
{
    number_of_subgroups = 1 + non_uniform_size / subgroup_size;
    last_subgroup_size = non_uniform_size % subgroup_size;
    workgroup_size = non_uniform_size;
}

void fill_and_shuffle_safe_values(std::vector<cl_ulong> &safe_values,
                                  int sb_size)
{
    // max product is 720, cl_half has enough precision for it
    const std::vector<cl_ulong> non_one_values{ 2, 3, 4, 5, 6 };

    if (sb_size <= non_one_values.size())
    {
        safe_values.assign(non_one_values.begin(),
                           non_one_values.begin() + sb_size);
    }
    else
    {
        safe_values.assign(sb_size, 1);
        std::copy(non_one_values.begin(), non_one_values.end(),
                  safe_values.begin());
    }

    std::mt19937 mersenne_twister_engine(10000);
    std::shuffle(safe_values.begin(), safe_values.end(),
                 mersenne_twister_engine);
}

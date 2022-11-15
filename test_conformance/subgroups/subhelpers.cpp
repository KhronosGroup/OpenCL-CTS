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

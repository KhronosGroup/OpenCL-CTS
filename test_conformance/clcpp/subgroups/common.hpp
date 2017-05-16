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
#ifndef TEST_CONFORMANCE_CLCPP_SG_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_SG_COMMON_HPP

#include <string>
#include <vector>
#include <limits>

enum class work_group_op : int {
    add, min, max
};

std::string to_string(work_group_op op)
{
    switch (op)
    {
        case work_group_op::add:
            return "add";
        case work_group_op::min:
            return "min";
        case work_group_op::max:
            return "max";
        default:
            break;
    }
    return "";
}

template <class CL_INT_TYPE, work_group_op op>
std::vector<CL_INT_TYPE> generate_input(size_t count, size_t wg_size)
{
    std::vector<CL_INT_TYPE> input(count, CL_INT_TYPE(1));
    switch (op)
    {
        case work_group_op::add:
            return input;
        case work_group_op::min:
            {
                size_t j = wg_size;
                for(size_t i = 0; i < count; i++)
                {
                    input[i] = static_cast<CL_INT_TYPE>(j);
                    j--;
                    if(j == 0)
                    {
                        j = wg_size;
                    }
                }
            }
            break;
        case work_group_op::max:
            {
                size_t j = 0;
                for(size_t i = 0; i < count; i++)
                {
                    input[i] = static_cast<CL_INT_TYPE>(j);
                    j++;
                    if(j == wg_size)
                    {
                        j = 0;
                    }
                }
            }
    }
    return input;
}

template <class CL_INT_TYPE, work_group_op op>
std::vector<CL_INT_TYPE> generate_output(size_t count, size_t wg_size)
{
    switch (op)
    {
        case work_group_op::add:
            return std::vector<CL_INT_TYPE>(count, CL_INT_TYPE(0));
        case work_group_op::min:
            return std::vector<CL_INT_TYPE>(count, (std::numeric_limits<CL_INT_TYPE>::max)());
        case work_group_op::max:
            return std::vector<CL_INT_TYPE>(count, (std::numeric_limits<CL_INT_TYPE>::min)());
    }
    return std::vector<CL_INT_TYPE>(count, CL_INT_TYPE(0));
}

#endif // TEST_CONFORMANCE_CLCPP_SG_COMMON_HPP

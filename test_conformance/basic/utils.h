//
// Copyright (c) 2023 The Khronos Group Inc.
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

#ifndef _BASIC_UTILS_H_
#define _BASIC_UTILS_H_

#include <memory>
#include <string>
#include <CL/cl_half.h>

extern cl_half_rounding_mode halfRoundingMode;

inline std::string concat_kernel(const char *sstr[], int num)
{
    std::string res;
    for (int i = 0; i < num; i++) res += std::string(sstr[i]);
    return res;
}

template <typename... Args>
inline std::string str_sprintf(const std::string &str, Args... args)
{
    int str_size = std::snprintf(nullptr, 0, str.c_str(), args...) + 1;
    if (str_size <= 0) throw std::runtime_error("Formatting error.");
    size_t s = static_cast<size_t>(str_size);
    std::unique_ptr<char[]> buffer(new char[s]);
    std::snprintf(buffer.get(), s, str.c_str(), args...);
    return std::string(buffer.get(), buffer.get() + s - 1);
}

#define HFF(num) cl_half_from_float(num, halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

#endif // _BASIC_UTIL_H_

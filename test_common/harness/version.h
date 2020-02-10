//
// Copyright (c) 2020 The Khronos Group Inc.
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

#ifndef HARNESS_VERSION_H
#define HARNESS_VERSION_H

#include <string>
#include <sstream>

#include <CL/cl.h>

class Version {
public:
    Version(): m_major(0), m_minor(0) {}
    Version(int major, int minor): m_major(major), m_minor(minor) {}
    bool operator>(const Version& rhs) const { return to_int() > rhs.to_int(); }
    bool operator<(const Version& rhs) const { return to_int() < rhs.to_int(); }
    bool operator<=(const Version& rhs) const
    {
        return to_int() <= rhs.to_int();
    }
    bool operator>=(const Version& rhs) const
    {
        return to_int() >= rhs.to_int();
    }
    bool operator==(const Version& rhs) const
    {
        return to_int() == rhs.to_int();
    }
    int to_int() const { return m_major * 10 + m_minor; }
    std::string to_string() const
    {
        std::stringstream ss;
        ss << m_major << "." << m_minor;
        return ss.str();
    }

private:
    int m_major;
    int m_minor;
};

Version get_device_cl_version(cl_device_id device);

#endif // #ifndef HARNESS_VERSION_H

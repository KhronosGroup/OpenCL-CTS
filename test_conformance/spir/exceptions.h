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
#ifndef __EXCEPTIONS_H
#define __EXCEPTIONS_H

#include <stdexcept>
#include "miniz/miniz.h"

namespace Exceptions
{
    /**
    Exception thrown on error in command line parameters
    */
    class CmdLineError : public std::runtime_error
    {
    public:
        CmdLineError (const std::string& msg): std::runtime_error(msg){}
    };

    /**
    Exception thrown on error in test run
    */
    class TestError : public std::runtime_error
    {
    public:
        TestError (const std::string& msg, int errorCode = 1): std::runtime_error(msg), m_errorCode(errorCode){}

        int getErrorCode() const { return m_errorCode; }
    private:
        int m_errorCode;
    };

    class ArchiveError : public std::runtime_error
    {
    public:
        ArchiveError(int errCode): std::runtime_error(mz_error(errCode)){}
    };
}

#endif
